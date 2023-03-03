import os
import wandb
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.losses import DiceCELoss
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

from base.base_modules import TensorBuffer, NegativeSamplingPixelContrastiveLoss
from base.base_segmentation import BaseSegmentationModel
from base.base_wandb_model import WandBModel
from models.networks import ProjectorUNet
from models.transform import FullAugmentor
from utils.metric_calculator import calculate_Dice_score, calculate_Hasudorff_distance, calculate_avg_surface_distance
from utils.iteration.iterator import PolynomialLRWithWarmUp, MetricMeter
from utils.ddp_utils import gather_object_across_processes


class SemiSupervisedContrastiveSegmentationModel(BaseSegmentationModel, WandBModel):
    """
    Proposed Rectified Contrastive Pseudo Supervision for semi-supervised medical image segmentation
    """
    def __init__(self, cfg, num_classes, amp=False):
        """
        Args:
            cfg: training configurations
            num_classes: number of classes
            amp: bool, whether to enable PyTorch native automatic mixed-precision training
        """
        BaseSegmentationModel.__init__(self, cfg, num_classes, amp)
        WandBModel.__init__(self, cfg)
        # define network
        self.network = ProjectorUNet(num_classes=num_classes, leaky=cfg['MODEL']['LEAKY'],
                                     norm=cfg['MODEL']['NORM']).to(self.device)
        self.network = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(self.network), device_ids=[self.device])
        # define loss function, optimizer and scheduler
        pos_weight = cfg['TRAIN']['CLASS_WEIGHT']
        lambda_ce = (1 + len(pos_weight)) / (1 + np.sum(pos_weight))
        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_ce=lambda_ce,
                                    ce_weight=torch.as_tensor([1] + pos_weight).to(self.device))
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=cfg['TRAIN']['LR'],
                                         weight_decay=cfg['TRAIN']['DECAY'], momentum=cfg['TRAIN']['MOMENTUM'])
        self.scheduler = PolynomialLRWithWarmUp(self.optimizer, total_steps=cfg['TRAIN']['EPOCHS'],
                                                   max_lr_steps=cfg['TRAIN']['BURN'],
                                                   warmup_steps=cfg['TRAIN']['BURN_IN'])
        # define augmentor to augment the inputs
        self.augmentor = FullAugmentor()
        # contrastive loss functions
        self.contrastive_loss = NegativeSamplingPixelContrastiveLoss(sample_num=cfg['TRAIN']['SAMPLE_NUM'],
                                                                     bidirectional=True, temperature=0.1)
        # deep supervision output list
        self.ds_list = ['level3', 'level2', 'level1', 'out']

        # visualization
        self.visual_pairs = [
            {'name': 'name_l', 'type': 'Pred', 'image': 'image_l', 'mask': 'pred_l'},
            {'name': 'name_l', 'type': 'GT', 'image': 'image_l', 'mask': 'label_l'},
            {'name': 'name_u', 'type': 'Pred', 'image': 'image_u', 'mask': 'pred_u'},
        ]
        self.loss_names = ['seg_loss', 'cps_l_loss', 'cps_u_loss', 'contrastive_l_loss', 'contrastive_u_loss',
                           'cosine_l_loss', 'cosine_u_loss']
        # wandb Table for val vis
        self.val_table = wandb.Table(columns=['ID'] + [pair['type'] for pair in self.visual_pairs])
        # define tensor buffer to calculate contrastive loss
        self.prepare_tensor_buffer()

    def prepare_tensor_buffer(self):
        """
        Define tensor buffers for negative sampling
        """
        self.project_l_negative = TensorBuffer(buffer_size=self.cfg['TRAIN']['BUFFER_SIZE'], concat_dim=0)
        self.project_u_negative = TensorBuffer(buffer_size=self.cfg['TRAIN']['BUFFER_SIZE'], concat_dim=0)
        self.map_l_negative = TensorBuffer(buffer_size=self.cfg['TRAIN']['BUFFER_SIZE'], concat_dim=0)
        self.map_u_negative = TensorBuffer(buffer_size=self.cfg['TRAIN']['BUFFER_SIZE'], concat_dim=0)

    def initialize_metric_meter(self, class_list):
        """
        Define training & validation loss/metric logger
        Args:
            class_list: the classes to be segmented
        """
        self.class_list = class_list
        self.metric_meter = MetricMeter(metrics=['dice', 'hd95', 'asd'], class_names=class_list, subject_names=['name_l'])
        self.train_loss = MetricMeter(metrics=self.loss_names, class_names=['train'])
        self.val_loss = MetricMeter(metrics=['loss'], class_names=['val'])

    def set_input(self, batch_data_labeled, batch_data_unlabeled):
        """
        Set input data during training
        Args:
            batch_data_labeled: labeled data batch, expects a dict
            batch_data_unlabeled: unlabeled data batch, expects a dict
        """
        self.image_l = batch_data_labeled['image'].to(self.device)
        self.label_l = batch_data_labeled['label'].to(self.device)
        self.name_l = batch_data_labeled['name']
        self.image_u = batch_data_unlabeled['image'].to(self.device)
        self.name_u = batch_data_unlabeled['name']

    def set_test_input(self, batch_data_labeled):
        """
        Set input data during testing
        Args:
            batch_data_labeled: labeled data batch, expects a dict
        """
        self.image_l = batch_data_labeled['image'].to(self.device)
        self.label_l = batch_data_labeled['label'].to(self.device)
        self.name_l = batch_data_labeled['name']

    def forward(self):
        """
        Perform basic forward process
        Returns:
            out: dict containing network outputs
        """
        out = self.network(self.image_l)
        self.pred_l = out['out']
        return out

    def predictor(self, inputs):
        """
        Predictor function for `monai.inferers.sliding_window_inference`
        Args:
            inputs: input data

        Returns:
            output: network final output
        """
        output = self.network(inputs)['out']
        return output

    def get_multi_loss(self, out_dict, label, is_ds=True, key_list=None):
        """
        Calculate the segmentation loss with deep supervision.
        Args:
            out_dict: dict of tensors, containing outputs
            label: ground truth tensor
            is_ds: bool, whether to calculate the deep supervision loss
            key_list: list, keys that will be used to calculate the deep supervision loss

        Returns:
            multi_loss: tensor, loss value
        """
        keys = key_list if key_list is not None else list(out_dict.keys())
        if is_ds:
            multi_loss = sum([self.criterion(out_dict[key], label) for key in keys])
        else:
            multi_loss = self.criterion(out_dict['out'], label)
        return multi_loss

    def sigmoid_rampup(self, current):
        """
        Exponential rampup from https://arxiv.org/abs/1610.02242
        Args:
            current: current epoch index
        """
        rampup_length = self.cfg['TRAIN']['RAMPUP']
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def deallocate_batch_dict(self, input_dict, batch_idx_start, batch_idx_end):
        """
        Deallocate the dict containing multiple batches into a dict with multiple items
        """
        out_dict = {}
        for key, value in input_dict.items():
            out_dict[key] = value[batch_idx_start:batch_idx_end]
        return out_dict

    def forward_semi(self):
        # prepare augmented views of the same images
        self.batch_size = self.image_l.shape[0]
        self.image_l_t1 = self.augmentor.forward_image(self.image_l)
        self.image_l_t2 = self.augmentor.forward_image(self.image_l)
        self.image_u_t1 = self.augmentor.forward_image(self.image_u)
        self.image_u_t2 = self.augmentor.forward_image(self.image_u)
        # In ddp training, the buffer of running mean and running variance in batch normalization is updated after
        # EACH forward pass, thus if you call backward after multiple forward pass, since the buffer has been modified,
        # an error will be thrown
        # forward and deallocate the dict outputs
        inputs = torch.cat([
            self.image_l, self.image_l_t1, self.image_l_t2, self.image_u, self.image_u_t1, self.image_u_t2], dim=0)
        outputs = self.network(inputs)
        self.out_l = self.deallocate_batch_dict(outputs, 0, self.batch_size)
        self.out_l_t1 = self.deallocate_batch_dict(outputs, self.batch_size, 2 * self.batch_size)
        self.out_l_t2 = self.deallocate_batch_dict(outputs, 2 * self.batch_size, 3 * self.batch_size)
        self.out_u = self.deallocate_batch_dict(outputs, 3 * self.batch_size, 4 * self.batch_size)
        self.out_u_t1 = self.deallocate_batch_dict(outputs, 4 * self.batch_size, 5 * self.batch_size)
        self.out_u_t2 = self.deallocate_batch_dict(outputs, 5 * self.batch_size, 6 * self.batch_size)
        # define predictions
        self.pred_l = self.out_l['out']  # pseudo label for labeled data
        self.pred_l1 = self.out_l_t1['out']
        self.pred_l2 = self.out_l_t2['out']
        self.pred_u = self.out_u['out']  # pseudo label for unlabeled data
        self.pred_u1 = self.out_u_t1['out']
        self.pred_u2 = self.out_u_t2['out']
        # define projector variables
        # positive pair comes from the two views of the same image, we use two augmented views rather than the original
        # view to encourage the max difference of the positive pairs
        self.project_l_t1 = self.out_l_t1['project']  # view 1 as the input
        self.project_l_t2 = self.out_l_t2['project']  # view 2 as the positive anchor
        self.project_l_negative.update(self.out_u['project'])  # use the unlabeled image as the negative
        # positive label, the label of the original view is the most accurate one
        self.map_l_positive = self.out_l['project_map']
        # negative label, the label of the original view is the most accurate one
        self.map_l_negative.update(self.out_u['project_map'])
        # define the variables for the unlabeled data just the same as the labeled one
        self.project_u_t1 = self.out_u_t1['project']
        self.project_u_t2 = self.out_u_t2['project']
        self.project_u_negative.update(self.out_l['project'])
        self.map_u_positive = self.out_u['project_map']
        self.map_u_negative.update(self.out_l['project_map'])

    def uncertainty_loss(self, inputs, targets):
        """
        Uncertainty rectified pseudo supervised loss
        """
        # detach from the computational graph
        pseudo_label = F.softmax(targets / self.cfg['TRAIN']['TEMP'], dim=1).detach()
        vanilla_loss = F.cross_entropy(inputs, pseudo_label, reduction='none')
        # uncertainty rectification
        kl_div = torch.sum(F.kl_div(F.log_softmax(inputs, dim=1), F.softmax(targets, dim=1).detach(), reduction='none'), dim=1)
        uncertainty_loss = (torch.exp(-kl_div) * vanilla_loss).mean() + kl_div.mean()
        return uncertainty_loss

    def dict_loss(self, loss_fn, inputs, targets, key_list=None, **kwargs):
        """
        Perform a certain loss function in a data dict
        Args:
            loss_fn: loss function
            inputs: input data dict
            targets: ground truth
            key_list: specify which keys should be taken into computation
            **kwargs: other args for loss function
        """
        loss = 0.0
        keys = key_list if key_list is not None else list(inputs.keys())
        for key in keys:
            loss += loss_fn(inputs[key], targets, **kwargs)
        return loss

    def consist_loss(self, inputs, targets, key_list=None):
        """
        Consistency regularization between two augmented views
        """
        loss = 0.0
        keys = key_list if key_list is not None else list(inputs.keys())
        for key in keys:
            loss += (1.0 - F.cosine_similarity(inputs[key], targets[key], dim=1)).mean()
        return loss

    def optimize_parameters(self, epoch_idx):
        if not self.metric_meter:
            raise RuntimeError('The metric meter has not been initialized, '
                               'please initialize metric meter before training.')
        with autocast(enabled=self.is_mixed):
            self.forward_semi()
            self.seg_loss = self.get_multi_loss(self.out_l, self.label_l, is_ds=True, key_list=self.ds_list)
            self.cps_l_loss = self.dict_loss(self.uncertainty_loss, self.out_l_t1, self.pred_l, key_list=self.ds_list) + \
                              self.dict_loss(self.uncertainty_loss, self.out_l_t2, self.pred_l, key_list=self.ds_list)
            self.cosine_l_loss = self.consist_loss(self.out_l_t1, self.out_l_t2, key_list=self.ds_list)
            self.cps_u_loss = self.dict_loss(self.uncertainty_loss, self.out_u_t1, self.pred_u, key_list=self.ds_list) + \
                              self.dict_loss(self.uncertainty_loss, self.out_u_t2, self.pred_u, key_list=self.ds_list)
            self.cosine_u_loss = self.consist_loss(self.out_u_t1, self.out_u_t2, key_list=self.ds_list)
            # use PyTorch checkpointing to save video memory usage
            self.contrastive_l_loss = torch.utils.checkpoint.checkpoint(
                self.contrastive_loss, self.project_l_t1, self.project_l_t2, self.project_l_negative.values,
                self.map_l_positive, self.map_l_negative.values)
            self.contrastive_u_loss = torch.utils.checkpoint.checkpoint(
                self.contrastive_loss, self.project_u_t1, self.project_u_t2, self.project_u_negative.values,
                self.map_u_positive, self.map_u_negative.values)
            tau = self.sigmoid_rampup(epoch_idx)
            final_loss = self.seg_loss + \
                         self.cfg['TRAIN']['CON_RATIO'] * tau * (self.contrastive_u_loss + self.contrastive_l_loss) \
                         + self.cfg['TRAIN']['CPS_RATIO'] * tau * (self.cps_u_loss + self.cps_l_loss + self.cosine_l_loss + self.cosine_u_loss)
        self.optimizer.zero_grad()
        if self.is_mixed:
            # backward
            self.scaler.scale(final_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            final_loss.backward()
            self.optimizer.step()

    @torch.inference_mode()
    def evaluate_one_step(self, save2disk=True, save_dir=None, affine_matrix=None, patch_based_inference=False):
        """
        Evaluation pass on one batch
        Args:
            save2disk: bool, save the predictions to disk
            save_dir: save path
            affine_matrix: save affine matrix
            patch_based_inference: bool, whether to infer with patches
        """
        if patch_based_inference:
            self.pred_l = sliding_window_inference(self.image_l, roi_size=self.cfg['TEST']['PATCH_SIZE'],
                                                   sw_batch_size=self.cfg['TEST']['BATCHSIZE'], predictor=self.predictor,
                                                   overlap=self.cfg['TEST']['PATCH_OVERLAP'], mode='gaussian')
        else:
            self.forward()
        multi_loss = self.criterion(self.pred_l, self.label_l)
        self.val_loss.update({'val_loss': multi_loss.item()})
        predictions = one_hot(torch.argmax(self.pred_l, dim=1, keepdim=True), self.num_classes)
        ground_truth = one_hot(self.label_l, self.num_classes)
        # compute metrics
        metric = {'name_l': self.name_l}
        for index, cls in enumerate(self.class_list):
            dice = calculate_Dice_score(y_pred=predictions[:, index:index + 1, ...],
                                        y=ground_truth[:, index:index + 1, ...]).cpu().numpy().tolist()
            hd95 = calculate_Hasudorff_distance(y_pred=predictions[:, index:index + 1, ...],
                                                y=ground_truth[:, index:index + 1, ...],
                                                directed=False, percentile=95).cpu().numpy().tolist()
            asd = calculate_avg_surface_distance(y_pred=predictions[:, index:index + 1, ...],
                                                 y=ground_truth[:, index:index + 1, ...]).cpu().numpy().tolist()
            metric['{}_dice'.format(cls)] = dice
            metric['{}_hd95'.format(cls)] = hd95
            metric['{}_asd'.format(cls)] = asd
        metric_list = gather_object_across_processes(metric)
        self.metric_meter.update(metric_list)
        if save2disk:
            batch_pred2save = torch.argmax(predictions, dim=1)  # (N, H, W, D)
            for i in range(batch_pred2save.shape[0]):
                pred2save = batch_pred2save[i, ...]
                data = pred2save.cpu().numpy().astype(np.float32)
                nib.save(nib.Nifti1Image(data, affine=affine_matrix), os.path.join(save_dir, self.name_l[i]))

    def save_intermediate_plots(self, epoch_idx, visualization_dir, affine_matrix):
        """
        Save intermediate results to disk
        """
        for name in self.visual_names:
            item = getattr(self, name)
            if 'pred' in name:
                item = torch.argmax(item, dim=1, keepdim=True)
            for i in range(item.shape[0]):
                item2save = item[i, ...]
                # detach the tensor and convert to channel last
                data = item2save.detach().permute(1, 2, 3, 0).squeeze().cpu().numpy().astype(np.float32)
                nib.save(nib.Nifti1Image(data, affine=affine_matrix),
                         os.path.join(visualization_dir, 'Epoch_{}_Type_{}_{}'.format(epoch_idx, name, self.name_l[i])))

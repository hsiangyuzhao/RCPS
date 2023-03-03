import wandb
import yaml
import tqdm
import argparse
import torch
import torch.distributed as dist
from itertools import cycle
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import *
from configs.experiment import prepare_experiment, update_config_file, makedirs
from models.segmentation_models import SemiSupervisedContrastiveSegmentationModel
from utils.iteration.load_data_v2 import TrainValDataPipeline
from utils.iteration.iterator import set_random_seed
from utils.ddp_utils import init_distributed_mode
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='la',
                        help='experiment task, currently support "pancreas" and "la"')
    parser.add_argument('--eval_interval', type=int, default=5, help='interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=10, help='interval for checkpoint saving')
    parser.add_argument('--mixed', action="store_true",
                        help='whether to use PyTorch native mixed precision training')
    parser.add_argument('-pc', '--pretrain_ckpt', type=str, help='model checkpoint used for fine tuning')
    parser.add_argument('--benchmark', action="store_true",
                        help='whether to use cudnn benchmark to speed up convolution operations')
    parser.add_argument('--ncpu', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--verbose', action='store_true', help='print progress bar while training')
    parser.add_argument('--exp_name', type=str, default='running', help='experiment name to save logs')
    parser.add_argument('--debug', action="store_true", help='enable debug mode')
    parser.add_argument('--wandb', action='store_true', help='use WandB for experiment logging')
    parser.add_argument('--entity', type=str, help='WandB entity when logging')
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    return parser.parse_args()


def main():
    args = parse_args()
    ngpu = torch.cuda.device_count()
    init_distributed_mode(args)
    print('-' * 30)
    print('Semi-Supervised Medical Image Segmentation Training')
    print('Mixed Precision - {}; CUDNN Benchmark - {}; Num GPU - {}; Num Worker - {}'.format(
        args.mixed, args.benchmark, ngpu, args.ncpu))

    # load the cfg file
    cfg_file = 'configs/{}.cfg'.format(args.task)
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)
    # update task-specific configurations
    cfg = update_config_file(args, cfg)
    # set important hyper-parameters
    seed = cfg['TRAIN']['SEED']  # random seed
    batch_size = cfg['TRAIN']['BATCHSIZE']  # batch size
    num_epochs = cfg['TRAIN']['EPOCHS']  # number of epochs
    ratio = cfg['TRAIN']['RATIO']  # the ratio of labeled images in the training dataset, ignored for task 'tbi'
    # define experiment name
    full_exp_name = args.exp_name + '-task_{}-ratio_{}'.format(args.task, ratio)
    cfg['EXP_NAME'] = full_exp_name  # experiment name

    if args.debug:
        num_epochs = 2
    # set random seed for reproductivity
    set_random_seed(seed=seed, benchmark=args.benchmark)

    # define training & validation transforms
    train_aug = Compose([
        LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        AddChanneld(keys=['image', 'label'], allow_missing_keys=True),
        NormalizeIntensityd(keys=['image'], allow_missing_keys=False),
        EnsureTyped(keys=['image', 'label'], allow_missing_keys=True),
        RandGridDistortiond(keys=['image', 'label'], allow_missing_keys=True, mode=['bilinear', 'nearest'],
                            distort_limit=0.1, device=torch.device('cuda')),
        RandSpatialCropd(keys=['image', 'label'], allow_missing_keys=True,
                         roi_size=cfg['TRAIN']['PATCH_SIZE'], random_size=False),
    ])

    val_aug = Compose([
        LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        AddChanneld(keys=['image', 'label'], allow_missing_keys=True),
        NormalizeIntensityd(keys=['image'], allow_missing_keys=False),
        EnsureTyped(keys=['image', 'label'], allow_missing_keys=True),
    ])

    save_root_path = './experiments'
    # prepare the experiments
    image_root, num_classes, class_names, affine, = prepare_experiment(args.task)
    save_dir, metric_savedir, infer_save_dir, vis_save_dir = makedirs(args.task, full_exp_name, save_root_path)
    # define dataset
    data_pipeline = TrainValDataPipeline(image_root, 'labeled', label_ratio=ratio, random_seed=seed)
    trainset, unlabeled_set, valset = data_pipeline.get_dataset(train_aug, val_aug, cache_dataset=False)
    train_sampler = DistributedSampler(trainset, shuffle=True)
    unlabeled_sampler = DistributedSampler(unlabeled_set, shuffle=True)
    val_sampler = DistributedSampler(valset)
    # define tasks-specific information
    print('Task {} prepared. Num labeled subjects: {}; Num unlabeled subjects: {}; '
          'Num validation subjects: {}'.format(args.task, len(trainset), len(unlabeled_set), len(valset)))

    # define devices and loaders
    train_loader = DataLoader(trainset, batch_size=batch_size,  num_workers=args.ncpu,
                              sampler=train_sampler, persistent_workers=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, num_workers=args.ncpu,
                                  sampler=unlabeled_sampler, persistent_workers=True)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, sampler=val_sampler)
    # define models
    model = SemiSupervisedContrastiveSegmentationModel(cfg, num_classes=num_classes, amp=args.mixed)
    if args.pretrain_ckpt:
        model.load_networks(args.pretrain_ckpt, resume_training=False)
        print('Checkpoint {} loaded'.format(args.pretrain_ckpt))
    # classes for evaluation
    model.initialize_metric_meter(class_names)

    if args.wandb and dist.get_rank() == 0:
        wandb.init(project=cfg['PROJECT'], entity=args.entity, reinit=True, name=cfg['EXP_NAME'])
        wandb.config.update(cfg)

    # starts training
    print('Start training, please wait...')
    for epoch in range(model.start_epoch + 1, num_epochs):
        train_sampler.set_epoch(epoch)
        unlabeled_sampler.set_epoch(epoch)
        # define progress bar
        tbar = tqdm.tqdm(range(len(unlabeled_loader))) if args.verbose else range(len(unlabeled_loader))
        print("Epoch {}/{}, current lr {}".format(epoch + 1, num_epochs, model.optimizer.param_groups[0]['lr']))
        model.train()
        data_loader = iter(zip(cycle(train_loader), unlabeled_loader))
        for step in tbar:
            batch_data_labeled, batch_data_unlabeled = next(data_loader)
            model.set_input(batch_data_labeled, batch_data_unlabeled)
            # forward and backward
            model.optimize_parameters(epoch)
            stats = model.update_loss_meter(print=False)  # update the training loss meter
            if args.verbose:
                tbar.set_postfix_str(stats)  # set progress bar postfix
        if not args.verbose:
            model.update_loss_meter(print=True)
        if args.wandb and dist.get_rank() == 0:
            model.log_train_loss(step=epoch + 1)
        model.scheduler.step()  # update learning rate

        if dist.get_rank() == 0 and (epoch + 1) % args.save_interval == 0:
            if args.wandb:
                model.log_vis('train_visualization', step=epoch + 1)
            else:
                model.save_intermediate_plots(epoch, visualization_dir=vis_save_dir, affine_matrix=affine)

        # evaluation loop
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) > num_epochs - 5:
            print('Evaluating, plz wait...')
            model.eval()
            val_loader = tqdm.tqdm(val_loader) if args.verbose else val_loader
            for step, batch_data in enumerate(val_loader):
                model.set_test_input(batch_data)
                model.evaluate_one_step(True if (epoch + 1) % args.save_interval == 0 else False,
                                        infer_save_dir, affine, patch_based_inference=True)
                # uncomment following lines to log the validation visualization during inference
                # if args.wandb:
                #     model.update_val_visualization()

            current_metric = model.metric_meter.pop_mean_metric()['dice']
            if args.wandb and dist.get_rank() == 0:  # log data to wandb
                model.log_val_loss(step=epoch + 1)
                # uncomment following line to log the validation visualization during inference
                # model.log_val_visualization(step=epoch + 1)
                model.log_scaler('val/val_metric_mean', current_metric, step=epoch + 1)

            model.metric_meter.report(print_stats=True)  # print stats
            # save the metric at the end of training
            model.metric_meter.save(metric_savedir, '{}_Epoch_{}.csv'.format(full_exp_name, epoch + 1))
            # re-initialize the metric meter every time when performing evaluation
            model.metric_meter.initialization()
            model.val_loss.initialization()
            if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) > num_epochs - 5:
                model.save_networks(epoch, save_dir)  # save checkpoints



if __name__ == '__main__':
    main()

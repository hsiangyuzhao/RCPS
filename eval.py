import yaml
import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import *
from configs.experiment import prepare_experiment, update_config_file, makedirs
from models.segmentation_models import SemiSupervisedContrastiveSegmentationModel
from utils.iteration.load_data_v2 import TrainValDataPipeline
from utils.iteration.iterator import set_random_seed
from utils.ddp_utils import init_distributed_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pancreas',
                        help='experiment task, currently supports "pancreas" and "la"')
    parser.add_argument('--mixed', action="store_true",
                        help='whether to use PyTorch native mixed precision training')
    parser.add_argument('-pc', '--pretrain_ckpt', type=str, help='model checkpoint', required=True)
    parser.add_argument('--benchmark', action="store_true",
                        help='whether to use cudnn benchmark to speed up convolution operations')
    parser.add_argument('--ncpu', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--verbose', action='store_true', help='print progress bar while training')
    parser.add_argument('--exp_name', type=str, default='running', help='experiment name to save logs')
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    return parser.parse_args()


def main():
    args = parse_args()
    ngpu = torch.cuda.device_count()
    init_distributed_mode(args)
    print('-' * 30)
    print('Semi-Supervised Medical Image Segmentation Evaluation')
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
    ratio = cfg['TRAIN']['RATIO']  # the ratio of labeled images in the training dataset, ignored for task 'tbi'
    # define experiment name
    full_exp_name = 'Inference_' + args.exp_name + '-task_{}-ratio_{}'.format(args.task, ratio)
    cfg['EXP_NAME'] = full_exp_name  # experiment name

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
        ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=cfg['TRAIN']['PATCH_SIZE'],
                             allow_missing_keys=True, mode='constant'),
    ])

    val_aug = Compose([
        LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        AddChanneld(keys=['image', 'label'], allow_missing_keys=True),
        NormalizeIntensityd(keys=['image'], allow_missing_keys=False),
        EnsureTyped(keys=['image', 'label'], allow_missing_keys=True),
    ])

    save_root_path = '/mnt/shared_storage/zhaoxiangyu/experiments/SemiSeg'
    # prepare the experiments
    image_root, num_classes, class_names, affine = prepare_experiment(args.task)
    save_dir, metric_savedir, infer_save_dir, vis_save_dir = makedirs(args.task, full_exp_name, save_root_path)
    # define dataset
    data_pipeline = TrainValDataPipeline(image_root, 'labeled', label_ratio=ratio, random_seed=seed)
    trainset, unlabeled_set, valset = data_pipeline.get_dataset(train_aug, val_aug, cache_dataset=False)
    val_sampler = DistributedSampler(valset)
    # define tasks-specific information
    print('Task {} prepared. Num labeled subjects: {}; Num unlabeled subjects: {}; '
          'Num validation subjects: {}'.format(args.task, len(trainset), len(unlabeled_set), len(valset)))

    # define devices and loaders
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, sampler=val_sampler)
    # define models
    model = SemiSupervisedContrastiveSegmentationModel(cfg, num_classes=num_classes, amp=args.mixed)
    model.load_networks(args.pretrain_ckpt, resume_training=False)
    print('Checkpoint {} loaded'.format(args.pretrain_ckpt))
    # classes for evaluation
    model.initialize_metric_meter(class_names)

    # starts evaluation
    print('Start evaluation, please wait...')
    loader = tqdm.tqdm(val_loader) if args.verbose else val_loader
    model.eval()
    for step, batch_data in enumerate(loader):
        model.set_test_input(batch_data)
        model.evaluate_one_step(True, infer_save_dir, affine, patch_based_inference=True)

    model.metric_meter.report(print_stats=True)  # print stats
    # save the metric at the end of training
    model.metric_meter.save(metric_savedir, 'Evaluation_{}.csv'.format(full_exp_name))


if __name__ == '__main__':
    main()

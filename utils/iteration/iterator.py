import os
import random
import math
import torch
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler


affine_matrix = np.array([[1, 0, 0, -85], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    D = size[4]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cut_d = np.int(D * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(D)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)

    return bbx1, bby1, bbz1, bbx2, bby2, bbz2


def set_random_seed(seed=512, benchmark=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class MetricMeter:
    """
    Metric logger, receives metrics when validation and print results.
    We provide support for saving metric file to local, ddp metric logging and metric logging to WandB.
    This class can be used for both training loss logging and evaluation metric logging, depending on how to use it.
    """
    def __init__(self, metrics, class_names, subject_names=('name')):
        """
        Args:
            metrics: the list of metric names
            class_names: the list of class names
            subject_names: the list of subject names, in case one sample has multiple subject names in the experiment
        """
        self.metrics = metrics
        self.class_names = class_names
        self.subject_names = subject_names
        self.initialization()

    def initialization(self):
        """
        Initialize the metric logger, must call this method before logging
        """
        for metric in self.metrics:
            for class_name in self.class_names:
                setattr(self, '{}_{}'.format(class_name, metric), [])
        for name in self.subject_names:
            setattr(self, '{}'.format(name), [])

    def update(self, metric_dict_list):
        """
        Update the metric
        Args:
            metric_dict_list: a list of metric dict
        """
        if not isinstance(metric_dict_list, (list, tuple)):
            metric_dict_list = [metric_dict_list]
        for metric_dict in metric_dict_list:
            for metric_key, metric_value in metric_dict.items():
                attr = getattr(self, metric_key)
                if isinstance(metric_value, (list, tuple)):
                    attr.extend(metric_value)
                else:
                    attr.append(metric_value)

    def report(self, print_stats=True, mean_only=False):
        """
        Report the mean and variance of the metrics during training or inference
        Args:
            print_stats: bool, whether to print the metrics
            mean_only: bool, whether to calculate the mean value only
        """
        report_str = ''
        for metric in self.metrics:
            for class_name in self.class_names:
                metric_mean = np.nanmean(getattr(self, '{}_{}'.format(class_name, metric)), axis=0)
                metric_std = np.nanstd(getattr(self, '{}_{}'.format(class_name, metric)), axis=0)
                if not mean_only:
                    stats = '{}_{}: {} ± {}; '.format(class_name, metric, np.around(metric_mean, decimals=4),
                                                     np.around(metric_std, decimals=4))
                else:
                    stats = '{}_{}: {}; '.format(class_name, metric, np.around(metric_mean, decimals=4))
                report_str += stats
                if print_stats:
                    print(stats, end=' ')
        if print_stats:
            print('\n')
        return report_str

    def save(self, savedir='./metrics', filename='metric.csv'):
        """
        Save the metrics to disk using pandas
        Args:
            savedir: save path
            filename: filename of the saved file
        """
        os.makedirs(savedir, exist_ok=True)
        series = [pd.Series(getattr(self, name)) for name in self.subject_names]
        columns = [name for name in self.subject_names]
        for metric in self.metrics:
            for class_name in self.class_names:
                data = getattr(self, '{}_{}'.format(class_name, metric))
                series.append(pd.Series(np.around(data, decimals=6)))
                columns.append('{}_{}'.format(class_name, metric))
        df = pd.concat(series, axis=1)
        df.columns = columns
        df.to_csv(os.path.join(savedir, filename), index=False)

    def to_df(self):
        """
        Convert saved data to pandas.Dataframe
        """
        series = [pd.Series(getattr(self, name)) for name in self.subject_names]
        columns = [name for name in self.subject_names]
        for metric in self.metrics:
            for class_name in self.class_names:
                data = getattr(self, '{}_{}'.format(class_name, metric))
                series.append(pd.Series(np.around(data, decimals=5)))
                columns.append('{}_{}'.format(class_name, metric))
        df = pd.concat(series, axis=1)
        df.columns = columns
        return df

    def pop_data(self, mean_only=False):
        """
        Pop logged data, typically used for loss logging
        Args:
            mean_only: bool, pop mean value only
        """
        data_dict = {}
        for metric in self.metrics:
            for class_name in self.class_names:
                metric_mean = np.nanmean(getattr(self, '{}_{}'.format(class_name, metric)), axis=0)
                metric_std = np.nanstd(getattr(self, '{}_{}'.format(class_name, metric)), axis=0)
                mean_key = "{}_{}_mean".format(class_name, metric)
                std_key = "{}_{}_std".format(class_name, metric)
                data_dict[mean_key] = np.around(metric_mean, decimals=7)
                if not mean_only:
                    data_dict[std_key] = np.around(metric_std, decimals=7)
        return data_dict

    def pop_mean_metric(self, abandon_background=True):
        """
        Pop logged mean value, typically used for evaluation metric printing
        Args:
            abandon_background: ignore the metrics of class 'bg'
        """
        data_dict = {}
        for metric in self.metrics:
            sum_value = 0
            idx = 0
            for class_name in self.class_names:
                if class_name == 'bg' and abandon_background:
                    continue
                else:
                    metric_mean = np.nanmean(getattr(self, '{}_{}'.format(class_name, metric)), axis=0)
                    sum_value += metric_mean
                    idx += 1
            mean_value = sum_value / idx
            data_dict['{}'.format(metric)] = mean_value
        return data_dict


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, power=0.9, last_epoch=-1, verbose=False):
        self.iter_max = total_iters
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch, verbose)

    def polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            # or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]


class PolynomialLRWithWarmUp(_LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, power: float = 0.9,  warmup_steps: int = 0,
                 max_lr_steps: int = 0, warmup_ratio: float = 0.01, last_epoch: int = -1, verbose=False):
        assert warmup_steps < total_steps, 'The warm up steps should be less than total steps'
        assert warmup_steps + max_lr_steps <= total_steps, \
            'The sum of warm up and burn steps should be no more than total steps'
        self.warmup_steps = warmup_steps  # warmup step size
        self.max_lr_steps = max_lr_steps
        self.total_steps = total_steps  # first cycle step size
        self.warmup_ratio = warmup_ratio
        self.power = power
        super(PolynomialLRWithWarmUp, self).__init__(optimizer, last_epoch, verbose)

    def polynomial_decay(self, lr):
        delayed_step = self.last_epoch - (self.warmup_steps + self.max_lr_steps)
        delayed_total_steps = self.total_steps - (self.warmup_steps + self.max_lr_steps)
        return lr * (1 - float(delayed_step) / delayed_total_steps) ** self.power

    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr * self.warmup_ratio for base_lr in self.base_lrs]
        if self.last_epoch > self.total_steps:
            return [0.0 for _ in self.base_lrs]
        if self.last_epoch <= self.warmup_steps:
            return [base_lr * self.warmup_ratio + (base_lr - base_lr * self.warmup_ratio)
                    / self.warmup_steps * self.last_epoch for base_lr in self.base_lrs]
        if self.last_epoch > self.warmup_steps and self.last_epoch <= (self.warmup_steps + self.max_lr_steps):
            return self.base_lrs
        if self.last_epoch > (self.warmup_steps + self.max_lr_steps):
            return [self.polynomial_decay(lr) for lr in self.base_lrs]


class CosineAnnealingWithWarmUp(_LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0,
                 max_lr_steps: int = 0, warmup_ratio: float = 0.01, last_epoch: int = -1, verbose=False):
        assert warmup_steps < total_steps, 'The warm up steps should be less than total steps'
        assert warmup_steps + max_lr_steps <= total_steps, \
            'The sum of warm up and burn steps should be no more than total steps'
        self.warmup_steps = warmup_steps  # warmup step size
        self.max_lr_steps = max_lr_steps
        self.total_steps = total_steps  # first cycle step size
        self.warmup_ratio = warmup_ratio
        super(CosineAnnealingWithWarmUp, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr * self.warmup_ratio for base_lr in self.base_lrs]
        if self.last_epoch > self.total_steps:
            return [0.0 for _ in self.base_lrs]
        if self.last_epoch <= self.warmup_steps:
            return [base_lr * self.warmup_ratio + (base_lr - base_lr * self.warmup_ratio)
                    / self.warmup_steps * self.last_epoch for base_lr in self.base_lrs]
        if self.last_epoch > self.warmup_steps and self.last_epoch <= (self.warmup_steps + self.max_lr_steps):
            return self.base_lrs
        if self.last_epoch > (self.warmup_steps + self.max_lr_steps):
            return [base_lr * (1 + math.cos(math.pi * (
                    self.last_epoch - self.warmup_steps - self.max_lr_steps) / (
                    self.total_steps - self.warmup_steps - self.max_lr_steps))) / 2 for base_lr in self.base_lrs]

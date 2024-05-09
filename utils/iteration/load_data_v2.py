import os
import random
import numpy as np
from monai.data import Dataset, CacheDataset


simple_affine = np.eye(4, 4)


class TrainValDataPipeline:
    """
    Data pipeline that specifies training and testing data in semi-supervised training, used to evaluate the effectiveness of 
    SSL by changing the ratio of labeled data.

    Note that when using `TrainValDataPipeline`, all of the images should have labels; if you are working on real semi-supervision 
    scenarios (i.e., some of the data are unlabeled), you should use `RealSemiSupervisionPipeline` instead.
    """
    def __init__(self, image_root: str, label_ratio=1.0, random_seed: int = None):
        """
        Args:
            image_root: the data root for training and validation data, should contain 4 folders:
                train_images, train_labels, val_images, val_labels;
            label_ratio: flaot, the ratio of labeled data in SSL, remaining training data is viewed as unlabeled,
                and their segmentation labels remain untouched during training;
            random_seed: random seed for spliting labeled and unlabeled training data.
        """
        assert label_ratio > 0 and label_ratio < 1, 'The raio of labeled data in semi-supervised training should be within 0 to 1.'
        self.image_root = image_root
        self.label_ratio = label_ratio
        self.seed = random_seed

        self.prepare_train_val_subjects()

    def prepare_train_val_subjects(self):
        train_subjects_all = self.get_subjects('train')
        num_labeled_subjects = int(len(train_subjects_all) * self.label_ratio)
        random.seed(self.seed)
        self.train_subjects = random.sample(train_subjects_all, num_labeled_subjects)
        # dict is not hashable, thus use list generator rather than set
        self.unlabeled_train_subjects = [subject for subject in train_subjects_all if subject not in self.train_subjects]
        self.val_subjects = self.get_subjects('val')

    def get_subjects(self, split_mode):
        subjects = []
        image_list = os.listdir(os.path.join(self.image_root, '{}_images'.format(split_mode)))
        for index, filename in enumerate(image_list):
            subject = {
                'image': os.path.join(self.image_root, '{}_images'.format(split_mode), filename),
                'name': filename
            }
            subject['label'] = os.path.join(self.image_root, '{}_labels'.format(split_mode), filename),
            subjects.append(subject)
        return subjects

    def get_dataset(self, train_transform, val_transform, cache_dataset=False, unlabeled_transform=None):
        dataset = CacheDataset if cache_dataset else Dataset
        trainset = dataset(data=self.train_subjects, transform=train_transform)
        unlabeled_trainset = dataset(data=self.unlabeled_train_subjects,
                                     transform=unlabeled_transform if unlabeled_transform else train_transform)
        valset = dataset(data=self.val_subjects, transform=val_transform)
        return trainset, unlabeled_trainset, valset


class RealSemiSupervisionPipeline:
    """
    Data pipeline instance that specifies training and testing data in semi-supervised training.

    This class is different from `TrainValDataPipeline`, which is used for real semi-supervision scenarios.

    Note:
        1. In `TrainValDataPipeline`, we require that all of the training images are labeled, and by changing 
           the `label_ratio` argument, we change the ratio of training data that are viewed as unlabeled data.
        2. In `RealSemiSupervisionPipeline`, labeled data and unlabeled data should be stored separately, and 
           SSL is utilized to enhance segmentation performance with given unlabeled data.
    """
    def __init__(self, labeled_image_root: str, unlabeled_image_root: str):
        """
        Args:
            labeled_image_root: the data root for labeled images, should contain 4 folders:
                train_images, train_labels, val_images, val_labels;
            unlabeled_image_root: the data root for unlabeled images, should contain 1 folder:
                train_images
        """
        self.labeled_root = labeled_image_root
        self.unlabeled_root = unlabeled_image_root
        self.prepare_train_val_subjects()

    def prepare_train_val_subjects(self):
        self.train_subjects = self.get_subjects(self.labeled_root, 'train', True)
        self.val_subjects = self.get_subjects(self.labeled_root, 'val', True)
        self.unlabeled_train_subjects = self.get_subjects(self.unlabeled_root, 'train', False)

    def get_subjects(self, data_path: str, split_mode: str, is_labeled: bool = False):
        subjects = []
        image_list = os.listdir(os.path.join(data_path, '{}_images'.format(split_mode)))
        for index, filename in enumerate(image_list):
            subject = {
                'image': os.path.join(data_path, '{}_images'.format(split_mode), filename),
                'name': filename
            }
            if is_labeled:
                subject['label'] = os.path.join(data_path, '{}_labels'.format(split_mode), filename),
            subjects.append(subject)
        return subjects

    def get_dataset(self, train_transform, val_transform, cache_dataset=False, unlabeled_transform=None):
        dataset = CacheDataset if cache_dataset else Dataset
        trainset = dataset(data=self.train_subjects, transform=train_transform)
        unlabeled_trainset = dataset(data=self.unlabeled_train_subjects,
                                     transform=unlabeled_transform if unlabeled_transform else train_transform)
        valset = dataset(data=self.val_subjects, transform=val_transform)
        return trainset, unlabeled_trainset, valset
    

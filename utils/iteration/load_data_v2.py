import os
import random
import numpy as np
from monai.data import Dataset, CacheDataset


simple_affine = np.eye(4, 4)


class TrainValDataPipeline:
    def __init__(self, image_root: str, mode: str = 'labeled', label_ratio=1.0, random_seed=None):
        assert mode in ('labeled', 'unlabeled')
        self.image_root = image_root
        self.mode = mode
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
            if self.mode != 'unlabeled':
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

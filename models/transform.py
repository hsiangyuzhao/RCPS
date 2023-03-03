import random
import torch
import numpy as np
from abc import ABC, abstractmethod


class RandTransform(ABC):
    @abstractmethod
    def randomize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply_transform(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_image(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def invert_label(self, *args, **kwargs):
        raise NotImplementedError


class RandIntensityDisturbance(RandTransform):
    def __init__(self, p: float = 0.1, brightness_limit: float = 0.5, contrast_limit: float = 0.5, clip: bool = False,
                 beta_by_max: bool = True):
        self.beta = (-brightness_limit, brightness_limit)
        self.alpha = (1 - contrast_limit, 1 + contrast_limit)
        self.clip = clip
        self.beta_by_max = beta_by_max
        self.p = p

        self.alpha_value = None
        self.beta_value = None

        self._do_transform = False

    def randomize(self):
        if random.uniform(0, 1) < self.p:
            self._do_transform = True
            self.alpha_value = random.uniform(self.alpha[0], self.alpha[1])
            self.beta_value = random.uniform(self.beta[0], self.beta[1])

    def apply_transform(self, inputs):
        """
        Apply brightness and contrast transform on image
            Args: inputs, torch.tensor, shape (B, C, H, W)
        """
        if self._do_transform:
            img_t = self.alpha_value * inputs
            if self.beta_by_max:
                img_t = img_t + self.beta_value
            else:
                img_t = img_t + self.beta_value * torch.mean(img_t)
            return torch.clamp(img_t, 0, 1) if self.clip else img_t
        else:
            return inputs

    def inverse_transform(self, img_t):
        raise NotImplementedError

    def forward_image(self, image, randomize=True):
        if randomize:
            self.randomize()
        return self.apply_transform(image)

    def invert_label(self, label_t):
        return label_t


class RandGaussianNoise(RandTransform):
    def __init__(self, p: float = 0.2, mean: float = 0.0, std: float = 0.1, clip: bool = False):
        self.p = p
        self.mean = mean
        self.std = std
        self.clip = clip

        self.std_value = None
        self._do_transform = False

    def randomize(self, inputs):
        if random.uniform(0, 1) < self.p:
            self._do_transform = True
            self.std_value = random.uniform(0, self.std)
            self.noise = torch.normal(self.mean, self.std_value, size=inputs.shape)

    def apply_transform(self, inputs):
        if self._do_transform:
            added = inputs + self.noise.to(inputs.device)
            return torch.clamp(added, 0, 1) if self.clip else added
        else:
            return inputs

    def inverse_transform(self, img_t):
        raise NotImplementedError

    def forward_image(self, image, randomize=True):
        if randomize:
            self.randomize(image)
        return self.apply_transform(image)

    def invert_label(self, label_t):
        return label_t


class CutOut(RandTransform):
    def __init__(self, p: float = 0.1, num_holes: int = 5, hole_ratio: float = 0.1, value: float = 0.0):
        # hole 5 and ratio 0.05 test fine
        self.p = p
        self.num_holes = num_holes
        self.hole_ratio = hole_ratio
        self.value = value

        self.hole_list = None

    def rand_bbox(self, input_shape):
        W = input_shape[2]
        H = input_shape[3]
        D = input_shape[4]
        cut_w = np.int(W * self.hole_ratio)
        cut_h = np.int(H * self.hole_ratio)
        cut_d = np.int(D * self.hole_ratio)

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

    def fill_hole(self, inputs, location, value):
        processed = inputs.clone()
        processed[:, :, location[0]:location[3], location[1]:location[4], location[2]:location[5]] = value
        return processed

    def randomize(self, inputs):
        inputs_shape = inputs.shape
        hole_list = []
        for i in range(self.num_holes):
            loc = self.rand_bbox(inputs_shape)
            hole_list.append(loc)
        return hole_list

    def apply_transform(self, inputs, hole_list):
        for hole in hole_list:
            inputs = self.fill_hole(inputs, hole, self.value)
        return inputs

    def inverse_transform(self):
        raise NotImplementedError

    def forward_image(self, inputs, randomize=True):
        if randomize:
            self.hole_list = self.randomize(inputs)
        outputs = self.apply_transform(inputs, self.hole_list)
        return outputs

    def invert_label(self):
        raise NotImplementedError


class FullAugmentor:
    """
    Augmentor to generate augmented views of the input, support intensity shift or scaling, Gaussian noise and
    CutOut (optional)
    """
    def __init__(self):
        self.intensity = RandIntensityDisturbance(p=1, clip=False, beta_by_max=False)
        self.gaussian = RandGaussianNoise(p=0.5, clip=False)
        self.cutout = CutOut(p=0.1)

    def forward_image(self, image):
        image = self.intensity.forward_image(image)
        image = self.gaussian.forward_image(image)
        # image = self.cutout.forward_image(image)
        return image

import wandb
import torch
import numpy as np
from monai.visualize import blend_images


class WandBModel:
    """
    Enable WandB features to the model using multiple inheritance
    """
    def __init__(self, *args, **kwargs):
        # the following attributes should be initialized by class `BaseSegmentationModel`
        self.visual_pairs = None
        self.train_loss = None
        self.val_loss = None
        self.metric_meter = None
        self.name = None
        # the following attributes should be initialized by the child class
        self.val_table = None

    def volume2videos(self, time_dim=3, tag=''):
        """
        Convert 3D volumes to video in favor of WandB logging
        Args:
            time_dim: the spatial dimension to be converted as the time dimension, default is the axial axis (dim 3)
            tag: extra information for logging
        """
        videos = []
        for image_pair in self.visual_pairs:
            try:
                pair_name = getattr(self, image_pair['name'])
                image = getattr(self, image_pair['image'])
                mask = getattr(self, image_pair['mask'])
                vis_type = image_pair['type']
            except:
                continue
            for i in range(image.shape[0]):  # deallocate the batch dim
                image2save = image[i, ...]
                mask2save = mask[i, ...]
                item_name = pair_name[i]
                # detach the tensor, format [C, H, W, D]
                image_numpy = image2save.detach()
                mask_numpy = mask2save.detach()
                if mask_numpy.shape[0] > 1:
                    mask_numpy = torch.argmax(mask_numpy, dim=0, keepdim=True)
                # (C, H, W, D), torch.Tensor on device
                pair_blend = blend_images(image_numpy, mask_numpy, alpha=0.5) * 255
                # permute the axes to (time, channel, height, width)
                spatial_dim = list(range(1, len(pair_blend.shape[1:]) + 1))
                spatial_dim.remove(time_dim)
                pair_blend = pair_blend.permute([time_dim, 0] + spatial_dim).cpu().numpy().astype(np.uint8)
                # record in the wandb.Video class
                video = wandb.Video(pair_blend, fps=8, caption='{}_{}{}'.format(item_name, vis_type, tag))
                videos.append(video)
        return videos

    def log_scaler(self, key, value, step=None):
        """
        Log manually defined scaler data
        """
        wandb.log({key: np.round(value, decimals=4)}, step=step)

    def log_train_loss(self, step=None):
        """
        Log train loss
        """
        data_dict = self.train_loss.pop_data(True)
        for key, value in data_dict.items():
            wandb.log({'train/{}'.format(key): value}, step=step)

    def log_val_loss(self, step=None):
        """
        Log val loss
        """
        data_dict = self.val_loss.pop_data(True)
        for key, value in data_dict.items():
            wandb.log({'val/{}'.format(key): value}, step=step)

    def log_metrics(self, step=None):
        """
        Log validation metrics as wandb.Table
        """
        df = self.metric_meter.to_df()
        wandb.log({'val/metrics': wandb.Table(dataframe=df)}, step=step)

    def log_vis(self, key, step=None, time_dim=3, tag=''):
        """
        Log training intermediate visualizations
        """
        videos = self.volume2videos(time_dim, tag)
        wandb.log({key: videos}, step=step)

    def update_val_visualization(self, time_dim=3, tag=''):
        """
        Update the validation visualization to buffer, called every step of evaluation
        """
        videos = self.volume2videos(time_dim, tag)
        self.val_table.add_data(self.name, *videos)

    def log_val_visualization(self, step=None):
        """
        Log validation visualization
        """
        wandb.log({'val/visualization': self.val_table}, step=step)
        # re-initialize the table for next logging
        del self.val_table
        self.val_table = wandb.Table(columns=['ID'] + [pair['type'] for pair in self.visual_pairs])

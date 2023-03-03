import torch.nn as nn
import torch.nn.functional as F
from utils.ddp_utils import *


norm_dict = {'BATCH': nn.BatchNorm3d, 'INSTANCE': nn.InstanceNorm3d, 'GROUP': nn.GroupNorm}
__all__ = ['ConvNorm', 'ConvBlock', 'ConvBottleNeck', 'ResBlock', 'ResBottleneck',
           'GumbelTopK', 'PixelContrastiveLoss', 'TensorBuffer', 'DDPTensorBuffer',
           'NegativeSamplingPixelContrastiveLoss']


class GumbelTopK(nn.Module):
    """
    Perform top-k or Gumble top-k on given data
    """
    def __init__(self, k: int, dim: int = -1, gumble: bool = False):
        """
        Args:
            k: int, number of chosen
            dim: the dimension to perform top-k
            gumble: bool, whether to introduce Gumble noise when sampling
        """
        super().__init__()
        self.k = k
        self.dim = dim
        self.gumble = gumble

    def forward(self, logits):
        # logits shape: [B, N], B denotes batch size, and N denotes the multiplication of channel and spatial dim
        if self.gumble:
            u = torch.rand(size=logits.shape, device=logits.device)
            z = - torch.log(- torch.log(u))
            return torch.topk(logits + z, self.k, dim=self.dim)
        else:
            return torch.topk(logits, self.k, dim=self.dim)


class TensorBuffer:
    """
    A buffer to store tensors. Used to enlarge the number of negative samples when calculating contrastive loss.
    """
    def __init__(self, buffer_size: int, concat_dim: int, retain_gradient: bool = True):
        """
        Args:
            buffer_size: int, the number of stored tensors
            concat_dim: specify a dimension to concatenate the stored tensors, usually the batch dim
            retain_gradient: whether to detach the tensor from the computational graph, must set `retain_graph=True`
                            during backward
        """
        self.buffer_size = buffer_size
        self.concat_dim = concat_dim
        self.retain_gradient = retain_gradient
        self.tensor_list = []

    def update(self, tensor):
        if len(self.tensor_list) >= self.buffer_size:
            self.tensor_list.pop(0)
        if self.retain_gradient:
            self.tensor_list.append(tensor)
        else:
            self.tensor_list.append(tensor.detach())

    @property
    def values(self):
        return torch.cat(self.tensor_list, dim=self.concat_dim)


class DDPTensorBuffer:
    """
    Extend tensor buffer to DDP, thus all the processes share the same large buffer
    Note: dist.all_gather will detach the tensors, thus all the tensors have no gradient
    """
    def __init__(self, buffer_size, concat_dim):
        """
        Args:
            buffer_size: int, the number of stored tensors
            concat_dim: specify a dimension to concatenate the stored tensors, usually the batch dim
        """
        self.buffer_size = buffer_size
        self.concat_dim = concat_dim
        self.tensor_list = []

    def update(self, tensor):
        tensor_gather = self.concat_all_gather(tensor)
        if len(self.tensor_list) >= self.buffer_size:
            self.tensor_list.pop(0)
        self.tensor_list.append(tensor_gather)

    @property
    def values(self):
        return torch.cat(self.tensor_list, dim=self.concat_dim)

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output


class PixelContrastiveLoss(nn.Module):
    """
    Pixel contrastive loss for segmentation. In this loss, we utilize two feature maps generated from two different
    images to calculate the contrastive loss. We treat every pixel pair belonging to the same classes as positives,
    and pixels belonging to different classes are treated as negatives.
    Since the calculation is performed on the whole image, we can sample the top-N confident pixels to calculate the
    loss and ignore other pixels.
    """
    def __init__(self, temperature: float = 0.07, sample: bool = False, sample_num: int = None):
        """
        Args:
            temperature: hyperparameter
            sample: bool, whether to sample negative pixels
            sample_num: the number of negative samples
        """
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
        self.sample = False if sample_num is None else sample
        self.topk = GumbelTopK(k=sample_num) if self.sample else None

    def check_input(self, input_logits, target_logits, input_seg, target_seg):
        """
        Check whether the input is valid (we expect paired inputs and only one paired inputs)
        Args:
            input_logits: input feature map class logits, shape (B, NumCls, H, W, [D])
            target_logits: target feature map class logits, shape (B, NumCls, H, W, [D])
            input_seg: input feature map class segmentation, shape (B, H, W, [D])
            target_seg: target feature map class segmentation, shape (B, H, W, [D])

        Returns:
            input_seg, target_seg, final_logits
        """
        if input_logits is not None and target_logits is not None:
            assert input_seg is None and target_seg is None, \
                'When the logits are provided, do not provide segmentation.'
            if input_logits.shape[1] == 1:  # single-class segmentation, use sigmoid activation
                input_max_probs = torch.sigmoid(input_logits)
                target_max_probs = torch.sigmoid(target_logits)
                input_cls_seg = (input_max_probs > 0.5).to(torch.float32)
                target_cls_seg = (target_max_probs > 0.5).to(torch.float32)
            else:  # multi-class segmentation case, use softmax activation
                input_max_probs, input_cls_seg = torch.max(F.softmax(input_logits, dim=1), dim=1)
                target_max_probs, target_cls_seg = torch.max(F.softmax(target_logits, dim=1), dim=1)
            return input_cls_seg, target_cls_seg, input_max_probs, target_max_probs
        elif input_seg is not None and target_seg is not None:
            assert input_logits is None and target_logits is None, \
                'When the segmentation are provided, do not provide logits.'
            confident_probs = torch.ones_like(input_seg)
            return input_seg, target_seg, confident_probs, confident_probs
        else:
            raise ValueError('The logits/segmentation are not paired, please check the input')

    def forward(self, input, target, input_logits=None, target_logits=None, input_seg=None, target_seg=None):
        """
        calculate pixel contrastive loss
        Args:
            input: input projected feature, shape (B, C, H, W, [D])
            target: target projected feature, shape (B, C, H, W, [D])
            input_logits: input feature map class logits, shape (B, NumCls, H, W, [D])
            target_logits: target feature map class logits, shape (B, NumCls, H, W, [D])
            input_seg: input feature map class segmentation, shape (B, H, W, [D])
            target_seg: target feature map class segmentation, shape (B, H, W, [D])

        Returns:
            contrastive loss
        """
        B, C, *spatial_size = input.shape  # N = H * W * D
        spatial_dims = len(spatial_size)

        # input/target shape: B * N, C
        norm_input = F.normalize(input.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_target = F.normalize(target.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        pred_input, pred_target, prob_map1, prob_map2 = self.check_input(input_logits, target_logits, input_seg, target_seg)
        prob_map = (prob_map1 + prob_map2) / 2
        pred_input = pred_input.flatten()  # B * N
        pred_target = pred_target.flatten()  # B * N
        prob_map = prob_map.flatten()  # B * N

        if self.topk:
            indices = self.topk(prob_map).indices
            norm_input = norm_input[indices, :]
            norm_target = norm_target[indices, :]
            pred_input = pred_input[indices]
            pred_target = pred_target[indices]

        # mask matrix indicating whether the pixel-pair in input and target belongs to the same class or not
        diff_cls_matrix = (pred_input.unsqueeze(0) != pred_target.unsqueeze(1)).to(torch.float32)  # B * N, B * N
        # similarity matrix, which calculate the cosine distance of every pixel-pair in input and target
        # dim0 represents target and dim1 represents input
        sim_matrix = F.cosine_similarity(norm_input.unsqueeze(0), norm_target.unsqueeze(1), dim=2)
        sim_exp = torch.exp(sim_matrix / self.tau)
        nominator = torch.sum((1 - diff_cls_matrix) * sim_exp, dim=0)
        denominator = torch.sum(diff_cls_matrix * sim_exp, dim=0) + nominator
        loss = -torch.log(nominator / (denominator + 1e-8)).mean()
        return loss


class NegativeSamplingPixelContrastiveLoss(nn.Module):
    """
    Pixel contrastive loss for segmentation. This loss is different from the loss above, as we use two different views
    of the same image to construct the positive pairs. Negative pixels are sampled from different images according to
    the pseudo labels.
    We support bidirectional computation, which will force the negative samples to be far from both the two augmented
    views, rather than only the view chosen as anchor
    """
    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
        self.topk = GumbelTopK(k=sample_num, dim=0)
        self.bidir = bidirectional

    def check_input(self, input_logits, target_logits, input_seg, target_seg):
        """
        Check whether the input is valid (we expect paired inputs and only one paired inputs)
        Args:
            input_logits: input feature map class logits, shape (B, NumCls, H, W, [D])
            target_logits: target feature map class logits, shape (B, NumCls, H, W, [D])
            input_seg: input feature map class segmentation, shape (B, H, W, [D])
            target_seg: target feature map class segmentation, shape (B, H, W, [D])

        Returns:
            input_seg, target_seg, final_logits
        """
        if input_logits is not None and target_logits is not None:
            assert input_seg is None and target_seg is None, \
                'When the logits are provided, do not provide segmentation.'
            if input_logits.shape[1] == 1:  # single-class segmentation, use sigmoid activation
                input_max_probs = torch.sigmoid(input_logits)
                target_max_probs = torch.sigmoid(target_logits)
                input_cls_seg = (input_max_probs > 0.5).to(torch.float32)
                target_cls_seg = (target_max_probs > 0.5).to(torch.float32)
            else:  # multi-class segmentation case, use softmax activation
                input_max_probs, input_cls_seg = torch.max(F.softmax(input_logits, dim=1), dim=1)
                target_max_probs, target_cls_seg = torch.max(F.softmax(target_logits, dim=1), dim=1)
            return input_cls_seg, target_cls_seg, input_max_probs, target_max_probs
        elif input_seg is not None and target_seg is not None:
            assert input_logits is None and target_logits is None, \
                'When the segmentation are provided, do not provide logits.'
            confident_probs = torch.ones_like(input_seg)
            return input_seg, target_seg, confident_probs, confident_probs
        else:
            raise ValueError('The logits/segmentation are not paired, please check the input')

    def forward(self, input, positive, negative, input_logits=None, negative_logits=None, input_seg=None, negative_seg=None):
        """
        calculate pixel contrastive loss
        Args:
            input: input projected feature, shape (B, C, H, W, [D])
            positive: projected feature from the other view, shape (B, C, H, W, [D])
            negative: negative projected feature, shape (B, C, H, W, [D])
            input_logits: input feature map class logits, shape (B, NumCls, H, W, [D])
            negative_logits: negative feature map class logits, shape (B, NumCls, H, W, [D])
            input_seg: input feature map class segmentation, shape (B, H, W, [D])
            negative_seg: negative feature map class segmentation, shape (B, H, W, [D])

        Returns:
            contrastive loss
        """
        B, C, *spatial_size = input.shape  # N = H * W * D
        spatial_dims = len(spatial_size)

        # input/target shape: B * N, C
        norm_input = F.normalize(input.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_positive = F.normalize(positive.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_negative = F.normalize(negative.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        seg_input, seg_negative, input_prob, negative_prob = self.check_input(
            input_logits, negative_logits, input_seg, negative_seg)

        # calculate the segmentation map to determine which pixel pairs are negative
        seg_input = seg_input.flatten()  # B * N
        seg_negative = seg_negative.flatten()  # B * N
        # input_prob = input_prob.flatten()  # B * N
        negative_prob = negative_prob.flatten()  # B * N

        # calculate the similarity between every positive pair
        positive = F.cosine_similarity(norm_input, norm_positive, dim=-1)  # B * N

        # binary matrix indicating which pixel pair is negative pair
        # seg_input is fixed in dim0, seg_target is fixed in dim1
        # in other words, dim0 represents target and dim1 represents input
        diff_cls_matrix = (seg_input.unsqueeze(0) != seg_negative.unsqueeze(1)).to(torch.float32)  # B * N, B * N
        prob_matrix = negative_prob.unsqueeze(1).expand_as(diff_cls_matrix)  # B * N, B * N
        masked_target_prob_matrix = diff_cls_matrix * prob_matrix  # mask positive pairs, sample negative only

        # sample the top-K negative pixels in the target for every pixel in the input
        sampled_negative_indices = self.topk(masked_target_prob_matrix).indices  # K, B * N
        sampled_negative = norm_negative[sampled_negative_indices]  # K, B * N, C
        negative_sim_matrix = F.cosine_similarity(norm_input.unsqueeze(0).expand_as(sampled_negative),
                                                  sampled_negative, dim=-1)  # K, B * N

        nominator = torch.exp(positive / self.tau)
        denominator = torch.exp(negative_sim_matrix / self.tau).sum(dim=0) + nominator
        loss = -torch.log(nominator / (denominator + 1e-8)).mean()
        if self.bidir:  # negative samples should be far from the other view as well
            alter_negative_sim_matrix = F.cosine_similarity(norm_positive.unsqueeze(0).expand_as(sampled_negative),
                                                            sampled_negative, dim=-1)  # K, B * N
            alter_denominator = torch.exp(alter_negative_sim_matrix / self.tau).sum(dim=0) + nominator
            alter_loss = -torch.log(nominator / (alter_denominator + 1e-8)).mean()
            loss = loss + alter_loss
        return loss


class Identity(nn.Module):
    """
    Identity mapping for building a residual connection
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ConvNorm(nn.Module):
    """
    Convolution and normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=True, norm='BATCH', activation=True):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        padding = (kernel_size - 1) // 2

        # activation, support PReLU and common ReLU
        if activation:
            self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
            # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)
        else:
            self.act = None

        # instantiate layers
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(4, in_channels)

    def basic_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        if self.act:
            x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)


class ConvBlock(nn.Module):
    """
    Convolutional blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        # activation, support PReLU and common ReLU
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True)
        self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ResBlock(nn.Module):
    """
    Residual blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1) if use_dropout else None
        # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True)
        self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

        need_map = in_channels != out_channels or stride != 1
        self.id = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False) if need_map else Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.id(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        return out


class ConvBottleNeck(nn.Module):
    """
    Convolutional bottleneck blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        middle_channels = in_channels // 4
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, middle_channels, 1, 1, leaky, norm, True)
        self.conv2 = ConvNorm(middle_channels, middle_channels, 3, stride, leaky, norm, True)
        self.conv3 = ConvNorm(middle_channels, out_channels, 1, 1, leaky, norm, False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ResBottleneck(nn.Module):
    """
    Residual bottleneck blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        middle_channels = in_channels // 4
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1) if use_dropout else None
        # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, middle_channels, 1, 1, leaky, norm, True)
        self.conv2 = ConvNorm(middle_channels, middle_channels, 3, stride, leaky, norm, True)
        self.conv3 = ConvNorm(middle_channels, out_channels, 1, 1, leaky, norm, False)

        need_map = in_channels != out_channels or stride != 1
        self.id = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False) if need_map else Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        identity = self.id(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        return out

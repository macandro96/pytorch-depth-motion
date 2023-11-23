import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class MotionVectorModel(nn.Module):
    """
    Predicts 3D motion field and pose estimates from a pair of RGBD images.
    Sources:
        - https://github.com/bolianchen/pytorch_depth_from_videos_in_the_wild
    """

    def __init__(self, input_shape, learn_intrinsics: bool = False, auto_mask=True):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.learn_intrinsics = learn_intrinsics

        self.input_channels = 8  # pair of RGBD
        self.num_ch_conv = [self.input_channels, 16, 32, 64, 128, 256, 512, 1024]
        self.motion_encoder = nn.ModuleDict()

        self.auto_mask = auto_mask

        self.padding = []
        self.feature_dims = []
        H, W = self.input_shape
        kernel_size = 3
        stride = 2
        for i, (in_ch, out_ch) in enumerate(zip(self.num_ch_conv, self.num_ch_conv[1:])):
            self.motion_encoder[f"conv_{i}"] = MotionConvBlock(in_ch, out_ch, (W % 2, 1, H % 2, 1))
            pad = 2 * (kernel_size // 2)
            H = (H - kernel_size + pad) // stride + 1
            W = (W - kernel_size + pad) // stride + 1
            self.feature_dims.append((H, W))

        self.global_pooling = nn.AdaptiveAvgPool2d(output_size=1)

        self.background_motion_pred = nn.Conv2d(self.num_ch_conv[-1], 6, kernel_size=1, stride=1, bias=False)

        self.residual_translation_pred = nn.Conv2d(6, 3, kernel_size=1, stride=1)

        self.motion_refinement = nn.ModuleDict()
        for i, (num_ch, feature_dim) in enumerate(
            zip(reversed(self.num_ch_conv), reversed([self.input_shape] + self.feature_dims))
        ):
            self.motion_refinement[f"refine_{i}"] = MotionRefinementBlock(num_ch, feature_dim)

        self.scaler = RotTransScaler()

        if self.learn_intrinsics:
            self.intrinsics_layer = nn.Sequential(nn.Conv2d(1024, 2, kernel_size=1, stride=1), nn.Softplus())
            self.intrinsics_layer_offset = nn.Conv2d(1024, 2, kernel_size=1, stride=1)

    def _intrinsic_layer(self, x, h, w):
        batch_size = x.shape[0]
        offsets = self.intrinsics_layer_offset(x)
        focal_lengths = self.intrinsics_layer(x)
        focal_lengths = focal_lengths.squeeze(2).squeeze(2) + 0.5
        focal_lengths = focal_lengths * torch.tensor([[w, h]], dtype=x.dtype, device=x.device)
        offsets = offsets.squeeze(2).squeeze(2) + 0.5
        offsets = offsets * torch.tensor([[w, h]], dtype=x.dtype, device=x.device)
        foci = torch.diagflat(focal_lengths[0]).unsqueeze(0)

        for b in range(1, batch_size):
            foci = torch.cat((foci, torch.diagflat(focal_lengths[b]).unsqueeze(0)), dim=0)
        intrinsic_mat = torch.cat([foci, torch.unsqueeze(offsets, -1)], dim=2)
        last_row = torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(batch_size, 1, 1).to(device=x.device)
        intrinsic_mat = torch.cat([intrinsic_mat, last_row], dim=1)
        return intrinsic_mat

    def _mask(self, x):
        sq_x = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        mean_sq_x = torch.mean(sq_x, dim=(0, 2, 3))
        mask_x = (sq_x > mean_sq_x).type(x.dtype)
        x = x * mask_x
        return x

    def forward(self, x, background_motion=None):
        features = [x]
        for i in range(len(self.num_ch_conv) - 1):
            x = self.motion_encoder[f"conv_{i}"](x)
            features.append(x)

        bottleneck = self.global_pooling(features[-1])
        rotation, background_translation = None, None
        use_background_motion = background_motion is not None
        if not use_background_motion:
            background_motion = self.background_motion_pred(bottleneck)

            rotation = background_motion[:, :3, 0, 0]
            background_translation = background_motion[:, 3:]
        else:
            background_translation = background_motion[:, 3:]

        # residual_translation = self.residual_translation_pred(background_motion)
        residual_translation = background_translation
        for i, feature in enumerate(reversed(features)):
            residual_translation = self.motion_refinement[f"refine_{i}"](residual_translation, feature)

        residual_translation = self.scaler(residual_translation, "trans")
        if not use_background_motion:
            rotation = self.scaler(rotation, "rot")
            background_translation = self.scaler(background_translation, "trans")

        intrinsics_mat = None
        if self.learn_intrinsics:
            intrinsics_mat = self._intrinsic_layer(bottleneck, self.input_shape[0], self.input_shape[1])

        # automasking
        if self.auto_mask:
            residual_translation = self._mask(residual_translation)

        return (rotation, background_translation, residual_translation, intrinsics_mat)


class MotionRefinementBlock(nn.Module):
    def __init__(self, num_channels, dims, num_motion_fields: int = 3):
        super().__init__()

        self.num_channels = num_channels
        self.dims = dims
        self.num_motion_fields = num_motion_fields

        self.num_mid_channels = max(4, self.num_channels)

        self.conv1 = nn.Conv2d(
            self.num_channels + self.num_motion_fields, self.num_mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2a = nn.Conv2d(
            self.num_channels + self.num_motion_fields, self.num_mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2b = nn.Conv2d(self.num_mid_channels, self.num_mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_mid_channels * 2, self.num_motion_fields, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, motion_field, feature):
        unsampled_motion_field = F.interpolate(motion_field, size=self.dims, mode="bilinear")
        x = torch.cat((unsampled_motion_field, feature), dim=1)
        out1 = self.relu(self.conv1(x))

        out2 = self.relu(self.conv2a(x))
        out2 = self.relu(self.conv2b(out2))

        out = torch.cat((out1, out2), dim=1)
        out = unsampled_motion_field + self.conv3(out)
        return out


class RotTransScaler(nn.Module):
    """
    The network to learn a scale factor shared by rotation and translation
    """

    def __init__(self, minimum=0.001):
        super().__init__()
        self.rot_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.trans_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.minimum = minimum
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rot_or_trans):
        if rot_or_trans == "rot":
            scale = self.rot_scale
        elif rot_or_trans == "trans":
            scale = self.trans_scale
        else:
            raise NotImplementedError(f"{rot_or_trans} mode does not exist.")
        # scale = self.relu(scale - self.minimum) + self.minimum
        return x * scale


class MotionConvBlock(nn.Module):
    """
    Layer to perform a convolution followed by ReLU
    """

    def __init__(self, in_channels, out_channels, padding):
        super().__init__()

        self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.nonlin(out)
        return out

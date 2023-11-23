import logging

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv1x1

from models.layers import make_randomized_layernorm

log = logging.getLogger(__name__)


class DepthModel(nn.Module):
    """
    Sources:
        - https://github.com/bolianchen/pytorch_depth_from_videos_in_the_wild
    """

    def __init__(
        self,
        pretrained=True,
        num_layers: int = 18,
        use_layernorm: bool = False,
        layernorm_noise_rampup_steps: int = 10000,
    ):
        super(DepthModel, self).__init__()

        if use_layernorm:
            self._norm_layer = make_randomized_layernorm(noise_rampup_steps=layernorm_noise_rampup_steps)
        else:
            self._norm_layer = nn.BatchNorm2d

        self.depth_encoder = DepthEncoder(pretrained, num_layers, self._norm_layer)

        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc)

    def forward(self, x):
        features = self.depth_encoder(x)
        return self.depth_decoder(features)


class DepthEncoder(nn.Module):
    def __init__(self, pretrained, num_layers: int = 18, norm_layer=None, use_norm_in_downsample=True):
        super(DepthEncoder, self).__init__()

        self.use_norm_in_downsample = use_norm_in_downsample

        # channels info will be passed to decoder
        self.num_ch_enc = [64, 64, 128, 256, 512]

        self.encoder = self._build_encoder(num_layers, pretrained=pretrained, norm_layer=norm_layer)

        # remove the unused avgpool and fc layers
        self.encoder.avgpool = nn.Sequential()
        self.encoder.fc = nn.Sequential()
        self.pretrained = pretrained

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def _build_encoder(self, num_layers, pretrained=False, **kwargs):
        """Build a Resent Encoder
        Refactor the resent and _resnet functions from
        the torchvision.models.resnet module
        """
        # information to build pretrained resnet-based encoders
        model_ingredients = {
            "resnet18": (BasicBlock, [2, 2, 2, 2]),
            "resnet34": (BasicBlock, [3, 4, 6, 3]),
            "resnet50": (Bottleneck, [3, 4, 6, 3]),
        }

        model_name = "resnet" + str(num_layers)
        assert model_name in model_ingredients, "{} is not a valid number of resnet layers"
        ingredients = model_ingredients[model_name]
        block, layers = ingredients[0], ingredients[1]

        def _resnet(arch, block, layers, pretrained, **kwargs):
            model = self._depth_resnet()(block, layers, **kwargs)
            resnet_models = {
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
            }
            if pretrained:
                state_dict = resnet_models[arch](weights="IMAGENET1K_V1").state_dict()
                # ignore norm_layer related weights while using layernorm
                model.load_state_dict(state_dict, strict=False)
            return model

        return _resnet(model_name, block, layers, pretrained, **kwargs)

    def _depth_resnet(self):
        """Return a Customized ResNet for depth encoder"""

        use_norm_in_downsample = self.use_norm_in_downsample

        class DepthResNet(ResNet):
            """Replace batchnorm with layernorm"""

            def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
                norm_layer = self._norm_layer
                downsample = None
                previous_dilation = self.dilation
                if dilate:
                    self.dilation *= stride
                    stride = 1
                if stride != 1 or self.inplanes != planes * block.expansion:
                    if use_norm_in_downsample:
                        downsample = nn.Sequential(
                            conv1x1(self.inplanes, planes * block.expansion, stride),
                            norm_layer(planes * block.expansion),
                        )
                    else:
                        downsample = nn.Sequential(
                            conv1x1(self.inplanes, planes * block.expansion, stride),
                        )

                layers = []
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        stride,
                        downsample,
                        self.groups,
                        self.base_width,
                        previous_dilation,
                        norm_layer,
                    )
                )
                self.inplanes = planes * block.expansion
                # single iteration for resnet18
                for _ in range(1, blocks):
                    # replace batchnorm with layernorm for every layer
                    layers.append(
                        block(
                            self.inplanes,
                            planes,
                            groups=self.groups,
                            base_width=self.base_width,
                            dilation=self.dilation,
                            norm_layer=norm_layer,
                        )
                    )
                return nn.Sequential(*layers)

        return DepthResNet

    def forward(self, x):
        if self.pretrained:
            x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        features = []
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        return features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(1), num_output_channels=1, use_skip_connections=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skip_connections = use_skip_connections
        self.scales = scales

        self.num_ch_enc = num_ch_enc  # [64, 64, 128, 256, 512]
        self.num_ch_dec = [16, 32, 64, 128, 256]

        # decoder
        self.depth_decoder = nn.ModuleDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.depth_decoder[f"upconv_{i}_0"] = TransposeConv3x3(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skip_connections and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.depth_decoder[f"upconv_{i}_1"] = DepthConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.depth_decoder[f"depthconv_{s}"] = nn.Sequential(
                Conv3x3(self.num_ch_dec[s], self.num_output_channels), nn.Softplus()
            )

    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = [self.depth_decoder[f"upconv_{i}_0"](x)[:, :, :-1, :-1]]
            if self.use_skip_connections and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.depth_decoder[f"upconv_{i}_1"](x)
            if i in self.scales:
                self.outputs[("depth", i)] = self.depth_decoder[f"depthconv_{i}"](x)

        output = self.outputs[("depth", 0)]
        output = torch.clamp(output, min=1e-3, max=80)  # clip depth range
        return output


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""

    def __init__(self, in_channels, out_channels, use_refl=False):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            # constant zero padding
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
        )

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class TransposeConv3x3(nn.Module):
    """
    Upsampling layer
    """

    def __init__(self, in_channels, out_channels):
        super(TransposeConv3x3, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthConvBlock(nn.Module):
    """
    Layer to perform a convolution followed by ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DepthConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

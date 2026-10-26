# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn is True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn is True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn is True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
    ):
        """Init.

        Args:
            features (int): number of features
            has_residual (bool): if False, skip creation of resConfUnit1
                (used for the top fusion block which receives no skip input).
        """
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand is True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1
        )

        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            assert self.has_residual, "FeatureFusionBlock got skip input but has_residual=False"
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output.contiguous(), **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None, has_residual=True):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_frames,
        model_tubelet_size,
        grid_size,
        input_size,
        task_type="depth_estimation",
        num_classes=1,
        max_depth=10.0,
        use_bn=False,
        use_clstoken=False,
    ):
        super(DPTHead, self).__init__()

        assert model_tubelet_size == 1, "DPTHead currently only supports model_tubelet_size == 1"

        self.use_clstoken = use_clstoken
        self.task_type = task_type
        self.num_frames = num_frames
        self.t_size = num_frames // model_tubelet_size
        self.grid_size = grid_size
        self.input_size = input_size
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.patch_size = input_size // grid_size

        features = 256 if in_channels >= 1024 else 128
        out_channels = [256, 512, 1024, 1024] if in_channels >= 1024 else [96, 192, 384, 768]

        if task_type == "depth_estimation":
            num_output_channels = 1
        elif task_type == "semantic_segmentation":
            num_output_channels = num_classes
        else:
            num_output_channels = 1  # classification uses forward_features, output_conv2 unused

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        # refinenet4 receives no skip input, so resConfUnit1 would be unused (breaks DDP).
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, has_residual=False)

        head_features_1 = features
        head_features_2 = 32
        self.mid_channels = head_features_1 // 2

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        # output_conv2 is only used for dense prediction; classification pools after output_conv1.
        if task_type != "classification":
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, num_output_channels, kernel_size=1, stride=1, padding=0),
            )

    def forward_features(self, out_features, patch_h, patch_w):
        """Run DPT fusion up to (and including) output_conv1.

        Returns a feature map of shape (B*T, mid_channels, H_pre, W_pre) at the
        pre-final-interpolate resolution (typically patch_h*8, patch_w*8).
        """
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        return self.scratch.output_conv1(path_1)

    def forward(self, features):
        """
        features: list of 4 backbone layer outputs, each of shape (B, T*HW, C)
                  where T = t_size = num_frames // model_tubelet_size, HW = grid_size**2.
        Returns task-specific output:
          - depth_estimation: (B, num_frames, input_size, input_size)
          - semantic_segmentation: (B, num_classes, num_frames, input_size, input_size)
          - classification: (B, mid_channels) pooled features (linear head applied by Probe)
        """
        B = features[0].shape[0]
        T = self.t_size
        HW = self.grid_size * self.grid_size
        # (B, T*HW, C) -> (B*T, HW, C) so DPT processes each frame independently
        feats = [f.reshape(B * T, HW, -1) for f in features]

        mid = self.forward_features(feats, self.grid_size, self.grid_size)  # (B*T, mid, H_pre, W_pre)

        if self.task_type == "classification":
            # spatial GAP then temporal mean -> (B, mid_channels)
            pooled = mid.mean(dim=(2, 3)).reshape(B, T, -1).mean(dim=1)
            return pooled

        mid = F.interpolate(
            mid,
            (self.input_size, self.input_size),
            mode="bilinear",
            align_corners=True,
        )
        dense = self.scratch.output_conv2(mid)  # (B*T, num_out, input_size, input_size)

        if self.task_type == "depth_estimation":
            out = dense.squeeze(1).reshape(B, T, self.input_size, self.input_size)
            return torch.sigmoid(out) * self.max_depth  # (B, num_frames, H, W)
        elif self.task_type == "semantic_segmentation":
            out = dense.reshape(B, T, self.num_classes, self.input_size, self.input_size)
            return out.permute(0, 2, 1, 3, 4)  # (B, num_classes, num_frames, H, W)
        else:
            raise NotImplementedError(f"DPT forward not implemented for task_type={self.task_type}")

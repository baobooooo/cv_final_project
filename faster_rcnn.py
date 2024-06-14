import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            feat_shape = inner_lateral.shape[-2:]
            last_inner = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + last_inner
            results.insert(0, self.layer_blocks[idx](last_inner))
        return results

class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()
        self.resnet = ResNet34()
        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 256)
        self.out_channels = 256  # Adding the required out_channels attribute

    def forward(self, x):
        features = self.resnet(x)
        fpn_output = self.fpn(features)
        out_dict = {str(i): feature for i, feature in enumerate(fpn_output)}
        return out_dict

class CustomFasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomFasterRCNN, self).__init__()

        # 使用自定义的 backbone
        backbone = CustomBackbone()

        # 定义 RPN 的 anchor generator
        anchor_sizes = ((32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        # 创建 anchor generator
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

        # 定义 RoI align layer
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # 创建 Faster R-CNN 模型
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        # 定义 GeneralizedRCNNTransform
        # self.transform = CustomGeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    def forward(self, images, targets=None):
        # images, targets = self.transform(images, targets)
        return self.model(images, targets)

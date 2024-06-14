import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF


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

class AnchorGenerator(nn.Module):
    def __init__(self, sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5):
        super(AnchorGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def forward(self, feature_maps):
        anchors = []
        for size, aspect_ratios in zip(self.sizes, self.aspect_ratios):
            for feature_map in feature_maps:
                anchors.append(self.generate_anchors(feature_map, size, aspect_ratios))
        return anchors

    def generate_anchors(self, feature_map, size, aspect_ratios):
        anchors = []
        height, width = feature_map.shape[-2:]
        for y in range(height):
            for x in range(width):
                for aspect_ratio in aspect_ratios:
                    w = size[0] * aspect_ratio ** 0.5
                    h = size[0] / aspect_ratio ** 0.5
                    anchors.append([x, y, w, h])
        return torch.tensor(anchors).float()

class MultiScaleRoIAlign(nn.Module):
    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(MultiScaleRoIAlign, self).__init__()
        self.featmap_names = featmap_names
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio

    def forward(self, x, proposals, image_shapes):
        pooled_features = []
        for proposal, image_shape in zip(proposals, image_shapes):
            pooled_feature = self.roi_align(x, proposal, image_shape)
            pooled_features.append(pooled_feature)
        return pooled_features

    def roi_align(self, x, proposal, image_shape):
        return F.adaptive_max_pool2d(x[0], self.output_size)

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.cls_logits(x), self.bbox_pred(x)

class FasterRCNN(nn.Module):
    def __init__(self, backbone, num_classes, rpn_anchor_generator, box_roi_pool):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn_anchor_generator = rpn_anchor_generator
        self.box_roi_pool = box_roi_pool

        self.rpn_head = RPNHead(256, 3)
        
        # 添加 box_predictor 分类和回归分支
        self.box_predictor_cls = nn.Linear(256 * box_roi_pool.output_size * box_roi_pool.output_size, num_classes)
        self.box_predictor_reg = nn.Linear(256 * box_roi_pool.output_size * box_roi_pool.output_size, 4)

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images)
        features = self.backbone(images)
        feature_maps = features
        anchors = self.rpn_anchor_generator(feature_maps)
        rpn_cls_logits, rpn_bbox_pred = self.rpn_head(feature_maps[0])
        proposals = [rpn_bbox_pred]
        box_features = self.box_roi_pool(feature_maps, proposals, [img.shape[-2:] for img in images])
        box_features = torch.cat(box_features, dim=0)

        # 计算分类和回归预测
        class_logits = self.box_predictor_cls(box_features.view(box_features.size(0), -1))
        box_regression = self.box_predictor_reg(box_features.view(box_features.size(0), -1))

        if self.training:
            labels = torch.cat([t["labels"] for t in targets], dim=0)
            boxes = torch.cat([t["boxes"] for t in targets], dim=0)

            # 匹配分类 logits 和标签
            loss_classifier = F.cross_entropy(class_logits, labels[:class_logits.size(0)])

            # 计算回归损失
            loss_box_reg = F.smooth_l1_loss(box_regression, boxes[:box_regression.size(0)])

            loss_objectness = F.binary_cross_entropy_with_logits(rpn_cls_logits, torch.zeros_like(rpn_cls_logits))
            loss_rpn_box_reg = F.smooth_l1_loss(rpn_bbox_pred, torch.zeros_like(rpn_bbox_pred))

            loss_dict = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
            return loss_dict
        else:
            return class_logits


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        device = images[0].device  # 获取第一个图像的设备
        images = [self.normalize(image) for image in images]
        images = [self.resize(image) for image in images]
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images).to(device)  # 确保批处理后的图像在同一个设备上
        if targets is not None:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        return images, targets

    def normalize(self, image):
        device = image.device  # 获取图像的设备
        mean = torch.tensor(self.image_mean).to(device).view(-1, 1, 1)
        std = torch.tensor(self.image_std).to(device).view(-1, 1, 1)
        return (image - mean) / std

    def resize(self, image):
        size = self.min_size
        image = TF.resize(image, size)
        return image

    def batch_images(self, images, size_divisible=32):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = (max_size[1] + stride - 1) // stride * stride
        max_size[2] = (max_size[2] + stride - 1) // stride * stride
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_zeros(batch_shape)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        return batched_imgs

class CustomFasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomFasterRCNN, self).__init__()

        # 使用 resnet_fpn 作为 backbone
        backbone = self.resnet_fpn_backbone()

        # 定义 RPN 的 anchor generator
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
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
        self.transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    def resnet_fpn_backbone(self):
        resnet = ResNet34()
        in_channels_list = [64, 128, 256, 512]
        out_channels = 256

        fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        return nn.Sequential(resnet, fpn)

    def forward(self, images, targets=None):
        images, targets = self.transform(images, targets)
        return self.model(images, targets)

# 使用示例
num_classes = 91  # 类别数量
model = CustomFasterRCNN(num_classes)

# 创建虚拟输入
images = [torch.randn(3, 800, 800), torch.randn(3, 600, 600)]  # Batch size 2
targets = [{"boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32), "labels": torch.tensor([1])} for _ in range(2)]

# 传递给模型并输出结果
output = model(images, targets)
print(output)

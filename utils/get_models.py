import os
import sys
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.utils
import torchvision
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import copy


def get_model(arch, class_num, pretrained=False):
    # pytorch resnet
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained, num_classes=class_num)
    if arch == 'resnet34':
        model = models.resnet34(pretrained=pretrained, num_classes=class_num)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained, num_classes=class_num)
    elif arch == 'resnet152':
        model = models.resnet152(pretrained=pretrained, num_classes=class_num)
    elif arch == 'resnext101':
        model = models.resnext101(pretrained=pretrained, num_classes=class_num)
    elif arch == 'resnet152':
        model = models.resnet152(pretrained=pretrained, num_classes=class_num)
    # pytorch vgg
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=pretrained, num_classes=class_num)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=pretrained, num_classes=class_num)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=pretrained, num_classes=class_num)
    elif arch == 'vgg16_bn':
        model = models.vgg16(pretrained=pretrained, num_classes=class_num)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=pretrained, num_classes=class_num)
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=pretrained, num_classes=class_num)
    # pytorch densenet
    elif arch == "densenet121":
        model = models.densenet121(pretrained=pretrained, num_classes=class_num)
    elif arch == "densenet169":
        model = models.densenet169(pretrained=pretrained, num_classes=class_num)
    
    # https://github.com/kuangliu/pytorch-cifar
    elif arch == "cifar10_resnet18":
        sys.path.append("../pytorch-cifar/models")
        from resnet import ResNet18

        model = ResNet18()
    elif arch == "cifar10_vgg16_bn":
        sys.path.append("../pytorch-cifar/models")
        from vgg import VGG

        model = VGG("VGG16")
    elif arch == "cifar10_densenet121":
        sys.path.append("../pytorch-cifar/models")
        from densenet import densenet_cifar

        model = densenet_cifar()
        
    # https://github.com/weiaicunzai/pytorch-cifar100
    elif arch == "cifar100_resnet18":
        sys.path.append("pytorch-cifar100/models")
        from resnet import resnet18

        model = resnet18()
    elif arch == "cifar100_vgg16_bn":
        sys.path.append("pytorch-cifar100/models")
        from vgg import vgg16_bn

        model = vgg16_bn()
    elif arch == "cifar100_densenet121":
        sys.path.append("pytorch-cifar100/models")
        from densenet import densenet121

        model = densenet121()
    
    # timm models
    elif arch == "xception":
        model = timm.create_model(
            "xception", pretrained=pretrained, num_classes=class_num
        )
    elif arch == "vit_base_patch16_224":
        print(f"{arch} is pre-trained on ImageNet-21k.")
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=class_num
        )
    elif arch == "BiT_M":
        print(f"{arch} is pre-trained on ImageNet-21k.")
        model = timm.create_model(
            "resnetv2_101x1_bitm",
            pretrained=pretrained,
            num_classes=class_num,
        )
    elif arch == "resnext101_32x8d_wsl":
        model = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return model


def load_model(model, file_name):
    assert os.path.exists(file_name), "No exps found. {}".format(file_name)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded ... ", file_name)
    best_acc, best_epoch = checkpoint["acc"], checkpoint["epoch"]
    print("best_acc:{} at epoch {}".format(best_acc, best_epoch))
    return model, best_acc, best_epoch


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        hook_layers,
        return_dict=True,
    ):
        super(FeatureExtractor, self).__init__()
        self.return_dict = return_dict

        self.model = copy.deepcopy(model)

        if isinstance(hook_layers, list):
            self.hook_layers = hook_layers
            self.hook_layers_dict = {k: k for k in hook_layers}
        elif isinstance(hook_layers, dict):
            self.hook_layers = [k for k, v in hook_layers.items()]
            self.hook_layers_dict = hook_layers
            # hook_layers_dict = {original_name: return_name}
        print("hook_layers:", hook_layers)

        added_layer_names = []
        for name, module in self.model.named_modules():
            if name in self.hook_layers:
                module.register_forward_hook(self.extract())
                added_layer_names += [name]

        assert len(added_layer_names) == len(
            hook_layers
        ), f"Some layer did not exist. {set(added_layer_names) - set(hook_layers)},{set(hook_layers) - set(added_layer_names)}"

        self.features = []

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def avg_pool_feature(self, o):
        """o: cpu feature map"""
        if len(o.shape) == 4:
            feat = self.avgpool(o).reshape(o.shape[0], -1).data
        elif len(o.shape) == 3:
            feat = torch.mean(o, 1).data
        elif len(o.shape) == 2:
            feat = o.data
        else:
            print(k, o.shape)
            raise ValueError
        return feat

    def extract(self):
        def _extract(module, f_in, f_out):
            f_out = self.avg_pool_feature(f_out)
            self.features.append(f_out)

        return _extract

    def forward(self, input):
        _ = self.model(input)
        assert len(self.features) == len(self.hook_layers), (
            "Something's wrong.",
            len(self.features),
            len(self.hook_layers),
        )
        if self.return_dict:
            d = {
                self.hook_layers_dict[k]: feat
                for k, feat in zip(self.hook_layers, self.features)
            }
            self.features = []
            return d
        else:
            features = self.features
            self.features = []
            return features

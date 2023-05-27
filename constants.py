import torchvision


# layer names: basically, each block.
def get_layers_name(
    model_name, model=None, add_logits=True, get_all=False, get_layer_idx=False
):
    if get_all:
        assert model is not None
        if model_name == "cifar100_vgg16_bn" or "cifar10_" in model_name:
            return_nodes = [name for name, module in model.named_modules()]
            return_nodes = [
                l
                for l in return_nodes
                if l not in ["", "features", "classifier", "size", "view", "getitem"]
            ]
            print(return_nodes)
            return return_nodes

        return_nodes = torchvision.models.feature_extraction.get_graph_node_names(
            model
        )[0]
        print(return_nodes)
        return return_nodes

    if "cifar100_resnet18" == model_name:
        return_nodes = ["conv1", "conv2_x", "conv3_x", "conv4_x", "conv5_x"]
    elif "cifar100_vgg16_bn" == model_name:
        return_nodes = [
            "features.6",
            "features.13",
            "features.23",
            "features.33",
            "features.43",
        ]
    elif "cifar100_densenet121" == model_name:
        return_nodes = [
            "conv1",
            "features.transition_layer_0",
            "features.transition_layer_1",
            "features.transition_layer_2",
            "avgpool",
        ]

    elif "cifar100_mobilenetv2_x0_5" == model_name:
        return_nodes = []
        for j in range(0, 19):
            name = f"features.{j}"
            return_nodes.append(name)

    elif "resnet" in model_name:
        return_nodes = [
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]

    elif "vgg16" in model_name:
        return_nodes = [
            "features.4",
            "features.9",
            "features.16",
            "features.23",
            "features.30",
        ]

    elif "densenet" in model_name:
        return_nodes = [
            "features.pool0",
            "features.transition1",
            "features.transition2",
            "features.transition3",
            "features.norm5",
        ]

    elif "xception" == model_name:
        return_nodes = ["act2"]
        for i in range(1, 13):
            name = f"block{i}"
            return_nodes.append(name)
        return_nodes.append("global_pool")

    elif "vit_base_patch16_224" in model_name:
        return_nodes = []
        for i in range(0, 12):
            name = f"blocks.{i}"
            return_nodes.append(name)
        return_nodes.append("fc_norm")

    elif "resnext101_32x8d_wsl" in model_name:
        return_nodes = [
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]

    elif "BiT" in model_name:
        return_nodes = ["stem"]
        for j in range(0, 3):
            name = f"stages.0.blocks.{j}"
            return_nodes.append(name)
        for j in range(0, 4):
            name = f"stages.1.blocks.{j}"
            return_nodes.append(name)
        for j in range(0, 23):
            name = f"stages.2.blocks.{j}"
            return_nodes.append(name)
        for j in range(0, 3):
            name = f"stages.3.blocks.{j}"
            return_nodes.append(name)
        return_nodes.append("norm")

    else:
        raise NotImplementedError

    if add_logits:
        if model is None:
            return_nodes += ["logits"]
        else:
            return_nodes = {k: k for k in return_nodes}
            last_layer_name = [k for k, v in model.named_modules()][-1]
            return_nodes[last_layer_name] = "logits"

    if get_layer_idx:
        assert model is not None
        all_nodes = torchvision.models.feature_extraction.get_graph_node_names(model)[0]
        print("all:", [(i, l) for i, l in enumerate(all_nodes)])
        layer_idx = []
        for i, l in enumerate(all_nodes):
            if l in return_nodes:
                layer_idx += [i]
            elif i == len(all_nodes) - 1 and add_logits:
                layer_idx += [i]
        print(layer_idx, len(all_nodes))
        return return_nodes, (layer_idx, len(all_nodes))

    return return_nodes

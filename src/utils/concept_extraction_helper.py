import torch
import torchvision.transforms

import eval_model
import numpy as np
from PIL import Image
import os
import timm
from models.convnet import ConvNet, ConvNetTiny








def patchify_images(inputs, patch_size, strides):
    assert len(inputs.shape) == 4, "Input data must be of shape (n_samples, channels, height, width)."
    assert inputs.shape[2] == inputs.shape[3], "Input data must be square."

    image_size = inputs.shape[2]

    # extract patches from the input data, keep patches on cpu
    strides = int(patch_size * 0.80)

    patches = torch.nn.functional.unfold(inputs, kernel_size=patch_size, stride=strides)
    patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(4, 4)
    # for axi, ax in enumerate(axes.flatten()):
    #     ax.axis('off')
    #     ax.imshow(patches[axi].permute(1, 2, 0))
    # plt.show()
    return patches


def load_feature_extraction_layers(model, feature_layer_params):
    out = {}
    feature_layer_version = feature_layer_params['feature_layer_version']

    if isinstance(model, timm.models.resnet.ResNet):
        if feature_layer_version == 'v0':
            out['layer_type'] = 'layer'
            out['layers'] = [model.layer1, model.layer2, model.layer3, model.layer4]
            out['layer_names'] = ['layer1', 'layer2', 'layer3', 'layer4']

            def post_activation_func(x):
                return x.mean((-1, -2))
            out['post_activation_func'] = post_activation_func

        elif feature_layer_version == 'v1':
            layers = []
            layer_names = []
            for name, module in model.named_modules():
                if type(module) == torch.nn.modules.ReLU:
                    layers.append(module)
                    layer_names.append(name)
            out['layers'] = layers
            out['layer_names'] = layer_names
            out['layer_type'] = 'relu'

            def post_activation_func(x):
                return x.mean((-1, -2))
            out['post_activation_func'] = post_activation_func

        else:
            raise ValueError(f'Unknown feature_layer_version: {feature_layer_version}')

    elif isinstance(model, timm.models.vision_transformer.VisionTransformer):
        if feature_layer_version == 'v0':
            out['layer_type'] = 'blmlp'
            out['layers'] = [model.blocks[i] for i in range(len(model.blocks))]
            out['layer_names'] = [f'block{i}' for i in range(len(model.blocks))]
            def post_activation_func(x):
                return x[:, 0, :] # take the first token
            out['post_activation_func'] = post_activation_func

        elif feature_layer_version == 'cls_tok_ll':
            out['layer_type'] = 'blmlp'
            out['layers'] = [model.blocks[len(model.blocks) - 1]]
            out['layer_names'] = [f'block{len(model.blocks)-1}']
            def post_activation_func(x):
                return x[:, 0, :] # take the first token
            out['post_activation_func'] = post_activation_func

        elif feature_layer_version == 'v1':
            out['layer_type'] = 'norm'
            out['layers'] = [model.norm]
            out['layer_names'] = ['norm']
            def post_activation_func(x):
                return x[:, 0, :]
            out['post_activation_func'] = post_activation_func

        elif feature_layer_version == 'output_logits':
            out['layer_type'] = 'head'
            out['layers'] = [model.head]
            out['layer_names'] = ['head']
            def post_activation_func(x):
                return x
            out['post_activation_func'] = post_activation_func
        else:
            raise ValueError(f'Unknown feature_layer_version: {feature_layer_version}')

    elif isinstance(model, ConvNet) or isinstance(model, ConvNetTiny):
        if feature_layer_version == 'v0':
            out['layer_type'] = 'conv'
            out['layers'] = [model.layer2]
            out['layer_names'] = ['layer2']
            def post_activation_func(x):
                return x.mean((-1, -2))
            out['post_activation_func'] = post_activation_func
        else:
            raise ValueError(f'Unknown feature_layer_version: {feature_layer_version}')
    else:
        raise ValueError(f'Unknown model type: {type(model)}')

    return out


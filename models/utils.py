import torch
import torchvision
from models.mlp import MLP
from models.convnet import ConvNet, ConvNetTiny
# from models.concept_transformer import ConceptTransformer
import timm

vision_model_opts = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101,
}

model_weights = {
    'resnet18': torchvision.models.ResNet18_Weights,
    'resnet34': torchvision.models.ResNet34_Weights,
    'resnet50': torchvision.models.ResNet50_Weights,
    'resnet101': torchvision.models.ResNet101_Weights,
}


def construct_model(model_type, params=None):
    if model_type in vision_model_opts:
        weights = None
        pretrained = params['pretrained']
        if pretrained:
            weights = model_weights[model_type]
        model = vision_model_opts[model_type](weights=weights)
    elif model_type == 'mlp':
        model = MLP(input_size=params['input_size'], hidden_size=params['hidden_size'],
                    output_size=params['output_size'], num_layers=params['num_layers'])
    # elif model_type == 'concept_transformer':
    #    model = ConceptTransformer(d_model=params['d_model'], n_head=params['n_head'], num_layers=params['num_layers'],
    #                               input_emb_dim=params['input_emb_dim'], num_inputs=params['num_inputs'],
    #                               output_size=params['output_size'])
    elif model_type == 'mnist_conv':
        model = ConvNet(num_classes=params['num_classes'])
    elif model_type == 'mnist_conv_tiny':
        model = ConvNetTiny(num_classes=params['num_classes'])

    elif model_type == 'timm_model':
        model = timm.create_model(params['model_name'], pretrained=params['pretrained'])

    else:
        raise NotImplementedError

    return model


def modify_model_output_layer(model, num_classes):
    if 'torchvision.models.resnet' in str(type(model)) or 'ConvNet' in str(type(model)):
        if num_classes is None:
            model.fc = torch.nn.Identity()
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif 'timm.models' in str(type(model)):
        if 'resnet' in str(type(model)):
            if num_classes is None:
                model.fc = torch.nn.Identity()
            else:
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif 'vision_transformer' in str(type(model)):
            if hasattr(model.head, 'in_features'):
                if num_classes is None:
                    model.head = torch.nn.Identity()
                else:
                    model.head = torch.nn.Linear(model.head.in_features, num_classes)
            else:
                if num_classes is None:
                    model.head = torch.nn.Identity()
                else:
                    model.head = torch.nn.Linear(model.embed_dim, num_classes)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

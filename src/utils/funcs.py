from datasets.utils.dataset_loader import get_dataset

import torch
import numpy as np
import random
import torchvision
from math import ceil
from sklearn.utils._testing import ignore_warnings
import json
import eval_model
from PIL import Image
import os

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@ignore_warnings(category=UserWarning)
def _batch_inference(model, dataset, batch_size=128, resize=None, device='cuda'):
    '''
    Code from CRAFT repository
    Used to perform inference on a dataset in batches.
    '''
    nb_batchs = ceil(len(dataset) / batch_size)
    start_ids = [i * batch_size for i in range(nb_batchs)]

    results = []

    with torch.no_grad():
        for i in start_ids:
            x = torch.tensor(dataset[i:i + batch_size])
            x = x.to(device)

            if resize:
                x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)

            results.append(model(x).cpu())

    results = torch.cat(results)
    return results

def _group_images(method, params):
    '''
    Group images based on the specified method and parameters.
    :param method: str, one of 'craft', 'topkprobs', 'topkconfusion', 'ground_truth'
    :param params: dict, parameters required for the specified method
    :return:
    '''

    if method == 'craft':
        # group images by their predicted class (CRAFT format)

        predictions = params['predictions']
        pred_label_groups = eval_model.convert_predictions_to_label_groups(predictions)

        # subsample N images per class
        num_images = params['num_images']
        seed = params['seed']
        if num_images is not None:
            rng = np.random.default_rng(seed)
            subsampled_label_groups = {}
            for i in pred_label_groups.keys():
                path_list = sorted(pred_label_groups[i])
                rng.shuffle(path_list)
                subsampled_label_groups[i] = path_list[:num_images]

            pred_label_groups = subsampled_label_groups

        output = pred_label_groups

    elif method == 'topkprobs':
        # images belong to a class group if it is one of their top k probabilities and is above the threshold

        probs = params['probs']
        k = params['k']
        threshold = params['threshold']
        label_groups = eval_model.convert_probs_to_label_groups(probs, k=k, threshold=threshold)

        # subsample N images per class
        num_images = params['num_images']
        seed = params['seed']
        if num_images is not None:
            rng = np.random.default_rng(seed)
            subsampled_label_groups = {}
            for i in label_groups.keys():
                path_list = sorted(label_groups[i])
                rng.shuffle(path_list)
                subsampled_label_groups[i] = path_list[:num_images]

            label_groups = subsampled_label_groups
        output = label_groups
    elif method == 'topkconfusion':
        # images belong to a class group if it is one of their top k probabilities in the confusion matrix
        # (i.e. average over all images in the class)

        probs = params['probs']
        out = params['model_out']
        target_classes = params['target_classes']
        if target_classes is None:
            target_classes = np.unique(out['labels'])
        labels = np.array(out['labels'])
        filenames = np.stack(list(probs.keys()))
        prob_matrix = np.stack(list(probs.values()))
        prob_conf_matrix = np.zeros((prob_matrix.shape[1], prob_matrix.shape[1]))
        for i in range(prob_conf_matrix.shape[0]):
            mask = labels == i
            prob_conf_matrix[i] = prob_matrix[mask].mean(0)
        k = params['k']
        threshold = params['threshold']

        # subsample N images per class
        num_images = params['num_images']
        seed = params['seed']
        rng = np.random.default_rng(seed)
        label_groups = {}
        for i in target_classes:
            top_k_vals, top_k_ind = torch.topk(torch.tensor(prob_conf_matrix[i]), k=k)
            label_groups[i] = []
            for ind in top_k_ind:
                ind_mask = labels == ind.item()
                path_list = filenames[ind_mask]
                if num_images is not None:
                    _ni = min(num_images, len(path_list))
                    if ind.item() == i:
                        path_list = rng.choice(path_list, _ni, replace=False).tolist()
                    else:
                        image_inds = prob_matrix[ind_mask][:, i].argsort()[-_ni:]
                        path_list = path_list[image_inds].tolist()
                else:
                    path_list = path_list.tolist()
                label_groups[i].extend(path_list)


        output = label_groups

    elif method == 'ground_truth':
        # group images by their ground truth labels

        paths = params['paths']
        labels = params['labels']
        target_classes = params['target_classes']
        if target_classes is None:
            target_classes = np.unique(labels)
        labels = np.array(labels)
        paths = np.array(paths)
        label_groups = {}
        for un_labels in target_classes:
            label_groups[un_labels] = paths[labels == un_labels].tolist()

        num_images = params['num_images']
        seed = params['seed']
        if num_images is not None:
            rng = np.random.default_rng(seed)
            subsampled_label_groups = {}
            for i in label_groups.keys():
                path_list = sorted(label_groups[i])
                rng.shuffle(path_list)
                subsampled_label_groups[i] = path_list[:num_images]

            label_groups = subsampled_label_groups

        output = label_groups
    else:
        raise ValueError(f'Unknown method: {method}')

    return output


def create_image_group(strategy, param_dicts, return_eval_dict=False, return_labels=False):
    # calls _group_images with the appropriate parameters based on the strategy, handles merging of image groups if needed

    if strategy == 'craft':
        # Load/compute model predictions
        dataset_params = param_dicts['dataset_params']
        dataset = dataset_params['dataset_name']
        dataset_seed = dataset_params['seed']
        split = dataset_params['split']
        model_name, ckpt_path = param_dicts['model']
        model_eval_path = f'model_evaluation/{dataset}/{model_name}_{split}.json'
        try:
            with open(model_eval_path, 'r') as f:
                eval_dict = json.load(f)
                predictions = eval_dict['predictions']
        except FileNotFoundError:
            predictions = eval_model.main(model_name, dataset, split, ckpt_path, data_root='./data')['predictions']

        # load images
        print('Loading images')
        image_group = _group_images(method='craft', params={'predictions': predictions,
                                                               'num_images': dataset_params['num_images'],
                                                               'seed': dataset_seed})
    elif strategy == 'topkprobs':
        dataset_params = param_dicts['dataset_params']
        fe_params = param_dicts['feature_extraction_params']
        dataset = dataset_params['dataset_name']
        dataset_seed = dataset_params['seed']
        split = dataset_params['split']
        model_name, ckpt_path = param_dicts['model']
        model_probs_path = f'model_evaluation/{dataset}/{model_name}_probs_{split}.pth'
        try:
            probs_dict = torch.load(model_probs_path)
        except Exception as e:
            print(f'Could not load probs from {model_probs_path}')
            print('Computing probs...')
            _, probs_dict = eval_model.main(model_name, dataset, split, ckpt_path, data_root='./data',
                                            return_probs=True)

        # load images
        print('Loading images')
        image_group = _group_images(method='topkprobs', params={'probs': probs_dict,
                                                                   'num_images': dataset_params['num_images'],
                                                                   'k': fe_params['topk_group_size'],
                                                                   'threshold': fe_params['topk_prob_thresh'],
                                                                   'seed': dataset_seed})
    elif strategy == 'ground_truth':
        paths = param_dicts['paths']
        num_images = param_dicts['num_images']
        labels = param_dicts['labels']
        dataset_seed = param_dicts['dataset_seed']
        target_classes = param_dicts.get('target_classes', None)
        # load images
        print('Loading images')
        image_group = _group_images(method='ground_truth',
                                       params={
                                           'paths': paths,
                                           'labels': labels,
                                           'target_classes': target_classes,
                                           'num_images': num_images,
                                           'seed': dataset_seed})


    elif strategy == 'topkprobs-union':  # requires two models

        data_group_params = param_dicts[0]
        dataset = data_group_params['dataset_name']
        dataset_seed = data_group_params['seed']
        split = data_group_params['split']
        num_images = data_group_params['num_images']
        topk_group_size = data_group_params['topk_group_size']
        threshold = data_group_params['topk_prob_thresh']

        image_groups = []
        eval_dict = []
        for pd in param_dicts[1:]:

            model_name, ckpt_path = pd['model']
            model_probs_path = f'model_evaluation/{dataset}/{model_name}_probs_{split}.pth'
            try:
                probs_dict = torch.load(model_probs_path)
            except Exception as e:
                print(f'Could not load probs from {model_probs_path}')
                print('Computing probs...')
                _, probs_dict = eval_model.main(model_name, dataset, split, ckpt_path, data_root='./data',
                                                return_probs=True)

            # load images
            print('Loading images')
            image_group = _group_images(method='topkprobs', params={'probs': probs_dict,
                                                                       'num_images': num_images,
                                                                       'k': topk_group_size,
                                                                       'threshold': threshold,
                                                                       'seed': dataset_seed})
            image_groups.append(image_group)
        # TODO why is there no union for this method?

    elif strategy == 'topkconfusion-union':
        data_group_params = param_dicts[0]
        dataset = data_group_params['dataset_name']
        dataset_seed = data_group_params['seed']
        split = data_group_params['split']
        num_images = data_group_params['num_images']
        topk_group_size = data_group_params['topk_group_size']
        target_classes = data_group_params['target_classes']
        threshold = data_group_params['topk_prob_thresh']

        image_groups = []
        eval_dict = []
        for pd in param_dicts[1:]:

            model_name, ckpt_path = pd['model']
            model_probs_path = f'model_evaluation/{dataset}/{model_name}_probs_{split}.pth'
            model_out_path = f'model_evaluation/{dataset}/{model_name}_{split}.json'
            try:
                print('Loading probs...')
                probs_dict = torch.load(model_probs_path)
                with open(model_out_path, 'r') as f:
                    model_out = json.load(f)
            except Exception as e:
                print(f'Could not load probs from {model_probs_path}')
                print('Computing probs...')
                model_out, probs_dict = eval_model.main(model_name, dataset, split, ckpt_path, data_root='./data',
                                                        return_probs=True)

            # load images
            print('Loading images')
            image_group = _group_images(method='topkconfusion', params={'probs': probs_dict,
                                                                           'model_out': model_out,
                                                                           'num_images': num_images,
                                                                           'target_classes': target_classes,
                                                                           'k': topk_group_size,
                                                                           'threshold': threshold,
                                                                           'seed': dataset_seed})
            image_groups.append(image_group)

        image_group = merge_image_groups('union', image_groups[0], image_groups[1])
        labels = []
        path_label_map = dict(zip(model_out['predictions'].keys(), model_out['labels']))
        for i in image_group.keys():
            img_paths = image_group[i]
            labels.extend([path_label_map[path] for path in img_paths])

    elif strategy == 'union-craft':  # requires two models
        image_groups = []
        eval_dict = []
        for pd in param_dicts:
            # Load/compute model predictions
            dataset_params = pd['dataset_params']
            dataset = dataset_params['dataset_name']
            dataset_seed = dataset_params['seed']
            split = dataset_params['split']
            model_name, ckpt_path = pd['model']
            model_eval_path = f'model_evaluation/{dataset}/{model_name}_{split}.json'
            try:
                with open(model_eval_path, 'r') as f:
                    _eval_dict = json.load(f)
                    predictions = _eval_dict['predictions']
                    eval_dict.append(_eval_dict)
            except FileNotFoundError:
                predictions = eval_model.main(model_name, dataset, split, ckpt_path, data_root='./data')[
                    'predictions']

            # load images
            print('Loading images')
            image_group = _group_images(method='craft', params={'predictions': predictions,
                                                                   'num_images': None,
                                                                   'seed': dataset_seed})
            image_groups.append(image_group)

        image_group = merge_image_groups('union', image_groups[0], image_groups[1])

        # subsample N images per class
        assert param_dicts[0]['dataset_params']['num_images'] == param_dicts[1]['dataset_params']['num_images']
        assert param_dicts[0]['dataset_params']['seed'] == param_dicts[1]['dataset_params']['seed']
        target_num_images = param_dicts[0]['dataset_params']['num_images']
        seed = param_dicts[0]['dataset_params']['seed']
        if target_num_images is not None:
            rng = np.random.default_rng(seed)
            subsampled_label_groups = {}
            for i in image_group.keys():
                path_list = sorted(image_group[i])
                if target_num_images > len(path_list):
                    print(f'Warning: class {i} has only {len(path_list)} / {target_num_images} images')
                    num_images = len(path_list)
                else:
                    num_images = target_num_images
                ### TO MATCH PREVIOUS VERSION RESULTS
                # rng.shuffle(path_list)
                # subsampled_label_groups[i] = path_list[:num_images]
                ####
                subsampled_label_groups[i] = list(rng.choice(path_list, size=num_images, replace=False))
                # print(path_list[:num_images])
                # print(len(path_list[:num_images]))
            image_group = subsampled_label_groups

    else:
        raise ValueError(f'Unknown image_group_strategy: {strategy}')

    if return_eval_dict:
        return image_group, eval_dict

    if return_labels:
        return image_group, labels

    return image_group

def merge_image_groups(strategy, image_group1, image_group2):
    image_group = {}
    keys = image_group1.keys()
    for key in keys:
        if strategy == 'union':
            image_group[key] = list(set(image_group1.get(key, []) + image_group2.get(key, [])))
        elif strategy == 'intersection':
            image_group[key] = list(set(image_group1[key]).intersection(image_group2[key]))
        elif strategy == 'model1':
            image_group[key] = image_group1[key]
        elif strategy == 'model2':
            image_group[key] = image_group2[key]
        else:
            raise ValueError(f'Unknown strategy: {strategy}')
    return image_group


def get_image_group(image_group_strategy, config, igs_params=None, return_dataset=False, old_output_format=False):
    igs = image_group_strategy

    targets = None
    if igs == 'ground_truth':
        dataset_params = {
            'dataset_name': config['dataset_name'],
            'data_root': config['dataset_root'],
            'batch_size': 64,
            'num_workers': 0,

            'load_images': True,
            'class_list': config.get('class_list', None),
            'shuffle_train': False,
            'shuffle_test': False,
            'get_part_map': False,

            'transform_params': {
                'dataset_name': config['dataset_name'],
                'synthetic_concept_config': config,
            },
        }
        if 'cub_processed_data_root' in config:
            dataset_params['cub_processed_data_root'] = config['cub_processed_data_root']
        dataset_dict = get_dataset(params=dataset_params)
        if config["dataset_split"] is not None:
            dataset = dataset_dict[f'{config["dataset_split"]}_dataset']
        else:
            dataset = dataset_dict['dataset']
        pd = dict(dataset_params=dataset_params, dataset_seed=config['seed'], paths=dataset.paths, labels=dataset.labels,
                  num_images=config['num_images'], target_classes=igs_params['data_group_params'].get('target_classes', None),
                  )
        image_group = create_image_group(strategy=igs, param_dicts=pd)
        test_transform = dataset_dict.get('test_transform', None)
    elif igs == 'full_dataset':
        dataset_params = {
            'dataset_name': config['dataset_name'],
            'data_root': config['dataset_root'],
            'batch_size': 64,
            'num_workers': 0,

            'load_images': True,
            'class_list': config.get('class_list', None),
            'shuffle_train': False,
            'shuffle_test': False,
            'get_part_map': False,

            'transform_params': {
                'dataset_name': config['dataset_name'],
                'synthetic_concept_config': config,
            },
        }
        if 'cub_processed_data_root' in config:
            dataset_params['cub_processed_data_root'] = config['cub_processed_data_root']
        dataset_dict = get_dataset(params=dataset_params)
        if config["dataset_split"] is not None:
            dataset = dataset_dict[f'{config["dataset_split"]}_dataset']
        else:
            dataset = dataset_dict['dataset']
        image_group = {config['dataset_name']: dataset.paths}
        targets = dataset.labels
        test_transform = dataset_dict.get('test_transform', None)
    else:
        data_group_params = igs_params['data_group_params']
        param_dicts1 = igs_params['param_dicts1']
        param_dicts2 = igs_params['param_dicts2']
        image_group, targets = create_image_group(strategy=igs, param_dicts=[data_group_params, param_dicts1, param_dicts2], return_labels=True)
        dataset = None
        test_transform = None

    if config.get('group_all_images', None) and igs != 'full_dataset':
        image_list = []
        targets = []
        for key in image_group:
            image_list.extend(image_group[key])
            targets.extend([key] * len(image_group[key]))
        key = f"{config['dataset_name']}_subset_grouped"
        image_group = {key: image_list}

    if old_output_format and return_dataset:
        return image_group, targets, dataset
    elif old_output_format:
        return image_group, targets

    return dict(image_group=image_group, labels=targets, dataset=dataset, test_transform=test_transform)


def load_images(image_path_list, data_root, transform, return_raw_images=False, raw_data=None):

    sel_paths = image_path_list
    is_dummy_path = sel_paths[0].startswith('root')
    gt_labels = np.array([path.split('/')[-2] for path in sel_paths])
    basic_transform = None
    if return_raw_images:
        basic_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])

    images = []
    transformed_images = []
    print(transform)
    for img_path in sel_paths:
        if not is_dummy_path:
            if 'data/stanford_cars' in img_path:
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.open(os.path.join(data_root, img_path.lstrip('/'))).convert('RGB')
        else:
            index = int(img_path.split('/')[-1])
            image = raw_data[index]

        if basic_transform:
            images.append(basic_transform(image))
        transformed_images.append(transform(image))

    images = torch.stack(images, 0) if return_raw_images else None
    images_preprocessed = torch.stack(transformed_images, 0)

    out = {
        'image_paths': sel_paths,
        'gt_labels': gt_labels,
        'images_preprocessed': images_preprocessed,
        'num_images': len(sel_paths),
        'image_size': images_preprocessed.shape[2],
        'images': images,
    }
    return out
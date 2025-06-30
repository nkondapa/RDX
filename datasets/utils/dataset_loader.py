import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from datasets.utils.build_transform import get_transform
from datasets.nabirds import NABirds
from datasets.imagenet import ImageNetModified
from datasets.funny_birds import FunnyBirds
from datasets.butterflies import ButterfliesDataset
from datasets.mnist import MNISTDataset
from datasets.inatdl import INatDL
from datasets import derma_data
from datasets import cub
import numpy as np
import os


'''def custom_collate(original_batch):
    input = []
    target = []

    for item in original_batch:
        input.append(item['input'])
        target.append(item['target'])

    return dict(input=input, target=target)'''


def get_dataset(params):

    dataset_name = params['dataset_name']

    if dataset_name == 'nabirds':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        class_list = params.get('class_list', None)

        train_dataset = NABirds(root=data_root, transform=transform, train=True, class_list=class_list)
        test_dataset = NABirds(root=data_root, transform=test_transform, train=False, class_list=class_list)
        dataset = NABirds(root=data_root, transform=transform, train=None, class_list=class_list)

        num_classes = train_dataset.num_classes

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, dataset=dataset,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'nabirds_modified':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])

        transform = transform_dict['transform']
        modified_transform = transform_dict['modified_transform']
        test_transform = transform_dict['test_transform']
        modified_test_transform = transform_dict['modified_test_transform']
        preprocessing = transform_dict['preprocessing']

        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)

        modified_params = {'transform': modified_transform, 'modified_classes': params['transform_params']['synthetic_concept_config']['train']['concept_params']['classes']}
        train_dataset = NABirds(root=data_root, transform=transform, train=True, modified_params=modified_params)
        modified_test_params = {'transform': modified_test_transform, 'modified_classes': params['transform_params']['synthetic_concept_config']['test']['concept_params']['classes']}
        test_dataset = NABirds(root=data_root, transform=test_transform, train=False, modified_params=modified_test_params)
        dataset = NABirds(root=data_root, transform=transform, train=None, modified_params=modified_params)

        num_classes = train_dataset.num_classes

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, dataset=dataset,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'imagenet':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        concept_params = params.get('concept_params', None)
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)

        # train_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
        # test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=test_transform)
        # dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
        train_dataset = ImageNetModified(root=data_root, split='train', transform=transform, concept_params=concept_params)
        test_dataset = ImageNetModified(root=data_root, split='val', transform=test_transform, concept_params=concept_params)
        dataset = train_dataset + test_dataset

        num_classes = len(train_dataset.classes)

        train_sampler = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, dataset=dataset,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'stanford_cars':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        concept_params = params.get('concept_params', None)
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)

        # train_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
        # test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=test_transform)
        # dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
        train_dataset = torchvision.datasets.StanfordCars(root=data_root, split='train', transform=transform)
        test_dataset = torchvision.datasets.StanfordCars(root=data_root, split='test', transform=test_transform)
        dataset = train_dataset + test_dataset

        num_classes = len(train_dataset.classes)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, all_dataset=dataset,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)
    
    if dataset_name == 'funny_birds' or dataset_name == 'funny_birds_hc':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        concept_params = params.get('concept_params', None)
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        get_part_map = params.get('get_part_map', False)

        train_dataset = FunnyBirds(data_root, 'train', get_part_map=get_part_map, transform=transform)
        test_dataset = FunnyBirds(data_root, 'test', get_part_map=get_part_map, transform=transform)

        num_classes = len(train_dataset.classes)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        
        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'chinese_chars':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)

        train_dataset = ChineseCharsDataset(data_root, split='train', transform=transform)
        dataset = ChineseCharsDataset(data_root, split=None, transform=transform)
        test_dataset = ChineseCharsDataset(data_root, split='test', transform=test_transform)

        num_classes = len(train_dataset.classes)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)

        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset,
                    dataset=dataset, all_loader=all_loader,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'butterflies' or dataset_name.split('_')[0] == 'butterflies':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)

        train_dataset = ButterfliesDataset(data_root, split='train', transform=transform)
        test_dataset = ButterfliesDataset(data_root, split='test', transform=test_transform)
        dataset = ButterfliesDataset(data_root, split=None, transform=test_transform)

        if dataset_name == 'butterflies_sh1':
            mask = train_dataset.labels != 0
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh2':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3)
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh3':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3) & (train_dataset.labels != 2)
            train_dataset.labels[mask] = 1

        num_classes = len(train_dataset.classes)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)

        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset,
                    dataset=dataset, all_loader=all_loader,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

    if 'mnist' in dataset_name:
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        mod_params = {}
        if 'mnist_hflip' in dataset_name:
            mod_params = {'hflip': True}
        elif 'mnist_vflip' in dataset_name:
            mod_params = {'vflip': True}
        elif 'mnist_835' == dataset_name:
            mod_params = {'classes': [8, 3, 5]}
        if 'nl' in dataset_name:
            mod_params['new_labels'] = True

        train_dataset = MNISTDataset(data_root, train=True, transform=transform, mod_params=mod_params)
        test_dataset = MNISTDataset(data_root, train=False, transform=test_transform, mod_params=mod_params)
        # dataset = MNISTDataset(data_root, split=None, transform=test_transform)

        if dataset_name == 'mnist_m35':
            mask = (train_dataset.labels == 3) | (train_dataset.labels == 5)
            train_dataset.labels[mask] = 3
        elif dataset_name == 'mnist_m49':
            mask = (train_dataset.labels == 4) | (train_dataset.labels == 9)
            train_dataset.labels[mask] = 4

        num_classes = len(train_dataset.classes)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        # all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test,
        #                          num_workers=num_workers)

        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset,
                    # dataset=dataset, all_loader=all_loader,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == "ham10000":
        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        params = {
            'seed': params.get('seed', 0),
            'batch_size': params['batch_size'],
            'num_workers': params['num_workers'],
            'transform': transform,
            'data_root': params['data_root']

        }
        out = derma_data.load_ham_data(params)
        idx_to_class = out['idx_to_class']
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())
        return dict(train_loader=out['train_loader'], test_loader=out['val_loader'],
                    train_dataset=out['trainset'], test_dataset=out['valset'],
                    dataset=None, all_loader=None,
                    num_classes=len(classes), idx_to_class=idx_to_class,
                    class_to_idx=class_to_idx, classes=classes, test_transform=test_transform,
                    transform=transform, preprocessing=None)
    elif dataset_name == "cub_pcbm":
        # from .constants import CUB_PROCESSED_DIR, CUB_DATA_DIR
        from torchvision import transforms

        # transform_dict = get_transform(params['transform_params'])
        # transform = transform_dict['transform']
        # test_transform = transform_dict['test_transform']
        # params = {
        #     'seed': params.get('seed', 0),
        #     'batch_size': params['batch_size'],
        #     'num_workers': params['num_workers'],
        #     'transform': transform,
        #     'data_root': params['data_root']
        #
        # }
        cub_processed_data_root = params['cub_processed_data_root']
        num_classes = 200
        TRAIN_PKL = os.path.join(cub_processed_data_root, "train.pkl")
        TEST_PKL = os.path.join(cub_processed_data_root, "test.pkl")
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        train_loader = cub.load_cub_data([TRAIN_PKL], use_attr=False, no_img=False,
                                     batch_size=params['batch_size'], uncertain_label=False, image_dir=params['data_root'],
                                     resol=224, normalizer=normalizer,
                                     n_classes=num_classes, resampling=True)

        test_loader = cub.load_cub_data([TEST_PKL], use_attr=False, no_img=False,
                                    batch_size=params['batch_size'], uncertain_label=False, image_dir=params['data_root'],
                                    resol=224, normalizer=normalizer,
                                    n_classes=num_classes, resampling=True)

        classes = open(os.path.join(params['data_root'], "classes.txt")).readlines()
        classes = [a.split(".")[1].strip() for a in classes]
        idx_to_class = {i: classes[i] for i in range(num_classes)}
        classes = [classes[i] for i in range(num_classes)]
        # print(len(classes), "num classes for cub")
        # print(len(train_loader.dataset), "training set size")
        # print(len(test_loader.dataset), "test set size")

        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_loader.dataset, test_dataset=test_loader.dataset,
                    dataset=None, all_loader=None,
                    num_classes=len(classes), idx_to_class=idx_to_class,
                    classes=classes, test_transform=test_loader.dataset.transform,
                    transform=train_loader.dataset.transform, preprocessing=None)

    if dataset_name == 'inatdl':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        class_list = params.get('class_list', None)

        train_dataset = INatDL(root=data_root, split="train", transform=transform, class_list=class_list)
        test_dataset = INatDL(root=data_root, split="test", transform=test_transform, class_list=class_list)
        dataset = INatDL(root=data_root, split=None, transform=transform, class_list=class_list)

        num_classes = train_dataset.num_classes

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, dataset=dataset,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

def get_triplet_dataset(params):
    dataset_name = params['dataset_name']

    if dataset_name == 'butterflies' or dataset_name.split('_')[0] == 'butterflies':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        teacher_model = params.get('teacher_model', None)
        student_model = params.get('student_model', None)
        triplet_sampling_config = params.get('triplet_sampling_config', None)
        skip_dataset_list = params.get('skip_dataset_list', [])

        train_dataset, test_dataset, dataset = None, None, None
        if 'train' not in skip_dataset_list:
            train_dataset = ButterfliesTeacherTripletDataset(data_root, teacher_model, student_model, triplet_sampling_config, split='train', transform=transform)
        if 'test' not in skip_dataset_list:
            test_dataset = ButterfliesTeacherTripletDataset(data_root, teacher_model, student_model, triplet_sampling_config, split='test', transform=test_transform)
        if 'all' not in skip_dataset_list:
            dataset = ButterfliesTeacherTripletDataset(data_root, teacher_model, student_model, triplet_sampling_config, split=None, transform=test_transform)

        if dataset_name == 'butterflies_sh1':
            mask = train_dataset.labels != 0
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh2':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3)
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh3':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3) & (train_dataset.labels != 2)
            train_dataset.labels[mask] = 1

        num_classes = len(train_dataset.classes)
        train_loader = None
        test_loader = None
        all_loader = None
        if 'train' not in skip_dataset_list:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers)
        if 'test' not in skip_dataset_list:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers)
        if 'all' not in skip_dataset_list:
            all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test,
                                    num_workers=num_workers)

        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset,
                    dataset=dataset, all_loader=all_loader,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)


def get_grid_dataset(params):
    dataset_name = params['dataset_name']

    if dataset_name == 'butterflies' or dataset_name.split('_')[0] == 'butterflies':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        teacher_model = params.get('teacher_model', None)
        student_model = params.get('student_model', None)
        grid_sampling_config = params.get('sampling_config', None)
        skip_dataset_list = params.get('skip_dataset_list', [])

        train_dataset, test_dataset, dataset = None, None, None
        if 'train' not in skip_dataset_list:
            train_dataset = ButterfliesTeacherGridDataset(data_root, teacher_model, student_model, grid_sampling_config, split='train', transform=transform)
        if 'test' not in skip_dataset_list:
            test_dataset = ButterfliesTeacherGridDataset(data_root, teacher_model, student_model, grid_sampling_config, split='test', transform=test_transform)
        if 'all' not in skip_dataset_list:
            dataset = ButterfliesTeacherGridDataset(data_root, teacher_model, student_model, grid_sampling_config, split=None, transform=test_transform)

        if dataset_name == 'butterflies_sh1':
            mask = train_dataset.labels != 0
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh2':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3)
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh3':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3) & (train_dataset.labels != 2)
            train_dataset.labels[mask] = 1

        num_classes = len(train_dataset.classes)
        train_loader = None
        test_loader = None
        all_loader = None
        if 'train' not in skip_dataset_list:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers)
        if 'test' not in skip_dataset_list:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers)
        if 'all' not in skip_dataset_list:
            all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test,
                                    num_workers=num_workers)

        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset,
                    dataset=dataset, all_loader=all_loader,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)


def get_teacher_clf_dataset(params):
    dataset_name = params['dataset_name']

    if dataset_name == 'butterflies' or dataset_name.split('_')[0] == 'butterflies':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        teacher_model = (params.get('teacher_model', None), params.get('teacher_model_clf', None))
        student_model = (params.get('student_model', None), params.get('student_model_clf', None))
        sampling_config = params.get('sampling_config', None)
        skip_dataset_list = params.get('skip_dataset_list', [])

        # @TODO add model splitting code and let dataset split the model into backbone and head on its own

        train_dataset, test_dataset, dataset = None, None, None
        if 'train' not in skip_dataset_list:
            train_dataset = ButterfliesTeacherClfDataset(data_root, teacher_model, student_model, sampling_config, split='train', transform=transform)
        if 'test' not in skip_dataset_list:
            test_dataset = ButterfliesTeacherClfDataset(data_root, teacher_model, student_model, sampling_config, split='test', transform=test_transform)
        if 'all' not in skip_dataset_list:
            dataset = ButterfliesTeacherClfDataset(data_root, teacher_model, student_model, sampling_config, split=None, transform=test_transform)

        if dataset_name == 'butterflies_sh1':
            mask = train_dataset.labels != 0
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh2':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3)
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh3':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3) & (train_dataset.labels != 2)
            train_dataset.labels[mask] = 1

        num_classes = len(train_dataset.classes)
        train_loader = None
        test_loader = None
        all_loader = None
        if 'train' not in skip_dataset_list:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers)
        if 'test' not in skip_dataset_list:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers)
        if 'all' not in skip_dataset_list:
            all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test,
                                    num_workers=num_workers)

        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset,
                    dataset=dataset, all_loader=all_loader,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)

def get_teacher_concept_dl_dataset(params):
    dataset_name = params['dataset_name']

    if dataset_name == 'butterflies' or dataset_name.split('_')[0] == 'butterflies':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        teacher_model = (params.get('teacher_model', None), params.get('teacher_model_clf', None))
        student_model = (params.get('student_model', None), params.get('student_model_clf', None))
        sampling_config = params.get('sampling_config', None)
        skip_dataset_list = params.get('skip_dataset_list', [])

        # @TODO add model splitting code and let dataset split the model into backbone and head on its own

        train_dataset, test_dataset, dataset = None, None, None
        if 'train' not in skip_dataset_list:
            train_dataset = ButterfliesTeacherDLDataset(data_root, teacher_model, student_model, sampling_config, split='train', transform=transform)
        if 'test' not in skip_dataset_list:
            test_dataset = ButterfliesTeacherDLDataset(data_root, teacher_model, student_model, sampling_config, split='test', transform=test_transform)
        if 'all' not in skip_dataset_list:
            dataset = ButterfliesTeacherDLDataset(data_root, teacher_model, student_model, sampling_config, split=None, transform=test_transform)

        if dataset_name == 'butterflies_sh1':
            mask = train_dataset.labels != 0
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh2':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3)
            train_dataset.labels[mask] = 1
        elif dataset_name == 'butterflies_sh3':
            mask = (train_dataset.labels != 0) & (train_dataset.labels != 3) & (train_dataset.labels != 2)
            train_dataset.labels[mask] = 1

        num_classes = len(train_dataset.classes)
        train_loader = None
        test_loader = None
        all_loader = None
        if 'train' not in skip_dataset_list:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers)
        if 'test' not in skip_dataset_list:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers)
        if 'all' not in skip_dataset_list:
            all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test,
                                    num_workers=num_workers)

        return dict(train_loader=train_loader, test_loader=test_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset,
                    dataset=dataset, all_loader=all_loader,
                    num_classes=num_classes, test_transform=test_transform,
                    transform=transform, preprocessing=preprocessing)
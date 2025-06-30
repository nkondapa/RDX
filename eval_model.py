import os
import numpy as np
import torch
from datasets.imagenet import imagenet, imagenet_modified
from datasets.nabirds import NABirds
import tqdm
import json
from src.utils.model_loader import load_model
from datasets.utils.dataset_loader import get_dataset


def eval_model(model, dataloader, device, criterion, out_transform=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    class_acc = {}

    root = dataloader.dataset.root
    predictions = {}
    probs = {}
    labels_all = []
    for data in tqdm.tqdm(dataloader):
        inputs, labels, paths = data['input'], data['target'], data['path']
        inputs = inputs.to(device)
        labels = labels.to(device)
        paths = [path.replace(root, '') for path in paths]

        # plt.figure()
        # plt.imshow(inputs[1].permute(1, 2, 0).cpu().numpy())
        # plt.title(labels[1])
        # plt.show()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            if out_transform:
                outputs = out_transform(outputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            for t, (path, pred) in enumerate(zip(paths, preds)):
                predictions[path] = pred.cpu().item()
                probs[path] = torch.softmax(outputs[t].cpu(), dim=0).detach().numpy()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        for i in range(len(labels)):
            sc = int(preds[i] == labels[i])
            if labels[i].item() in class_acc:
                class_acc[labels[i].item()][0] += sc
                class_acc[labels[i].item()][1] += 1
            else:
                class_acc[labels[i].item()] = [sc, 1]

        labels_all.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    sorted_class_acc = {}
    sorted_keys = sorted(class_acc.keys())
    for k in sorted_keys:
        v = class_acc[k]
        sorted_class_acc[k] = v
        print(f'Class {k} acc: {v[0] / v[1]}')

    return epoch_loss, epoch_acc.item(), sorted_class_acc, predictions, probs, labels_all


def compute_stats(model_name, eval_dict):
    # with open(f'./model_evaluation/{dataset}/{model_name}_probs_{split}.pth', 'rb') as f:
    #     prob_dict = torch.load(f)

    labels = eval_dict['labels']
    labels = np.array(labels)
    # probs = np.stack(list(prob_dict.values()))

    os.makedirs(f'visualizations/{model_name}/confusion_matrices/', exist_ok=True)
    class_pred = np.array(list(eval_dict['predictions'].values()))

    stats = {}
    for target_class in range(len(np.unique(labels))):
        class_mask = labels == target_class

        # class_pred = np.argmax(probs, axis=1)
        tp = np.sum((class_pred == target_class) & (labels == target_class))
        fp = np.sum((class_pred == target_class) & (labels != target_class))
        fn = np.sum((class_pred != target_class) & (labels == target_class))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f_score = 2 * prec * rec / (prec + rec)
        acc = np.mean(class_pred[class_mask] == target_class)

        # class_probs = probs[class_mask]

        stats[target_class] = {'prec': prec, 'rec': rec, 'acc': acc, 'f1': f_score}

    return stats


def convert_probs_to_label_groups(probs, k=5, threshold=0):
    label_dict = {}
    for pi, path in enumerate(probs):
        cur_probs = probs[path]
        top_k_vals, top_k_ind = torch.topk(torch.tensor(cur_probs), k=k)
        top_k_vals = np.array(top_k_vals)
        top_k_ind = np.array(top_k_ind)
        for i in range(k):
            if top_k_vals[i] > threshold:
                if top_k_ind[i] not in label_dict:
                    label_dict[top_k_ind[i]] = []
                label_dict[top_k_ind[i]].append(path)
            else:
                break

    return label_dict

def convert_predictions_to_label_groups(predictions):
    label_dict = {}
    for pi, path in enumerate(predictions):
        class_pred = predictions[path]
        if class_pred not in label_dict:
            label_dict[class_pred] = []
        label_dict[class_pred].append(path)

    return label_dict


def main(model_name, dataset_name, split='val', ckpt_path=None, model_type=None, post_model_load=None,
         out_transform=None, save_root='model_evaluation', data_root='../data', modifier_params=None, return_probs=False,
         batch_size=512):
    model_dict = load_model(model_name, ckpt_path, model_type=model_type)
    test_transform = model_dict['test_transform']
    model = model_dict['model']
    if post_model_load is not None:
        model = post_model_load(model)

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    if dataset_name == 'imagenet':
        dataset = imagenet_modified(split, test_transform, os.path.join(data_root, 'imagenet'))
    elif dataset_name == 'nabirds':
        dataset = NABirds(os.path.join(data_root, 'nabirds'), train=False, download=False, transform=test_transform)
    elif dataset_name == 'nabirds_modified':
        dparams = model_dict['lightning_model'].dataset_params
        if modifier_params is not None:
            dparams['transform_params'].update(modifier_params)
        dataset_dict = get_dataset(params=dparams)
        dataset = dataset_dict[f'{split}_dataset']
    elif dataset_name == 'nabirds_stanford_cars':
        dparams = model_dict['lightning_model'].dataset_params
        dparams['dataset_name'] = 'nabirds_stanford_cars'
        dparams['transform_params']['use_test_transform_for_train'] = True
        dparams['transform_params']['dataset_name'] = 'nabirds_stanford_cars'
        dataset_dict = get_dataset(params=dparams)
        dataset = dataset_dict[f'{split}_dataset']
    else:
        raise ValueError(f'Unknown dataset_name: {dataset_name}')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    loss, acc, class_acc, predictions, probs, labels = eval_model(model, dataloader, device, criterion,
                                                                  out_transform=out_transform)

    out = {'loss': loss, 'acc': acc, 'class_acc': class_acc, 'predictions': predictions, 'labels': labels}
    os.makedirs(f'{save_root}/{dataset_name}', exist_ok=True)
    with open(f'{save_root}/{dataset_name}/{model_name}_{split}.json', 'w') as f:
        json.dump(out, f, indent=2)

    with open(f'{save_root}/{dataset_name}/{model_name}_probs_{split}.pth', 'wb') as f:
        torch.save(probs, f)

    stats = compute_stats(model_name, out)
    with open(f'./{save_root}/{dataset_name}/{model_name}_stats_{split}.json', 'w') as f:
        json.dump(stats, f)

    if return_probs:
        return out, probs
    return out


def compute_stats_main(model_name, dataset_name, split='val', ckpt_path=None, model_type=None, post_model_load=None,
                       out_transform=None, save_root='model_evaluation', data_root='../data'):
    with open(f'{save_root}/{dataset_name}/{model_name}_{split}.json', 'r') as f:
        out = json.load(f)

    stats = compute_stats(model_name, out)
    with open(f'./{save_root}/{dataset_name}/{model_name}_stats_{split}.json', 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    model_names = [
        # "nabirds_r18_fs_a3_seed=4834586_rh_nbsc",
        # "nabirds_stanford_cars_r18_fs_seed=4834586_rh_nbsc",
        "vit_base_patch14_reg4_dinov2.lvd142m",
        "vit_base_patch16_224.dino"
    ]
    for model_name in model_names:
        # main(model_name, dataset_name='nabirds_stanford_cars', split='test', ckpt_path=f'./checkpoints/{model_name}/last.ckpt', data_root='./data')
        main(model_name, dataset_name='imagenet', split='val', data_root='./data', batch_size=128)
import json
import copy
import os.path

root_folder = '.'
experiment_folder = os.path.abspath('./').split('/')[-1]
with open('./base_config.json', 'r') as f:
    base = json.load(f)

sweep_param_keys_list = [
    ['representation_params'],
]
sweep_param_values = [
    {
        "repr_0": {
            "model": "mnist_m35",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_m35/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        },
        "repr_1": {
            "model": "mnist_expert",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_expert/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        }
    },

    {
        "repr_0": {
            "model": "mnist_m49",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_m49/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        },
        "repr_1": {
            "model": "mnist_expert",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_expert/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        }
    },

    {
        "repr_0": {
            "model": "mnist_m35",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_m35/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        },
        "repr_1": {
            "model": "mnist_m49",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_m49/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        }
    },

    {
        "repr_0": {
            "model": "mnist_hflip",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_hflip/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        },
        "repr_1": {
            "model": "mnist_hflip_nl",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_hflip_nl/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        }
    },

    {
        "repr_0": {
            "model": "mnist_vflip",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_vflip/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        },
        "repr_1": {
            "model": "mnist_vflip_nl",
            "model_ckpt": f"{root_folder}/checkpoints/mnist_vflip_nl/last.ckpt",
            "feature_extraction_params": {"feature_layer_version": "v0"}
        }
    },

]
dataset_names = ['mnist', 'mnist', 'mnist', 'mnist_hflip_nl', 'mnist_vflip_nl']
num_images = [500, 500, 500, 250, 250]
name_list = ['mnist_m35_vs_mnist', 'mnist_m49_vs_mnist', 'mnist_m35_vs_mnist_m49', 'mnist_hflip_vs_mnist_hflip_nl', 'mnist_vflip_vs_mnist_vflip_nl']
# name_fn = lambda x: f'{experiment_folder}_ed={x}.json'
configs = []

for i, sweep_param_value in enumerate(sweep_param_values):
    sweep_params = copy.deepcopy(base)
    for sweep_param_keys in sweep_param_keys_list:
        tmp = sweep_params
        for key in sweep_param_keys[:-1]:
            print(key)
            tmp = tmp[key]
        tmp[sweep_param_keys[-1]] = sweep_param_value
        # name = name_fn(sweep_param_value)
        name = name_list[i] + '.json'
    sweep_params["image_selection"]["dataset_name"] = dataset_names[i]
    sweep_params["image_selection"]["num_images"] = num_images[i]
    configs.append(os.path.abspath(name))
    with open(f'./{name}', 'w') as f:
        json.dump(sweep_params, f, indent=2)

with open('configs.json', 'w') as f:
    json.dump(configs, f)

print(configs)
bash_output_folder = '../../'
with open(f'{os.path.join(bash_output_folder, experiment_folder)}.sh', 'w') as f:
    f.write(f'#!/bin/bash\n')
    for config in configs:
        f.write(
            f'python generate_comparison_explanations.py --comparison_config "{config}" --comparison_output_root outputs/{experiment_folder} --save_m0_representation  --save_m1_representation \n')

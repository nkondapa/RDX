import json
import copy
import os.path

root_folder = '/home/nkondapa/PycharmProjects/ConceptDiff'
experiment_folder = os.path.abspath('./').split('/')[-1]
with open('./base_config.json', 'r') as f:
    base = json.load(f)

sweep_param_keys_list = [
    ['representation_params', 'repr_0', 'model_ckpt'],
]
sweep_param_values = [
    "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=0-step=0.ckpt",
    "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=0-step=32.ckpt",
    "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=0-step=56.ckpt",
    "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=0-step=136.ckpt",
    "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=1-step=160.ckpt",
    "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=1-step=184.ckpt",
    "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=1-step=272.ckpt",
    # "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=2-step=408.ckpt",
    # "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=3-step=544.ckpt",
    # "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/mnist_835/epoch=4-step=680.ckpt",
]
# name_list = ['mnist_m35_vs_mnist', 'mnist_m49_vs_mnist', 'mnist_m35_vs_mnist_m49', 'mnist_hflip_vs_mnist_hflip_nl', 'mnist_vflip_vs_mnist_vflip_nl']
name_fn = lambda x: f'{experiment_folder}_ckpt={x.split("/")[-1].split(".ckpt")[0]}.json'
configs = []

for i, sweep_param_value in enumerate(sweep_param_values):
    sweep_params = copy.deepcopy(base)
    for sweep_param_keys in sweep_param_keys_list:
        tmp = sweep_params
        for key in sweep_param_keys[:-1]:
            print(key)
            tmp = tmp[key]
        tmp[sweep_param_keys[-1]] = sweep_param_value
        name = name_fn(sweep_param_value)
        # name = name_list[i] + '.json'
    configs.append(os.path.abspath(name))
    with open(f'./{name}', 'w') as f:
        json.dump(sweep_params, f, indent=2)

with open('configs.json', 'w') as f:
    json.dump(configs, f)

print(configs)
with open(f'{os.path.join(root_folder, experiment_folder)}.sh', 'w') as f:
    f.write(f'#!/bin/bash\n')
    for config in configs:
        f.write(
            f'python generate_comparison_explanations.py --comparison_config "{config}" --comparison_output_root outputs2/{experiment_folder} --save_m0_representation  --save_m1_representation \n')

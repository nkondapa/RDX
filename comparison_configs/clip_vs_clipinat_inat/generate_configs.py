import json
import copy
import os.path


root_folder = '.'
experiment_folder = os.path.abspath('./').split('/')[-1]
with open('./base_config.json', 'r') as f:
    base = json.load(f)


sweep_param_keys_list = [
    ['image_selection', "target_classes"],
    ]
sweep_param_values = [
    [0, 6],  [3, 5],  [1, 2, 4, 7]
]
names = ['gator', 'corvid', "maple"]
configs = []

for si, sweep_param_value in enumerate(sweep_param_values):
    sweep_params = copy.deepcopy(base)
    for sweep_param_keys in sweep_param_keys_list:
        tmp = sweep_params
        for key in sweep_param_keys[:-1]:
            print(key)
            tmp = tmp[key]
        tmp[sweep_param_keys[-1]] = sweep_param_value
        name = f'{experiment_folder}_{names[si]}.json'
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
        f.write(f'python generate_comparison_explanations.py --comparison_config "{config}" --comparison_output_root outputs/{experiment_folder} --save_m0_representation  --save_m1_representation \n')


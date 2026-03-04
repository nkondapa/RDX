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

with open(f'{os.path.join(bash_output_folder, experiment_folder)}.sh', 'a') as f:
    f.write('\n\n\n')
    for config in configs:
        name = config.split('/')[-1].split('.')[0]
        f.write(f'python interactive_cluster_viz_general.py --rdx_output_dir outputs/{experiment_folder}/{name}/inatdl_subset_grouped/rdx_nb_lb_spectral '
                f'--repr_0_path outputs/{experiment_folder}/{name}/m0_rep.pkl '
                f'--repr_1_path outputs/{experiment_folder}/{name}/m1_rep.pkl '
                f'--repr_0_name "CLIP" --repr_1_name "CLIP-iNat" '
                f'--image_paths outputs/{experiment_folder}/{name}/image_paths.pkl '
                f'--data_root ./data/inat_subset/ '
                f'--K 12 --K_matrix 12 --thumb_size 224\n')


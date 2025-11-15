# Representational Difference Explanations (RDX)
[![Project Page](https://img.shields.io/badge/Project%20Page-Link)](https://nkondapa.github.io/rdx-page/)
[![OpenReview](https://img.shields.io/badge/OpenReview-a61717)](https://openreview.net/forum?id=ngvolie2nH)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)]([https://arxiv.org/pdf/2505.23917](https://arxiv.org/pdf/2505.23917))


#### Updates
- Paper was accepted to NeurIPS 2025!
- NLMCD, TopK-SAE, and USAE added as baseline options
  - tested on mnist modification and cub experiments


### Setup
```sh

conda create -n "RDX" python=3.10.15
conda activate RDX
bash setup.sh
```
### Downloading Checkpoints
Download checkpoints for the MNIST experiments here: ```bash download_checkpoints.sh```

### Downloading Datasets
1) MNIST: You can download the MNIST dataset with ```bash download_mnist.sh```.
2) INaturalist (Subset): You can download the iNaturalist subset with ```bash download_inaturalist.sh```.
3) CUB: You can download CUB and supplementary files for CUB CBMs with ```bash download_cub.sh```.
4) ImageNet: You can download imagenet with ```bash download_imagenet.sh```.

If you have already downloaded some of these datasets, you can symlink them to the `data/` directory. See symlinks.txt 
for examples.

### Experiments
1) To reproduce the MNIST experiments, download the checkpoints and the mnist dataset. 
   - ```bash mnist_835_experiment.sh``` for the MNIST subset experiment with only 3s, 5s, and 8s. 
   - ```bash mnist_modification_experiment_k=3.sh``` for the MNIST training modification experiments, with k=3.
2) To reproduce the CUB PCBM experiments, download the CUB dataset and run:
   - ```bash cub_pcbm_v_cub_masked_pcbm.sh```
3) To reproduce the ImageNet experiments, download the ImageNet dataset and run:
   - ```bash dino_vs_dinov2_imagenet_ar.sh``` (aligned)
   - ```bash dino_vs_dinov2_imagenet.sh``` (unaligned)
4) To reproduce the iNaturalist experiments, download the iNaturalist subset and run:
   - ```bash clip_vs_clipinat_ar.sh``` (aligned)
   - ```bash clip_vs_clipinat.sh``` (unaligned)

### Minimal Example
The smallest dataset is the INat Subset. The fastest way to run a minimal example is to download the iNaturalist subset 
and run Experiment 4. 

### Visualizations
To visualize the results of the experiments, you can run:
```python analyze_explanations.py```. By default this will analyze the inat subset experiment (aligned) (Exp. 4a).
There are several commented functions in the script that you can uncomment to visualize the results of other experiments.


### Citation
```
@article{kondapaneni2025repdiffexp,
  title={Representational Difference Explanations},
  author={Kondapaneni, Neehar and Mac Aodha, Oisin and Perona, Pietro},
  journal={arXiv preprint arXiv:2505.23917},
  year={2025}
}
```

### Note on Reproducibility
Our original code seeded once at the start of comparisons for all methods, however, we realized it is likely to cause
inconsistencies due to arbitrary choices made in order of running the different comparisons. The new code re-seeds at the
beginning of each comparison for all methods. This may lead to slightly different results than those reported in the paper,
but we have checked that the trends remain the same. We apologize for any inconvenience this may cause.

### Note on Concept Selection
We stayed as close as possible to the original concept selection strategies for each baseline method. However, TopK-SAEs,
NLMCD, and USAE concept selection strategies were modified for fair comparison on our comparison tasks. 

Let k = the number of concepts shown to the user.
- TopK-SAE uses 50 latents during training with the top k remaining active. TopK-SAE are trained on each representation 
independently. After training, we select the k concepts per model with the largest mean activations to show to the user. 
- NLMCD uses HDBSCAN clustering and generates an arbitrary number of concepts for each representation. We measure concept
similarity across models and select the top k most dissimilar concepts for our comparisons.  
- USAE learns an internal representation of 8 * (representation dimension), much larger than k. To select k
concepts we measure firing entropy for each concept and select the k concepts per model with the lowest firing entropy. Firing
entropy is defined in the USAE paper and measures how evenly a concept activates across the different models. Low entropy
indicates that a concept is more specific to certain models, and is thus more likely to be useful for distinguishing them.
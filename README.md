# MorphoMod: Visible Watermark Removal with Morphological Dilation
These experiments were conducted on a macOS 13.5.2 with an Apple M2 Max processor with 64 GB of memory.

## Installation
1. Install all necessary dependencies and folders
```
conda create -n morpho pip python=3.8 && conda activate morpho && pip install -r requirements.txt
```
## Datasets
   
   a. Colored Large-scale Watermark Dataset from [here](https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij/view?usp=sharing):

   b. LOGO-Gray, LOGO-L, and LOGO-H from [here](https://github.com/vinthony/deep-blind-watermark-removal#Resources).

   c. Alpha1-S and Alpha1-L: *(Due to anonymity, this will be provided for the final publication.)*
   
## Test
| Experiment | Script Locations |
| -------- | -------- |
| **Experiment 1** | `scripts/experiment_1` |
| **Experiment 2** | `scripts/experiment_2` |
| **Eval SAM** | `scripts/sam` |
| **Eval Prompts** | `scripts/eval_diffusion_steps.sh` |
| **Eval Pre-Fill** | `scripts/eval_fill.sh` |
| **Eval Inpaint Model** | `scripts/eval_models.sh` |
| **Steg Disorientation** | `scripts/steg_disorient` |


#### Directories
> `perception_stats`: files to get BRISQUE, NIQE, and PIQE metrics

> `src`: main experiments folder

> `scripts`: easy to run training and testing scripts

> `steg_disorient`: Steganographic Disorientation

> `sam`: Files used to eval sam



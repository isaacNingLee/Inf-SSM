# Inf-SSM

Official code for **[Exemplar-Free Continual Learning for State Space Models](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Exemplar-Free_Continual_Learning_for_State_Space_Models_CVPR_2026_paper.html)** (CVPR 2026).

Built on [hustvl/Vim](https://github.com/hustvl/Vim) (Vision Mamba); the continual-learning
setup and protocol follow [aimagelab/mammoth](https://github.com/aimagelab/mammoth).

## Layout

| Path | What |
|---|---|
| [vim/main.py](vim/main.py) | Class-incremental training/eval entry point |
| [vim/cl_methods/inf_ssm.py](vim/cl_methods/inf_ssm.py) | Inf-SSM regularizer + training loop (`--cl_method reg`) |
| [vim/cl_methods/seq.py](vim/cl_methods/seq.py) | Vanilla sequential finetuning baseline (`--cl_method seq`) |
| [vim/models_mamba.py](vim/models_mamba.py) | Vim backbones, incl. `state_ac_vim_small_*` which exposes `(A, C)` |
| [mamba-1p1p1/](mamba-1p1p1/) | Mamba fork: `bimamba_type="v2_state_ac"` returns the per-token states |
| [causal-conv1d/](causal-conv1d/) | Depthwise causal conv kernel |
| [vim/scripts/](vim/scripts/) | One launcher per benchmark |

The state-extraction path is `v2_state_ac` in
[mamba-1p1p1/mamba_ssm/modules/mamba_simple.py](mamba-1p1p1/mamba_ssm/modules/mamba_simple.py).
It runs the usual bidirectional scan and additionally exposes the per-token `(A, C)` pairs
as `(positions, dstate)` — one row per token, one column per state channel — which is the
layout the observability Gramian expects.

## Environment

- Python 3.10.13 — `conda create -n inf-ssm python=3.10.13`
- PyTorch 2.1.1 + cu118 — `conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia`
  - if cuda/11.8 is unavailable: `conda install cuda-nvcc cuda-toolkit -c nvidia/label/cuda-11.8.0`
- `pip install -r vim/vim_requirements.txt && pip install wandb`
  - on `undefined symbol: iJIT_NotifyEvent`: `conda install mkl==2024.0.0`
- Kernels:
  - `pip install causal_conv1d==1.1.1` (or `pip install -e causal-conv1d --no-build-isolation`)
  - `pip install -e mamba-1p1p1 --no-build-isolation`  ← must be this fork, not upstream `mamba-ssm`

## Datasets

The scripts look under `$DATA_ROOT` (default `../dataset`):

```bash
mkdir -p dataset

# ImageNet-R
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvf imagenet-r.tar -C dataset/

# Caltech-256
wget https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar
tar -xvf 256_ObjectCategories.tar -C dataset/caltech256/
```

CIFAR-100 downloads itself into `$DATA_ROOT/cifar100` on first run.

## Pretrained / init weights

Vim backbones from [hustvl/Vim](https://github.com/hustvl/Vim):

| Model | #param. | Top-1 | Source |
|---|---|---|---|
| Vim-tiny | 7M | 76.1 | `wget https://huggingface.co/hustvl/Vim-tiny-midclstok/resolve/main/vim_t_midclstok_76p1acc.pth` |
| Vim-small | 26M | 80.5 | `wget https://huggingface.co/hustvl/Vim-small-midclstok/resolve/main/vim_s_midclstok_80p5acc.pth` |

Continual learning starts from a task-0 checkpoint (Vim-small trained on the first task
with `--cl_method seq`). Ours are on
[`ileening0022/Inf-SSM`](https://huggingface.co/ileening0022/Inf-SSM).

| Benchmark | Directory |
|---|---|
| ImageNet-R 5 / 10 task | `5task/`, `10task/` |
| CIFAR-100 5 / 10 task | `cifar100_5task/`, `cifar100_10task/` |
| Caltech-256 5 / 10 task | `caltech256_5task/`, `caltech256_10task/` |

The scripts read them from `$INIT_ROOT` (default `../checkpoints`), which mirrors the hub
repo layout exactly, so the whole thing can be pulled in one go:

```bash
pip install -U "huggingface_hub[cli]"
hf download ileening0022/Inf-SSM --local-dir checkpoints
```

Or fetch just the one you need:

```bash
mkdir -p checkpoints/cifar100_10task
wget -P checkpoints/cifar100_10task \
  https://huggingface.co/ileening0022/Inf-SSM/resolve/main/cifar100_10task/task_0_checkpoint_seed_0.pth
```

## Running

```bash
conda activate inf-ssm
export WANDB_API_KEY=...          # scripts read it from the environment
export DATA_ROOT=../dataset       # optional, these are the defaults
export INIT_ROOT=../checkpoints

cd vim
bash scripts/imagenetr_5task.sh   # also: imagenetr_10task, cifar100_{5,10}task, caltech256_{5,10}task
```

Results land in `vim/output/inf_ssm/<run>/` (`accum_result_final.json` holds the
per-task accuracies and forgetting), logs in `vim/logs/inf_ssm/`.

## Key flags

| Flag | Meaning |
|---|---|
| `--cl_method` | `reg` = Inf-SSM, `seq` = vanilla sequential finetuning |
| `--reg_mode` | `inf_ssm` = Grassmannian observability distance, `mse` = plain MSE on `(A, C)` ablation |
| `--reg_lambda` | Regularizer weight (benchmark-specific; see the scripts) |
| `--n_tasks` | Number of tasks in the class-incremental split |
| `--finetune` | Task-0 checkpoint to start from |
| `--task_lr_scaling` | Multiplies the LR at each task boundary (`0.25` in the reported runs) |
| `--save_all` | Keep a checkpoint per task instead of overwriting one |

## Citation

```bibtex
@InProceedings{Lee_2026_CVPR,
    author    = {Lee, Isaac Ning and Mahmoodi, Leila and Le, Trung and Harandi, Mehrtash},
    title     = {Exemplar-Free Continual Learning for State Space Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {25077-25087}
}
```

## Acknowledgements

Built on [Vim](https://github.com/hustvl/Vim), [Mamba](https://github.com/state-spaces/mamba)
and [DeiT](https://github.com/facebookresearch/deit); the continual-learning portion refers
to [Mammoth](https://github.com/aimagelab/mammoth).

[Claude](https://claude.com/claude-code) was used to clean up and reorganize this code for
release, extracting it from a larger research codebase.

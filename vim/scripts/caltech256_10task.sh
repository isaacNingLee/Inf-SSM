#!/usr/bin/env bash

# Inf-SSM: Grassmannian (A, C) regularization.
# Caltech-256, 10 tasks, 3 seeds.

export MODEL="state_ac_vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
export BATCH_SIZE=128
export WARMUP_LR=1e-5
export TASK_LR_SCALING=0.5
export LR=1e-4
export MIN_LR=1e-5
export DROP_PATH=0.0
export WEIGHT_DECAY=0.1
export NUM_WORKERS=8
export N_TASKS=10
export DATA_PATH="${DATA_ROOT:-../dataset}/caltech256"
export OUTPUT_DIR_small="./output/inf_ssm"
export FINETUNE_PATH=("${INIT_ROOT:-../checkpoints}/caltech256_10task/task_0_checkpoint_seed_0.pth" \
    "${INIT_ROOT:-../checkpoints}/caltech256_10task/task_0_checkpoint_seed_10.pth" \
    "${INIT_ROOT:-../checkpoints}/caltech256_10task/task_0_checkpoint_seed_100.pth")
export EPOCHS=40
export DATA_SET="SeqCaltech256"
export CL_METHOD="reg"
export WANDB_PROJECT="inf-ssm_small-caltech256-${N_TASKS}task"
# Set WANDB_API_KEY in your environment (export WANDB_API_KEY=...) before running.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export LOG_DIR="./logs/inf_ssm"
export DEVICE="0"
export REG_LAMBDA=2500000
export REG_MODE="inf_ssm"

NO_AMP="--no_amp"
WANDB_FLAG="--wandb"

SEEDS=(0 10 100)

mkdir -p "${OUTPUT_DIR_small}"
mkdir -p "${LOG_DIR}"

for i in "${!SEEDS[@]}"; do

    CURRENT_RUN_NAME="${N_TASKS}task_${CL_METHOD}_small_SEED_${SEEDS[$i]}_${REG_LAMBDA}"

    CURRENT_OUTPUT_DIR="${OUTPUT_DIR_small}/caltech256_${CURRENT_RUN_NAME}"
    CURRENT_OUT_LOG="${LOG_DIR}/caltech256_${CURRENT_RUN_NAME}.out"
    CURRENT_ERR_LOG="${LOG_DIR}/caltech256_${CURRENT_RUN_NAME}.err"

    echo "-----------------------------------------------------"
    echo "Starting run: ${CURRENT_RUN_NAME}"
    echo "  model:      ${MODEL}"
    echo "  logs:       ${CURRENT_OUT_LOG}, ${CURRENT_ERR_LOG}"
    echo "-----------------------------------------------------"

    python main.py \
        --model                "${MODEL}" \
        --batch-size           "${BATCH_SIZE}" \
        --warmup-lr            "${WARMUP_LR}" \
        --lr                   "${LR}" \
        --task_lr_scaling      "${TASK_LR_SCALING}" \
        --min-lr               "${MIN_LR}" \
        --drop-path            "${DROP_PATH}" \
        --weight-decay         "${WEIGHT_DECAY}" \
        --num_workers          "${NUM_WORKERS}" \
        --n_tasks              "${N_TASKS}" \
        --data-path            "${DATA_PATH}" \
        --output_dir           "${CURRENT_OUTPUT_DIR}" \
        --epochs               "${EPOCHS}" \
        ${NO_AMP} \
        --data-set              "${DATA_SET}" \
        --finetune               "${FINETUNE_PATH[$i]}" \
        --cl_method               "${CL_METHOD}" \
        --wandb_project           "${WANDB_PROJECT}" \
        --wandb_run_name          "${CURRENT_RUN_NAME}" \
        --wandb_api_key           "${WANDB_API_KEY}" \
        ${WANDB_FLAG} \
        --seed                    "${SEEDS[$i]}" \
        --device                  "${DEVICE}" \
        --reg_lambda              "${REG_LAMBDA}" \
        --reg_mode                "${REG_MODE}" \
        --save_all \
        > "${CURRENT_OUT_LOG}" 2> "${CURRENT_ERR_LOG}"

    echo "Done with run: ${CURRENT_RUN_NAME}"
    echo
done

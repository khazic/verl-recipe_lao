#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================================
# Vietnamese MCQ RLVR Training Script (GRPO)
# ============================================================================
# Usage:
#   # Single node, 8 GPUs (Ray will be started automatically)
#   bash train_grpo.sh
#
#   # Multi-node: First start Ray cluster using start_multinode.sh
#   # On Head Node:
#   #   NODE_RANK=0 NNODES=2 bash start_multinode.sh
#   # On Worker Nodes:
#   #   NODE_RANK=1 HEAD_IP=<head_ip> bash start_multinode.sh
#   # Then on Head Node, run training:
#   #   NNODES=2 bash train_grpo.sh
# ============================================================================

# Project settings
project_name=${PROJECT_NAME:-"RLVR_Vi"}
exp_name=${EXP_NAME:-"RLVR-Vi-MCQ"}

# Algorithm settings
adv_estimator=grpo
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

# Clip ratio for PPO/GRPO
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

# Sequence length settings
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}

# Batch size settings
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-256}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-16}

# Distributed settings
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Path settings
DATA_HOME=${DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${DATA_HOME}/models/Qwen2.5-7B-Instruct"}  # Change to your model
CKPTS_DIR=${CKPTS_DIR:-"${DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_HOME}/data/rlvr_vi/rl/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_HOME}/data/rlvr_vi/rl/val.parquet"}
PROMPT_KEY=${PROMPT_KEY:-prompt}

# Sampling settings
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}  # 0 for HF rollout, -1 for vLLM rollout
val_top_p=${VAL_TOP_P:-0.7}

# Parallelism settings
sp_size=${SP_SIZE:-1}  # Sequence parallel size (increase for long sequences)
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$((max_prompt_length + max_response_length))}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$((max_prompt_length + max_response_length))}
offload=${OFFLOAD:-False}  # Set True for large models to offload to CPU
gen_tp=${GEN_TP:-1}  # Tensor parallel for generation
fsdp_size=${FSDP_SIZE:--1}  # FSDP sharding size (-1 for auto)

# Get script directory for reward function path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key="${PROMPT_KEY}" \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${SCRIPT_DIR}/reward_function.py" \
    custom_reward_function.name=mcq_reward_function \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto

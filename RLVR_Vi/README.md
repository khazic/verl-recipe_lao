# Vietnamese MCQ RLVR (Reinforcement Learning with Verifiable Rewards)

## Introduction

This recipe implements RLVR for Vietnamese multiple-choice questions (MCQ). The task trains a language model to answer MCQ questions where the answer is a single letter (A, B, C, D, or E).

The reward function verifies whether the model's answer matches the ground truth, supporting various output formats:
- `\boxed{A}` - LaTeX boxed format
- `<answer>A</answer>` - XML tag format
- `Answer: A` or `Đáp án: A` - Explicit answer line
- Trailing choice letter at the end

## Dataset Format

### Input JSON Format
```json
[
    {
        "question": "Chỉ trả lời bằng MỘT chữ cái: A, B, C, D hoặc E...",
        "response": "B",
        "_source_file": "train.jsonl"
    },
    ...
]
```

### Output Parquet Format (for RL training)
- `prompt`: List of messages `[{"role": "user", "content": question}]`
- `data_source`: String identifier
- `ability`: String identifier  
- `reward_model`: Dict with `{"style": "rule", "ground_truth": "B"}`
- `extra_info`: Dict with additional info

## Quick Start

### 1. Install verl
```bash
pip install verl
```

### 2. Prepare Dataset

Convert your JSON file to parquet format:
```bash
python create_dataset.py \
    --input_file /path/to/your/rlvr_vi.json \
    --output_dir ~/verl/data/rlvr_vi \
    --train_ratio 0.9 \
    --seed 42
```

This will create:
- `~/verl/data/rlvr_vi/rl/train.parquet`
- `~/verl/data/rlvr_vi/rl/val.parquet`

### 3. Run GRPO Training

#### Single Node (8 GPUs)
```bash
# Set your model and data paths
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export TRAIN_FILE=~/verl/data/rlvr_vi/rl/train.parquet
export VAL_FILE=~/verl/data/rlvr_vi/rl/val.parquet

bash train_grpo.sh
```

#### Multi-Node Training (Using Ray)

verl uses **Ray** for distributed training. Follow these steps:

**Step 1: Start Ray Cluster**

```bash
# On Head Node (Node 0):
NODE_RANK=0 NNODES=2 bash start_multinode.sh

# On Worker Node (Node 1):
NODE_RANK=1 HEAD_IP=<head_node_ip> bash start_multinode.sh
```

**Step 2: Run Training (on Head Node)**

```bash
# After all nodes joined, run on Head Node:
export NNODES=2
export MODEL_PATH=/path/to/model
bash train_grpo.sh
```

**Alternative: Submit as Ray Job**

```bash
# From any machine that can access the Ray dashboard
ray job submit --address="http://<head_node_ip>:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- bash /path/to/train_grpo.sh
```

**Verify Cluster Status**

```bash
# Check Ray cluster status
ray status

# View Ray dashboard
# Open browser: http://<head_node_ip>:8265
```

## Configuration

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `~/verl/models/Qwen2.5-7B-Instruct` | Path to pretrained model |
| `TRAIN_FILE` | `~/verl/data/rlvr_vi/rl/train.parquet` | Training data path |
| `VAL_FILE` | `~/verl/data/rlvr_vi/rl/val.parquet` | Validation data path |
| `NNODES` | `1` | Number of nodes |
| `NGPUS_PER_NODE` | `8` | GPUs per node |
| `TRAIN_PROMPT_BSZ` | `256` | Training batch size |
| `N_RESP_PER_PROMPT` | `8` | Responses per prompt (for GRPO) |
| `MAX_PROMPT_LENGTH` | `2048` | Max prompt length |
| `MAX_RESPONSE_LENGTH` | `512` | Max response length |
| `OFFLOAD` | `False` | CPU offload (for large models) |
| `GEN_TP` | `1` | Tensor parallel for generation |

### For Large Models (e.g., 72B)

```bash
export OFFLOAD=True
export GEN_TP=8
export SP_SIZE=8
export FSDP_SIZE=32
export TRAIN_PROMPT_BSZ=128
export TRAIN_PROMPT_MINI_BSZ=8
bash train_grpo.sh
```

## Reward Function

The `reward_function.py` implements a rule-based reward:

```python
def mcq_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    # Extract model's answer (supports multiple formats)
    pred = _extract_choice(solution_str)
    
    # Compare with ground truth
    if pred == ground_truth:
        return 1  # Correct
    else:
        return 0  # Wrong
```

## Expected Results

- After 1-2 epochs, you should see validation accuracy improvements
- The model learns to output answers in a consistent format
- Monitor `reward/mean` and `val/reward_mean` in logs

## Troubleshooting

### Out of Memory
- Reduce `TRAIN_PROMPT_BSZ` and `TRAIN_PROMPT_MINI_BSZ`
- Enable `OFFLOAD=True`
- Reduce `N_RESP_PER_PROMPT`

### Slow Training
- Increase `GEN_TP` for faster generation
- Enable chunked prefill (already enabled by default)
- Use more GPUs

### Low Reward
- Check if model outputs match expected format
- Verify ground truth labels are correct (A/B/C/D/E)
- Consider SFT warmup before RLVR

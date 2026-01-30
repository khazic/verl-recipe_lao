# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert Vietnamese MCQ JSON dataset to parquet format for RLVR training.

Input JSON format:
[
    {
        "question": "...",
        "response": "A",  # or B, C, D, E
        "_source_file": "train.jsonl"
    },
    ...
]

Output parquet format (for RL training):
- prompt: list of messages [{"role": "user", "content": question}]
- data_source: string identifier
- ability: string identifier
- reward_model: dict with ground_truth
- extra_info: dict with additional info
"""

import argparse
import json
import os
import random

import pandas as pd


def load_json_data(json_path: str) -> list[dict]:
    """Load JSON data from file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def create_rl_dataset(data: list[dict], data_source: str = "rlvr_vi") -> pd.DataFrame:
    """
    Convert raw data to RL dataset format.
    
    Args:
        data: List of dicts with 'question' and 'response' keys
        data_source: Identifier for the data source
    
    Returns:
        DataFrame with columns: prompt, data_source, ability, reward_model, extra_info
    """
    rl_dataset = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
    }

    for item in data:
        question = item.get("question", "")
        response = item.get("response", "").strip().upper()
        
        if not question or not response:
            continue
        
        # Format prompt as chat messages
        prompt_with_template = [
            {
                "role": "user",
                "content": question,
            }
        ]

        rl_dataset["prompt"].append(prompt_with_template)
        rl_dataset["data_source"].append(data_source)
        rl_dataset["ability"].append("mcq")
        rl_dataset["reward_model"].append(
            {"style": "rule", "ground_truth": response}
        )
        rl_dataset["extra_info"].append(
            {"response": response, "source_file": item.get("_source_file", "")}
        )

    return pd.DataFrame(data=rl_dataset)


def create_sft_dataset(data: list[dict]) -> pd.DataFrame:
    """
    Convert raw data to SFT dataset format.
    
    Args:
        data: List of dicts with 'question' and 'response' keys
    
    Returns:
        DataFrame with 'messages' column
    """
    sft_dataset = {"messages": []}

    for item in data:
        question = item.get("question", "")
        response = item.get("response", "").strip().upper()
        
        if not question or not response:
            continue

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        sft_dataset["messages"].append(messages)

    return pd.DataFrame(data=sft_dataset)


def main():
    parser = argparse.ArgumentParser(description="Convert MCQ JSON to parquet for RLVR")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/rlvr_vi",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training (rest for validation)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="rlvr_vi",
        help="Data source identifier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--create_sft",
        action="store_true",
        help="Also create SFT dataset",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Expand user path
    output_dir = os.path.expanduser(args.output_dir)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_json_data(args.input_file)
    print(f"Loaded {len(data)} samples")

    # Shuffle data
    random.shuffle(data)

    # Split into train and val
    train_size = int(len(data) * args.train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Create RL dataset
    rl_output_dir = os.path.join(output_dir, "rl")
    os.makedirs(rl_output_dir, exist_ok=True)

    train_rl_df = create_rl_dataset(train_data, args.data_source)
    val_rl_df = create_rl_dataset(val_data, args.data_source)

    train_rl_path = os.path.join(rl_output_dir, "train.parquet")
    val_rl_path = os.path.join(rl_output_dir, "val.parquet")

    train_rl_df.to_parquet(train_rl_path)
    val_rl_df.to_parquet(val_rl_path)

    print(f"Saved RL train dataset to {train_rl_path} ({len(train_rl_df)} samples)")
    print(f"Saved RL val dataset to {val_rl_path} ({len(val_rl_df)} samples)")

    # Optionally create SFT dataset
    if args.create_sft:
        sft_output_dir = os.path.join(output_dir, "sft")
        os.makedirs(sft_output_dir, exist_ok=True)

        train_sft_df = create_sft_dataset(train_data)
        val_sft_df = create_sft_dataset(val_data)

        train_sft_path = os.path.join(sft_output_dir, "train.parquet")
        val_sft_path = os.path.join(sft_output_dir, "val.parquet")

        train_sft_df.to_parquet(train_sft_path)
        val_sft_df.to_parquet(val_sft_path)

        print(f"Saved SFT train dataset to {train_sft_path} ({len(train_sft_df)} samples)")
        print(f"Saved SFT val dataset to {val_sft_path} ({len(val_sft_df)} samples)")

    print("Done!")


if __name__ == "__main__":
    main()

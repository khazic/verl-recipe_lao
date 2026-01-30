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
Rule-based reward function for Vietnamese multiple-choice (A/B/C/D/E) RLVR.

This reward function extracts the model's answer from various formats:
1. \boxed{A} - LaTeX boxed format
2. <answer>A</answer> - XML tag format
3. Answer: A - Explicit answer line
4. Trailing choice letter at the end

And compares it with the ground truth to give a binary reward (0 or 1).
"""

import re
from typing import Any

# Regex patterns for extracting choices
_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Ea-e])\s*\}")
_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Ea-e])\s*</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_LINE_RE = re.compile(r"^\s*(?:answer|đáp án|trả lời)\s*[:：]\s*([A-Ea-e])\b", re.IGNORECASE | re.MULTILINE)
_TRAILING_CHOICE_RE = re.compile(r"\b([A-Ea-e])\b(\s*[).。]?\s*)$", re.MULTILINE)


def _extract_choice(text: str) -> str | None:
    """
    Extract the choice letter from model output.
    
    Tries multiple patterns in order of specificity:
    1. \boxed{X}
    2. <answer>X</answer>
    3. Answer: X (supports Vietnamese "đáp án", "trả lời")
    4. Trailing choice letter
    5. Last standalone choice letter anywhere
    
    Returns:
        Uppercase letter (A-E) or None if not found
    """
    if not text:
        return None

    # Try \boxed{} format first (most explicit)
    match = _BOXED_RE.search(text)
    if match:
        return match.group(1).upper()

    # Try <answer> tag format
    match = _ANSWER_TAG_RE.search(text)
    if match:
        return match.group(1).upper()

    # Try "Answer:" or "Đáp án:" line format
    match = _ANSWER_LINE_RE.search(text)
    if match:
        return match.group(1).upper()

    # Try trailing choice letter at the end
    match = _TRAILING_CHOICE_RE.search(text.strip())
    if match:
        return match.group(1).upper()

    # Fallback: last standalone choice letter anywhere in the text
    matches = re.findall(r"\b([A-Ea-e])\b", text)
    if matches:
        return matches[-1].upper()

    return None


def _normalize_gt(ground_truth: Any) -> str | None:
    """
    Normalize ground truth to uppercase letter.
    
    Handles various formats:
    - Integer (0-4 -> A-E)
    - String (direct letter)
    - Dict (with 'target', 'answer', 'response' keys)
    - List/Tuple (takes first element)
    
    Returns:
        Uppercase letter or None if cannot normalize
    """
    if ground_truth is None:
        return None

    # Handle integer (0=A, 1=B, etc.)
    if isinstance(ground_truth, int):
        if 0 <= ground_truth <= 4:
            return "ABCDE"[ground_truth]
        return str(ground_truth).strip().upper()

    # Handle dict with various keys
    if isinstance(ground_truth, dict):
        for key in (
            "target",
            "answer",
            "label",
            "gold",
            "correct",
            "correct_answer",
            "response",
        ):
            if key in ground_truth:
                value = ground_truth[key]
                if isinstance(value, int) and 0 <= value <= 4:
                    return "ABCDE"[value]
                return str(value).strip().upper()
        return None

    # Handle list/tuple (take first element)
    if isinstance(ground_truth, (list, tuple)) and ground_truth:
        return str(ground_truth[0]).strip().upper()

    # Handle string directly
    return str(ground_truth).strip().upper()


def mcq_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """
    Reward function for multiple-choice questions.
    
    Args:
        data_source: Identifier for the data source (unused but required by interface)
        solution_str: Model's generated response
        ground_truth: Expected correct answer
        extra_info: Optional dict with additional info (may contain 'response' key)
    
    Returns:
        1 if model's answer matches ground truth, 0 otherwise
    """
    try:
        # Extract model's predicted choice
        pred = _extract_choice(solution_str)
        
        # Get ground truth, preferring extra_info.response if available
        gt = None
        if extra_info is not None and isinstance(extra_info, dict) and "response" in extra_info:
            gt = _normalize_gt(extra_info["response"])
        if gt is None:
            gt = _normalize_gt(ground_truth)
        
        # Cannot evaluate if either is missing
        if pred is None or gt is None:
            return 0
        
        # Binary reward: exact match
        return 1 if pred == gt else 0
    except Exception:
        return 0

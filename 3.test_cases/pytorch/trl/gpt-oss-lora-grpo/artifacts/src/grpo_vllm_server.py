# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
GRPO Training Script with vLLM Server Mode + DeepSpeed ZeRO-3

Fine-tunes an SFT model using TRL's GRPOTrainer with vLLM for fast generation.
Uses DeepSpeed ZeRO-3 for multi-node distributed training.

Architecture:
    - vLLM server (separate pod): handles generation on dedicated GPUs
    - Training workers (this script): DeepSpeed ZeRO-3 across N nodes
    - Weight sync: trainer pushes updated weights to vLLM server after each step

Model loading:
    - Uses pre-merged SFT model at /fsx/models/gpt-oss-20b-sft-merged/
      (openai/gpt-oss-20b base + SFT LoRA merged, saved as bf16)
    - Model passed as STRING to GRPOTrainer — DeepSpeed ZeRO-3 handles sharding
    - New LoRA applied on top via peft_config for GRPO training
    - vLLM server loads the same merged model for generation

Usage:
    torchrun --nproc_per_node=4 --nnodes=3 \
        --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        grpo_vllm_server.py \
        --model_name_or_path /fsx/models/gpt-oss-20b-sft-merged \
        --vllm_server_host <VLLM_POD_IP> \
        --output_dir /fsx/checkpoints/grpo-vllm
"""

import os
import re
import sys
import argparse
import torch
from datetime import datetime
from typing import List
from datasets import load_dataset

# Language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("WARNING: langdetect not installed")


LANG_CODE_MAP = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
}

SUPPORTED_LANGUAGES = list(LANG_CODE_MAP.keys())


def extract_reasoning(response: str) -> str:
    """Extract reasoning section from model response."""
    match = re.search(r'assistantanalysis\s*(.*?)\s*assistantfinal', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response[:500] if len(response) > 500 else response


def extract_final_answer(response: str) -> str:
    """Extract final answer section."""
    match = re.search(r'assistantfinal\s*(.*)$', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response[-200:] if len(response) > 200 else response


def detect_language(text: str) -> str:
    """Detect language of text."""
    if not HAS_LANGDETECT:
        return "unknown"
    try:
        clean_text = re.sub(r'[0-9\+\-\*\/\=\%\$\€\£]', '', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        if len(clean_text.strip()) < 20:
            return "too_short"
        return detect(clean_text)
    except Exception:
        return "error"


def count_sentences(text: str) -> int:
    """Count number of sentences in text."""
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def language_reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Language-aware reward function for GRPO.

    Priority order:
    1. Answer language (most important): +5.0 / -5.0
    2. Reasoning language: +1.5 / -1.5
    3. Final answer brevity (<=2 sentences): +0.5 / -1.0

    Max score: +7.0, Min score: -7.5
    """
    rewards = []

    for i, completion in enumerate(completions):
        reward = 0.0
        reasoning = extract_reasoning(completion)
        final_answer = extract_final_answer(completion)
        reasoning_lang = detect_language(reasoning)
        output_lang = detect_language(final_answer)

        # Determine expected language from prompt
        expected_code = "en"
        if prompts and i < len(prompts):
            prompt = prompts[i].lower()
            for lang_name, lang_code in LANG_CODE_MAP.items():
                if f"reasoning language: {lang_name.lower()}" in prompt:
                    expected_code = lang_code
                    break

        # 1. ANSWER LANGUAGE (most important)
        if output_lang == expected_code:
            reward += 5.0
        else:
            reward -= 5.0

        # 2. REASONING LANGUAGE
        if reasoning_lang == expected_code:
            reward += 1.5
        else:
            reward -= 1.5

        # 3. FINAL ANSWER BREVITY
        answer_sentences = count_sentences(final_answer)
        if answer_sentences <= 2:
            reward += 0.5
        else:
            reward -= 1.0

        rewards.append(reward)

    return rewards


def detect_question_language(text: str) -> str:
    """Detect language of question text and return language name."""
    if not HAS_LANGDETECT:
        return "English"
    try:
        clean_text = re.sub(r'[0-9\+\-\*\/\=\%\$\€\£]', '', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        if len(clean_text.strip()) < 10:
            return "English"
        detected_code = detect(clean_text)
        code_to_name = {v: k for k, v in LANG_CODE_MAP.items()}
        return code_to_name.get(detected_code, "English")
    except Exception:
        return "English"


def prepare_dataset(dataset_name: str, tokenizer, eval_size: int = 20):
    """Prepare dataset with question language detection."""
    dataset = load_dataset(dataset_name)

    def format_prompt(example):
        messages = example.get("messages", [])
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break

        detected_language = detect_question_language(user_prompt)
        if detected_language not in SUPPORTED_LANGUAGES:
            detected_language = "English"

        chat_messages = [
            {"role": "system", "content": f"reasoning language: {detected_language}\nanswer language: {detected_language}"},
            {"role": "user", "content": user_prompt}
        ]

        try:
            formatted_prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = (
                f"System: reasoning language: {detected_language}\n"
                f"answer language: {detected_language}\n"
                f"User: {user_prompt}\nAssistant:"
            )

        return {"prompt": formatted_prompt}

    print("Processing dataset...")
    formatted_dataset = dataset["train"].map(
        format_prompt,
        remove_columns=dataset["train"].column_names,
    )

    total_size = len(formatted_dataset)
    eval_indices = list(range(0, min(eval_size, total_size)))
    train_indices = list(range(eval_size, total_size))

    eval_dataset = formatted_dataset.select(eval_indices)
    train_dataset = formatted_dataset.select(train_indices)

    print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="GRPO training with vLLM server mode")
    parser.add_argument("--model_name_or_path", type=str, default="/fsx/models/gpt-oss-20b-sft-merged",
                        help="Path to pre-merged SFT model (bf16)")
    parser.add_argument("--vllm_server_host", type=str, required=True,
                        help="IP address or hostname of the vLLM server pod")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                        help="Port of the vLLM server")
    parser.add_argument("--output_dir", type=str, default="/fsx/checkpoints/grpo-vllm")
    parser.add_argument("--num_generations", type=int, default=8, help="K generations per prompt")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=20)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--deepspeed", type=str, default="/app/configs/ds_config_zero3.json",
                        help="Path to DeepSpeed ZeRO-3 config JSON")
    args = parser.parse_args()

    # Get distributed info (set by torchrun)
    world_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (world_rank == 0)

    if is_main:
        print("=" * 60)
        print("GRPO Training with vLLM Server Mode + DeepSpeed ZeRO-3")
        print("=" * 60)
        print(f"Model: {args.model_name_or_path}")
        print(f"vLLM Server: {args.vllm_server_host}:{args.vllm_server_port}")
        print(f"Output: {args.output_dir}")
        print(f"K (num_generations): {args.num_generations}")
        print(f"Max completion length: {args.max_completion_length}")
        print(f"Epochs: {args.num_epochs}")
        print(f"World Size: {world_size}, Local Rank: {local_rank}")
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        print("=" * 60)

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"ERROR: {e}")
        print("GRPOTrainer requires TRL >= 0.14.0")
        return

    from peft import LoraConfig
    from transformers import AutoTokenizer

    # Load tokenizer (needed for dataset preparation)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(
        "HuggingFaceH4/Multilingual-Thinking",
        tokenizer,
        eval_size=args.eval_size,
    )

    # LoRA config for GRPO training (applied on top of merged SFT model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Calculate steps per epoch
    dataset_size = len(train_dataset)
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, dataset_size // effective_batch)

    # generation_batch_size must be divisible by num_generations.
    # At config time, DeepSpeed hasn't initialized so world_size=1 in GRPOConfig.__post_init__.
    # We compute the correct value ourselves: per_device_bs * world_size * grad_accum_steps.
    generation_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    # Round up to nearest multiple of num_generations if needed
    remainder = generation_batch_size % args.num_generations
    if remainder != 0:
        generation_batch_size += args.num_generations - remainder

    if is_main:
        print(f"Train dataset size: {dataset_size}")
        print(f"Effective batch size: {effective_batch}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps for {args.num_epochs} epochs: {steps_per_epoch * args.num_epochs}")
        print(f"Generation batch size: {generation_batch_size} (must be divisible by {args.num_generations})")

    # GRPO config with vLLM server mode + DeepSpeed ZeRO-3
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=1,
        save_steps=steps_per_epoch,
        save_total_limit=args.num_epochs + 2,
        num_generations=args.num_generations,
        generation_batch_size=generation_batch_size,
        max_completion_length=args.max_completion_length,
        temperature=0.7,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        # DeepSpeed ZeRO-3 — HF Trainer native integration
        deepspeed=args.deepspeed,
        # vLLM server mode configuration
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=600.0,  # 10 min timeout for first startup
        # Importance sampling correction for train/inference mismatch
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="sequence_mask",
        vllm_importance_sampling_cap=3.0,
        # DeepSpeed ZeRO-3 handles model sharding
        ds3_gather_for_generation=True,
        # Model init kwargs — passed to from_pretrained() when loading model string
        # device_map=None is REQUIRED: TRL defaults to device_map="auto" which is
        # incompatible with DeepSpeed ZeRO-3 (ZeRO-3 handles sharding itself)
        model_init_kwargs={
            "trust_remote_code": True,
            "device_map": None,
        },
    )

    if is_main:
        print(f"\nGRPO Config:")
        print(f"  use_vllm: {grpo_config.use_vllm}")
        print(f"  vllm_mode: {grpo_config.vllm_mode}")
        print(f"  vllm_server: {grpo_config.vllm_server_host}:{grpo_config.vllm_server_port}")
        print(f"  bf16: {grpo_config.bf16}")
        print(f"  gradient_checkpointing: {grpo_config.gradient_checkpointing}")
        print(f"  deepspeed: {grpo_config.deepspeed}")
        import json
        if grpo_config.deepspeed and isinstance(grpo_config.deepspeed, str):
            try:
                with open(grpo_config.deepspeed) as f:
                    ds_cfg = json.load(f)
                print(f"  deepspeed config: {json.dumps(ds_cfg, indent=2)}")
            except Exception as e:
                print(f"  deepspeed config read error: {e}")
        print(f"\nInitializing GRPOTrainer...")

    # Pass model as STRING — DeepSpeed ZeRO-3 handles loading and sharding.
    # The merged SFT model at /fsx/models/gpt-oss-20b-sft-merged/ is bf16
    # (41.9 GB, 9 safetensors shards). ZeRO-3 shards it across all GPUs.
    # LoRA is applied on top via peft_config.
    trainer = GRPOTrainer(
        model=args.model_name_or_path,  # String, not a loaded model
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=language_reward_fn,
        peft_config=peft_config,
    )

    if is_main:
        print("Starting GRPO training with vLLM server mode...")
        print(f"  Training nodes: {world_size} GPUs across {world_size // 4} nodes")
        print(f"  Generation: vLLM server at {args.vllm_server_host}:{args.vllm_server_port}")

    trainer.train()

    if is_main:
        print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if is_main:
        print("=" * 60)
        print("GRPO training complete!")
        print(f"Model saved to: {args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()

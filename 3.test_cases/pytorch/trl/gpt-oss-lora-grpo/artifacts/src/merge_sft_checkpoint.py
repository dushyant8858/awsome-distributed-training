# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Merge SFT LoRA Checkpoint into Base Model

One-time script to merge the SFT LoRA adapter into the base GPT-OSS 20B model
and save the merged model to disk. This is required for the vLLM server mode
where both the vLLM server and trainer need to load the same merged model.

Usage:
    python merge_sft_checkpoint.py \
        --model_name_or_path openai/gpt-oss-20b \
        --peft_checkpoint /fsx/checkpoints/checkpoint-1000 \
        --output_dir /fsx/models/gpt-oss-20b-sft-merged

Takes ~5-10 minutes (load 40GB model, merge 15MB adapter, save 40GB merged model).
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge SFT LoRA adapter into base model")
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b",
                        help="Base model name or path")
    parser.add_argument("--peft_checkpoint", type=str, default="/fsx/checkpoints/checkpoint-1000",
                        help="Path to PEFT/LoRA checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="/fsx/models/gpt-oss-20b-sft-merged",
                        help="Output directory for merged model")
    args = parser.parse_args()

    print("=" * 60)
    print("Merging SFT LoRA Checkpoint into Base Model")
    print("=" * 60)
    print(f"Base model: {args.model_name_or_path}")
    print(f"PEFT checkpoint: {args.peft_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Validate checkpoint
    if not os.path.exists(args.peft_checkpoint):
        raise FileNotFoundError(f"PEFT checkpoint not found: {args.peft_checkpoint}")

    adapter_files = [
        os.path.join(args.peft_checkpoint, "adapter_model.safetensors"),
        os.path.join(args.peft_checkpoint, "adapter_model.bin"),
    ]
    if not any(os.path.exists(f) for f in adapter_files):
        raise FileNotFoundError(f"No adapter weights found in: {args.peft_checkpoint}")

    # Load base model on CPU to avoid GPU memory issues
    print("\n[1/4] Loading base model (this takes ~2-3 min for 20B params)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    print(f"  Base model loaded: {model.config.num_hidden_layers} layers, dtype={model.dtype}")

    # Load tokenizer
    print("\n[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load and merge PEFT adapter
    print(f"\n[3/4] Loading and merging PEFT adapter from {args.peft_checkpoint}...")
    model = PeftModel.from_pretrained(model, args.peft_checkpoint)
    model = model.merge_and_unload()
    print("  LoRA weights merged into base model!")

    # Save merged model
    print(f"\n[4/4] Saving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Verify saved files
    saved_files = os.listdir(args.output_dir)
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in saved_files
        if os.path.isfile(os.path.join(args.output_dir, f))
    )
    print(f"\n  Saved {len(saved_files)} files, total size: {total_size / 1e9:.1f} GB")
    for f in sorted(saved_files):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"    {f}: {size / 1e6:.1f} MB")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print(f"Merged model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

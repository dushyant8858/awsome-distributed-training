# TRL Instance Profiles

Instance profiles configure GPU count and EFA networking per EC2 instance type.
The profile is auto-detected at runtime by `_detect.sh`.

## Detection Order

1. `INSTANCE_PROFILE` env var (explicit override)
2. `INSTANCE_TYPE` env var
3. EC2 Instance Metadata API
4. GPU name from nvidia-smi

## Available Profiles

| Profile | Instance | GPU | VRAM | Status |
|---------|----------|-----|------|--------|
| [p5en-48xlarge.env](p5en-48xlarge.env) | p5en.48xlarge | 8x H200 | 141 GB | Tested |
| [p4de-24xlarge.env](p4de-24xlarge.env) | p4de.24xlarge | 8x A100 | 80 GB | Untested |
| [g5-12xlarge.env](g5-12xlarge.env) | g5.12xlarge | 4x A10G | 24 GB | Untested |
| [g6e-12xlarge.env](g6e-12xlarge.env) | g6e.12xlarge | 4x L40S | 48 GB | Untested |

## Notes

The `grpo-math-reasoning` script splits nodes between training (accelerate +
DeepSpeed) and vLLM serving. The `GPUS_PER_NODE` from the profile drives:
- `--num_processes` for accelerate (TRAIN_NODES * GPUS_PER_NODE)
- `TENSOR_PARALLEL` for vLLM (auto-computed from model's attention heads)

The `gpt-oss-lora-grpo` sub-case uses K8s manifests that are currently
hardcoded to g6e.12xlarge. See its README for manual adjustment instructions.

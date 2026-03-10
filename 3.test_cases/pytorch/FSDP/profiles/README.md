# FSDP Instance Profiles

Instance profiles configure GPU count, EFA networking, NCCL settings, and FSDP
memory optimizations per EC2 instance type. The profile is auto-detected at
runtime so the same sbatch script works across different clusters.

## How Detection Works

The detection script (`_detect.sh`) selects the profile automatically:

1. **`INSTANCE_PROFILE`** env var — explicit override (e.g., `g5-12xlarge`)
2. **`INSTANCE_TYPE`** env var — from your `env_vars` (e.g., `g5.12xlarge`)
3. **EC2 Instance Metadata API** — works on bare metal and K8s with host networking
4. **GPU name from nvidia-smi** — fallback when metadata is unavailable

To override: `export INSTANCE_PROFILE=g5-12xlarge` before running sbatch.

## Available Profiles

| Profile | Instance | GPU | VRAM | Compatible Models | Status |
|---------|----------|-----|------|------------------|--------|
| [p5en-48xlarge.env](p5en-48xlarge.env) | p5en.48xlarge | 8x H200 | 141 GB | All 9 models | Tested |
| [p5-48xlarge.env](p5-48xlarge.env) | p5.48xlarge | 8x H100 | 80 GB | All 9 models | Tested |
| [p4de-24xlarge.env](p4de-24xlarge.env) | p4de.24xlarge | 8x A100 | 80 GB | All 9 models | Tested |
| [g5-12xlarge.env](g5-12xlarge.env) | g5.12xlarge | 4x A10G | 24 GB | 1B, 3B; tight for 7B-8B | Tested |
| [g6e-12xlarge.env](g6e-12xlarge.env) | g6e.12xlarge | 4x L40S | 48 GB | 1B-8B; tight for 13B | Untested |

## What the Profile Controls

Settings that change per instance (set by profile):

| Setting | Description |
|---------|-------------|
| `GPUS_PER_NODE` | Number of GPUs (4 for g5/g6e, 8 for p4de/p5/p5en) |
| `FI_PROVIDER`, `FI_EFA_*` | EFA networking (set for EFA instances, unset for g5/g6e) |
| `EFA_PER_NODE` | EFA adapter count for K8s resource requests |
| `NCCL_SOCKET_IFNAME` | NCCL network interface filter |
| `FSDP_CPU_OFFLOAD` | Whether to offload FSDP parameters to CPU |
| `FSDP_CPU_OFFLOAD_MIN_LAYERS` | Only enable cpu_offload if model has >= this many layers. Avoids the ~30% offloading overhead on small models that fit without it (e.g., 1B/3B on g5). |

Settings that stay the same across instances (set in `models/*.txt`):

| Setting | Description |
|---------|-------------|
| Model architecture args | `hidden_width`, `num_layers`, `num_heads`, etc. |
| `--train_batch_size` | Per-GPU micro batch size |
| `--offload_activations` | Activation offloading (always 1) |
| `--sharding_strategy` | FSDP sharding (always `full`) |

## Model Compatibility by Instance

| Model | p5en/p5 (80-141 GB) | p4de (80 GB) | g6e (48 GB) | g5 (24 GB) |
|-------|---------------------|--------------|-------------|------------|
| Llama 3.2 1B | OK | OK | OK | OK |
| Llama 3.2 3B | OK | OK | OK | OK |
| Llama 2 7B | OK | OK | OK | Tight (cpu_offload) |
| Llama 3.1 8B | OK | OK | OK | Tight (cpu_offload) |
| Mathstral 7B | OK | OK | OK | Tight (cpu_offload) |
| Llama 2 13B | OK | OK | Tight | Won't fit |
| Llama 2 70B | OK | OK | Won't fit | Won't fit |
| Llama 3.1 70B | OK | OK | Won't fit | Won't fit |
| Mixtral 8x7B | OK | OK | Won't fit | Won't fit |

## Creating a New Profile

1. Copy the closest existing profile:
   ```bash
   cp profiles/p5en-48xlarge.env profiles/p4d-24xlarge.env
   ```
2. Adjust GPU count, EFA settings, and FSDP overrides
3. Run detection test:
   ```bash
   INSTANCE_TYPE=p4d.24xlarge bash profiles/_detect.sh profiles/
   ```

# Instance Hardware Profiles

This directory contains hardware specifications and distributed training
guidance for each EC2 accelerated instance family relevant to this repository.

Each profile includes GPU/accelerator specs (VRAM, compute TFLOPS, memory
bandwidth), interconnect topology (NVLink, NeuronLink), network configuration
(EFA generation, GPUDirect RDMA), and practical guidance for distributed
training workloads.

## NVIDIA GPU Instances

### G-Family (PCIe-attached, no NVSwitch)

| Profile | GPU | Architecture | VRAM | BF16 TFLOPS | Memory BW | GPUDirect RDMA | Max EFA |
|---------|-----|-------------|------|-------------|-----------|----------------|---------|
| [g4dn](g4dn.md) | T4 | Turing | 16 GB GDDR6 | 65 (FP16) | 320 GB/s | No | 1 (EFAv1) |
| [g5](g5.md) | A10G | Ampere | 24 GB GDDR6 | 125 | 600 GB/s | No | 1 (EFAv1) |
| [g6](g6.md) | L4 | Ada Lovelace | 24 GB GDDR6 | 121 | 300 GB/s | No | 1 (EFAv2) |
| [g6e](g6e.md) | L40S | Ada Lovelace | 48 GB GDDR6 | 362 | 864 GB/s | No | 4 (EFAv2) |
| [g7e](g7e.md) | RTX PRO 6000 | Blackwell | 96 GB GDDR7 | 125 | 1,597 GB/s | **Yes** | 4 (EFAv4) |

### P-Family (NVSwitch full-mesh, GPUDirect RDMA)

| Profile | GPU | Architecture | VRAM | BF16 TFLOPS | Memory BW | NVLink BW/GPU | Max EFA |
|---------|-----|-------------|------|-------------|-----------|---------------|---------|
| [p4d](p4d.md) | A100 40 GB | Ampere | 40 GB HBM2 | 312 | 1,555 GB/s | 600 GB/s (v3) | 4 (EFAv1) |
| [p4de](p4de.md) | A100 80 GB | Ampere | 80 GB HBM2e | 312 | 2,039 GB/s | 600 GB/s (v3) | 4 (EFAv1) |
| [p5](p5.md) | H100 | Hopper | 80 GB HBM3 | 990 | 3,350 GB/s | 900 GB/s (v4) | 32 (EFAv2) |
| [p5e](p5e.md) | H200 | Hopper | 141 GB HBM3e | 990 | 4,800 GB/s | 900 GB/s (v4) | 32 (EFAv2) |
| [p5en](p5en.md) | H200 | Hopper | 141 GB HBM3e | 990 | 4,800 GB/s | 900 GB/s (v4) | 16 (EFAv3) |
| [p6-b200](p6-b200.md) | B200 | Blackwell | 179 GB HBM3e | 2,250 | 8,000 GB/s | 1,800 GB/s (v5) | 8 (EFAv4) |
| [p6-b300](p6-b300.md) | B300 | Blackwell Ultra | 288 GB HBM3e | 2,250 | 8,000 GB/s | 1,800 GB/s (v5) | 17 (EFAv4) |
| [p6e-gb200](p6e-gb200.md) | B200 (Grace) | Blackwell | 185 GB HBM3e | 2,500 | 8,000 GB/s | 1,800 GB/s (v5) | 17 (EFAv4) |

## AWS Trainium / Inferentia Instances

| Profile | Accelerator | Cores/Chip | HBM/Chip | BF16/Chip | Interconnect | Max EFA |
|---------|-------------|------------|----------|-----------|--------------|---------|
| [trn1](trn1.md) | Trainium v1 | 2 (NeuronCore v2) | 32 GB | ~190 TFLOPS | NeuronLink v2 (768 GB/s) | 16 (EFAv2) |
| [trn2](trn2.md) | Trainium v2 | 8 (NeuronCore v3) | 96 GiB | ~667 TFLOPS | NeuronLink v3 (1,024 GB/s/chip) | 16 (EFAv3) |
| [inf2](inf2.md) | Inferentia v2 | 2 (NeuronCore v2) | 32 GB | 190 TFLOPS | NeuronLink v2 (192 GB/s) | None |

## Quick Selection Guide

**By VRAM per GPU (ascending):**
T4 (16 GB) → L4/A10G (24 GB) → A100-40 (40 GB) → L40S (48 GB) →
A100-80/H100 (80 GB) → RTX PRO 6000 (96 GB) → H200 (141 GB) →
B200 (179-185 GB) → B300 (288 GB)

**By use case:**
- **Experimentation / fine-tuning (small models)**: g5, g6, g4dn
- **Development / medium models**: g6e, g7e, p4d
- **Production training**: p4de, p5, p5e, p5en
- **Large-scale foundation models**: p6-b200, p6-b300, p6e-gb200
- **Neuron SDK training**: trn1n, trn2, trn2u
- **Inference**: inf2, g4dn, g6

**By GPUDirect RDMA support:**
- **Yes**: p4d, p4de, p5, p5e, p5en, p6-b200, p6-b300, p6e-gb200, g7e
- **No**: g4dn, g5, g6, g6e

## NCCL / EFA Configuration Reference

For NCCL environment variables, EFA settings, and version-specific tuning
guidance, see the [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

The key configuration difference between instance families:

```bash
# Instances WITH GPUDirect RDMA (p-family, g7e):
export FI_EFA_USE_DEVICE_RDMA=1
# NCCL_PROTO can use default (LL/LL128)

# Instances WITHOUT GPUDirect RDMA (g4dn, g5, g6, g6e):
export FI_EFA_USE_DEVICE_RDMA=0
export NCCL_PROTO=simple
```

## EFA Generations

| Generation | Instances | Key Improvement |
|------------|-----------|-----------------|
| EFAv1 | g4dn, g5, p4d, p4de | First generation |
| EFAv2 | g6, g6e, p5, p5e | Higher throughput |
| EFAv3 | p5en, trn2, trn2u | 35% lower latency vs EFAv2 |
| EFAv4 | g7e, p6-b200, p6-b300, p6e-gb200 | Latest generation |

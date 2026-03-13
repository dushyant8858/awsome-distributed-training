# trn2 / trn2u Instance Family — AWS Trainium v2

> Covers: trn2.48xlarge, trn2u.48xlarge (UltraServer)

## Hardware at a Glance

| Spec | trn2.48xlarge | trn2u.48xlarge (UltraServer) |
|------|---------------|------------------------------|
| Accelerator | AWS Trainium v2 | AWS Trainium v2 |
| Chips | 16 | 64 (4 nodes x 16) |
| NeuronCores (v3) | 128 | 512 |
| HBM per chip | 96 GiB (HBM3) | 96 GiB (HBM3) |
| Total accelerator memory | 1.5 TB | 6 TB |
| Memory bandwidth per chip | 2,900 GB/s | 2,900 GB/s |
| BF16 TFLOPS per chip | ~632 | ~632 |
| FP8 TFLOPS per chip | ~1,300 | ~1,300 |
| FP32 TFLOPS per chip | ~160 | ~160 |
| Chip interconnect | NeuronLink v3, 2D torus (1 TB/s per chip) | NeuronLink v3, 2D torus + inter-node ring (1 TB/s per chip) |
| EFA | 16 adapters (EFAv3) | 64 adapters (EFAv3) |
| Network bandwidth | 3,200 Gbps | 12,800 Gbps |
| vCPUs | 192 | 768 (4 x 192) |
| System memory | 2 TiB | 8 TiB |
| Local NVMe | ~7.6 TB | ~32 TB |

## Key Characteristics

- **Trainium v2** — 3.3x the BF16 compute and 4.7x the memory bandwidth of
  Trainium v1; supports cFP8 (configurable FP8)
- **NeuronCores v3** — 8 cores per chip (vs 2 on v1); fundamentally redesigned
  for higher throughput and larger models
- **1 TB/s NeuronLink per chip** — significantly higher than v1's 768 GB/s
  aggregate; enables efficient model parallelism
- **trn2u UltraServer** — 4 trn2 nodes connected via NeuronLink v3 into a
  single 64-chip domain; scheduled as one unit, not 4 separate instances

## UltraServer Topology (trn2u)

The trn2u.48xlarge is a **logical grouping of 4 interconnected trn2 nodes**:

- **Within each node**: 16 chips connected in a 2D torus via NeuronLink v3
- **Across nodes**: Chips at corresponding positions connected in a ring
  topology, creating a unified 64-chip fabric
- **Benefit**: Faster collective communication for model parallelism compared
  to 4 separate trn2.48xlarge instances connected via EFA
- **6 TB total HBM** — enables hosting and training models with 400B+
  parameters within the NeuronLink domain

## Distributed Training Considerations

- **Software stack**: Same as trn1 — `torch-neuronx`, NxD Training,
  `optimum-neuron`; requires Neuron SDK 2.21+
- **Max model size**: 100B+ with NxD Training across a single trn2u
  UltraServer; 400B+ across multiple UltraServers
- **Trainium v2 vs v1**: 3.3x BF16 compute, 3x HBM per chip, 4.7x memory
  bandwidth — a generational leap, not an incremental upgrade
- **UltraServer scheduling**: trn2u is scheduled as a unit (all 4 nodes);
  cannot use individual nodes independently

## Configuration

```bash
# Neuron runtime configuration
export NEURON_RT_NUM_CORES=128  # trn2.48xlarge (32 for trn2u per node)
export NEURON_CC_FLAGS="--model-type=transformer"

# EFA for multi-node / multi-UltraServer
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
```

## Comparison: Trainium v1 vs v2

| Aspect | trn1n.32xlarge | trn2.48xlarge | trn2u.48xlarge |
|--------|----------------|---------------|----------------|
| NeuronCores/chip | 2 (v2) | 8 (v3) | 8 (v3) |
| HBM/chip | 32 GB | 96 GiB (3x) | 96 GiB (3x) |
| Memory BW/chip | 613 GB/s | 2,900 GB/s (4.7x) | 2,900 GB/s (4.7x) |
| BF16/chip | ~190 TFLOPS | ~632 TFLOPS (3.3x) | ~632 TFLOPS (3.3x) |
| Chips | 16 | 16 | 64 |
| EFA bandwidth | 1,600 Gbps | 3,200 Gbps | 12,800 Gbps |
| EFA generation | EFAv2 | EFAv3 | EFAv3 |
| NeuronLink | v2 (768 GB/s) | v3 (1 TB/s/chip) | v3 (1 TB/s/chip, 64-chip domain) |

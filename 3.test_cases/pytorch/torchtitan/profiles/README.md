# torchtitan Instance Profiles

Instance profiles configure GPU count and EFA networking variables for each
supported EC2 instance type. Training-specific parameters (model config, batch
size, etc.) are handled by torchtitan's TOML configuration files.

## Auto-detection

The launch script auto-detects the running instance type via the EC2 instance
metadata service and sources the matching `.env` profile. Detection order
follows the same logic used by the FSDP and veRL profiles.

To override auto-detection:

```bash
export INSTANCE_PROFILE=g5-12xlarge
```

See [docs/instance-compatibility.md](../../../../docs/instance-compatibility.md)
for full details on the detection mechanism and supported instances.

## Available Profiles

| Profile | Instance | GPUs | VRAM | EFA | Status |
|---------|----------|------|------|-----|--------|
| `p5en-48xlarge.env` | p5en.48xlarge | 8x H200 | 141 GB | 32 adapters | Supported |
| `p4de-24xlarge.env` | p4de.24xlarge | 8x A100 | 80 GB | 4 adapters | Supported |
| `g6e-12xlarge.env` | g6e.12xlarge | 4x L40S | 48 GB | None | Supported |
| `g5-12xlarge.env` | g5.12xlarge | 4x A10G | 24 GB | None | Supported (small configs) |

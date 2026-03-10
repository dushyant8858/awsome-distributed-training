#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=4 # number of nodes to use  # can also run with 1 node
#SBATCH --job-name=torchtitan #name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex;

###########################
###### Instance Profile ###
###########################
# Auto-detect instance type and source the matching profile.
# Profiles set: GPUS_PER_NODE, EFA vars, NCCL settings.
# Override with: export INSTANCE_PROFILE=g5-12xlarge (before sbatch)
# See profiles/README.md for details.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="${SCRIPT_DIR}/../profiles"
PROFILE_LOADED=0

if [[ -d "$PROFILES_DIR" ]]; then
    if PROFILE_ENV=$("${PROFILES_DIR}/_detect.sh" "${PROFILES_DIR}"); then
        echo "Sourcing instance profile: $PROFILE_ENV"
        source "$PROFILE_ENV"
        PROFILE_LOADED=1
    else
        echo "WARNING: Profile detection failed. Using defaults (8 GPU, EFA enabled)."
    fi
else
    echo "WARNING: No profiles/ directory found. Using defaults (8 GPU, EFA enabled)."
fi

# Fallback defaults when no profile is loaded (assumes P5-class instance)
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

###########################
## Environment Variables ##
###########################

# EFA networking — configured by profile or defaults.
if [[ "$PROFILE_LOADED" == "1" ]]; then
    # Profile was sourced — trust its EFA settings.
    true
else
    # No profile — use legacy EFA defaults (P4/P5)
    export FI_PROVIDER=efa
    export FI_EFA_USE_HUGE_PAGE=0
    export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
fi

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"^docker,lo,veth"}

# LD_PRELOAD is required for PyTorch to find the NCCL library
# This path assumes you are using the Deep Learning AMI
# If you are not using the DLAMI, you may need to update this path
export LD_PRELOAD=/usr/local/cuda-12.1/lib/libnccl.so

## Set HuggingFace metadata timeout (in seconds) for large clusters
export HF_HUB_ETAG_TIMEOUT=60

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

export TORCHRUN=./pt_torchtitan/bin/torchrun
export TRAIN_SCRIPT=./torchtitan/torchtitan/train.py

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/torchtitan/models/llama/train_configs/llama3_8b.toml"}

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun  ${AUTO_RESUME} -l ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT  --job.config_file ${CONFIG_FILE}

#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=4 # number of nodes to use
#SBATCH --job-name=llama3_2_3b-FSDP # name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --error=logs/%x_%j.err # logfile for stderr, remove it to merge both outputs
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex;

###########################
###### Instance Profile ###
###########################
# Auto-detect instance type and source the matching profile.
# Profiles set: GPUS_PER_NODE, FI_PROVIDER, EFA_PER_NODE, FSDP_CPU_OFFLOAD, etc.
# Override with: export INSTANCE_PROFILE=g5-12xlarge (before sbatch)
# See profiles/README.md for details.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="${SCRIPT_DIR}/../profiles"
PROFILE_LOADED=0

if [[ -d "$PROFILES_DIR" ]]; then
    # _detect.sh prints diagnostics to stderr and the profile path to stdout.
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

###############################
###### Container Variable #####
###############################
# Uncomment if you want to use a container instead of Virtual Environment.
#export CONTAINER_IMAGE=$(pwd)/pytorch-fsdp.sqsh
export DATA_PATH=/fsx
export FSX_MOUNT=$(pwd):$DATA_PATH

###########################
## Environment Variables ##
###########################

# EFA networking — configured by profile or defaults.
# Non-EFA instances (g5, g6e) have EFA_PER_NODE=0 in their profile,
# which skips these variables. EFA instances set FI_PROVIDER in the profile.
if [[ "$PROFILE_LOADED" == "1" ]]; then
    # Profile was sourced — trust its EFA settings.
    # If profile set FI_PROVIDER, EFA is enabled. Otherwise, it's a non-EFA instance.
    true
else
    # No profile — use legacy EFA defaults (P4/P5)
    export FI_PROVIDER=efa
    export FI_EFA_USE_HUGE_PAGE=0
    export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
fi

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"^docker,lo,veth,eth"}

# LD_PRELOAD is required for PyTorch to find the NCCL library
# This path assumes you are using the Deep Learning AMI
# If you are not using the DLAMI, you may need to update this path
export LD_PRELOAD=/usr/local/cuda-12.8/lib/libnccl.so

## Set HuggingFace metadata timeout (in seconds) for large clusters
export HF_HUB_ETAG_TIMEOUT=60


########################
####### Container ######
########################

if [ ! -z $CONTAINER_IMAGE ];
then
    export TRAIN_SCRIPT=./train.py
    
    declare -a ARGS=(
        --container-image $CONTAINER_IMAGE
        --container-mounts $FSX_MOUNT
    )
fi

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

export TORCHRUN=torchrun
export TRAIN_SCRIPT=${TRAIN_SCRIPT:=../src/train.py}

############################
# llama3_2_3b Training Params ##
############################

declare -a TRAINING_ARGS=(
    --max_context_width=8192
    --num_key_value_heads=2
    --intermediate_size=8192
    --hidden_width=3072
    --num_layers=28
    --num_heads=24
    --model_type=llama_v3
    --tokenizer=hf-internal-testing/llama-tokenizer
    --checkpoint_freq=50
    --validation_freq=100
    --max_steps=100
    --checkpoint_dir=./checkpoints
    --dataset=allenai/c4
    --dataset_config_name=en
    --resume_from_checkpoint=./checkpoints
    --train_batch_size=1
    --val_batch_size=1
    --sharding_strategy=full # https://pytorch.org/docs/stable/fsdp.html
    --offload_activations=1
)

# Conditionally append cpu_offload from instance profile.
# The profile sets FSDP_CPU_OFFLOAD=1 for memory-constrained instances (e.g., g5)
# and FSDP_CPU_OFFLOAD_MIN_LAYERS as a threshold so small models (1B, 3B) that
# fit without offloading aren't penalized. Offloading adds ~30% overhead.
if [[ "${FSDP_CPU_OFFLOAD:-0}" == "1" ]]; then
    NUM_LAYERS=""
    for arg in "${TRAINING_ARGS[@]}"; do
        if [[ "$arg" == --num_layers=* ]]; then
            NUM_LAYERS="${arg#*=}"
            break
        fi
    done
    MIN_LAYERS="${FSDP_CPU_OFFLOAD_MIN_LAYERS:-0}"
    if [[ -n "$NUM_LAYERS" && "$NUM_LAYERS" -ge "$MIN_LAYERS" ]]; then
        echo "Enabling --cpu_offload=1 (num_layers=$NUM_LAYERS >= threshold=$MIN_LAYERS)"
        TRAINING_ARGS+=(--cpu_offload=1)
    else
        echo "Skipping cpu_offload (num_layers=${NUM_LAYERS:-unknown} < threshold=$MIN_LAYERS)"
    fi
fi

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi
srun ${AUTO_RESUME} -l "${ARGS[@]}" ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"
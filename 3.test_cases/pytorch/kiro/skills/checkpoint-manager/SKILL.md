# Checkpoint Manager Skill

## Overview

The checkpoint-manager skill provides utilities for managing training checkpoints in Kubernetes environments. It handles PVC creation, checkpoint discovery, validation, and cleanup operations.

## Usage

```python
from checkpoint_manager import (
    create_pvc,
    find_latest_checkpoint,
    list_checkpoints,
    validate_checkpoint,
    cleanup_old_checkpoints
)

# Create a PVC for checkpoint storage
pvc_name = create_pvc(
    name="training-checkpoints",
    storage_size="100Gi",
    storage_class="gp3",
    namespace="default"
)

# Find the latest checkpoint
checkpoint_dir = "/checkpoints"
latest = find_latest_checkpoint(checkpoint_dir)

# List all checkpoints
checkpoints = list_checkpoints(checkpoint_dir)

# Validate a checkpoint
is_valid = validate_checkpoint("/checkpoints/step_1000")

# Cleanup old checkpoints, keeping only the last 3
cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3)
```

## Functions

### create_pvc(name, storage_size, storage_class, namespace)
Creates a Kubernetes PersistentVolumeClaim for checkpoint storage.

**Parameters:**
- `name` (str): Name of the PVC
- `storage_size` (str): Storage size (e.g., "100Gi")
- `storage_class` (str): Storage class name
- `namespace` (str): Kubernetes namespace

**Returns:** str - Name of the created PVC

### find_latest_checkpoint(checkpoint_dir)
Finds the most recent checkpoint in a directory.

**Parameters:**
- `checkpoint_dir` (str): Directory containing checkpoints

**Returns:** str or None - Path to latest checkpoint

### list_checkpoints(checkpoint_dir)
Lists all checkpoints in a directory.

**Parameters:**
- `checkpoint_dir` (str): Directory containing checkpoints

**Returns:** list - Sorted list of checkpoint paths

### validate_checkpoint(checkpoint_path)
Validates that a checkpoint directory contains required files.

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint directory

**Returns:** bool - True if valid

### cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
Removes old checkpoints, keeping only the most recent N.

**Parameters:**
- `checkpoint_dir` (str): Directory containing checkpoints
- `keep_last_n` (int): Number of checkpoints to keep

**Returns:** int - Number of checkpoints removed

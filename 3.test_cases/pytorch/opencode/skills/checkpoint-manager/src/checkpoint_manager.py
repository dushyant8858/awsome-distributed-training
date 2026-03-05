"""Checkpoint Manager - Main implementation for checkpoint operations."""

import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import logging
logger = logging.getLogger("checkpoint_manager")
if not logger.handlers:
    import sys as _sys
    _h = logging.StreamHandler(_sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def create_pvc(
    name: str,
    storage_size: str,
    storage_class: str,
    namespace: str = "default"
) -> str:
    """Create a Kubernetes PersistentVolumeClaim for checkpoint storage.
    
    Args:
        name: Name of the PVC
        storage_size: Storage size (e.g., "100Gi")
        storage_class: Storage class name
        namespace: Kubernetes namespace
        
    Returns:
        Name of the created PVC
        
    Raises:
        RuntimeError: If PVC creation fails
    """
    pvc_manifest = f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {name}
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: {storage_class}
  resources:
    requests:
      storage: {storage_size}
"""
    
    try:
        # Apply the PVC manifest using kubectl
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=pvc_manifest,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Created PVC '{name}' in namespace '{namespace}'")
        logger.debug(f"kubectl output: {result.stdout}")
        
        # Wait for PVC to be bound
        wait_result = subprocess.run(
            ["kubectl", "wait", "--for=jsonpath={.status.phase}=Bound",
             f"pvc/{name}", "-n", namespace, "--timeout=60s"],
            capture_output=True,
            text=True
        )
        
        if wait_result.returncode == 0:
            logger.info(f"PVC '{name}' is now Bound")
        else:
            logger.warning(f"PVC '{name}' may not be bound yet: {wait_result.stderr}")
        
        return name
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create PVC: {e.stderr}")
        raise RuntimeError(f"PVC creation failed: {e.stderr}")
    except FileNotFoundError:
        logger.error("kubectl not found in PATH")
        raise RuntimeError("kubectl is required but not found in PATH")


def _extract_step_number(checkpoint_name: str) -> Optional[int]:
    """Extract step number from checkpoint directory name.
    
    Supports formats like:
    - step_1000
    - checkpoint-1000
    - model_step_1000
    - 1000 (just a number)
    
    Args:
        checkpoint_name: Name of the checkpoint directory
        
    Returns:
        Step number or None if not found
    """
    # Try various patterns
    patterns = [
        r'step[_-]?(\d+)',
        r'checkpoint[_-]?(\d+)',
        r'model[_-]?step[_-]?(\d+)',
        r'^(\d+)$',
        r'ckpt[_-]?(\d+)',
        r'iter[_-]?(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, checkpoint_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


def _get_checkpoint_mtime(checkpoint_path: Path) -> datetime:
    """Get the modification time of a checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Modification time
    """
    try:
        return datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
    except (OSError, IOError):
        return datetime.min


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    """List all checkpoints in a directory.
    
    Checkpoints are sorted by step number if available, otherwise by modification time.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Sorted list of checkpoint paths (newest first)
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return []
    
    if not checkpoint_path.is_dir():
        logger.warning(f"Checkpoint path is not a directory: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for item in checkpoint_path.iterdir():
        if item.is_dir():
            checkpoints.append(item)
    
    if not checkpoints:
        logger.info(f"No checkpoints found in {checkpoint_dir}")
        return []
    
    # Sort by step number if available, otherwise by modification time
    def sort_key(cp: Path):
        step = _extract_step_number(cp.name)
        if step is not None:
            return (0, -step)  # Negative for descending order
        else:
            return (1, -_get_checkpoint_mtime(cp).timestamp())
    
    checkpoints.sort(key=sort_key)
    
    result = [str(cp) for cp in checkpoints]
    logger.debug(f"Found {len(result)} checkpoints in {checkpoint_dir}")
    
    return result


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        return None
    
    latest = checkpoints[0]
    logger.info(f"Latest checkpoint: {latest}")
    
    return latest


def find_latest_checkpoint_on_pod(
    head_pod: str,
    checkpoint_dir: str,
    namespace: str = "default",
) -> Optional[Tuple[str, int]]:
    """Find the latest checkpoint inside a Kubernetes pod.

    Runs ``kubectl exec`` to list the checkpoint directory on the pod,
    filters for ``global_step_*`` entries, and returns the one with the
    highest step number.

    Args:
        head_pod: Name of the pod to check.
        checkpoint_dir: Path to checkpoint directory inside the pod.
        namespace: Kubernetes namespace.

    Returns:
        Tuple of ``(checkpoint_path, step_number)`` or ``None`` if no
        checkpoints are found.
    """
    checkpoints = list_checkpoints_on_pod(head_pod, checkpoint_dir, namespace)
    if not checkpoints:
        return None
    # checkpoints is sorted descending; first element is latest
    return checkpoints[0]


def list_checkpoints_on_pod(
    head_pod: str,
    checkpoint_dir: str,
    namespace: str = "default",
) -> List[Tuple[str, int]]:
    """List all ``global_step_*`` checkpoints inside a Kubernetes pod.

    The results are sorted by step number in **descending** order (newest
    first).

    Args:
        head_pod: Name of the pod to check.
        checkpoint_dir: Path to checkpoint directory inside the pod.
        namespace: Kubernetes namespace.

    Returns:
        List of ``(checkpoint_path, step_number)`` tuples sorted descending
        by step number.  Returns an empty list when no checkpoints are found
        or the command fails.
    """
    try:
        result = subprocess.run(
            [
                "kubectl", "exec", head_pod,
                "-n", namespace,
                "--", "ls", "-1", checkpoint_dir,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Failed to list checkpoints on pod {head_pod}: {e.stderr}"
        )
        return []
    except FileNotFoundError:
        logger.error("kubectl not found in PATH")
        return []

    entries = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    pattern = re.compile(r"^global_step_(\d+)$")
    checkpoints: List[Tuple[str, int]] = []
    for entry in entries:
        match = pattern.match(entry)
        if match:
            step = int(match.group(1))
            path = f"{checkpoint_dir}/global_step_{step}"
            checkpoints.append((path, step))

    # Sort descending by step number
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        f"Found {len(checkpoints)} checkpoints on pod {head_pod} "
        f"in {checkpoint_dir}"
    )
    return checkpoints


def validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate that a checkpoint directory contains required files.
    
    Checks for common checkpoint files:
    - PyTorch: *.pt, *.pth, *.bin, model.safetensors
    - TensorFlow: checkpoint, *.ckpt, *.h5
    - Common: config.json, optimizer_state.pt
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        True if checkpoint appears valid
    """
    cp_path = Path(checkpoint_path)
    
    if not cp_path.exists():
        logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
        return False
    
    if not cp_path.is_dir():
        logger.warning(f"Checkpoint path is not a directory: {checkpoint_path}")
        return False
    
    # Check for model files
    model_extensions = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5"}
    has_model_file = False
    
    for ext in model_extensions:
        if list(cp_path.glob(f"*{ext}")):
            has_model_file = True
            break
    
    # Check for TensorFlow checkpoint file
    if (cp_path / "checkpoint").exists():
        has_model_file = True
    
    if not has_model_file:
        logger.warning(f"No model files found in checkpoint: {checkpoint_path}")
        return False
    
    # Check for common metadata files (optional but recommended)
    metadata_files = ["config.json", "training_args.bin", "scheduler.pt"]
    found_metadata = [f for f in metadata_files if (cp_path / f).exists()]
    
    logger.info(f"Checkpoint validated: {checkpoint_path}")
    logger.debug(f"Found {len(found_metadata)} metadata files: {found_metadata}")
    
    return True


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> int:
    """Remove old checkpoints, keeping only the most recent N.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
        
    Returns:
        Number of checkpoints removed
        
    Raises:
        ValueError: If keep_last_n is less than 1
    """
    if keep_last_n < 1:
        raise ValueError("keep_last_n must be at least 1")
    
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if len(checkpoints) <= keep_last_n:
        logger.info(f"No cleanup needed. Found {len(checkpoints)} checkpoints, keeping {keep_last_n}")
        return 0
    
    to_remove = checkpoints[keep_last_n:]
    removed_count = 0
    
    for cp_path in to_remove:
        try:
            shutil.rmtree(cp_path)
            logger.info(f"Removed old checkpoint: {cp_path}")
            removed_count += 1
        except (OSError, IOError) as e:
            logger.error(f"Failed to remove checkpoint {cp_path}: {e}")
    
    logger.info(f"Cleanup complete. Removed {removed_count} checkpoints, kept {keep_last_n}")
    
    return removed_count


def get_checkpoint_info(checkpoint_path: str) -> dict:
    """Get detailed information about a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Dictionary with checkpoint information
    """
    cp_path = Path(checkpoint_path)
    
    info = {
        "path": str(cp_path),
        "exists": cp_path.exists(),
        "name": cp_path.name,
        "step_number": _extract_step_number(cp_path.name),
        "is_valid": False,
        "size_bytes": 0,
        "file_count": 0,
        "files": [],
    }
    
    if not cp_path.exists():
        return info
    
    info["is_valid"] = validate_checkpoint(checkpoint_path)
    
    if cp_path.is_dir():
        for item in cp_path.iterdir():
            if item.is_file():
                try:
                    size = item.stat().st_size
                    info["size_bytes"] += size
                    info["file_count"] += 1
                    info["files"].append({
                        "name": item.name,
                        "size_bytes": size,
                    })
                except (OSError, IOError):
                    pass
    
    return info

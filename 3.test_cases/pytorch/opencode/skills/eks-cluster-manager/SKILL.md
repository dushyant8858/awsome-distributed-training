# EKS Cluster Manager Skill

Create, validate, and manage EKS clusters and supporting infrastructure for HyperPod distributed training workloads.

## Overview

This skill provisions the complete EKS infrastructure stack using direct boto3 API calls (modeled after the CFN nested stacks in `1.architectures/7.sagemaker-hyperpod-eks/`). It provides:

- **VPC creation** with public/private subnets, NAT gateways, secondary CIDR for HyperPod
- **Security groups** with safe defaults (intra-SG only, no public ingress)
- **VPC endpoints** to keep traffic off the public internet
- **IAM roles** for EKS control plane and SageMaker execution
- **S3 bucket** for lifecycle scripts (encrypted, no public access)
- **EKS cluster** with full logging, core add-ons, and private endpoint
- **Cluster validation** (status, nodes, GPUs, EFA, add-ons)

## Security Guardrails

These are enforced by default and cannot be accidentally bypassed:

- Private EKS endpoint enabled
- No `0.0.0.0/0` ingress in security groups
- VPC endpoints for S3, ECR, STS, CloudWatch, AMP (no public internet traffic)
- S3 buckets encrypted with AES256 and public access blocked
- No public observability endpoints

## Functions

### Full-Stack Provisioning

### `provision_eks_infrastructure(name_prefix, eks_cluster_name=None, vpc_cidr="10.192.0.0/16", private_subnet_cidr="10.1.0.0/16", kubernetes_version="1.32", availability_zone_id=None, endpoint_private_access=True, endpoint_public_access=True, create_endpoints=True, region=None)`

Single-call orchestrator that creates the complete infrastructure in 6 steps:

1. VPC (subnets, NAT gateways, route tables)
2. Security group
3. S3 bucket
4. IAM roles (EKS + SageMaker execution)
5. VPC endpoints
6. EKS cluster with core add-ons

```python
from manage_cluster import provision_eks_infrastructure

result = provision_eks_infrastructure(
    name_prefix="my-training",
    kubernetes_version="1.32",
    availability_zone_id="usw2-az2",  # pin HyperPod subnet to a specific AZ
)
# result contains: vpc, security_group_id, s3_bucket, eks_cluster_role_arn,
# sagemaker_execution_role_arn, vpc_endpoints, eks_cluster
```

### VPC

### `create_vpc(name_prefix="sagemaker-hyperpod", vpc_cidr="10.192.0.0/16", public_subnet1_cidr="10.192.10.0/24", public_subnet2_cidr="10.192.11.0/24", private_subnet_cidr="10.1.0.0/16", eks_subnet1_cidr="10.192.7.0/28", eks_subnet2_cidr="10.192.8.0/28", availability_zone_id=None, region=None)`

Creates a complete VPC:
- VPC with DNS support + secondary CIDR for HyperPod
- Internet gateway
- 2 public subnets (for NAT gateways, in different AZs)
- 2 NAT gateways (HA)
- 1 private subnet (HyperPod instances, from secondary CIDR)
- 2 EKS private subnets (small /28 for cross-account ENIs)
- Route tables with proper routing

Returns dict with `vpc_id`, `public_subnet_ids`, `private_subnet_id`, `eks_subnet_ids`, `nat_gateway_ids`, `route_table_ids`.

### Security

### `create_security_group(vpc_id, name_prefix="sagemaker-hyperpod", region=None)`

Creates a security group with:
- Intra-SG all traffic (required for NCCL/EFA inter-node comms)
- FSx Lustre ports TCP 988, 1018-1023 (intra-SG)
- Internet egress (for image pulls, model downloads)
- **NO `0.0.0.0/0` ingress**

Returns security group ID.

### `create_vpc_endpoints(vpc_id, subnet_ids, security_group_id, route_table_id, services=None, name_prefix="sagemaker-hyperpod", region=None)`

Creates VPC endpoints so training traffic stays off the public internet:
- S3 (Gateway endpoint)
- ECR API + DKR (Interface, for image pulls)
- STS (Interface, for IAM/Pod Identity)
- CloudWatch Logs (Interface)
- AMP workspaces (Interface, for observability)

Skips endpoints that already exist. Returns dict mapping service name to endpoint ID.

### IAM Roles

### `create_eks_cluster_role(name_prefix="sagemaker-hyperpod", region=None)`

Creates IAM role for the EKS control plane with `AmazonEKSClusterPolicy`. Returns role ARN. Idempotent (returns existing role if already created).

### `create_sagemaker_execution_role(s3_bucket_name, name_prefix="sagemaker-hyperpod", region=None)`

Creates IAM execution role for SageMaker HyperPod instances with:
- `AmazonSageMakerClusterInstanceRolePolicy` (managed)
- `AmazonEKS_CNI_Policy` (managed)
- Inline policy for: EC2 networking, ECR pull, EKS Pod Identity, S3 lifecycle script access, CloudWatch

Returns role ARN. Idempotent.

### S3

### `create_s3_bucket(name_prefix="sagemaker-hyperpod", region=None)`

Creates an encrypted S3 bucket for lifecycle scripts. Bucket name: `{prefix}-{account_id}-{region}`. Encrypted with AES256, all public access blocked. Idempotent. Returns bucket name.

### EKS Cluster

### `create_eks_cluster(cluster_name, eks_subnet_ids, security_group_id, cluster_role_arn, kubernetes_version="1.32", endpoint_private_access=True, endpoint_public_access=True, name_prefix="sagemaker-hyperpod", region=None)`

Creates an EKS cluster with:
- Full control plane logging (api, audit, authenticator, controllerManager, scheduler)
- `API_AND_CONFIG_MAP` auth mode
- Core add-ons: `vpc-cni`, `kube-proxy`, `coredns`, `eks-pod-identity-agent`
- Auto-grants cluster admin to the calling IAM identity
- Updates local kubeconfig

Waits for ACTIVE status (10-15 minutes). Returns dict with `name`, `arn`, `endpoint`, `status`.

### `create_access_entry(cluster_name, principal_arn, policy_arn="...AmazonEKSClusterAdminPolicy", region=None)`

Add an IAM principal (user or role) as an EKS cluster admin. Idempotent.

### Validation

### `validate_cluster(cluster_name, region=None)`

Validates an EKS cluster is ready for training:
- Cluster is ACTIVE
- Core add-ons installed (vpc-cni, kube-proxy, coredns, eks-pod-identity-agent)
- Nodes are Ready (via kubectl)
- GPU resources available (`nvidia.com/gpu`)
- EFA resources available (`vpc.amazonaws.com/efa`)

```python
from manage_cluster import validate_cluster

report = validate_cluster("my-cluster")
# report = {
#   "cluster_name": "my-cluster",
#   "overall": "PASS",  # or "WARN" or "FAIL"
#   "checks": {
#     "cluster_status": {"status": "PASS", "detail": "..."},
#     "core_addons": {"status": "PASS", "detail": "..."},
#     "nodes": {"status": "PASS", "detail": "4/4 nodes Ready, 16 GPUs, 4 EFA devices"},
#   }
# }
```

## Usage

```python
from manage_cluster import (
    provision_eks_infrastructure,
    create_vpc, create_security_group, create_vpc_endpoints,
    create_eks_cluster_role, create_sagemaker_execution_role,
    create_s3_bucket, create_eks_cluster, create_access_entry,
    validate_cluster,
)

# Option 1: Full stack in one call (recommended)
result = provision_eks_infrastructure(name_prefix="my-training")

# Option 2: Step by step
vpc = create_vpc(name_prefix="my-training")
sg_id = create_security_group(vpc["vpc_id"], "my-training")
bucket = create_s3_bucket("my-training")
eks_role = create_eks_cluster_role("my-training")
sm_role = create_sagemaker_execution_role(bucket, "my-training")
endpoints = create_vpc_endpoints(vpc["vpc_id"], vpc["eks_subnet_ids"], sg_id, vpc["private_route_table_id"])
cluster = create_eks_cluster("my-eks", vpc["eks_subnet_ids"], sg_id, eks_role)

# Add another admin
create_access_entry("my-eks", "arn:aws:iam::123456789012:role/TeamRole")

# Validate
report = validate_cluster("my-eks")
```

## Resource Naming

All resources are tagged with `ManagedBy: opencode-eks-cluster-manager` and named with the `{name_prefix}-*` pattern:

| Resource | Name Pattern |
|----------|-------------|
| VPC | `{prefix}-VPC` |
| Public subnets | `{prefix}-Public1`, `{prefix}-Public2` |
| Private subnet | `{prefix}-Private-HyperPod` |
| EKS subnets | `{prefix}-EKS-Private1`, `{prefix}-EKS-Private2` |
| Security group | `{prefix}-training-sg` |
| EKS cluster role | `{prefix}-eks-cluster-role` |
| Execution role | `{prefix}-exec-role` |
| S3 bucket | `{prefix}-{account_id}-{region}` |

## Requirements

- boto3
- kubectl (for validation and kubeconfig update)
- AWS credentials with permissions for: EC2, EKS, IAM, S3, STS

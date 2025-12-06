# 🏗️ NEURECTOMY - Terraform Infrastructure Deployment

## Overview

This directory contains Infrastructure as Code (IaC) for deploying NEURECTOMY on AWS EKS with multi-cloud readiness.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS Account                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────── VPC (10.0.0.0/16) ─────────────────────┐   │
│  │                                                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│  │  │  Public     │  │  Public     │  │  Public     │          │   │
│  │  │  Subnet     │  │  Subnet     │  │  Subnet     │          │   │
│  │  │  (AZ-a)     │  │  (AZ-b)     │  │  (AZ-c)     │          │   │
│  │  │  10.0.48.0  │  │  10.0.64.0  │  │  10.0.80.0  │          │   │
│  │  │     ↓       │  │     ↓       │  │     ↓       │          │   │
│  │  │   NAT GW    │  │   NAT GW*   │  │   NAT GW*   │          │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │   │
│  │         │                │                │                   │   │
│  │  ┌──────▼────────────────▼────────────────▼──────────────┐  │   │
│  │  │                   EKS Cluster                          │  │   │
│  │  │  ┌──────────────────────────────────────────────────┐ │  │   │
│  │  │  │              Private Subnets                      │ │  │   │
│  │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐           │ │  │   │
│  │  │  │  │ Node    │  │ Node    │  │ Node    │           │ │  │   │
│  │  │  │  │ Group   │  │ Group   │  │ Group   │           │ │  │   │
│  │  │  │  │ General │  │ ML-Inf  │  │ GPU*    │           │ │  │   │
│  │  │  │  └─────────┘  └─────────┘  └─────────┘           │ │  │   │
│  │  │  └──────────────────────────────────────────────────┘ │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                                                               │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                │   │
│  │  │ EFS       │  │ S3        │  │ Secrets   │                │   │
│  │  │ Storage   │  │ Buckets   │  │ Manager   │                │   │
│  │  └───────────┘  └───────────┘  └───────────┘                │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
* = Optional/Prod only
```

## Prerequisites

### Required Tools

```bash
# AWS CLI v2
aws --version  # >= 2.0

# Terraform
terraform --version  # >= 1.6.0

# kubectl
kubectl version --client  # >= 1.28

# Helm
helm version  # >= 3.12
```

### AWS Configuration

```bash
# Configure AWS credentials
aws configure

# Or use SSO
aws sso login --profile neurectomy

# Verify access
aws sts get-caller-identity
```

## Quick Start

### 1. Bootstrap Backend (First Time Only)

```bash
# Create S3 bucket for state
aws s3api create-bucket \
    --bucket neurectomy-terraform-state \
    --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket neurectomy-terraform-state \
    --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
    --bucket neurectomy-terraform-state \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms"}}]}'

# Block public access
aws s3api put-public-access-block \
    --bucket neurectomy-terraform-state \
    --public-access-block-configuration \
    'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'

# Create DynamoDB table for locking
aws dynamodb create-table \
    --table-name neurectomy-terraform-locks \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1
```

### 2. Initialize Terraform

```bash
cd terraform

# Initialize with backend
terraform init

# For local state (development only)
# Comment out backend block in backend.tf first
terraform init
```

### 3. Plan Deployment

```bash
# Development environment
terraform plan -var-file="environments/dev.tfvars" -out=plan.tfplan

# Production environment
terraform plan -var-file="environments/prod.tfvars" -out=plan.tfplan
```

### 4. Apply Infrastructure

```bash
# Review the plan carefully, then apply
terraform apply plan.tfplan

# Or apply directly
terraform apply -var-file="environments/dev.tfvars"
```

### 5. Configure kubectl

```bash
# Get kubeconfig (output from terraform)
aws eks update-kubeconfig \
    --region us-east-1 \
    --name neurectomy-cluster-dev

# Verify connection
kubectl get nodes
kubectl get namespaces
```

## File Structure

```
terraform/
├── main.tf                    # Core infrastructure (VPC, EKS)
├── variables.tf               # Variable definitions
├── outputs.tf                 # Output values
├── backend.tf                 # Remote state configuration
├── external-secrets.tf        # External Secrets Operator IAM
├── environments/
│   ├── dev.tfvars            # Development settings
│   └── prod.tfvars           # Production settings
└── modules/                   # Custom modules (future)
```

## Environment-Specific Deployment

### Development

```bash
# Minimal resources, cost-optimized
terraform apply -var-file="environments/dev.tfvars"
```

Features:

- 2 AZs (cost savings)
- Single NAT Gateway
- Smaller instance types (t3.large)
- Spot instances enabled
- 7-day log retention
- No GPU nodes

### Production

```bash
# High availability, enterprise-grade
terraform apply -var-file="environments/prod.tfvars"
```

Features:

- 3 AZs (high availability)
- NAT Gateway per AZ
- Larger instance types (m6i.xlarge+)
- Multi-AZ RDS
- GuardDuty enabled
- 90-day log retention
- GPU nodes for ML training

## Post-Deployment Setup

### 1. Install External Secrets Operator

```bash
kubectl apply -k k8s/external-secrets/
```

### 2. Deploy ArgoCD

```bash
kubectl apply -k k8s/argocd/
```

### 3. Deploy Applications

```bash
# Development
kubectl apply -f k8s/argocd/applications.yaml

# Or use ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

## Cost Estimation

| Environment | Monthly Estimate |
| ----------- | ---------------- |
| Development | $150-300         |
| Staging     | $300-500         |
| Production  | $1,500-3,000     |

_Estimates based on default configurations. Actual costs vary._

## Troubleshooting

### Common Issues

**EKS cluster unreachable:**

```bash
# Check AWS credentials
aws sts get-caller-identity

# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name neurectomy-cluster-dev
```

**Terraform state locked:**

```bash
# Check for active operations
aws dynamodb scan --table-name neurectomy-terraform-locks

# Force unlock (use with caution!)
terraform force-unlock LOCK_ID
```

**Node group not scaling:**

```bash
# Check cluster autoscaler logs
kubectl logs -n kube-system -l app.kubernetes.io/name=cluster-autoscaler

# Check node group status
aws eks describe-nodegroup \
    --cluster-name neurectomy-cluster-dev \
    --nodegroup-name general-workloads
```

## Destroy Infrastructure

⚠️ **WARNING: This will destroy ALL resources!**

```bash
# Dry run
terraform plan -destroy -var-file="environments/dev.tfvars"

# Destroy
terraform destroy -var-file="environments/dev.tfvars"
```

## Security Best Practices

1. **State Security:** Always use remote state with encryption
2. **Credentials:** Never commit AWS credentials to git
3. **Access:** Use IAM roles with least privilege
4. **Networking:** Restrict cluster endpoint access in production
5. **Secrets:** Use External Secrets Operator, not plain Kubernetes secrets

## Contributing

1. Create a feature branch
2. Make infrastructure changes
3. Run `terraform fmt` and `terraform validate`
4. Create PR with plan output
5. Get approval before applying

## Links

- [Terraform AWS EKS Module](https://registry.terraform.io/modules/terraform-aws-modules/eks/aws)
- [AWS EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [NEURECTOMY Architecture Docs](../docs/architecture/)

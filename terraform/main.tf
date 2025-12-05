# ============================================================================
# NEURECTOMY Infrastructure - Terraform Configuration
# Multi-Cloud Ready Infrastructure as Code
# ============================================================================

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.24"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }

  # Remote state configuration - uncomment for production
  # backend "s3" {
  #   bucket         = "neurectomy-terraform-state"
  #   key            = "infrastructure/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "neurectomy-terraform-locks"
  # }
}

# ============================================================================
# Variables
# ============================================================================

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "neurectomy-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones for multi-AZ deployment"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "enable_gpu_nodes" {
  description = "Enable GPU node group for ML training"
  type        = bool
  default     = false
}

# ============================================================================
# Locals
# ============================================================================

locals {
  cluster_name = "${var.cluster_name}-${var.environment}"
  
  common_tags = {
    Project     = "neurectomy"
    Environment = var.environment
    ManagedBy   = "terraform"
    Repository  = "github.com/iamthegreatdestroyer/NEURECTOMY"
  }
  
  private_subnets = [for i, az in var.availability_zones : cidrsubnet(var.vpc_cidr, 4, i)]
  public_subnets  = [for i, az in var.availability_zones : cidrsubnet(var.vpc_cidr, 4, i + length(var.availability_zones))]
}

# ============================================================================
# Provider Configuration
# ============================================================================

provider "aws" {
  region = var.region
  
  default_tags {
    tags = local.common_tags
  }
}

# ============================================================================
# VPC Module
# ============================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = local.private_subnets
  public_subnets  = local.public_subnets

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment == "dev" ? true : false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # Kubernetes-specific tags for subnet auto-discovery
  public_subnet_tags = {
    "kubernetes.io/role/elb"                    = 1
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"           = 1
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
  }

  tags = local.common_tags
}

# ============================================================================
# EKS Cluster
# ============================================================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = local.cluster_name
  cluster_version = var.cluster_version

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Enable IRSA for pod-level IAM roles
  enable_irsa = true

  # Cluster add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa_role.iam_role_arn
    }
  }

  # Managed node groups
  eks_managed_node_groups = {
    # General workloads
    general = {
      name           = "general-workloads"
      instance_types = var.environment == "prod" ? ["m6i.xlarge", "m6i.2xlarge"] : ["t3.large"]
      
      min_size     = var.environment == "prod" ? 3 : 1
      max_size     = var.environment == "prod" ? 10 : 3
      desired_size = var.environment == "prod" ? 3 : 2

      labels = {
        workload-type = "general"
      }

      taints = []
    }

    # ML inference workloads
    ml-inference = {
      name           = "ml-inference"
      instance_types = var.environment == "prod" ? ["c6i.2xlarge", "c6i.4xlarge"] : ["c6i.xlarge"]
      
      min_size     = var.environment == "prod" ? 2 : 0
      max_size     = var.environment == "prod" ? 20 : 5
      desired_size = var.environment == "prod" ? 2 : 1

      labels = {
        workload-type = "ml-inference"
      }

      taints = [{
        key    = "ml-inference"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  # GPU node group (optional)
  eks_managed_node_groups_gpu = var.enable_gpu_nodes ? {
    gpu = {
      name           = "gpu-training"
      instance_types = ["g5.xlarge", "g5.2xlarge"]
      ami_type       = "AL2_x86_64_GPU"
      
      min_size     = 0
      max_size     = 4
      desired_size = 0

      labels = {
        workload-type = "gpu-training"
        "nvidia.com/gpu" = "true"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  } : {}

  tags = local.common_tags
}

# ============================================================================
# IRSA for EBS CSI Driver
# ============================================================================

module "ebs_csi_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${local.cluster_name}-ebs-csi"

  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = local.common_tags
}

# ============================================================================
# Outputs
# ============================================================================

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID for the cluster"
  value       = module.eks.cluster_security_group_id
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

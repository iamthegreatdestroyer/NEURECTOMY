# ============================================================================
# NEURECTOMY - External Secrets IAM Configuration
# AWS IAM role and policies for External Secrets Operator
# ============================================================================

# IAM Role for External Secrets (IRSA)
module "external_secrets_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${local.cluster_name}-external-secrets"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["external-secrets:external-secrets-sa"]
    }
  }

  role_policy_arns = {
    secrets_manager = aws_iam_policy.external_secrets_policy.arn
  }

  tags = local.common_tags
}

# IAM Policy for External Secrets
resource "aws_iam_policy" "external_secrets_policy" {
  name        = "${local.cluster_name}-external-secrets-policy"
  description = "Policy for External Secrets Operator to access AWS Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerRead"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
          "secretsmanager:ListSecretVersionIds"
        ]
        Resource = [
          "arn:aws:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:neurectomy/*"
        ]
      },
      {
        Sid    = "SecretsManagerList"
        Effect = "Allow"
        Action = [
          "secretsmanager:ListSecrets"
        ]
        Resource = "*"
      },
      {
        Sid    = "KMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = [
          aws_kms_key.secrets_key.arn
        ]
      }
    ]
  })

  tags = local.common_tags
}

# KMS Key for encrypting secrets
resource "aws_kms_key" "secrets_key" {
  description             = "KMS key for NEURECTOMY secrets encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow External Secrets"
        Effect = "Allow"
        Principal = {
          AWS = module.external_secrets_irsa_role.iam_role_arn
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_kms_alias" "secrets_key_alias" {
  name          = "alias/neurectomy-secrets"
  target_key_id = aws_kms_key.secrets_key.key_id
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}

# ============================================================================
# Initial Secrets Creation (for bootstrapping)
# ============================================================================

# ML Service secrets
resource "aws_secretsmanager_secret" "ml_service_database" {
  name        = "neurectomy/ml-service/database"
  description = "Database connection credentials for ML Service"
  kms_key_id  = aws_kms_key.secrets_key.arn

  tags = local.common_tags
}

resource "aws_secretsmanager_secret" "ml_service_redis" {
  name        = "neurectomy/ml-service/redis"
  description = "Redis connection credentials for ML Service"
  kms_key_id  = aws_kms_key.secrets_key.arn

  tags = local.common_tags
}

resource "aws_secretsmanager_secret" "ml_service_auth" {
  name        = "neurectomy/ml-service/auth"
  description = "Authentication secrets for ML Service"
  kms_key_id  = aws_kms_key.secrets_key.arn

  tags = local.common_tags
}

resource "aws_secretsmanager_secret" "ml_service_mlflow" {
  name        = "neurectomy/ml-service/mlflow"
  description = "MLflow tracking credentials"
  kms_key_id  = aws_kms_key.secrets_key.arn

  tags = local.common_tags
}

resource "aws_secretsmanager_secret" "database_postgres" {
  name        = "neurectomy/database/postgres"
  description = "PostgreSQL database credentials"
  kms_key_id  = aws_kms_key.secrets_key.arn

  tags = local.common_tags
}

resource "aws_secretsmanager_secret" "database_neo4j" {
  name        = "neurectomy/database/neo4j"
  description = "Neo4j graph database credentials"
  kms_key_id  = aws_kms_key.secrets_key.arn

  tags = local.common_tags
}

resource "aws_secretsmanager_secret" "api_keys" {
  name        = "neurectomy/api-keys/external"
  description = "External API keys (OpenAI, Anthropic, etc.)"
  kms_key_id  = aws_kms_key.secrets_key.arn

  tags = local.common_tags
}

# ============================================================================
# Outputs
# ============================================================================

output "external_secrets_role_arn" {
  description = "IAM role ARN for External Secrets Operator"
  value       = module.external_secrets_irsa_role.iam_role_arn
}

output "kms_key_arn" {
  description = "KMS key ARN for secrets encryption"
  value       = aws_kms_key.secrets_key.arn
}

# ============================================================================
# NEURECTOMY - Service Account IAM Roles (IRSA)
# Least-privilege IAM roles for each service account
# ============================================================================

# ML Service IAM Role
resource "aws_iam_role" "ml_service" {
  name = "${var.project_name}-ml-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:neurectomy:ml-service"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.project_name}-ml-service"
    Service = "ml-service"
  })
}

resource "aws_iam_policy" "ml_service" {
  name        = "${var.project_name}-ml-service-policy"
  description = "Policy for ML Service workload"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 access for model artifacts
      {
        Sid    = "S3ModelArtifacts"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-ml-artifacts",
          "arn:aws:s3:::${var.project_name}-ml-artifacts/*"
        ]
      },
      # Secrets Manager for application secrets
      {
        Sid    = "SecretsManager"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          "arn:aws:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/ml-service/*"
        ]
      },
      # SageMaker for model inference (optional)
      {
        Sid    = "SageMakerInference"
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint"
        ]
        Resource = [
          "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}:endpoint/${var.project_name}-*"
        ]
      },
      # CloudWatch for logging
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/neurectomy/ml-service:*"
        ]
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "ml_service" {
  role       = aws_iam_role.ml_service.name
  policy_arn = aws_iam_policy.ml_service.arn
}

# MLflow IAM Role
resource "aws_iam_role" "mlflow" {
  name = "${var.project_name}-mlflow"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:neurectomy:mlflow"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.project_name}-mlflow"
    Service = "mlflow"
  })
}

resource "aws_iam_policy" "mlflow" {
  name        = "${var.project_name}-mlflow-policy"
  description = "Policy for MLflow tracking server"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 access for artifact storage
      {
        Sid    = "S3ArtifactStorage"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-mlflow-artifacts",
          "arn:aws:s3:::${var.project_name}-mlflow-artifacts/*"
        ]
      },
      # Secrets Manager for database credentials
      {
        Sid    = "SecretsManager"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          "arn:aws:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/mlflow/*"
        ]
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "mlflow" {
  role       = aws_iam_role.mlflow.name
  policy_arn = aws_iam_policy.mlflow.arn
}

# PostgreSQL IAM Role (for RDS IAM authentication)
resource "aws_iam_role" "postgres" {
  name = "${var.project_name}-postgres"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:neurectomy:postgres"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.project_name}-postgres"
    Service = "postgres"
  })
}

resource "aws_iam_policy" "postgres" {
  name        = "${var.project_name}-postgres-policy"
  description = "Policy for PostgreSQL backup and snapshot operations"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 access for backups
      {
        Sid    = "S3Backups"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-postgres-backups",
          "arn:aws:s3:::${var.project_name}-postgres-backups/*"
        ]
      },
      # KMS for encryption
      {
        Sid    = "KMSEncryption"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = [
          aws_kms_key.database.arn
        ]
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "postgres" {
  role       = aws_iam_role.postgres.name
  policy_arn = aws_iam_policy.postgres.arn
}

# Prometheus IAM Role
resource "aws_iam_role" "prometheus" {
  name = "${var.project_name}-prometheus"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:monitoring:prometheus"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.project_name}-prometheus"
    Service = "prometheus"
  })
}

resource "aws_iam_policy" "prometheus" {
  name        = "${var.project_name}-prometheus-policy"
  description = "Policy for Prometheus metrics collection"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # CloudWatch for AWS metrics
      {
        Sid    = "CloudWatchMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      },
      # EC2 for service discovery
      {
        Sid    = "EC2Discovery"
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances"
        ]
        Resource = "*"
      },
      # AMP for remote write (if using Amazon Managed Prometheus)
      {
        Sid    = "AMPRemoteWrite"
        Effect = "Allow"
        Action = [
          "aps:RemoteWrite",
          "aps:GetSeries",
          "aps:GetLabels",
          "aps:GetMetricMetadata"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "prometheus" {
  role       = aws_iam_role.prometheus.name
  policy_arn = aws_iam_policy.prometheus.arn
}

# Grafana IAM Role
resource "aws_iam_role" "grafana" {
  name = "${var.project_name}-grafana"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:monitoring:grafana"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.project_name}-grafana"
    Service = "grafana"
  })
}

resource "aws_iam_policy" "grafana" {
  name        = "${var.project_name}-grafana-policy"
  description = "Policy for Grafana dashboards"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # CloudWatch for metrics visualization
      {
        Sid    = "CloudWatch"
        Effect = "Allow"
        Action = [
          "cloudwatch:DescribeAlarmsForMetric",
          "cloudwatch:DescribeAlarmHistory",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:ListMetrics",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:GetMetricData"
        ]
        Resource = "*"
      },
      # Logs for log visualization
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:DescribeLogGroups",
          "logs:GetLogGroupFields",
          "logs:StartQuery",
          "logs:StopQuery",
          "logs:GetQueryResults",
          "logs:GetLogEvents"
        ]
        Resource = "*"
      },
      # X-Ray for tracing
      {
        Sid    = "XRay"
        Effect = "Allow"
        Action = [
          "xray:GetTraceSummaries",
          "xray:BatchGetTraces",
          "xray:GetServiceGraph",
          "xray:GetTraceGraph",
          "xray:GetInsightSummaries",
          "xray:GetInsight",
          "xray:GetTimeSeriesServiceStatistics"
        ]
        Resource = "*"
      },
      # Athena for log queries (optional)
      {
        Sid    = "Athena"
        Effect = "Allow"
        Action = [
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "athena:StartQueryExecution",
          "athena:StopQueryExecution"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "grafana" {
  role       = aws_iam_role.grafana.name
  policy_arn = aws_iam_policy.grafana.arn
}

# KMS key for database encryption
resource "aws_kms_key" "database" {
  description             = "KMS key for database encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = merge(var.tags, {
    Name    = "${var.project_name}-database-kms"
    Purpose = "database-encryption"
  })
}

resource "aws_kms_alias" "database" {
  name          = "alias/${var.project_name}-database"
  target_key_id = aws_kms_key.database.key_id
}

# Outputs
output "ml_service_role_arn" {
  description = "ARN of the ML Service IAM role"
  value       = aws_iam_role.ml_service.arn
}

output "mlflow_role_arn" {
  description = "ARN of the MLflow IAM role"
  value       = aws_iam_role.mlflow.arn
}

output "postgres_role_arn" {
  description = "ARN of the PostgreSQL IAM role"
  value       = aws_iam_role.postgres.arn
}

output "prometheus_role_arn" {
  description = "ARN of the Prometheus IAM role"
  value       = aws_iam_role.prometheus.arn
}

output "grafana_role_arn" {
  description = "ARN of the Grafana IAM role"
  value       = aws_iam_role.grafana.arn
}

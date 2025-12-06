# ============================================================================
# NEURECTOMY - Velero IAM Configuration
# IAM roles and policies for Velero backup operations
# ============================================================================

# S3 bucket for backups (primary region)
resource "aws_s3_bucket" "velero_primary" {
  bucket = "${var.project_name}-velero-backups-primary"

  tags = merge(var.tags, {
    Name        = "${var.project_name}-velero-backups-primary"
    Purpose     = "velero-backups"
    Environment = var.environment
  })
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "velero_primary" {
  bucket = aws_s3_bucket.velero_primary.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "velero_primary" {
  bucket = aws_s3_bucket.velero_primary.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.velero.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# S3 bucket lifecycle rules
resource "aws_s3_bucket_lifecycle_configuration" "velero_primary" {
  bucket = aws_s3_bucket.velero_primary.id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "velero_primary" {
  bucket = aws_s3_bucket.velero_primary.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 bucket for DR region
resource "aws_s3_bucket" "velero_dr" {
  provider = aws.dr
  bucket   = "${var.project_name}-velero-backups-dr"

  tags = merge(var.tags, {
    Name        = "${var.project_name}-velero-backups-dr"
    Purpose     = "velero-dr-backups"
    Environment = var.environment
    Region      = var.dr_region
  })
}

resource "aws_s3_bucket_versioning" "velero_dr" {
  provider = aws.dr
  bucket   = aws_s3_bucket.velero_dr.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "velero_dr" {
  provider = aws.dr
  bucket   = aws_s3_bucket.velero_dr.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "velero_dr" {
  provider = aws.dr
  bucket   = aws_s3_bucket.velero_dr.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 replication for cross-region DR
resource "aws_s3_bucket_replication_configuration" "velero" {
  depends_on = [aws_s3_bucket_versioning.velero_primary]

  role   = aws_iam_role.velero_replication.arn
  bucket = aws_s3_bucket.velero_primary.id

  rule {
    id     = "replicate-to-dr"
    status = "Enabled"

    filter {
      prefix = "backups/"
    }

    destination {
      bucket        = aws_s3_bucket.velero_dr.arn
      storage_class = "STANDARD_IA"

      encryption_configuration {
        replica_kms_key_id = "arn:aws:kms:${var.dr_region}:${data.aws_caller_identity.current.account_id}:alias/velero-dr"
      }
    }

    delete_marker_replication {
      status = "Enabled"
    }
  }
}

# KMS key for Velero encryption
resource "aws_kms_key" "velero" {
  description             = "KMS key for Velero backup encryption"
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
        Sid    = "Allow Velero to use the key"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.velero.arn
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.project_name}-velero-kms"
    Purpose = "velero-encryption"
  })
}

resource "aws_kms_alias" "velero" {
  name          = "alias/${var.project_name}-velero"
  target_key_id = aws_kms_key.velero.key_id
}

# IAM role for Velero (IRSA)
resource "aws_iam_role" "velero" {
  name = "${var.project_name}-velero-role"

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
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:velero:velero"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.project_name}-velero-role"
    Purpose = "velero-irsa"
  })
}

# IAM policy for Velero
resource "aws_iam_policy" "velero" {
  name        = "${var.project_name}-velero-policy"
  description = "Policy for Velero backup operations"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EC2Permissions"
        Effect = "Allow"
        Action = [
          "ec2:DescribeVolumes",
          "ec2:DescribeSnapshots",
          "ec2:CreateTags",
          "ec2:CreateVolume",
          "ec2:CreateSnapshot",
          "ec2:DeleteSnapshot"
        ]
        Resource = "*"
      },
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:PutObject",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]
        Resource = [
          "${aws_s3_bucket.velero_primary.arn}/*",
          "${aws_s3_bucket.velero_dr.arn}/*"
        ]
      },
      {
        Sid    = "S3BucketList"
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.velero_primary.arn,
          aws_s3_bucket.velero_dr.arn
        ]
      },
      {
        Sid    = "KMSPermissions"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = [
          aws_kms_key.velero.arn
        ]
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "velero" {
  role       = aws_iam_role.velero.name
  policy_arn = aws_iam_policy.velero.arn
}

# IAM role for S3 replication
resource "aws_iam_role" "velero_replication" {
  name = "${var.project_name}-velero-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_policy" "velero_replication" {
  name = "${var.project_name}-velero-replication-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.velero_primary.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = "${aws_s3_bucket.velero_primary.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = "${aws_s3_bucket.velero_dr.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = aws_kms_key.velero.arn
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "velero_replication" {
  role       = aws_iam_role.velero_replication.name
  policy_arn = aws_iam_policy.velero_replication.arn
}

# Outputs
output "velero_iam_role_arn" {
  description = "ARN of the Velero IAM role"
  value       = aws_iam_role.velero.arn
}

output "velero_bucket_name" {
  description = "Name of the primary Velero S3 bucket"
  value       = aws_s3_bucket.velero_primary.id
}

output "velero_dr_bucket_name" {
  description = "Name of the DR Velero S3 bucket"
  value       = aws_s3_bucket.velero_dr.id
}

output "velero_kms_key_arn" {
  description = "ARN of the Velero KMS key"
  value       = aws_kms_key.velero.arn
}

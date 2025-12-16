"""
Comprehensive Backup Manager for Neurectomy
Handles automated backups with S3 integration and retention policies
"""

import asyncio
import subprocess
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import hashlib
import sys

# Add boto3 for S3 uploads (optional)
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("Warning: boto3 not installed. S3 uploads will be disabled.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages automated backups of all Neurectomy components
    
    Backs up:
    - PostgreSQL database
    - Ryot LLM model files
    - ΣVAULT encrypted storage
    - Configuration files
    """
    
    def __init__(
        self,
        backup_dir: Path = Path("/backups"),
        s3_bucket: Optional[str] = None,
        retention_days: int = 30,
        db_host: str = "postgres",
        db_user: str = "neurectomy",
        db_name: str = "neurectomy"
    ):
        self.backup_dir = backup_dir
        self.s3_bucket = s3_bucket
        self.retention_days = retention_days
        self.db_host = db_host
        self.db_user = db_user
        self.db_name = db_name
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if available
        if s3_bucket and S3_AVAILABLE:
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None
        
    async def create_full_backup(self) -> Dict[str, str]:
        """
        Create complete system backup
        
        Returns:
            Dictionary with backup manifest containing backup_id and component details
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"full_backup_{timestamp}"
        
        logger.info(f"Starting full backup: {backup_id}")
        
        manifest = {
            "backup_id": backup_id,
            "timestamp": timestamp,
            "components": {}
        }
        
        try:
            # Backup PostgreSQL database
            logger.info("Backing up database...")
            manifest["components"]["database"] = await self._backup_database(backup_id)
            
            # Backup Ryot LLM models
            logger.info("Backing up models...")
            manifest["components"]["models"] = await self._backup_models(backup_id)
            
            # Backup ΣVAULT storage
            logger.info("Backing up ΣVAULT...")
            manifest["components"]["sigmavault"] = await self._backup_sigmavault(backup_id)
            
            # Backup configurations
            logger.info("Backing up configurations...")
            manifest["components"]["configs"] = await self._backup_configs(backup_id)
            
            # Save manifest
            manifest_path = self.backup_dir / f"{backup_id}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
            logger.info(f"Backup manifest saved: {manifest_path}")
            
            # Upload to S3 if configured
            if self.s3_client and self.s3_bucket:
                logger.info("Uploading backup to S3...")
                await self._upload_to_s3(backup_id, manifest)
                
            # Clean old backups
            logger.info("Cleaning old backups...")
            await self._cleanup_old_backups()
            
            logger.info(f"Full backup completed successfully: {backup_id}")
            
            return manifest
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
        
    async def _backup_database(self, backup_id: str) -> Dict:
        """Backup PostgreSQL database using pg_dump"""
        backup_file = self.backup_dir / f"{backup_id}_database.sql.gz"
        
        cmd = [
            "pg_dump",
            "-h", self.db_host,
            "-U", self.db_user,
            "-d", self.db_name,
            "-F", "c",  # Custom format (compressed)
            "-Z", "9",  # Maximum compression
            "-f", str(backup_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Database backup failed: {result.stderr}")
            raise RuntimeError(f"Database backup failed: {result.stderr}")
            
        checksum = await self._calculate_checksum(backup_file)
        
        return {
            "file": str(backup_file),
            "size_bytes": backup_file.stat().st_size,
            "checksum": checksum,
            "type": "postgresql"
        }
        
    async def _backup_models(self, backup_id: str) -> Dict:
        """Backup Ryot LLM model files"""
        backup_file = self.backup_dir / f"{backup_id}_models.tar.gz"
        
        # Check if models directory exists
        models_dir = Path("/models")
        if not models_dir.exists():
            logger.warning("Models directory not found, skipping")
            return {"status": "skipped", "reason": "directory_not_found"}
        
        cmd = [
            "tar",
            "-czf",
            str(backup_file),
            "-C", "/",
            "models"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Models backup failed: {result.stderr}")
            raise RuntimeError(f"Models backup failed: {result.stderr}")
            
        checksum = await self._calculate_checksum(backup_file)
        
        return {
            "file": str(backup_file),
            "size_bytes": backup_file.stat().st_size,
            "checksum": checksum,
            "type": "tarball"
        }
        
    async def _backup_sigmavault(self, backup_id: str) -> Dict:
        """Backup ΣVAULT encrypted storage"""
        backup_file = self.backup_dir / f"{backup_id}_sigmavault.tar.gz"
        
        # Check if ΣVAULT data exists
        vault_dir = Path("/data/sigmavault")
        if not vault_dir.exists():
            logger.warning("ΣVAULT directory not found, skipping")
            return {"status": "skipped", "reason": "directory_not_found"}
        
        cmd = [
            "tar",
            "-czf",
            str(backup_file),
            "-C", "/data",
            "sigmavault"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"ΣVAULT backup failed: {result.stderr}")
            raise RuntimeError(f"ΣVAULT backup failed: {result.stderr}")
            
        checksum = await self._calculate_checksum(backup_file)
        
        return {
            "file": str(backup_file),
            "size_bytes": backup_file.stat().st_size,
            "checksum": checksum,
            "type": "tarball"
        }
        
    async def _backup_configs(self, backup_id: str) -> Dict:
        """Backup configuration files"""
        backup_file = self.backup_dir / f"{backup_id}_configs.tar.gz"
        
        # Backup Kubernetes configs, Helm values, and env files
        config_paths = [
            "infrastructure/kubernetes",
            "infrastructure/helm",
            ".env.production"
        ]
        
        # Filter to only existing paths
        existing_paths = [p for p in config_paths if Path(p).exists()]
        
        if not existing_paths:
            logger.warning("No configuration files found, skipping")
            return {"status": "skipped", "reason": "no_configs_found"}
        
        cmd = ["tar", "-czf", str(backup_file)] + existing_paths
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Config backup failed: {result.stderr}")
            raise RuntimeError(f"Config backup failed: {result.stderr}")
            
        checksum = await self._calculate_checksum(backup_file)
        
        return {
            "file": str(backup_file),
            "size_bytes": backup_file.stat().st_size,
            "checksum": checksum,
            "type": "tarball"
        }
        
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
                
        return sha256.hexdigest()
        
    async def _upload_to_s3(self, backup_id: str, manifest: Dict):
        """Upload backup files to S3"""
        if not self.s3_client or not self.s3_bucket:
            logger.warning("S3 not configured, skipping upload")
            return
            
        for component, details in manifest["components"].items():
            if details.get("status") == "skipped":
                continue
                
            file_path = Path(details["file"])
            s3_key = f"backups/{backup_id}/{file_path.name}"
            
            try:
                self.s3_client.upload_file(
                    str(file_path),
                    self.s3_bucket,
                    s3_key
                )
                logger.info(f"Uploaded {component} to s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload {component} to S3: {e}")
                
    async def _cleanup_old_backups(self):
        """Remove backups older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        logger.info(f"Cleaning backups older than {self.retention_days} days")
        
        removed_count = 0
        
        for backup_file in self.backup_dir.glob("full_backup_*"):
            try:
                # Extract timestamp from filename
                timestamp_str = backup_file.stem.split("_", 2)[2]
                backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                if backup_date < cutoff_date:
                    backup_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed old backup: {backup_file.name}")
                    
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse backup date: {backup_file.name} - {e}")
                
        logger.info(f"Removed {removed_count} old backups")


# CLI interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Neurectomy Backup Manager")
    parser.add_argument("--backup-dir", default="/backups", help="Backup directory path")
    parser.add_argument("--s3-bucket", help="S3 bucket for remote backups")
    parser.add_argument("--retention-days", type=int, default=30, help="Backup retention period")
    parser.add_argument("--db-host", default="postgres", help="PostgreSQL host")
    parser.add_argument("--db-user", default="neurectomy", help="PostgreSQL user")
    parser.add_argument("--db-name", default="neurectomy", help="PostgreSQL database name")
    
    args = parser.parse_args()
    
    manager = BackupManager(
        backup_dir=Path(args.backup_dir),
        s3_bucket=args.s3_bucket,
        retention_days=args.retention_days,
        db_host=args.db_host,
        db_user=args.db_user,
        db_name=args.db_name
    )
    
    try:
        manifest = await manager.create_full_backup()
        print("\n✅ Backup completed successfully!")
        print(f"\nBackup ID: {manifest['backup_id']}")
        print(f"Components backed up: {len(manifest['components'])}")
        
        for component, details in manifest['components'].items():
            if details.get("status") == "skipped":
                print(f"  - {component}: SKIPPED ({details.get('reason', 'unknown')})")
            else:
                size_mb = details['size_bytes'] / (1024 * 1024)
                print(f"  - {component}: {size_mb:.2f} MB")
                
    except Exception as e:
        print(f"\n❌ Backup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

"""
System Restore Manager
Handles point-in-time recovery from backups
"""

import asyncio
import subprocess
from typing import Dict, Optional, List
from pathlib import Path
import logging
import json
import hashlib
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RestoreManager:
    """
    Manages system restoration from backups
    """
    
    def __init__(
        self,
        backup_dir: Path = Path("/backups"),
        db_host: str = "postgres",
        db_user: str = "neurectomy",
        db_name: str = "neurectomy"
    ):
        self.backup_dir = backup_dir
        self.db_host = db_host
        self.db_user = db_user
        self.db_name = db_name
        
    async def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []
        
        for manifest_file in self.backup_dir.glob("*_manifest.json"):
            try:
                with open(manifest_file) as f:
                    manifest = json.load(f)
                    backups.append({
                        "backup_id": manifest["backup_id"],
                        "timestamp": manifest["timestamp"],
                        "components": list(manifest["components"].keys())
                    })
            except Exception as e:
                logger.warning(f"Could not read manifest {manifest_file}: {e}")
                
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
        
    async def restore_from_backup(
        self,
        backup_id: str,
        components: Optional[List[str]] = None,
        verify_checksums: bool = True
    ) -> Dict:
        """
        Restore system from backup
        
        Args:
            backup_id: ID of backup to restore from
            components: List of components to restore (None = all)
            verify_checksums: Verify file integrity before restoring
            
        Returns:
            Restoration results for each component
        """
        logger.info(f"Starting restoration from backup: {backup_id}")
        
        # Load manifest
        manifest_file = self.backup_dir / f"{backup_id}_manifest.json"
        
        if not manifest_file.exists():
            raise ValueError(f"Backup not found: {backup_id}")
            
        with open(manifest_file) as f:
            manifest = json.load(f)
            
        results = {}
        available_components = manifest["components"].keys()
        components_to_restore = components or available_components
        
        for component in components_to_restore:
            if component not in available_components:
                logger.warning(f"Component not in backup: {component}")
                results[component] = {"success": False, "error": "not_in_backup"}
                continue
                
            component_info = manifest["components"][component]
            
            # Skip components that were skipped during backup
            if component_info.get("status") == "skipped":
                logger.info(f"Skipping {component} (was skipped during backup)")
                results[component] = {"success": True, "skipped": True}
                continue
                
            logger.info(f"Restoring component: {component}")
            
            # Verify checksum if requested
            if verify_checksums:
                if not await self._verify_checksum(component_info):
                    results[component] = {"success": False, "error": "checksum_mismatch"}
                    logger.error(f"Checksum verification failed for {component}")
                    continue
                    
            try:
                if component == "database":
                    result = await self._restore_database(component_info)
                elif component == "models":
                    result = await self._restore_models(component_info)
                elif component == "sigmavault":
                    result = await self._restore_sigmavault(component_info)
                elif component == "configs":
                    result = await self._restore_configs(component_info)
                else:
                    result = {"success": False, "error": "unknown_component"}
                    
                results[component] = result
                
            except Exception as e:
                logger.error(f"Failed to restore {component}: {e}")
                results[component] = {"success": False, "error": str(e)}
                
        logger.info(f"Restoration completed from backup: {backup_id}")
        
        return results
        
    async def _verify_checksum(self, component_info: Dict) -> bool:
        """Verify file checksum matches backup manifest"""
        backup_file = Path(component_info["file"])
        expected_checksum = component_info["checksum"]
        
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
            
        sha256 = hashlib.sha256()
        with open(backup_file, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
                
        actual_checksum = sha256.hexdigest()
        
        if actual_checksum != expected_checksum:
            logger.error(
                f"Checksum mismatch for {backup_file.name}\n"
                f"Expected: {expected_checksum}\n"
                f"Actual: {actual_checksum}"
            )
            return False
            
        logger.info(f"Checksum verified for {backup_file.name}")
        return True
        
    async def _restore_database(self, backup_info: Dict) -> Dict:
        """Restore PostgreSQL database"""
        backup_file = Path(backup_info["file"])
        
        if not backup_file.exists():
            return {"success": False, "error": "backup_file_not_found"}
            
        logger.info("Restoring database...")
        
        cmd = [
            "pg_restore",
            "-h", self.db_host,
            "-U", self.db_user,
            "-d", self.db_name,
            "--clean",  # Drop objects before recreating
            "--if-exists",  # Don't error if objects don't exist
            str(backup_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Database restore failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
            
        logger.info("Database restored successfully")
        return {"success": True}
        
    async def _restore_models(self, backup_info: Dict) -> Dict:
        """Restore Ryot LLM models"""
        backup_file = Path(backup_info["file"])
        
        if not backup_file.exists():
            return {"success": False, "error": "backup_file_not_found"}
            
        logger.info("Restoring models...")
        
        cmd = [
            "tar",
            "-xzf",
            str(backup_file),
            "-C", "/"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Models restore failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
            
        logger.info("Models restored successfully")
        return {"success": True}
        
    async def _restore_sigmavault(self, backup_info: Dict) -> Dict:
        """Restore ΣVAULT storage"""
        backup_file = Path(backup_info["file"])
        
        if not backup_file.exists():
            return {"success": False, "error": "backup_file_not_found"}
            
        logger.info("Restoring ΣVAULT...")
        
        cmd = [
            "tar",
            "-xzf",
            str(backup_file),
            "-C", "/data"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"ΣVAULT restore failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
            
        logger.info("ΣVAULT restored successfully")
        return {"success": True}
        
    async def _restore_configs(self, backup_info: Dict) -> Dict:
        """Restore configuration files"""
        backup_file = Path(backup_info["file"])
        
        if not backup_file.exists():
            return {"success": False, "error": "backup_file_not_found"}
            
        logger.info("Restoring configurations...")
        
        cmd = [
            "tar",
            "-xzf",
            str(backup_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Config restore failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
            
        logger.info("Configurations restored successfully")
        return {"success": True}


# CLI interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Neurectomy Restore Manager")
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--restore", help="Backup ID to restore from")
    parser.add_argument("--components", nargs="+", help="Specific components to restore")
    parser.add_argument("--no-verify", action="store_true", help="Skip checksum verification")
    parser.add_argument("--backup-dir", default="/backups", help="Backup directory path")
    parser.add_argument("--db-host", default="postgres")
    parser.add_argument("--db-user", default="neurectomy")
    parser.add_argument("--db-name", default="neurectomy")
    
    args = parser.parse_args()
    
    manager = RestoreManager(
        backup_dir=Path(args.backup_dir),
        db_host=args.db_host,
        db_user=args.db_user,
        db_name=args.db_name
    )
    
    if args.list:
        backups = await manager.list_backups()
        
        if not backups:
            print("No backups found")
            return
            
        print("\nAvailable backups:\n")
        for backup in backups:
            print(f"  {backup['backup_id']}")
            print(f"    Timestamp: {backup['timestamp']}")
            print(f"    Components: {', '.join(backup['components'])}\n")
            
    elif args.restore:
        results = await manager.restore_from_backup(
            backup_id=args.restore,
            components=args.components,
            verify_checksums=not args.no_verify
        )
        
        print(f"\n✅ Restore from {args.restore}:\n")
        for component, result in results.items():
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {component}")
            if not result["success"]:
                print(f"      Error: {result.get('error', 'unknown')}")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())

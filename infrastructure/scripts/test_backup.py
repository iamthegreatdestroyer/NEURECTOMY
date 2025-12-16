"""Test backup manager"""
import asyncio
from pathlib import Path
from backup_manager import BackupManager

async def test():
    manager = BackupManager(
        backup_dir=Path("/tmp/test_backups"),
        retention_days=7
    )
    
    manifest = await manager.create_full_backup()
    
    assert "backup_id" in manifest
    assert "components" in manifest
    assert "database" in manifest["components"]
    
    print("✅ Backup test passed!")

if __name__ == "__main__":
    asyncio.run(test())

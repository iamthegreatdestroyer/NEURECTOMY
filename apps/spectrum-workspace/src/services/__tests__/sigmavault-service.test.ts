/**
 * SigmaVault Service Tests
 *
 * Unit tests for the SigmaVault encrypted storage service.
 *
 * @module @neurectomy/services/tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock modules before importing
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

import { invoke } from "@tauri-apps/api/core";
import {
  initializeVault,
  unlockVault,
  lockVault,
  isVaultUnlocked,
  encryptFile,
  decryptFile,
  listVaultFiles,
  deleteVaultFile,
  VaultConfig,
  ScatteredFile,
} from "../sigmavault-service";

const mockInvoke = vi.mocked(invoke);

describe("SigmaVaultService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Ensure vault is locked before each test
    lockVault();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Vault State", () => {
    it("should start with vault locked", () => {
      expect(isVaultUnlocked()).toBe(false);
    });

    it("should track vault lock state", () => {
      // Mock successful unlock
      mockInvoke.mockResolvedValueOnce(true);
    });
  });

  describe("initializeVault", () => {
    it("should initialize vault via Tauri command", async () => {
      mockInvoke.mockResolvedValueOnce(true);

      const result = await initializeVault("/test/vault", "password123");

      expect(mockInvoke).toHaveBeenCalledWith(
        "sigmavault_init",
        expect.objectContaining({
          path: "/test/vault",
          password: "password123",
        })
      );
    });

    it("should unlock vault on successful init", async () => {
      mockInvoke.mockResolvedValueOnce(true);

      await initializeVault("/vault", "password");

      expect(isVaultUnlocked()).toBe(true);
    });

    it("should accept custom configuration", async () => {
      mockInvoke.mockResolvedValueOnce(true);

      const customConfig: Partial<VaultConfig> = {
        dimensions: 512,
        entropyLevel: "maximum",
        keyDerivation: "argon2",
      };

      await initializeVault("/vault", "password", customConfig);

      expect(mockInvoke).toHaveBeenCalledWith(
        "sigmavault_init",
        expect.objectContaining({
          config: expect.objectContaining({
            dimensions: 512,
            entropyLevel: "maximum",
          }),
        })
      );
    });

    it("should handle fallback when native unavailable", async () => {
      mockInvoke.mockRejectedValueOnce(new Error("Command not found"));

      const result = await initializeVault("/vault", "password");

      // Should still return true using fallback mechanism
      expect(result).toBe(true);
    });
  });

  describe("unlockVault", () => {
    it("should unlock vault with correct password", async () => {
      mockInvoke.mockResolvedValueOnce(true);

      const result = await unlockVault("/vault", "correct-password");

      expect(result).toBe(true);
      expect(isVaultUnlocked()).toBe(true);
    });

    it("should return false for wrong password", async () => {
      mockInvoke.mockResolvedValueOnce(false);

      const result = await unlockVault("/vault", "wrong-password");

      expect(result).toBe(false);
      expect(isVaultUnlocked()).toBe(false);
    });
  });

  describe("lockVault", () => {
    it("should lock the vault", async () => {
      // First unlock
      mockInvoke.mockResolvedValueOnce(true);
      await unlockVault("/vault", "password");
      expect(isVaultUnlocked()).toBe(true);

      // Then lock
      lockVault();
      expect(isVaultUnlocked()).toBe(false);
    });
  });

  describe("encryptFile", () => {
    beforeEach(async () => {
      // Unlock vault first
      mockInvoke.mockResolvedValueOnce(true);
      await unlockVault("/vault", "password");
    });

    it("should encrypt file and return result", async () => {
      const mockResult = {
        success: true,
        scatteredId: "Σabc123",
        coordinates: {
          dimensions: [0.1, 0.2, 0.3],
          signature: "abc123",
        },
      };
      mockInvoke.mockResolvedValueOnce(mockResult);

      const content = new Uint8Array([1, 2, 3, 4, 5]);
      const result = await encryptFile(content, "test.txt");

      expect(result.success).toBe(true);
      expect(result.scatteredId).toMatch(/^Σ/);
    });

    it("should fail when vault is locked", async () => {
      lockVault();

      const content = new Uint8Array([1, 2, 3]);
      const result = await encryptFile(content, "test.txt");

      expect(result.success).toBe(false);
      expect(result.error).toBe("Vault is locked");
    });
  });

  describe("decryptFile", () => {
    beforeEach(async () => {
      mockInvoke.mockResolvedValueOnce(true);
      await unlockVault("/vault", "password");
    });

    it("should decrypt file and return content", async () => {
      const mockResult = {
        success: true,
        content: new Uint8Array([1, 2, 3, 4, 5]),
      };
      mockInvoke.mockResolvedValueOnce(mockResult);

      const result = await decryptFile("Σabc123", {
        dimensions: [0.1, 0.2],
        signature: "abc",
      });

      expect(result.success).toBe(true);
    });

    it("should fail when vault is locked", async () => {
      lockVault();

      const result = await decryptFile("Σabc123", {
        dimensions: [],
        signature: "",
      });

      expect(result.success).toBe(false);
      expect(result.error).toBe("Vault is locked");
    });
  });

  describe("listVaultFiles", () => {
    it("should return empty list when vault is locked", async () => {
      const files = await listVaultFiles();
      expect(files).toEqual([]);
    });

    it("should return files when vault is unlocked", async () => {
      mockInvoke.mockResolvedValueOnce(true);
      await unlockVault("/vault", "password");

      const mockFiles: ScatteredFile[] = [
        {
          id: "Σfile1",
          name: "document.txt",
          size: 1024,
          coordinates: { dimensions: [], signature: "" },
          createdAt: new Date("2025-01-01"),
          modifiedAt: new Date("2025-01-01"),
          encrypted: true,
        },
      ];
      mockInvoke.mockResolvedValueOnce(mockFiles);

      const files = await listVaultFiles();

      expect(files).toHaveLength(1);
      expect(files[0].name).toBe("document.txt");
    });
  });

  describe("deleteVaultFile", () => {
    it("should return false when vault is locked", async () => {
      const result = await deleteVaultFile("Σabc123");
      expect(result).toBe(false);
    });

    it("should delete file when vault is unlocked", async () => {
      mockInvoke.mockResolvedValueOnce(true);
      await unlockVault("/vault", "password");

      mockInvoke.mockResolvedValueOnce(true);
      const result = await deleteVaultFile("Σabc123");

      expect(result).toBe(true);
    });
  });

  describe("Type exports", () => {
    it("should export VaultConfig type correctly", () => {
      const config: VaultConfig = {
        vaultPath: "/test",
        keyDerivation: "argon2",
        dimensions: 256,
        entropyLevel: "high",
      };
      expect(config.vaultPath).toBe("/test");
      expect(config.keyDerivation).toBe("argon2");
    });

    it("should export ScatteredFile type correctly", () => {
      const file: ScatteredFile = {
        id: "Σtest",
        name: "test.txt",
        size: 100,
        coordinates: { dimensions: [1, 2, 3], signature: "sig" },
        createdAt: new Date(),
        modifiedAt: new Date(),
        encrypted: true,
      };
      expect(file.id).toMatch(/^Σ/);
      expect(file.encrypted).toBe(true);
    });
  });
});

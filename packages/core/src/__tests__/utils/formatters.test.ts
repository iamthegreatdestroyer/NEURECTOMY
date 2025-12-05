import { describe, it, expect } from "vitest";
import {
  formatDuration,
  formatBytes,
  formatNumber,
  formatPercentage,
  formatRelativeTime,
} from "../utils/formatters";

describe("formatDuration", () => {
  it("should format milliseconds", () => {
    expect(formatDuration(500)).toBe("500ms");
  });

  it("should format seconds", () => {
    expect(formatDuration(1500)).toBe("1.5s");
    expect(formatDuration(5000)).toBe("5s");
  });

  it("should format minutes", () => {
    expect(formatDuration(60000)).toBe("1m");
    expect(formatDuration(90000)).toBe("1m 30s");
  });

  it("should format hours", () => {
    expect(formatDuration(3600000)).toBe("1h");
    expect(formatDuration(3660000)).toBe("1h 1m");
  });

  it("should handle zero", () => {
    expect(formatDuration(0)).toBe("0ms");
  });
});

describe("formatBytes", () => {
  it("should format bytes", () => {
    expect(formatBytes(500)).toBe("500 B");
  });

  it("should format kilobytes", () => {
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });

  it("should format megabytes", () => {
    expect(formatBytes(1048576)).toBe("1 MB");
  });

  it("should format gigabytes", () => {
    expect(formatBytes(1073741824)).toBe("1 GB");
  });

  it("should respect decimal places", () => {
    expect(formatBytes(1536, 0)).toBe("2 KB");
    expect(formatBytes(1536, 2)).toBe("1.50 KB");
  });

  it("should handle zero", () => {
    expect(formatBytes(0)).toBe("0 B");
  });
});

describe("formatNumber", () => {
  it("should add thousand separators", () => {
    expect(formatNumber(1000)).toBe("1,000");
    expect(formatNumber(1000000)).toBe("1,000,000");
  });

  it("should handle decimals", () => {
    expect(formatNumber(1234.56)).toBe("1,234.56");
  });

  it("should handle negative numbers", () => {
    expect(formatNumber(-1234)).toBe("-1,234");
  });

  it("should respect locale", () => {
    expect(formatNumber(1234.56, "de-DE")).toBe("1.234,56");
  });
});

describe("formatPercentage", () => {
  it("should format decimal to percentage", () => {
    expect(formatPercentage(0.5)).toBe("50%");
    expect(formatPercentage(0.123)).toBe("12.3%");
  });

  it("should respect decimal places", () => {
    expect(formatPercentage(0.1234, 2)).toBe("12.34%");
    expect(formatPercentage(0.1234, 0)).toBe("12%");
  });

  it("should handle values over 1", () => {
    expect(formatPercentage(1.5)).toBe("150%");
  });

  it("should handle zero", () => {
    expect(formatPercentage(0)).toBe("0%");
  });
});

describe("formatRelativeTime", () => {
  it("should format seconds ago", () => {
    const date = new Date(Date.now() - 30000);
    expect(formatRelativeTime(date)).toMatch(/30s ago|just now/);
  });

  it("should format minutes ago", () => {
    const date = new Date(Date.now() - 300000);
    expect(formatRelativeTime(date)).toBe("5m ago");
  });

  it("should format hours ago", () => {
    const date = new Date(Date.now() - 7200000);
    expect(formatRelativeTime(date)).toBe("2h ago");
  });

  it("should format days ago", () => {
    const date = new Date(Date.now() - 172800000);
    expect(formatRelativeTime(date)).toBe("2d ago");
  });

  it("should handle Date objects and timestamps", () => {
    const timestamp = Date.now() - 60000;
    expect(formatRelativeTime(new Date(timestamp))).toBe("1m ago");
  });
});

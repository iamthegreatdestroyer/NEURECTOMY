import { describe, it, expect } from "vitest";
import { cn } from "../cn";

describe("cn", () => {
  it("should merge class names", () => {
    const result = cn("px-4", "py-2");
    expect(result).toBe("px-4 py-2");
  });

  it("should handle conditional classes", () => {
    const isActive = true;
    const isDisabled = false;

    const result = cn("base", isActive && "active", isDisabled && "disabled");
    expect(result).toBe("base active");
  });

  it("should deduplicate conflicting Tailwind classes", () => {
    const result = cn("px-4", "px-8");
    expect(result).toBe("px-8");
  });

  it("should handle arrays of classes", () => {
    const result = cn(["px-4", "py-2"], "mx-auto");
    expect(result).toBe("px-4 py-2 mx-auto");
  });

  it("should handle objects for conditional classes", () => {
    const result = cn({
      "bg-blue-500": true,
      "bg-red-500": false,
      "text-white": true,
    });
    expect(result).toBe("bg-blue-500 text-white");
  });

  it("should handle undefined and null values", () => {
    const result = cn("base", undefined, null, "end");
    expect(result).toBe("base end");
  });

  it("should merge Tailwind modifiers correctly", () => {
    const result = cn("hover:bg-blue-500", "hover:bg-red-500");
    expect(result).toBe("hover:bg-red-500");
  });

  it("should handle empty inputs", () => {
    const result = cn();
    expect(result).toBe("");
  });

  it("should handle complex combinations", () => {
    const variant = "primary";
    const size = "large";

    const result = cn(
      "base-class",
      variant === "primary" && "bg-primary",
      size === "large" ? "px-8 py-4" : "px-4 py-2",
      { "rounded-full": true }
    );

    expect(result).toBe("base-class bg-primary px-8 py-4 rounded-full");
  });
});

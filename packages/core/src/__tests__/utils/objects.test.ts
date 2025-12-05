import { describe, it, expect } from "vitest";
import { deepMerge, pick, omit } from "../utils/objects";

describe("deepMerge", () => {
  it("should merge flat objects", () => {
    const target = { a: 1, b: 2 };
    const source = { b: 3, c: 4 };
    const result = deepMerge(target, source);
    expect(result).toEqual({ a: 1, b: 3, c: 4 });
  });

  it("should deep merge nested objects", () => {
    const target = { a: { b: 1, c: 2 } };
    const source = { a: { c: 3, d: 4 } };
    const result = deepMerge(target, source);
    expect(result).toEqual({ a: { b: 1, c: 3, d: 4 } });
  });

  it("should not mutate original objects", () => {
    const target = { a: 1 };
    const source = { b: 2 };
    deepMerge(target, source);
    expect(target).toEqual({ a: 1 });
    expect(source).toEqual({ b: 2 });
  });

  it("should handle arrays by replacing", () => {
    const target = { arr: [1, 2, 3] };
    const source = { arr: [4, 5] };
    const result = deepMerge(target, source);
    expect(result).toEqual({ arr: [4, 5] });
  });

  it("should handle null and undefined", () => {
    const target = { a: 1, b: 2 };
    const source = { a: null, c: undefined };
    const result = deepMerge(target, source);
    expect(result.a).toBeNull();
    expect(result.b).toBe(2);
  });

  it("should handle deeply nested structures", () => {
    const target = { a: { b: { c: { d: 1 } } } };
    const source = { a: { b: { c: { e: 2 } } } };
    const result = deepMerge(target, source);
    expect(result).toEqual({ a: { b: { c: { d: 1, e: 2 } } } });
  });
});

describe("pick", () => {
  it("should pick specified keys", () => {
    const obj = { a: 1, b: 2, c: 3 };
    const result = pick(obj, ["a", "c"]);
    expect(result).toEqual({ a: 1, c: 3 });
  });

  it("should ignore missing keys", () => {
    const obj = { a: 1, b: 2 };
    const result = pick(obj, ["a", "c" as keyof typeof obj]);
    expect(result).toEqual({ a: 1 });
  });

  it("should return empty object for empty keys", () => {
    const obj = { a: 1, b: 2 };
    const result = pick(obj, []);
    expect(result).toEqual({});
  });

  it("should not mutate original object", () => {
    const obj = { a: 1, b: 2 };
    pick(obj, ["a"]);
    expect(obj).toEqual({ a: 1, b: 2 });
  });

  it("should work with nested values", () => {
    const obj = { a: { nested: true }, b: 2 };
    const result = pick(obj, ["a"]);
    expect(result).toEqual({ a: { nested: true } });
  });
});

describe("omit", () => {
  it("should omit specified keys", () => {
    const obj = { a: 1, b: 2, c: 3 };
    const result = omit(obj, ["b"]);
    expect(result).toEqual({ a: 1, c: 3 });
  });

  it("should handle missing keys", () => {
    const obj = { a: 1, b: 2 };
    const result = omit(obj, ["c" as keyof typeof obj]);
    expect(result).toEqual({ a: 1, b: 2 });
  });

  it("should return copy for empty keys", () => {
    const obj = { a: 1, b: 2 };
    const result = omit(obj, []);
    expect(result).toEqual({ a: 1, b: 2 });
  });

  it("should not mutate original object", () => {
    const obj = { a: 1, b: 2 };
    omit(obj, ["a"]);
    expect(obj).toEqual({ a: 1, b: 2 });
  });

  it("should omit multiple keys", () => {
    const obj = { a: 1, b: 2, c: 3, d: 4 };
    const result = omit(obj, ["a", "c"]);
    expect(result).toEqual({ b: 2, d: 4 });
  });
});

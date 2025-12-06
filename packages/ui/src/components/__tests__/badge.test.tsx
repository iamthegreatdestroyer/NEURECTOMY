import * as React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Badge } from "../badge";

describe("Badge", () => {
  it("should render with text content", () => {
    render(<Badge>New</Badge>);
    expect(screen.getByText("New")).toBeInTheDocument();
  });

  it("should render with default variant", () => {
    render(<Badge data-testid="badge">Default</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveClass("bg-primary");
  });

  it("should apply custom className", () => {
    render(<Badge className="custom-class" data-testid="badge" />);
    expect(screen.getByTestId("badge")).toHaveClass("custom-class");
  });

  describe("variants", () => {
    it("should render default variant", () => {
      render(
        <Badge variant="default" data-testid="badge">
          Default
        </Badge>
      );
      expect(screen.getByTestId("badge")).toHaveClass("bg-primary");
    });

    it("should render secondary variant", () => {
      render(
        <Badge variant="secondary" data-testid="badge">
          Secondary
        </Badge>
      );
      expect(screen.getByTestId("badge")).toHaveClass("bg-secondary");
    });

    it("should render destructive variant", () => {
      render(
        <Badge variant="destructive" data-testid="badge">
          Destructive
        </Badge>
      );
      expect(screen.getByTestId("badge")).toHaveClass("bg-destructive");
    });

    it("should render outline variant", () => {
      render(
        <Badge variant="outline" data-testid="badge">
          Outline
        </Badge>
      );
      expect(screen.getByTestId("badge")).toHaveClass("text-foreground");
      expect(screen.getByTestId("badge")).not.toHaveClass("bg-primary");
    });
  });

  it("should have rounded-full styling", () => {
    render(<Badge data-testid="badge">Rounded</Badge>);
    expect(screen.getByTestId("badge")).toHaveClass("rounded-full");
  });

  it("should have proper padding", () => {
    render(<Badge data-testid="badge">Padded</Badge>);
    expect(screen.getByTestId("badge")).toHaveClass("px-2.5");
    expect(screen.getByTestId("badge")).toHaveClass("py-0.5");
  });

  it("should be inline-flex", () => {
    render(<Badge data-testid="badge">Inline</Badge>);
    expect(screen.getByTestId("badge")).toHaveClass("inline-flex");
  });

  it("should accept additional HTML attributes", () => {
    render(
      <Badge data-testid="badge" role="status" aria-label="Status badge">
        Active
      </Badge>
    );
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveAttribute("role", "status");
    expect(badge).toHaveAttribute("aria-label", "Status badge");
  });

  it("should support children as React elements", () => {
    render(
      <Badge data-testid="badge">
        <span>Icon</span>
        <span>Text</span>
      </Badge>
    );
    expect(screen.getByText("Icon")).toBeInTheDocument();
    expect(screen.getByText("Text")).toBeInTheDocument();
  });
});

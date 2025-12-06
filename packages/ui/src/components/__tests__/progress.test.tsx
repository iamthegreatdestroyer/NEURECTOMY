import * as React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Progress } from "../progress";

describe("Progress", () => {
  it("should render with default value", () => {
    render(<Progress data-testid="progress" />);
    expect(screen.getByTestId("progress")).toBeInTheDocument();
  });

  it("should render with progressbar role", () => {
    render(<Progress />);
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("should have correct aria-valuenow attribute", () => {
    render(<Progress value={50} />);
    const progressbar = screen.getByRole("progressbar");
    expect(progressbar).toHaveAttribute("aria-valuenow", "50");
  });

  it("should have aria-valuemin of 0", () => {
    render(<Progress value={50} />);
    const progressbar = screen.getByRole("progressbar");
    expect(progressbar).toHaveAttribute("aria-valuemin", "0");
  });

  it("should have aria-valuemax of 100", () => {
    render(<Progress value={50} />);
    const progressbar = screen.getByRole("progressbar");
    expect(progressbar).toHaveAttribute("aria-valuemax", "100");
  });

  it("should render indicator at 0% when value is 0", () => {
    const { container } = render(<Progress value={0} />);
    const indicator = container.querySelector("[data-state]");
    expect(indicator).toHaveStyle({ transform: "translateX(-100%)" });
  });

  it("should render indicator at 50% when value is 50", () => {
    const { container } = render(<Progress value={50} />);
    const indicator = container.querySelector("[data-state]");
    expect(indicator).toHaveStyle({ transform: "translateX(-50%)" });
  });

  it("should render indicator at 100% when value is 100", () => {
    const { container } = render(<Progress value={100} />);
    const indicator = container.querySelector("[data-state]");
    expect(indicator).toHaveStyle({ transform: "translateX(0%)" });
  });

  it("should apply custom className", () => {
    render(<Progress className="custom-class" data-testid="progress" />);
    expect(screen.getByTestId("progress")).toHaveClass("custom-class");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<Progress ref={ref} />);
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it("should have correct base styling", () => {
    render(<Progress data-testid="progress" />);
    const progress = screen.getByTestId("progress");
    expect(progress).toHaveClass("relative");
    expect(progress).toHaveClass("h-4");
    expect(progress).toHaveClass("w-full");
    expect(progress).toHaveClass("overflow-hidden");
    expect(progress).toHaveClass("rounded-full");
  });

  it("should update aria-valuenow when value changes", () => {
    const { rerender } = render(<Progress value={25} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "25"
    );

    rerender(<Progress value={75} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "75"
    );
  });

  it("should handle undefined value gracefully", () => {
    render(<Progress value={undefined} />);
    const progressbar = screen.getByRole("progressbar");
    // Radix handles undefined by not setting aria-valuenow for indeterminate
    expect(progressbar).toBeInTheDocument();
  });

  it("should accept additional props", () => {
    render(<Progress data-testid="progress" aria-label="Loading progress" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-label",
      "Loading progress"
    );
  });
});

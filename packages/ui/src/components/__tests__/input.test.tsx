import * as React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Input } from "../input";

describe("Input", () => {
  it("should render with default props", () => {
    render(<Input placeholder="Enter text" />);
    const input = screen.getByPlaceholderText("Enter text");
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("type", "text");
  });

  it("should render with different types", () => {
    const { rerender } = render(<Input type="email" data-testid="input" />);
    expect(screen.getByTestId("input")).toHaveAttribute("type", "email");

    rerender(<Input type="password" data-testid="input" />);
    expect(screen.getByTestId("input")).toHaveAttribute("type", "password");

    rerender(<Input type="number" data-testid="input" />);
    expect(screen.getByTestId("input")).toHaveAttribute("type", "number");
  });

  it("should apply custom className", () => {
    render(<Input className="custom-class" data-testid="input" />);
    expect(screen.getByTestId("input")).toHaveClass("custom-class");
  });

  it("should handle value changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Input onChange={onChange} data-testid="input" />);

    const input = screen.getByTestId("input");
    await user.type(input, "hello");

    expect(onChange).toHaveBeenCalledTimes(5);
    expect(input).toHaveValue("hello");
  });

  it("should be disabled when disabled prop is true", () => {
    render(<Input disabled data-testid="input" />);
    expect(screen.getByTestId("input")).toBeDisabled();
  });

  it("should be readonly when readOnly prop is true", () => {
    render(<Input readOnly data-testid="input" />);
    expect(screen.getByTestId("input")).toHaveAttribute("readonly");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLInputElement>();
    render(<Input ref={ref} />);
    expect(ref.current).toBeInstanceOf(HTMLInputElement);
  });

  it("should accept aria attributes", () => {
    render(
      <Input
        aria-label="Username"
        aria-describedby="username-help"
        data-testid="input"
      />
    );
    const input = screen.getByTestId("input");
    expect(input).toHaveAttribute("aria-label", "Username");
    expect(input).toHaveAttribute("aria-describedby", "username-help");
  });

  it("should handle focus and blur events", async () => {
    const user = userEvent.setup();
    const onFocus = vi.fn();
    const onBlur = vi.fn();
    render(<Input onFocus={onFocus} onBlur={onBlur} data-testid="input" />);

    const input = screen.getByTestId("input");
    await user.click(input);
    expect(onFocus).toHaveBeenCalledTimes(1);

    await user.tab();
    expect(onBlur).toHaveBeenCalledTimes(1);
  });

  it("should support controlled value", () => {
    const { rerender } = render(
      <Input value="initial" data-testid="input" readOnly />
    );
    expect(screen.getByTestId("input")).toHaveValue("initial");

    rerender(<Input value="updated" data-testid="input" readOnly />);
    expect(screen.getByTestId("input")).toHaveValue("updated");
  });

  it("should support placeholder text", () => {
    render(<Input placeholder="Type here..." />);
    expect(screen.getByPlaceholderText("Type here...")).toBeInTheDocument();
  });

  it("should have correct base styling classes", () => {
    render(<Input data-testid="input" />);
    const input = screen.getByTestId("input");
    expect(input).toHaveClass("flex");
    expect(input).toHaveClass("rounded-md");
    expect(input).toHaveClass("border");
  });
});

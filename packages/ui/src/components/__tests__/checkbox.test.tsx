import * as React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Checkbox } from "../checkbox";

describe("Checkbox", () => {
  it("should render unchecked by default", () => {
    render(<Checkbox aria-label="Accept terms" />);
    const checkbox = screen.getByRole("checkbox", { name: "Accept terms" });
    expect(checkbox).toBeInTheDocument();
    expect(checkbox).not.toBeChecked();
  });

  it("should render checked when defaultChecked is true", () => {
    render(<Checkbox defaultChecked aria-label="Accept terms" />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeChecked();
  });

  it("should toggle on click", async () => {
    const user = userEvent.setup();
    render(<Checkbox aria-label="Accept terms" />);

    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).not.toBeChecked();

    await user.click(checkbox);
    expect(checkbox).toBeChecked();

    await user.click(checkbox);
    expect(checkbox).not.toBeChecked();
  });

  it("should call onCheckedChange when toggled", async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(<Checkbox onCheckedChange={onCheckedChange} aria-label="Test" />);

    const checkbox = screen.getByRole("checkbox");
    await user.click(checkbox);

    expect(onCheckedChange).toHaveBeenCalledTimes(1);
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it("should be disabled when disabled prop is true", () => {
    render(<Checkbox disabled aria-label="Disabled checkbox" />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeDisabled();
  });

  it("should not toggle when disabled", async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(
      <Checkbox disabled onCheckedChange={onCheckedChange} aria-label="Test" />
    );

    const checkbox = screen.getByRole("checkbox");
    await user.click(checkbox);

    expect(onCheckedChange).not.toHaveBeenCalled();
  });

  it("should apply custom className", () => {
    render(<Checkbox className="custom-class" aria-label="Test" />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toHaveClass("custom-class");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLButtonElement>();
    render(<Checkbox ref={ref} aria-label="Test" />);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  it("should support controlled checked state", () => {
    const { rerender } = render(
      <Checkbox checked={false} aria-label="Controlled" />
    );
    expect(screen.getByRole("checkbox")).not.toBeChecked();

    rerender(<Checkbox checked={true} aria-label="Controlled" />);
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  it("should have correct base styling", () => {
    render(<Checkbox aria-label="Test" />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toHaveClass("h-4");
    expect(checkbox).toHaveClass("w-4");
    expect(checkbox).toHaveClass("rounded-sm");
    expect(checkbox).toHaveClass("border");
  });

  it("should be accessible via keyboard", async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(<Checkbox onCheckedChange={onCheckedChange} aria-label="Test" />);

    const checkbox = screen.getByRole("checkbox");
    checkbox.focus();
    expect(checkbox).toHaveFocus();

    await user.keyboard(" ");
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it("should accept aria-describedby", () => {
    render(
      <>
        <Checkbox aria-label="Terms" aria-describedby="terms-help" />
        <span id="terms-help">Read our terms carefully</span>
      </>
    );
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toHaveAttribute("aria-describedby", "terms-help");
  });

  it("should support required attribute", () => {
    render(<Checkbox required aria-label="Required checkbox" />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeRequired();
  });

  it("should support name attribute for forms", () => {
    render(<Checkbox name="agreement" aria-label="Agreement" />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toHaveAttribute("name", "agreement");
  });

  it("should support value attribute for forms", () => {
    render(<Checkbox value="yes" aria-label="Option" />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toHaveAttribute("value", "yes");
  });
});

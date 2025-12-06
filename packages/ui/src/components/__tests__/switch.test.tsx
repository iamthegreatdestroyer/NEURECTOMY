import * as React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Switch } from "../switch";

describe("Switch", () => {
  it("should render unchecked by default", () => {
    render(<Switch aria-label="Toggle feature" />);
    const switchEl = screen.getByRole("switch");
    expect(switchEl).toBeInTheDocument();
    expect(switchEl).toHaveAttribute("data-state", "unchecked");
  });

  it("should render checked when defaultChecked is true", () => {
    render(<Switch defaultChecked aria-label="Toggle feature" />);
    const switchEl = screen.getByRole("switch");
    expect(switchEl).toHaveAttribute("data-state", "checked");
  });

  it("should toggle on click", async () => {
    const user = userEvent.setup();
    render(<Switch aria-label="Toggle feature" />);

    const switchEl = screen.getByRole("switch");
    expect(switchEl).toHaveAttribute("data-state", "unchecked");

    await user.click(switchEl);
    expect(switchEl).toHaveAttribute("data-state", "checked");

    await user.click(switchEl);
    expect(switchEl).toHaveAttribute("data-state", "unchecked");
  });

  it("should call onCheckedChange when toggled", async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(<Switch onCheckedChange={onCheckedChange} aria-label="Toggle" />);

    await user.click(screen.getByRole("switch"));

    expect(onCheckedChange).toHaveBeenCalledTimes(1);
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it("should be disabled when disabled prop is true", () => {
    render(<Switch disabled aria-label="Disabled switch" />);
    const switchEl = screen.getByRole("switch");
    expect(switchEl).toBeDisabled();
  });

  it("should not toggle when disabled", async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(
      <Switch disabled onCheckedChange={onCheckedChange} aria-label="Toggle" />
    );

    await user.click(screen.getByRole("switch"));
    expect(onCheckedChange).not.toHaveBeenCalled();
  });

  it("should apply custom className", () => {
    render(<Switch className="custom-class" aria-label="Toggle" />);
    expect(screen.getByRole("switch")).toHaveClass("custom-class");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLButtonElement>();
    render(<Switch ref={ref} aria-label="Toggle" />);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  it("should support controlled checked state", () => {
    const { rerender } = render(<Switch checked={false} aria-label="Toggle" />);
    expect(screen.getByRole("switch")).toHaveAttribute(
      "data-state",
      "unchecked"
    );

    rerender(<Switch checked={true} aria-label="Toggle" />);
    expect(screen.getByRole("switch")).toHaveAttribute("data-state", "checked");
  });

  it("should have correct base styling", () => {
    render(<Switch aria-label="Toggle" />);
    const switchEl = screen.getByRole("switch");
    expect(switchEl).toHaveClass("inline-flex");
    expect(switchEl).toHaveClass("h-6");
    expect(switchEl).toHaveClass("w-11");
    expect(switchEl).toHaveClass("rounded-full");
  });

  it("should be accessible via keyboard", async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(<Switch onCheckedChange={onCheckedChange} aria-label="Toggle" />);

    const switchEl = screen.getByRole("switch");
    switchEl.focus();
    expect(switchEl).toHaveFocus();

    await user.keyboard(" ");
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it("should accept aria-describedby", () => {
    render(
      <>
        <Switch aria-label="Dark mode" aria-describedby="dark-mode-help" />
        <span id="dark-mode-help">Enable dark mode for the application</span>
      </>
    );
    expect(screen.getByRole("switch")).toHaveAttribute(
      "aria-describedby",
      "dark-mode-help"
    );
  });

  it("should support required attribute", () => {
    render(<Switch required aria-label="Required switch" />);
    expect(screen.getByRole("switch")).toBeRequired();
  });

  it("should support name attribute for forms", () => {
    render(<Switch name="darkMode" aria-label="Dark Mode" />);
    expect(screen.getByRole("switch")).toHaveAttribute("name", "darkMode");
  });

  it("should support value attribute for forms", () => {
    render(<Switch value="enabled" aria-label="Feature" />);
    expect(screen.getByRole("switch")).toHaveAttribute("value", "enabled");
  });

  it("should render thumb element", () => {
    const { container } = render(<Switch aria-label="Toggle" />);
    const thumb = container.querySelector("[data-state]");
    expect(thumb).toBeInTheDocument();
  });

  it("should have visual feedback on focus", () => {
    render(<Switch aria-label="Toggle" />);
    const switchEl = screen.getByRole("switch");
    expect(switchEl).toHaveClass("focus-visible:ring-2");
    expect(switchEl).toHaveClass("focus-visible:ring-ring");
  });
});

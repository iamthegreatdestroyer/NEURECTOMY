import * as React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "../select";

const renderSelect = (defaultValue?: string) => {
  return render(
    <Select defaultValue={defaultValue}>
      <SelectTrigger aria-label="Select option">
        <SelectValue placeholder="Select an option" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="option1">Option 1</SelectItem>
        <SelectItem value="option2">Option 2</SelectItem>
        <SelectItem value="option3">Option 3</SelectItem>
      </SelectContent>
    </Select>
  );
};

describe("Select", () => {
  it("should render trigger button", () => {
    renderSelect();
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should show placeholder when no value selected", () => {
    renderSelect();
    expect(screen.getByText("Select an option")).toBeInTheDocument();
  });

  it("should show selected value", () => {
    renderSelect("option1");
    expect(screen.getByText("Option 1")).toBeInTheDocument();
  });

  it("should open dropdown on click", async () => {
    const user = userEvent.setup();
    renderSelect();

    await user.click(screen.getByRole("combobox"));

    await waitFor(() => {
      expect(screen.getByRole("listbox")).toBeInTheDocument();
    });
  });

  it("should display all options when opened", async () => {
    const user = userEvent.setup();
    renderSelect();

    await user.click(screen.getByRole("combobox"));

    await waitFor(() => {
      expect(
        screen.getByRole("option", { name: "Option 1" })
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "Option 2" })
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "Option 3" })
      ).toBeInTheDocument();
    });
  });

  it("should select option on click", async () => {
    const user = userEvent.setup();
    renderSelect();

    await user.click(screen.getByRole("combobox"));
    await waitFor(() => {
      expect(screen.getByRole("listbox")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("option", { name: "Option 2" }));

    await waitFor(() => {
      expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    });
    expect(screen.getByText("Option 2")).toBeInTheDocument();
  });

  it("should call onValueChange when selection changes", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();

    render(
      <Select onValueChange={onValueChange}>
        <SelectTrigger>
          <SelectValue placeholder="Select" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a">A</SelectItem>
          <SelectItem value="b">B</SelectItem>
        </SelectContent>
      </Select>
    );

    await user.click(screen.getByRole("combobox"));
    await waitFor(() => {
      expect(screen.getByRole("listbox")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("option", { name: "B" }));

    expect(onValueChange).toHaveBeenCalledWith("b");
  });

  it("should close on escape key", async () => {
    const user = userEvent.setup();
    renderSelect();

    await user.click(screen.getByRole("combobox"));
    await waitFor(() => {
      expect(screen.getByRole("listbox")).toBeInTheDocument();
    });

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    });
  });

  it("should support keyboard navigation", async () => {
    const user = userEvent.setup();
    renderSelect();

    const trigger = screen.getByRole("combobox");
    trigger.focus();

    await user.keyboard("{ArrowDown}");

    await waitFor(() => {
      expect(screen.getByRole("listbox")).toBeInTheDocument();
    });
  });
});

describe("SelectTrigger", () => {
  it("should be disabled when disabled prop is true", () => {
    render(
      <Select>
        <SelectTrigger disabled>
          <SelectValue placeholder="Select" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a">A</SelectItem>
        </SelectContent>
      </Select>
    );

    expect(screen.getByRole("combobox")).toBeDisabled();
  });

  it("should apply custom className", () => {
    render(
      <Select>
        <SelectTrigger className="custom-trigger">
          <SelectValue placeholder="Select" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a">A</SelectItem>
        </SelectContent>
      </Select>
    );

    expect(screen.getByRole("combobox")).toHaveClass("custom-trigger");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLButtonElement>();
    render(
      <Select>
        <SelectTrigger ref={ref}>
          <SelectValue placeholder="Select" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a">A</SelectItem>
        </SelectContent>
      </Select>
    );

    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  it("should have correct aria attributes", () => {
    renderSelect();
    const trigger = screen.getByRole("combobox");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
  });
});

describe("SelectItem", () => {
  it("should apply custom className", async () => {
    const user = userEvent.setup();
    render(
      <Select>
        <SelectTrigger>
          <SelectValue placeholder="Select" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a" className="custom-item">
            A
          </SelectItem>
        </SelectContent>
      </Select>
    );

    await user.click(screen.getByRole("combobox"));

    await waitFor(() => {
      expect(screen.getByRole("option")).toHaveClass("custom-item");
    });
  });

  it("should be disabled when disabled prop is true", async () => {
    const user = userEvent.setup();
    render(
      <Select>
        <SelectTrigger>
          <SelectValue placeholder="Select" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a" disabled>
            Disabled
          </SelectItem>
          <SelectItem value="b">Enabled</SelectItem>
        </SelectContent>
      </Select>
    );

    await user.click(screen.getByRole("combobox"));

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Disabled" })).toHaveAttribute(
        "data-disabled"
      );
    });
  });
});

describe("Select Controlled", () => {
  it("should support controlled value", async () => {
    const { rerender } = render(
      <Select value="option1">
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="option1">Option 1</SelectItem>
          <SelectItem value="option2">Option 2</SelectItem>
        </SelectContent>
      </Select>
    );

    expect(screen.getByText("Option 1")).toBeInTheDocument();

    rerender(
      <Select value="option2">
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="option1">Option 1</SelectItem>
          <SelectItem value="option2">Option 2</SelectItem>
        </SelectContent>
      </Select>
    );

    expect(screen.getByText("Option 2")).toBeInTheDocument();
  });
});

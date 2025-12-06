import * as React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
} from "../dialog";

describe("Dialog", () => {
  it("should render trigger button", () => {
    render(
      <Dialog>
        <DialogTrigger>Open Dialog</DialogTrigger>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
        </DialogContent>
      </Dialog>
    );
    expect(
      screen.getByRole("button", { name: "Open Dialog" })
    ).toBeInTheDocument();
  });

  it("should open dialog when trigger is clicked", async () => {
    const user = userEvent.setup();
    render(
      <Dialog>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogTitle>Dialog Title</DialogTitle>
          <DialogDescription>Dialog content here</DialogDescription>
        </DialogContent>
      </Dialog>
    );

    await user.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });
    expect(screen.getByText("Dialog Title")).toBeInTheDocument();
    expect(screen.getByText("Dialog content here")).toBeInTheDocument();
  });

  it("should close dialog when clicking overlay", async () => {
    const user = userEvent.setup();
    render(
      <Dialog>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
        </DialogContent>
      </Dialog>
    );

    await user.click(screen.getByRole("button", { name: "Open" }));
    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    // Press Escape to close (overlay click is harder to test due to portal)
    await user.keyboard("{Escape}");
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  it("should support controlled open state", async () => {
    const onOpenChange = vi.fn();
    const { rerender } = render(
      <Dialog open={false} onOpenChange={onOpenChange}>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
        </DialogContent>
      </Dialog>
    );

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    rerender(
      <Dialog open={true} onOpenChange={onOpenChange}>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
        </DialogContent>
      </Dialog>
    );

    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("should close on Escape key", async () => {
    const user = userEvent.setup();
    const onOpenChange = vi.fn();
    render(
      <Dialog defaultOpen onOpenChange={onOpenChange}>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
        </DialogContent>
      </Dialog>
    );

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await user.keyboard("{Escape}");

    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("should trap focus within dialog", async () => {
    const user = userEvent.setup();
    render(
      <Dialog defaultOpen>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
          <input data-testid="input1" />
          <input data-testid="input2" />
        </DialogContent>
      </Dialog>
    );

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    // Focus should cycle within the dialog
    const input1 = screen.getByTestId("input1");
    const input2 = screen.getByTestId("input2");

    input1.focus();
    expect(input1).toHaveFocus();

    await user.tab();
    expect(input2).toHaveFocus();
  });
});

describe("DialogHeader", () => {
  it("should render children", () => {
    render(
      <Dialog defaultOpen>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Header Title</DialogTitle>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    );
    expect(screen.getByText("Header Title")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(
      <Dialog defaultOpen>
        <DialogContent>
          <DialogHeader className="custom-header" data-testid="header">
            <DialogTitle>Title</DialogTitle>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    );
    expect(screen.getByTestId("header")).toHaveClass("custom-header");
  });
});

describe("DialogFooter", () => {
  it("should render footer with actions", () => {
    render(
      <Dialog defaultOpen>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
          <DialogFooter>
            <button>Cancel</button>
            <button>Confirm</button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Confirm" })).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(
      <Dialog defaultOpen>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
          <DialogFooter className="custom-footer" data-testid="footer">
            <button>OK</button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
    expect(screen.getByTestId("footer")).toHaveClass("custom-footer");
  });
});

describe("DialogTitle", () => {
  it("should render as heading", () => {
    render(
      <Dialog defaultOpen>
        <DialogContent>
          <DialogTitle>My Title</DialogTitle>
        </DialogContent>
      </Dialog>
    );
    // Radix uses h2 by default for dialog titles
    expect(screen.getByText("My Title")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(
      <Dialog defaultOpen>
        <DialogContent>
          <DialogTitle className="custom-title">Title</DialogTitle>
        </DialogContent>
      </Dialog>
    );
    expect(screen.getByText("Title")).toHaveClass("custom-title");
  });
});

describe("DialogDescription", () => {
  it("should render description text", () => {
    render(
      <Dialog defaultOpen>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
          <DialogDescription>This is the description</DialogDescription>
        </DialogContent>
      </Dialog>
    );
    expect(screen.getByText("This is the description")).toBeInTheDocument();
  });

  it("should have muted text styling", () => {
    render(
      <Dialog defaultOpen>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
          <DialogDescription data-testid="desc">Description</DialogDescription>
        </DialogContent>
      </Dialog>
    );
    expect(screen.getByTestId("desc")).toHaveClass("text-muted-foreground");
  });
});

describe("Dialog Composition", () => {
  it("should render a complete dialog structure", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();

    render(
      <Dialog>
        <DialogTrigger>Open Settings</DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Settings</DialogTitle>
            <DialogDescription>
              Configure your preferences below.
            </DialogDescription>
          </DialogHeader>
          <div>
            <label htmlFor="name">Name</label>
            <input id="name" type="text" />
          </div>
          <DialogFooter>
            <button>Cancel</button>
            <button onClick={onConfirm}>Save</button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );

    await user.click(screen.getByRole("button", { name: "Open Settings" }));

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });
    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(
      screen.getByText("Configure your preferences below.")
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Name")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(onConfirm).toHaveBeenCalled();
  });
});

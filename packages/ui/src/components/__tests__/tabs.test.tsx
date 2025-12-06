import * as React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../tabs";

describe("Tabs", () => {
  const renderTabs = (defaultValue = "tab1") => {
    return render(
      <Tabs defaultValue={defaultValue}>
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          <TabsTrigger value="tab2">Tab 2</TabsTrigger>
          <TabsTrigger value="tab3">Tab 3</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
        <TabsContent value="tab3">Content 3</TabsContent>
      </Tabs>
    );
  };

  it("should render all tab triggers", () => {
    renderTabs();
    expect(screen.getByRole("tab", { name: "Tab 1" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Tab 2" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Tab 3" })).toBeInTheDocument();
  });

  it("should show default tab content", () => {
    renderTabs("tab1");
    expect(screen.getByText("Content 1")).toBeInTheDocument();
    expect(screen.queryByText("Content 2")).not.toBeInTheDocument();
  });

  it("should switch tabs on click", async () => {
    const user = userEvent.setup();
    renderTabs("tab1");

    expect(screen.getByText("Content 1")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Tab 2" }));
    expect(screen.queryByText("Content 1")).not.toBeInTheDocument();
    expect(screen.getByText("Content 2")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Tab 3" }));
    expect(screen.queryByText("Content 2")).not.toBeInTheDocument();
    expect(screen.getByText("Content 3")).toBeInTheDocument();
  });

  it("should indicate active tab with aria-selected", async () => {
    const user = userEvent.setup();
    renderTabs("tab1");

    const tab1 = screen.getByRole("tab", { name: "Tab 1" });
    const tab2 = screen.getByRole("tab", { name: "Tab 2" });

    expect(tab1).toHaveAttribute("aria-selected", "true");
    expect(tab2).toHaveAttribute("aria-selected", "false");

    await user.click(tab2);

    expect(tab1).toHaveAttribute("aria-selected", "false");
    expect(tab2).toHaveAttribute("aria-selected", "true");
  });

  it("should support keyboard navigation", async () => {
    const user = userEvent.setup();
    renderTabs("tab1");

    const tab1 = screen.getByRole("tab", { name: "Tab 1" });
    tab1.focus();
    expect(tab1).toHaveFocus();

    // Arrow right should move to next tab
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Tab 2" })).toHaveFocus();

    // Arrow right again
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Tab 3" })).toHaveFocus();

    // Arrow left should go back
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "Tab 2" })).toHaveFocus();
  });

  it("should support controlled value", async () => {
    const { rerender } = render(
      <Tabs value="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          <TabsTrigger value="tab2">Tab 2</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
      </Tabs>
    );

    expect(screen.getByText("Content 1")).toBeInTheDocument();

    rerender(
      <Tabs value="tab2">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          <TabsTrigger value="tab2">Tab 2</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
      </Tabs>
    );

    expect(screen.getByText("Content 2")).toBeInTheDocument();
  });
});

describe("TabsList", () => {
  it("should have tablist role", () => {
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content</TabsContent>
      </Tabs>
    );
    expect(screen.getByRole("tablist")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(
      <Tabs defaultValue="tab1">
        <TabsList className="custom-list" data-testid="tablist">
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content</TabsContent>
      </Tabs>
    );
    expect(screen.getByTestId("tablist")).toHaveClass("custom-list");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(
      <Tabs defaultValue="tab1">
        <TabsList ref={ref}>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content</TabsContent>
      </Tabs>
    );
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });
});

describe("TabsTrigger", () => {
  it("should have tab role", () => {
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content</TabsContent>
      </Tabs>
    );
    expect(screen.getByRole("tab")).toBeInTheDocument();
  });

  it("should be disabled when disabled prop is true", () => {
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          <TabsTrigger value="tab2" disabled>
            Tab 2
          </TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
      </Tabs>
    );
    expect(screen.getByRole("tab", { name: "Tab 2" })).toBeDisabled();
  });

  it("should apply custom className", () => {
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1" className="custom-trigger">
            Tab 1
          </TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content</TabsContent>
      </Tabs>
    );
    expect(screen.getByRole("tab")).toHaveClass("custom-trigger");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLButtonElement>();
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1" ref={ref}>
            Tab 1
          </TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content</TabsContent>
      </Tabs>
    );
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });
});

describe("TabsContent", () => {
  it("should have tabpanel role", () => {
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
      </Tabs>
    );
    expect(screen.getByRole("tabpanel")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1" className="custom-content">
          Content
        </TabsContent>
      </Tabs>
    );
    expect(screen.getByRole("tabpanel")).toHaveClass("custom-content");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1" ref={ref}>
          Content
        </TabsContent>
      </Tabs>
    );
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it("should be focusable for accessibility", async () => {
    const user = userEvent.setup();
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content with focus</TabsContent>
      </Tabs>
    );

    // Tab into the content panel
    await user.tab();
    await user.tab();
    expect(screen.getByRole("tabpanel")).toHaveFocus();
  });
});

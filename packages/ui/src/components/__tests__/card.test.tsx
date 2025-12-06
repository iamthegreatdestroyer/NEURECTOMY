import * as React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "../card";

describe("Card", () => {
  it("should render Card with children", () => {
    render(<Card>Card Content</Card>);
    expect(screen.getByText("Card Content")).toBeInTheDocument();
  });

  it("should apply custom className to Card", () => {
    render(<Card className="custom-class" data-testid="card" />);
    expect(screen.getByTestId("card")).toHaveClass("custom-class");
  });

  it("should forward ref to Card", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<Card ref={ref}>Test</Card>);
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it("should have correct base styling", () => {
    render(<Card data-testid="card" />);
    const card = screen.getByTestId("card");
    expect(card).toHaveClass("rounded-lg");
    expect(card).toHaveClass("border");
    expect(card).toHaveClass("shadow-sm");
  });
});

describe("CardHeader", () => {
  it("should render CardHeader with children", () => {
    render(<CardHeader>Header Content</CardHeader>);
    expect(screen.getByText("Header Content")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(<CardHeader className="custom-header" data-testid="header" />);
    expect(screen.getByTestId("header")).toHaveClass("custom-header");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<CardHeader ref={ref}>Test</CardHeader>);
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it("should have padding styles", () => {
    render(<CardHeader data-testid="header" />);
    expect(screen.getByTestId("header")).toHaveClass("p-6");
  });
});

describe("CardTitle", () => {
  it("should render CardTitle with text", () => {
    render(<CardTitle>My Title</CardTitle>);
    expect(screen.getByRole("heading", { level: 3 })).toHaveTextContent(
      "My Title"
    );
  });

  it("should apply custom className", () => {
    render(<CardTitle className="custom-title">Title</CardTitle>);
    expect(screen.getByRole("heading")).toHaveClass("custom-title");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLParagraphElement>();
    render(<CardTitle ref={ref}>Test</CardTitle>);
    expect(ref.current).toBeInstanceOf(HTMLHeadingElement);
  });

  it("should have typography styles", () => {
    render(<CardTitle>Title</CardTitle>);
    const title = screen.getByRole("heading");
    expect(title).toHaveClass("text-2xl");
    expect(title).toHaveClass("font-semibold");
  });
});

describe("CardDescription", () => {
  it("should render CardDescription with text", () => {
    render(<CardDescription>Description text</CardDescription>);
    expect(screen.getByText("Description text")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(<CardDescription className="custom-desc" data-testid="desc" />);
    expect(screen.getByTestId("desc")).toHaveClass("custom-desc");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLParagraphElement>();
    render(<CardDescription ref={ref}>Test</CardDescription>);
    expect(ref.current).toBeInstanceOf(HTMLParagraphElement);
  });

  it("should have muted text styling", () => {
    render(<CardDescription data-testid="desc">Text</CardDescription>);
    expect(screen.getByTestId("desc")).toHaveClass("text-sm");
    expect(screen.getByTestId("desc")).toHaveClass("text-muted-foreground");
  });
});

describe("CardContent", () => {
  it("should render CardContent with children", () => {
    render(<CardContent>Content here</CardContent>);
    expect(screen.getByText("Content here")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(<CardContent className="custom-content" data-testid="content" />);
    expect(screen.getByTestId("content")).toHaveClass("custom-content");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<CardContent ref={ref}>Test</CardContent>);
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it("should have correct padding", () => {
    render(<CardContent data-testid="content" />);
    expect(screen.getByTestId("content")).toHaveClass("p-6");
    expect(screen.getByTestId("content")).toHaveClass("pt-0");
  });
});

describe("CardFooter", () => {
  it("should render CardFooter with children", () => {
    render(<CardFooter>Footer content</CardFooter>);
    expect(screen.getByText("Footer content")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(<CardFooter className="custom-footer" data-testid="footer" />);
    expect(screen.getByTestId("footer")).toHaveClass("custom-footer");
  });

  it("should forward ref", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<CardFooter ref={ref}>Test</CardFooter>);
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it("should have flex layout", () => {
    render(<CardFooter data-testid="footer" />);
    expect(screen.getByTestId("footer")).toHaveClass("flex");
    expect(screen.getByTestId("footer")).toHaveClass("items-center");
  });
});

describe("Card Composition", () => {
  it("should render a complete card structure", () => {
    render(
      <Card data-testid="card">
        <CardHeader>
          <CardTitle>Card Title</CardTitle>
          <CardDescription>Card description goes here</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Main content area</p>
        </CardContent>
        <CardFooter>
          <button>Action</button>
        </CardFooter>
      </Card>
    );

    expect(screen.getByTestId("card")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Card Title" })
    ).toBeInTheDocument();
    expect(screen.getByText("Card description goes here")).toBeInTheDocument();
    expect(screen.getByText("Main content area")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Action" })).toBeInTheDocument();
  });
});

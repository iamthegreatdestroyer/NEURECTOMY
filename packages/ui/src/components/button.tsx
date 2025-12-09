import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../utils/cn";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline:
          "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export type ButtonVariant =
  | "default"
  | "destructive"
  | "outline"
  | "secondary"
  | "ghost"
  | "link";
export type ButtonSize = "default" | "sm" | "lg" | "icon";

export interface ButtonProps
  extends
    React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  /** Shows loading state with aria-busy */
  isLoading?: boolean;
  /** Accessible label for the button (use when button has no visible text) */
  "aria-label"?: string;
  /** ID of element that describes this button */
  "aria-describedby"?: string;
  /** Indicates the button's pressed state (for toggle buttons) */
  "aria-pressed"?: boolean | "mixed";
  /** Indicates the button controls an expandable element */
  "aria-expanded"?: boolean;
  /** Indicates the button activates a popup (menu, listbox, dialog, etc.) */
  "aria-haspopup"?: boolean | "menu" | "listbox" | "tree" | "grid" | "dialog";
  /** ID of the element that this button controls */
  "aria-controls"?: string;
  /** Indicates if this is the currently active item in a set */
  "aria-current"?: boolean | "page" | "step" | "location" | "date" | "time";
  /** Indicates the button is busy/loading */
  "aria-busy"?: boolean;
  /** Indicates the button is disabled */
  "aria-disabled"?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, isLoading, disabled, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={disabled || isLoading}
        aria-busy={isLoading || props["aria-busy"]}
        aria-disabled={disabled || isLoading || props["aria-disabled"]}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };

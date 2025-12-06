import * as React from "react";
import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import { cn } from "../utils/cn";

/**
 * Enhanced ARIA props for Checkbox accessibility
 * Radix UI Checkbox already provides:
 * - role="checkbox"
 * - aria-checked state management
 */
export interface CheckboxAriaProps {
  /** Accessible label for the checkbox */
  "aria-label"?: string;
  /** ID of element that labels this checkbox */
  "aria-labelledby"?: string;
  /** ID of element that describes this checkbox */
  "aria-describedby"?: string;
  /** Indicates checkbox has an error */
  "aria-invalid"?: boolean;
}

export interface CheckboxProps
  extends
    React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>,
    CheckboxAriaProps {}

const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  CheckboxProps
>(({ className, ...props }, ref) => (
  <CheckboxPrimitive.Root
    ref={ref}
    className={cn(
      "peer h-4 w-4 shrink-0 rounded-sm border border-primary ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground",
      className
    )}
    {...props}
  >
    <CheckboxPrimitive.Indicator
      className={cn("flex items-center justify-center text-current")}
    >
      ✓
    </CheckboxPrimitive.Indicator>
  </CheckboxPrimitive.Root>
));
Checkbox.displayName = CheckboxPrimitive.Root.displayName;

export { Checkbox };

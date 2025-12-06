import * as React from "react";
import { cn } from "../utils/cn";

/**
 * Enhanced ARIA props for Input accessibility
 */
export interface InputAriaProps {
  /** Accessible label for the input */
  "aria-label"?: string;
  /** ID of element that labels this input */
  "aria-labelledby"?: string;
  /** ID of element that describes this input */
  "aria-describedby"?: string;
  /** Indicates input has an error */
  "aria-invalid"?: boolean;
  /** ID of error message element */
  "aria-errormessage"?: string;
  /** Indicates input is required */
  "aria-required"?: boolean;
  /** Indicates the input is read-only */
  "aria-readonly"?: boolean;
}

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement>, InputAriaProps {
  /** Visual indicator for error state */
  hasError?: boolean;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, hasError, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          hasError && "border-destructive focus-visible:ring-destructive",
          className
        )}
        ref={ref}
        aria-invalid={hasError || props["aria-invalid"]}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };

import * as React from "react";
import { cn } from "../utils/cn";

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Visual indicator for error state */
  hasError?: boolean;
  /** Accessible label for the textarea */
  "aria-label"?: string;
  /** ID of element that labels this textarea */
  "aria-labelledby"?: string;
  /** ID of element that describes this textarea */
  "aria-describedby"?: string;
  /** Indicates textarea has an error */
  "aria-invalid"?: boolean | "false" | "true" | "grammar" | "spelling";
  /** ID of error message element */
  "aria-errormessage"?: string;
  /** Indicates the textarea is required */
  "aria-required"?: boolean | "false" | "true";
  /** Indicates the textarea is read-only */
  "aria-readonly"?: boolean | "false" | "true";
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, hasError, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
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
Textarea.displayName = "Textarea";

export { Textarea };

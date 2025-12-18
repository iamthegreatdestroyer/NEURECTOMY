/**
 * Type declarations for @neurectomy/ui package
 *
 * This file provides type definitions for the shared UI component library.
 */

declare module "@neurectomy/ui" {
  import { ComponentProps } from "react";

  export const Button: React.FC<
    ComponentProps<"button"> & {
      variant?:
        | "default"
        | "destructive"
        | "outline"
        | "secondary"
        | "ghost"
        | "link";
      size?: "default" | "sm" | "lg" | "icon";
    }
  >;

  export const Input: React.FC<ComponentProps<"input">>;

  export const Label: React.FC<ComponentProps<"label">>;

  export const Textarea: React.FC<ComponentProps<"textarea">>;

  export const Progress: React.FC<{
    value?: number;
    max?: number;
    className?: string;
  }>;

  export const Alert: React.FC<
    ComponentProps<"div"> & {
      variant?: "default" | "destructive";
    }
  >;

  export const AlertDescription: React.FC<ComponentProps<"div">>;

  export const Tabs: React.FC<
    ComponentProps<"div"> & {
      value?: string;
      onValueChange?: (value: string) => void;
      defaultValue?: string;
    }
  >;

  export const TabsContent: React.FC<
    ComponentProps<"div"> & {
      value: string;
    }
  >;

  export const TabsList: React.FC<ComponentProps<"div">>;

  export const TabsTrigger: React.FC<
    ComponentProps<"button"> & {
      value: string;
    }
  >;

  export const Badge: React.FC<
    ComponentProps<"div"> & {
      variant?: "default" | "secondary" | "destructive" | "outline";
    }
  >;

  export const Select: React.FC<{
    value?: string;
    onValueChange?: (value: string) => void;
    children?: React.ReactNode;
  }>;

  export const SelectContent: React.FC<ComponentProps<"div">>;

  export const SelectItem: React.FC<
    ComponentProps<"div"> & {
      value: string;
    }
  >;

  export const SelectTrigger: React.FC<ComponentProps<"button">>;

  export const SelectValue: React.FC<{
    placeholder?: string;
  }>;

  export const Checkbox: React.FC<
    ComponentProps<"input"> & {
      checked?: boolean;
      onCheckedChange?: (checked: boolean) => void;
    }
  >;

  export const Card: React.FC<ComponentProps<"div">>;
  export const CardHeader: React.FC<ComponentProps<"div">>;
  export const CardTitle: React.FC<ComponentProps<"h3">>;
  export const CardDescription: React.FC<ComponentProps<"p">>;
  export const CardContent: React.FC<ComponentProps<"div">>;
  export const CardFooter: React.FC<ComponentProps<"div">>;
}

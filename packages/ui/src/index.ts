/**
 * @neurectomy/ui - Shared UI Component Library
 *
 * Provides a comprehensive set of accessible, themeable UI components
 * built on Radix UI primitives with Tailwind CSS styling.
 */

// Re-export utility functions
export { cn } from "./utils/cn";

// Export component types
export type {
  ButtonProps,
  ButtonVariant,
  ButtonSize,
} from "./components/button";
export type { InputProps } from "./components/input";
export type { CardProps } from "./components/card";
export type { DialogProps } from "./components/dialog";
export type { ToastProps } from "./components/toast";

// Export components
export { Button } from "./components/button";
export { Input } from "./components/input";
export {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "./components/card";
export {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
} from "./components/dialog";
export { Toast, ToastProvider, ToastViewport } from "./components/toast";
export { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/tabs";
export {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "./components/select";
export {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
} from "./components/tooltip";
export { ScrollArea, ScrollBar } from "./components/scroll-area";
export { Separator } from "./components/separator";
export { Label } from "./components/label";
export { Checkbox } from "./components/checkbox";
export { Switch } from "./components/switch";
export { Slider } from "./components/slider";
export { Progress } from "./components/progress";
export { Avatar, AvatarImage, AvatarFallback } from "./components/avatar";
export { Badge } from "./components/badge";
export {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "./components/accordion";
export {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "./components/dropdown-menu";
export {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
} from "./components/context-menu";

// Visualization components
export {
  InstancedNodeGeometry,
  LOD_CONFIGS,
  DEFAULT_NODE_THEME,
  type NodeData,
  type InstancedNodeGeometryProps,
  type LODLevel,
  type NodeTheme,
} from "./components/visualization";

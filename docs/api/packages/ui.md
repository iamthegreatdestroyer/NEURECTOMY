# @neurectomy/ui API Reference

Shared UI component library built on Radix UI primitives with Tailwind CSS styling.

## Installation

```bash
pnpm add @neurectomy/ui
```

## Utilities

### cn(...)

Merges Tailwind CSS classes with conflict resolution using `clsx` and `tailwind-merge`.

```typescript
import { cn } from "@neurectomy/ui";

// Basic usage
cn("px-2 py-1", "p-4");
// => 'p-4' (p-4 overrides px-2 py-1)

// Conditional classes
cn("base", condition && "conditional", "always");

// Object syntax
cn({ "bg-red-500": isError, "bg-green-500": !isError });

// Array syntax
cn(["class1", "class2", condition && "class3"]);
```

**Parameters:**

- `...inputs`: `ClassValue[]` - Class names, objects, or arrays

**Returns:** `string` - Merged class string

---

## Components

### Button

A polymorphic button component with multiple variants and sizes.

```typescript
import { Button } from '@neurectomy/ui';

// Default button
<Button>Click me</Button>

// With variant
<Button variant="destructive">Delete</Button>

// With size
<Button size="lg">Large Button</Button>

// Icon button
<Button variant="ghost" size="icon">
  <Icon />
</Button>

// Disabled state
<Button disabled>Disabled</Button>

// With custom className
<Button className="w-full">Full Width</Button>
```

#### Props

| Prop        | Type            | Default     | Description               |
| ----------- | --------------- | ----------- | ------------------------- |
| `variant`   | `ButtonVariant` | `'default'` | Visual style variant      |
| `size`      | `ButtonSize`    | `'default'` | Button size               |
| `asChild`   | `boolean`       | `false`     | Render as child component |
| `disabled`  | `boolean`       | `false`     | Disabled state            |
| `className` | `string`        | -           | Additional CSS classes    |

#### Variants

| Variant       | Description              |
| ------------- | ------------------------ |
| `default`     | Primary action button    |
| `destructive` | Dangerous/delete actions |
| `outline`     | Bordered button          |
| `secondary`   | Secondary actions        |
| `ghost`       | Minimal styling          |
| `link`        | Styled as a link         |

#### Sizes

| Size      | Description                    |
| --------- | ------------------------------ |
| `default` | Standard size (h-10)           |
| `sm`      | Small (h-9)                    |
| `lg`      | Large (h-11)                   |
| `icon`    | Square icon button (h-10 w-10) |

---

### Input

Text input with consistent styling.

```typescript
import { Input } from '@neurectomy/ui';

<Input placeholder="Enter text..." />
<Input type="email" placeholder="Email" />
<Input disabled value="Read only" />
```

#### Props

Extends `React.InputHTMLAttributes<HTMLInputElement>`.

---

### Card

Container component for grouped content.

```typescript
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from '@neurectomy/ui';

<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
    <CardDescription>Card description text</CardDescription>
  </CardHeader>
  <CardContent>
    Main content goes here
  </CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>
```

#### Components

| Component         | Description          |
| ----------------- | -------------------- |
| `Card`            | Root container       |
| `CardHeader`      | Header section       |
| `CardTitle`       | Title text           |
| `CardDescription` | Subtitle/description |
| `CardContent`     | Main content area    |
| `CardFooter`      | Footer with actions  |

---

### Dialog

Modal dialog component built on Radix UI Dialog.

```typescript
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@neurectomy/ui';

<Dialog>
  <DialogTrigger asChild>
    <Button>Open Dialog</Button>
  </DialogTrigger>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Dialog Title</DialogTitle>
      <DialogDescription>
        Description text here
      </DialogDescription>
    </DialogHeader>
    <div>Dialog content</div>
    <DialogFooter>
      <Button>Confirm</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
```

---

### Tabs

Tabbed interface component.

```typescript
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@neurectomy/ui';

<Tabs defaultValue="tab1">
  <TabsList>
    <TabsTrigger value="tab1">Tab 1</TabsTrigger>
    <TabsTrigger value="tab2">Tab 2</TabsTrigger>
  </TabsList>
  <TabsContent value="tab1">Content 1</TabsContent>
  <TabsContent value="tab2">Content 2</TabsContent>
</Tabs>
```

---

### Select

Dropdown select component.

```typescript
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from '@neurectomy/ui';

<Select>
  <SelectTrigger>
    <SelectValue placeholder="Select option" />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="option1">Option 1</SelectItem>
    <SelectItem value="option2">Option 2</SelectItem>
  </SelectContent>
</Select>
```

---

### Tooltip

Hover tooltip component.

```typescript
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
} from '@neurectomy/ui';

<TooltipProvider>
  <Tooltip>
    <TooltipTrigger>Hover me</TooltipTrigger>
    <TooltipContent>
      Tooltip content
    </TooltipContent>
  </Tooltip>
</TooltipProvider>
```

---

### Toast

Toast notification system.

```typescript
import { Toast, ToastProvider, ToastViewport } from '@neurectomy/ui';

<ToastProvider>
  <Toast>
    <ToastTitle>Notification</ToastTitle>
    <ToastDescription>Message here</ToastDescription>
  </Toast>
  <ToastViewport />
</ToastProvider>
```

---

### Form Components

#### Label

```typescript
import { Label } from '@neurectomy/ui';

<Label htmlFor="email">Email</Label>
```

#### Checkbox

```typescript
import { Checkbox } from '@neurectomy/ui';

<Checkbox checked={checked} onCheckedChange={setChecked} />
```

#### Switch

```typescript
import { Switch } from '@neurectomy/ui';

<Switch checked={enabled} onCheckedChange={setEnabled} />
```

#### Slider

```typescript
import { Slider } from '@neurectomy/ui';

<Slider value={[50]} max={100} step={1} />
```

---

### Display Components

#### Badge

```typescript
import { Badge } from '@neurectomy/ui';

<Badge>Default</Badge>
<Badge variant="secondary">Secondary</Badge>
<Badge variant="destructive">Error</Badge>
<Badge variant="outline">Outline</Badge>
```

#### Avatar

```typescript
import { Avatar, AvatarImage, AvatarFallback } from '@neurectomy/ui';

<Avatar>
  <AvatarImage src="/avatar.png" alt="User" />
  <AvatarFallback>JD</AvatarFallback>
</Avatar>
```

#### Progress

```typescript
import { Progress } from '@neurectomy/ui';

<Progress value={75} />
```

#### Separator

```typescript
import { Separator } from '@neurectomy/ui';

<Separator />
<Separator orientation="vertical" />
```

---

### Layout Components

#### ScrollArea

```typescript
import { ScrollArea, ScrollBar } from '@neurectomy/ui';

<ScrollArea className="h-[200px]">
  <div>Long content...</div>
  <ScrollBar orientation="vertical" />
</ScrollArea>
```

#### Accordion

```typescript
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from '@neurectomy/ui';

<Accordion type="single" collapsible>
  <AccordionItem value="item-1">
    <AccordionTrigger>Section 1</AccordionTrigger>
    <AccordionContent>Content 1</AccordionContent>
  </AccordionItem>
</Accordion>
```

---

### Menu Components

#### DropdownMenu

```typescript
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '@neurectomy/ui';

<DropdownMenu>
  <DropdownMenuTrigger asChild>
    <Button>Menu</Button>
  </DropdownMenuTrigger>
  <DropdownMenuContent>
    <DropdownMenuItem>Item 1</DropdownMenuItem>
    <DropdownMenuItem>Item 2</DropdownMenuItem>
  </DropdownMenuContent>
</DropdownMenu>
```

#### ContextMenu

```typescript
import {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
} from '@neurectomy/ui';

<ContextMenu>
  <ContextMenuTrigger>Right click here</ContextMenuTrigger>
  <ContextMenuContent>
    <ContextMenuItem>Action</ContextMenuItem>
  </ContextMenuContent>
</ContextMenu>
```

---

## Type Exports

```typescript
import type {
  ButtonProps,
  ButtonVariant,
  ButtonSize,
  InputProps,
  CardProps,
  DialogProps,
  ToastProps,
} from "@neurectomy/ui";
```

---

## Theming

Components use CSS variables for theming. Define these in your global CSS:

```css
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --primary: 222.2 47.4% 11.2%;
  --primary-foreground: 210 40% 98%;
  --secondary: 210 40% 96.1%;
  --secondary-foreground: 222.2 47.4% 11.2%;
  --destructive: 0 84.2% 60.2%;
  --destructive-foreground: 210 40% 98%;
  --muted: 210 40% 96.1%;
  --muted-foreground: 215.4 16.3% 46.9%;
  --accent: 210 40% 96.1%;
  --accent-foreground: 222.2 47.4% 11.2%;
  --ring: 222.2 84% 4.9%;
  --radius: 0.5rem;
}

.dark {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  /* ... dark mode variables */
}
```

---

## Peer Dependencies

- `react` >= 18.0.0
- `react-dom` >= 18.0.0
- `tailwindcss` >= 3.0.0

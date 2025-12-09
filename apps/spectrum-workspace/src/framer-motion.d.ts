import {
  HTMLMotionProps,
  MotionProps,
  AnimationProps,
  GestureHandlers,
  DraggableProps,
  LayoutProps,
} from "framer-motion";

declare module "framer-motion" {
  export interface HTMLMotionProps<T = any>
    extends
      React.HTMLAttributes<T>,
      MotionProps,
      AnimationProps,
      GestureHandlers,
      DraggableProps,
      LayoutProps {
    style?: React.CSSProperties;
  }
}

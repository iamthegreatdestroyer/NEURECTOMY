/**
 * Motion Component Wrappers for Framer Motion v10+
 *
 * Framer Motion v10+ has stricter TypeScript types that don't allow
 * className on motion.div. This module provides typed wrappers.
 *
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import { motion, HTMLMotionProps } from "framer-motion";
import { forwardRef, HTMLAttributes } from "react";

// Re-export motion with proper typing
export { motion, AnimatePresence } from "framer-motion";
export type { HTMLMotionProps, MotionProps } from "framer-motion";

// Combined type for motion components with className and all HTML attributes
type MotionDivProps = HTMLMotionProps<"div"> & HTMLAttributes<HTMLDivElement>;
type MotionSpanProps = HTMLMotionProps<"span"> &
  HTMLAttributes<HTMLSpanElement>;
type MotionButtonProps = HTMLMotionProps<"button"> &
  HTMLAttributes<HTMLButtonElement>;
type MotionUlProps = HTMLMotionProps<"ul"> & HTMLAttributes<HTMLUListElement>;
type MotionLiProps = HTMLMotionProps<"li"> & HTMLAttributes<HTMLLIElement>;

/**
 * Motion div with className support
 */
export const MotionDiv = forwardRef<HTMLDivElement, MotionDivProps>(
  (props, ref) => {
    return <motion.div ref={ref} {...props} />;
  }
);
MotionDiv.displayName = "MotionDiv";

/**
 * Motion span with className support
 */
export const MotionSpan = forwardRef<HTMLSpanElement, MotionSpanProps>(
  (props, ref) => {
    return <motion.span ref={ref} {...props} />;
  }
);
MotionSpan.displayName = "MotionSpan";

/**
 * Motion button with className support
 */
export const MotionButton = forwardRef<HTMLButtonElement, MotionButtonProps>(
  (props, ref) => {
    return <motion.button ref={ref} {...props} />;
  }
);
MotionButton.displayName = "MotionButton";

/**
 * Motion ul with className support
 */
export const MotionUl = forwardRef<HTMLUListElement, MotionUlProps>(
  (props, ref) => {
    return <motion.ul ref={ref} {...props} />;
  }
);
MotionUl.displayName = "MotionUl";

/**
 * Motion li with className support
 */
export const MotionLi = forwardRef<HTMLLIElement, MotionLiProps>(
  (props, ref) => {
    return <motion.li ref={ref} {...props} />;
  }
);
MotionLi.displayName = "MotionLi";

/**
 * MonacoEditor Component - Simplified for Desktop App
 * Uses textarea fallback to avoid Monaco Worker build issues
 */

import React from "react";

export const MonacoEditor: React.FC<{
  value?: string;
  onChange?: (value: string) => void;
  language?: string;
  readOnly?: boolean;
  height?: string | number;
}> = ({ value = "", onChange, language = "typescript", readOnly = false, height = "400px" }) => {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
      readOnly={readOnly}
      style={{
        width: "100%",
        height: typeof height === "string" ? height : `${height}px`,
        fontFamily: "'Courier New', Courier, monospace",
        fontSize: "13px",
        padding: "12px",
        border: "1px solid #333",
        borderRadius: "4px",
        backgroundColor: "#1e1e1e",
        color: "#d4d4d4",
        resize: "vertical",
      }}
      className="monaco-editor-fallback"
      spellCheck={false}
      data-language={language}
    />
  );
};

export default MonacoEditor;

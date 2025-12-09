/**
 * AI Chat Panel Component
 *
 * Right sidebar AI assistant with Continue-style context integration.
 * Supports streaming responses, markdown rendering, code blocks.
 *
 * @module @neurectomy/shell
 * @author @LINGUA @TENSOR
 */

import {
  useState,
  useRef,
  useEffect,
  useCallback,
  ReactNode,
  KeyboardEvent,
} from "react";
import { cn } from "@/lib/utils";
import {
  Send,
  Paperclip,
  AtSign,
  Sparkles,
  Bot,
  User,
  Copy,
  Check,
  RefreshCw,
  StopCircle,
  Trash2,
  ChevronDown,
  Code2,
  FileText,
  Terminal as TerminalIcon,
  Folder,
  Settings2,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export type MessageRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  context?: MessageContext[];
  codeBlocks?: CodeBlock[];
  streaming?: boolean;
}

export interface MessageContext {
  type: "file" | "code" | "terminal" | "folder" | "url";
  name: string;
  preview?: string;
  path?: string;
}

export interface CodeBlock {
  language: string;
  code: string;
  filename?: string;
}

export interface AIPanelProps {
  messages: ChatMessage[];
  onSendMessage: (content: string, context?: MessageContext[]) => void;
  onCancelStream?: () => void;
  onClearChat?: () => void;
  onRegenerateResponse?: (messageId: string) => void;
  isStreaming?: boolean;
  modelName?: string;
  className?: string;
}

// ============================================================================
// Main Component
// ============================================================================

export function AIPanel({
  messages,
  onSendMessage,
  onCancelStream,
  onClearChat,
  onRegenerateResponse,
  isStreaming = false,
  modelName = "Claude Sonnet",
  className,
}: AIPanelProps) {
  const [inputValue, setInputValue] = useState("");
  const [attachedContext, setAttachedContext] = useState<MessageContext[]>([]);
  const [showContextMenu, setShowContextMenu] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle send
  const handleSend = useCallback(() => {
    if (!inputValue.trim() || isStreaming) return;

    onSendMessage(
      inputValue.trim(),
      attachedContext.length > 0 ? attachedContext : undefined
    );
    setInputValue("");
    setAttachedContext([]);
  }, [inputValue, attachedContext, isStreaming, onSendMessage]);

  // Handle keyboard shortcuts
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }

    // @ for context menu
    if (e.key === "@") {
      setShowContextMenu(true);
    }
  };

  // Add context
  const addContext = (context: MessageContext) => {
    setAttachedContext((prev) => [...prev, context]);
    setShowContextMenu(false);
  };

  // Remove context
  const removeContext = (index: number) => {
    setAttachedContext((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className={cn("h-full flex flex-col", className)}>
      {/* Header */}
      <div className="h-10 flex items-center justify-between px-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <Sparkles size={16} className="text-primary" />
          <span className="text-sm font-medium">AI Assistant</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onClearChat}
            className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
            title="Clear Chat"
          >
            <Trash2 size={14} />
          </button>
          <button
            className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
            title="Settings"
          >
            <Settings2 size={14} />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="flex flex-col gap-4 p-3">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onRegenerate={onRegenerateResponse}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Attached Context */}
      {attachedContext.length > 0 && (
        <div className="px-3 py-2 border-t border-border">
          <div className="flex flex-wrap gap-1.5">
            {attachedContext.map((ctx, index) => (
              <ContextChip
                key={index}
                context={ctx}
                onRemove={() => removeContext(index)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-3 border-t border-border shrink-0">
        <div className="relative">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything... (@ for context)"
            rows={3}
            className={cn(
              "w-full resize-none rounded-lg border border-border",
              "bg-background px-3 py-2 pr-20 text-sm",
              "focus:outline-none focus:ring-2 focus:ring-ring",
              "placeholder:text-muted-foreground"
            )}
          />

          {/* Input Actions */}
          <div className="absolute right-2 bottom-2 flex items-center gap-1">
            <button
              onClick={() => setShowContextMenu(!showContextMenu)}
              className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
              title="Add Context (@)"
            >
              <AtSign size={16} />
            </button>
            {isStreaming ? (
              <button
                onClick={onCancelStream}
                className="p-1.5 rounded bg-destructive text-destructive-foreground hover:bg-destructive/90"
                title="Stop"
              >
                <StopCircle size={16} />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!inputValue.trim()}
                className={cn(
                  "p-1.5 rounded",
                  inputValue.trim()
                    ? "bg-primary text-primary-foreground hover:bg-primary/90"
                    : "bg-muted text-muted-foreground cursor-not-allowed"
                )}
                title="Send (Enter)"
              >
                <Send size={16} />
              </button>
            )}
          </div>

          {/* Context Menu */}
          {showContextMenu && (
            <ContextMenu
              onSelect={addContext}
              onClose={() => setShowContextMenu(false)}
            />
          )}
        </div>

        {/* Model Info */}
        <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
          <span>Model: {modelName}</span>
          <span>Shift+Enter for new line</span>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Message Bubble
// ============================================================================

interface MessageBubbleProps {
  message: ChatMessage;
  onRegenerate?: (id: string) => void;
}

function MessageBubble({ message, onRegenerate }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
      {/* Avatar */}
      <div
        className={cn(
          "w-7 h-7 rounded-full flex items-center justify-center shrink-0",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        )}
      >
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>

      {/* Content */}
      <div
        className={cn(
          "flex-1 max-w-[85%]",
          isUser ? "text-right" : "text-left"
        )}
      >
        {/* Context chips */}
        {message.context && message.context.length > 0 && (
          <div
            className={cn(
              "flex flex-wrap gap-1 mb-2",
              isUser ? "justify-end" : "justify-start"
            )}
          >
            {message.context.map((ctx, i) => (
              <ContextChip key={i} context={ctx} compact />
            ))}
          </div>
        )}

        {/* Message content */}
        <div
          className={cn(
            "inline-block rounded-lg px-3 py-2 text-sm",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted"
          )}
        >
          {message.streaming ? (
            <span className="flex items-center gap-2">
              <span>{message.content}</span>
              <span className="w-2 h-4 bg-current animate-pulse" />
            </span>
          ) : (
            <MessageContent content={message.content} />
          )}
        </div>

        {/* Actions */}
        {!isUser && !message.streaming && (
          <div className="flex items-center gap-1 mt-1">
            <button
              onClick={handleCopy}
              className="p-1 rounded hover:bg-accent text-muted-foreground"
              title="Copy"
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
            </button>
            {onRegenerate && (
              <button
                onClick={() => onRegenerate(message.id)}
                className="p-1 rounded hover:bg-accent text-muted-foreground"
                title="Regenerate"
              >
                <RefreshCw size={12} />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Message Content (with code block support)
// ============================================================================

function MessageContent({ content }: { content: string }) {
  // Simple code block parsing
  const parts = content.split(/(```[\s\S]*?```)/g);

  return (
    <div className="space-y-2">
      {parts.map((part, i) => {
        if (part.startsWith("```")) {
          const match = part.match(/```(\w*)\n?([\s\S]*?)```/);
          if (match) {
            const [, lang, code] = match;
            return (
              <CodeBlockComponent
                key={i}
                language={lang || "text"}
                code={code.trim()}
              />
            );
          }
        }
        return part ? (
          <p key={i} className="whitespace-pre-wrap">
            {part}
          </p>
        ) : null;
      })}
    </div>
  );
}

function CodeBlockComponent({
  language,
  code,
}: {
  language: string;
  code: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded overflow-hidden border border-border bg-background">
      <div className="flex items-center justify-between px-3 py-1.5 bg-muted/50 border-b border-border">
        <span className="text-xs text-muted-foreground">{language}</span>
        <button
          onClick={handleCopy}
          className="p-1 rounded hover:bg-accent text-muted-foreground"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
        </button>
      </div>
      <pre className="p-3 overflow-x-auto text-xs">
        <code>{code}</code>
      </pre>
    </div>
  );
}

// ============================================================================
// Context Components
// ============================================================================

interface ContextChipProps {
  context: MessageContext;
  onRemove?: () => void;
  compact?: boolean;
}

function ContextChip({ context, onRemove, compact }: ContextChipProps) {
  const Icon = {
    file: FileText,
    code: Code2,
    terminal: TerminalIcon,
    folder: Folder,
    url: Paperclip,
  }[context.type];

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded bg-muted text-xs",
        compact ? "px-1.5 py-0.5" : "px-2 py-1"
      )}
    >
      <Icon size={12} className="text-muted-foreground" />
      <span className="truncate max-w-[100px]">{context.name}</span>
      {onRemove && (
        <button onClick={onRemove} className="p-0.5 rounded hover:bg-accent">
          <Trash2 size={10} />
        </button>
      )}
    </div>
  );
}

interface ContextMenuProps {
  onSelect: (context: MessageContext) => void;
  onClose: () => void;
}

function ContextMenu({ onSelect, onClose }: ContextMenuProps) {
  const items: {
    type: MessageContext["type"];
    label: string;
    icon: typeof FileText;
  }[] = [
    { type: "file", label: "Current File", icon: FileText },
    { type: "code", label: "Selected Code", icon: Code2 },
    { type: "terminal", label: "Terminal Output", icon: TerminalIcon },
    { type: "folder", label: "Folder", icon: Folder },
  ];

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        className={cn(
          "absolute bottom-full left-0 mb-2 z-50",
          "bg-popover border border-border rounded-lg shadow-lg",
          "py-1 min-w-[180px]"
        )}
      >
        <div className="px-3 py-1.5 text-xs font-medium text-muted-foreground border-b border-border">
          Add Context
        </div>
        {items.map((item) => (
          <button
            key={item.type}
            onClick={() => onSelect({ type: item.type, name: item.label })}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-accent"
          >
            <item.icon size={14} />
            <span>@{item.type}</span>
            <span className="ml-auto text-xs text-muted-foreground">
              {item.label}
            </span>
          </button>
        ))}
      </div>
    </>
  );
}

// ============================================================================
// Empty State
// ============================================================================

function EmptyState() {
  return (
    <div className="h-full flex flex-col items-center justify-center p-6 text-center">
      <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
        <Sparkles size={32} className="text-primary" />
      </div>
      <h3 className="text-lg font-medium mb-2">AI Assistant</h3>
      <p className="text-sm text-muted-foreground mb-6">
        Ask questions, get code suggestions, and explore your codebase.
      </p>
      <div className="space-y-2 text-sm text-muted-foreground">
        <p>
          <kbd className="px-1.5 py-0.5 bg-muted rounded">@file</kbd> Reference
          current file
        </p>
        <p>
          <kbd className="px-1.5 py-0.5 bg-muted rounded">@code</kbd> Include
          selected code
        </p>
        <p>
          <kbd className="px-1.5 py-0.5 bg-muted rounded">@terminal</kbd>{" "}
          Include terminal output
        </p>
      </div>
    </div>
  );
}

export default AIPanel;

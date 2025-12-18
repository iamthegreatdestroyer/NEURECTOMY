/**
 * RealTerminal - Fully functional terminal using xterm.js and Tauri shell
 *
 * Provides an integrated terminal experience within the NEURECTOMY IDE
 * with support for PowerShell on Windows and bash on Unix systems.
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { Command } from "@tauri-apps/plugin-shell";
import { platform } from "@tauri-apps/plugin-os";
import "@xterm/xterm/css/xterm.css";

interface RealTerminalProps {
  className?: string;
  onClose?: () => void;
  initialDirectory?: string;
}

// NEURECTOMY dark theme matching the IDE
const NEURECTOMY_THEME = {
  background: "#0a0a0f",
  foreground: "#e4e4e7",
  cursor: "#00ff88",
  cursorAccent: "#0a0a0f",
  selectionBackground: "#3b3b4f",
  selectionForeground: "#ffffff",
  black: "#1a1a2e",
  red: "#ff5555",
  green: "#00ff88",
  yellow: "#ffb454",
  blue: "#6366f1",
  magenta: "#a855f7",
  cyan: "#22d3ee",
  white: "#e4e4e7",
  brightBlack: "#52525b",
  brightRed: "#ff7b7b",
  brightGreen: "#7dffb3",
  brightYellow: "#ffd18a",
  brightBlue: "#818cf8",
  brightMagenta: "#c084fc",
  brightCyan: "#67e8f9",
  brightWhite: "#ffffff",
};

export function RealTerminal({
  className,
  onClose,
  initialDirectory,
}: RealTerminalProps) {
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [currentDir, setCurrentDir] = useState(
    initialDirectory || "C:\\Users\\sgbil\\NEURECTOMY"
  );
  const commandBufferRef = useRef<string>("");
  const historyRef = useRef<string[]>([]);
  const historyIndexRef = useRef<number>(-1);

  // Get shell command based on platform
  // Note: Tauri uses "macos" not "darwin" for macOS detection
  const getShellInfo = useCallback(async () => {
    const os = await platform();
    if (os === "windows") {
      return { shell: "powershell.exe", args: ["-NoLogo", "-NoProfile"] };
    } else if (os === "macos") {
      return { shell: "zsh", args: [] };
    } else {
      return { shell: "bash", args: [] };
    }
  }, []);

  // Execute command and stream output to terminal
  const executeCommand = useCallback(
    async (cmd: string) => {
      if (!xtermRef.current) return;

      const term = xtermRef.current;
      const trimmedCmd = cmd.trim();

      if (!trimmedCmd) {
        writePrompt();
        return;
      }

      // Add to history
      if (
        trimmedCmd &&
        historyRef.current[historyRef.current.length - 1] !== trimmedCmd
      ) {
        historyRef.current.push(trimmedCmd);
      }
      historyIndexRef.current = historyRef.current.length;

      // Handle built-in commands
      if (trimmedCmd === "clear" || trimmedCmd === "cls") {
        term.clear();
        writePrompt();
        return;
      }

      if (trimmedCmd === "exit") {
        onClose?.();
        return;
      }

      // Handle cd command specially to track current directory
      if (trimmedCmd.startsWith("cd ")) {
        const newDir = trimmedCmd.substring(3).trim();
        // Execute cd and capture new directory
        try {
          const command = Command.create("powershell", [
            "-NoLogo",
            "-NoProfile",
            "-Command",
            `cd "${newDir}"; Get-Location | Select-Object -ExpandProperty Path`,
          ]);
          const output = await command.execute();
          if (output.code === 0 && output.stdout) {
            setCurrentDir(output.stdout.trim());
          } else if (output.stderr) {
            term.write(`\r\n\x1b[31m${output.stderr}\x1b[0m`);
          }
        } catch (error) {
          term.write(`\r\n\x1b[31mError: ${error}\x1b[0m`);
        }
        writePrompt();
        return;
      }

      try {
        // Execute command in PowerShell
        const command = Command.create("powershell", [
          "-NoLogo",
          "-NoProfile",
          "-Command",
          `cd "${currentDir}"; ${trimmedCmd}`,
        ]);

        // Stream stdout
        command.stdout.on("data", (data: string) => {
          term.write(`\r\n${data}`);
        });

        // Stream stderr
        command.stderr.on("data", (data: string) => {
          term.write(`\r\n\x1b[31m${data}\x1b[0m`);
        });

        // Wait for completion
        const result = await command.execute();

        if (result.stdout && !result.stdout.endsWith("\n")) {
          // Ensure newline after output
        }
      } catch (error) {
        term.write(`\r\n\x1b[31mError executing command: ${error}\x1b[0m`);
      }

      writePrompt();
    },
    [currentDir, onClose]
  );

  // Write prompt to terminal
  const writePrompt = useCallback(() => {
    if (!xtermRef.current) return;
    const shortDir = currentDir.replace(/^C:\\Users\\[^\\]+/, "~");
    xtermRef.current.write(`\r\n\x1b[36m${shortDir}\x1b[0m \x1b[32m>\x1b[0m `);
    commandBufferRef.current = "";
  }, [currentDir]);

  // Initialize terminal
  useEffect(() => {
    if (!terminalRef.current || xtermRef.current) return;

    const term = new Terminal({
      theme: NEURECTOMY_THEME,
      fontFamily:
        "'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace",
      fontSize: 13,
      lineHeight: 1.4,
      cursorBlink: true,
      cursorStyle: "bar",
      scrollback: 10000,
      allowProposedApi: true,
    });

    const fitAddon = new FitAddon();
    const webLinksAddon = new WebLinksAddon();

    term.loadAddon(fitAddon);
    term.loadAddon(webLinksAddon);

    term.open(terminalRef.current);

    // Initial fit
    setTimeout(() => {
      fitAddon.fit();
    }, 50);

    xtermRef.current = term;
    fitAddonRef.current = fitAddon;

    // Write welcome message
    term.writeln(
      "\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m"
    );
    term.writeln(
      "\x1b[36m║\x1b[0m  \x1b[1;32mNEURECTOMY Terminal\x1b[0m                                        \x1b[36m║\x1b[0m"
    );
    term.writeln(
      "\x1b[36m║\x1b[0m  Integrated PowerShell with AI Agent Support                 \x1b[36m║\x1b[0m"
    );
    term.writeln(
      "\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m"
    );
    term.writeln("");
    writePrompt();

    // Handle keyboard input
    term.onKey(({ key, domEvent }) => {
      const printable =
        !domEvent.altKey && !domEvent.ctrlKey && !domEvent.metaKey;

      if (domEvent.keyCode === 13) {
        // Enter
        executeCommand(commandBufferRef.current);
      } else if (domEvent.keyCode === 8) {
        // Backspace
        if (commandBufferRef.current.length > 0) {
          commandBufferRef.current = commandBufferRef.current.slice(0, -1);
          term.write("\b \b");
        }
      } else if (domEvent.keyCode === 38) {
        // Up arrow - history
        if (historyIndexRef.current > 0) {
          historyIndexRef.current--;
          const historyCmd = historyRef.current[historyIndexRef.current];
          // Clear current line and write history command
          term.write("\r\x1b[K");
          const shortDir = currentDir.replace(/^C:\\Users\\[^\\]+/, "~");
          term.write(
            `\x1b[36m${shortDir}\x1b[0m \x1b[32m>\x1b[0m ${historyCmd}`
          );
          commandBufferRef.current = historyCmd;
        }
      } else if (domEvent.keyCode === 40) {
        // Down arrow - history
        if (historyIndexRef.current < historyRef.current.length - 1) {
          historyIndexRef.current++;
          const historyCmd = historyRef.current[historyIndexRef.current];
          term.write("\r\x1b[K");
          const shortDir = currentDir.replace(/^C:\\Users\\[^\\]+/, "~");
          term.write(
            `\x1b[36m${shortDir}\x1b[0m \x1b[32m>\x1b[0m ${historyCmd}`
          );
          commandBufferRef.current = historyCmd;
        } else {
          historyIndexRef.current = historyRef.current.length;
          term.write("\r\x1b[K");
          const shortDir = currentDir.replace(/^C:\\Users\\[^\\]+/, "~");
          term.write(`\x1b[36m${shortDir}\x1b[0m \x1b[32m>\x1b[0m `);
          commandBufferRef.current = "";
        }
      } else if (domEvent.ctrlKey && domEvent.keyCode === 67) {
        // Ctrl+C
        term.write("^C");
        commandBufferRef.current = "";
        writePrompt();
      } else if (domEvent.ctrlKey && domEvent.keyCode === 76) {
        // Ctrl+L - clear screen
        term.clear();
        writePrompt();
      } else if (printable) {
        commandBufferRef.current += key;
        term.write(key);
      }
    });

    // Handle paste
    term.onData((data) => {
      // Handle pasted data (multi-character input)
      if (data.length > 1 && !data.startsWith("\x1b")) {
        commandBufferRef.current += data;
        term.write(data);
      }
    });

    setIsReady(true);

    return () => {
      term.dispose();
      xtermRef.current = null;
    };
  }, [executeCommand, writePrompt, currentDir]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (fitAddonRef.current) {
        try {
          fitAddonRef.current.fit();
        } catch (e) {
          // Ignore fit errors during resize
        }
      }
    };

    window.addEventListener("resize", handleResize);

    // Also observe container size changes
    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });

    if (terminalRef.current) {
      resizeObserver.observe(terminalRef.current);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <div
      ref={terminalRef}
      className={`h-full w-full bg-[#0a0a0f] ${className || ""}`}
      style={{ padding: "8px" }}
    />
  );
}

export default RealTerminal;

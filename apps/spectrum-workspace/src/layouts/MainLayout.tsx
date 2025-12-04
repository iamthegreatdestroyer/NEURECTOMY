import { Outlet } from 'react-router-dom';
import { Sidebar } from '@/components/sidebar';
import { TopBar } from '@/components/topbar';
import { CommandPalette } from '@/components/command-palette';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';

export function MainLayout() {
  useKeyboardShortcuts();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      {/* Sidebar Navigation */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top Bar */}
        <TopBar />

        {/* Page Content */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>

      {/* Global Command Palette (Ctrl/Cmd + K) */}
      <CommandPalette />
    </div>
  );
}

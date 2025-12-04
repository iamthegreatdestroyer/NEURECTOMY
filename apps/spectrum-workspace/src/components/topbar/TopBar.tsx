/**
 * TopBar Component
 * Header with search, notifications, and user menu
 */

import { useState } from 'react';
import {
  Search,
  Bell,
  Command,
  User,
  ChevronDown,
  Wifi,
  WifiOff,
  Moon,
  Sun,
  Settings,
  LogOut,
  HelpCircle,
  X,
} from 'lucide-react';
import { useAppStore } from '../../stores/app.store';
import { cn, formatRelativeTime } from '../../lib/utils';

// Notification dropdown
function NotificationDropdown({ onClose }: { onClose: () => void }) {
  const notifications = useAppStore((state) => state.notifications);
  const markNotificationRead = useAppStore((state) => state.markNotificationRead);
  const clearNotifications = useAppStore((state) => state.clearNotifications);

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <div className="absolute right-0 top-full mt-2 w-80 bg-card border border-border rounded-xl shadow-lg z-50">
      <div className="flex items-center justify-between p-4 border-b border-border">
        <h3 className="font-semibold">
          Notifications
          {unreadCount > 0 && (
            <span className="ml-2 px-1.5 py-0.5 bg-primary text-primary-foreground text-xs rounded-full">
              {unreadCount}
            </span>
          )}
        </h3>
        <div className="flex items-center gap-2">
          {notifications.length > 0 && (
            <button
              onClick={clearNotifications}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Clear all
            </button>
          )}
          <button onClick={onClose} className="p-1 hover:bg-muted rounded">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="max-h-80 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="p-8 text-center text-muted-foreground">
            <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No notifications</p>
          </div>
        ) : (
          notifications.slice(0, 5).map((notification) => (
            <div
              key={notification.id}
              onClick={() => markNotificationRead(notification.id)}
              className={cn(
                'p-4 border-b border-border cursor-pointer hover:bg-muted/50 transition-colors',
                !notification.read && 'bg-primary/5'
              )}
            >
              <div className="flex items-start gap-3">
                <div
                  className={cn(
                    'w-2 h-2 rounded-full mt-2 flex-shrink-0',
                    notification.type === 'success' && 'bg-green-500',
                    notification.type === 'error' && 'bg-red-500',
                    notification.type === 'warning' && 'bg-yellow-500',
                    notification.type === 'info' && 'bg-blue-500'
                  )}
                />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm">{notification.title}</p>
                  {notification.message && (
                    <p className="text-xs text-muted-foreground line-clamp-2">
                      {notification.message}
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground mt-1">
                    {formatRelativeTime(notification.timestamp)}
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {notifications.length > 5 && (
        <div className="p-3 border-t border-border">
          <button className="w-full py-2 text-sm text-primary hover:underline">
            View all notifications
          </button>
        </div>
      )}
    </div>
  );
}

// User dropdown menu
function UserDropdown({ onClose }: { onClose: () => void }) {
  const theme = useAppStore((state) => state.theme);
  const setTheme = useAppStore((state) => state.setTheme);

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  return (
    <div className="absolute right-0 top-full mt-2 w-56 bg-card border border-border rounded-xl shadow-lg z-50">
      <div className="p-3 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-primary to-violet-500 rounded-full flex items-center justify-center text-white font-bold">
            D
          </div>
          <div>
            <p className="font-semibold text-sm">Developer</p>
            <p className="text-xs text-muted-foreground">dev@neurectomy.ai</p>
          </div>
        </div>
      </div>

      <div className="p-2">
        <button
          onClick={toggleTheme}
          className="w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-muted rounded-lg transition-colors"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </button>
        <button className="w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-muted rounded-lg transition-colors">
          <Settings className="w-4 h-4" />
          Settings
        </button>
        <button className="w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-muted rounded-lg transition-colors">
          <HelpCircle className="w-4 h-4" />
          Help & Support
        </button>
      </div>

      <div className="p-2 border-t border-border">
        <button className="w-full flex items-center gap-3 px-3 py-2 text-sm text-destructive hover:bg-destructive/10 rounded-lg transition-colors">
          <LogOut className="w-4 h-4" />
          Sign Out
        </button>
      </div>
    </div>
  );
}

export function TopBar() {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  
  const openCommandPalette = useAppStore((state) => state.openCommandPalette);
  const isConnected = useAppStore((state) => state.isConnected);
  const notifications = useAppStore((state) => state.notifications);

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <header className="h-14 bg-card border-b border-border flex items-center justify-between px-4">
      {/* Left section - Search */}
      <div className="flex items-center gap-4 flex-1">
        <button
          onClick={openCommandPalette}
          className="flex items-center gap-3 px-4 py-2 bg-muted rounded-lg text-muted-foreground hover:text-foreground transition-colors w-full max-w-md"
        >
          <Search className="w-4 h-4" />
          <span className="text-sm">Search anything...</span>
          <div className="ml-auto flex items-center gap-1 text-xs bg-background px-1.5 py-0.5 rounded border border-border">
            <Command className="w-3 h-3" />
            <span>K</span>
          </div>
        </button>
      </div>

      {/* Right section */}
      <div className="flex items-center gap-2">
        {/* Connection status */}
        <div
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm',
            isConnected ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'
          )}
        >
          {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
          <span className="text-xs font-medium">{isConnected ? 'Connected' : 'Offline'}</span>
        </div>

        {/* Notifications */}
        <div className="relative">
          <button
            onClick={() => {
              setShowNotifications(!showNotifications);
              setShowUserMenu(false);
            }}
            className="relative p-2 hover:bg-muted rounded-lg transition-colors"
          >
            <Bell className="w-5 h-5" />
            {unreadCount > 0 && (
              <span className="absolute top-1 right-1 w-4 h-4 bg-primary text-primary-foreground text-[10px] font-bold rounded-full flex items-center justify-center">
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
          </button>

          {showNotifications && (
            <NotificationDropdown onClose={() => setShowNotifications(false)} />
          )}
        </div>

        {/* User menu */}
        <div className="relative">
          <button
            onClick={() => {
              setShowUserMenu(!showUserMenu);
              setShowNotifications(false);
            }}
            className="flex items-center gap-2 p-1.5 hover:bg-muted rounded-lg transition-colors"
          >
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-violet-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
              D
            </div>
            <ChevronDown className="w-4 h-4 text-muted-foreground" />
          </button>

          {showUserMenu && <UserDropdown onClose={() => setShowUserMenu(false)} />}
        </div>
      </div>
    </header>
  );
}

export default TopBar;

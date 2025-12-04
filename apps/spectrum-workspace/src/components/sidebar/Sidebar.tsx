/**
 * Sidebar Component
 * Main navigation sidebar for NEURECTOMY
 */

import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Boxes,
  Server,
  Brain,
  BookOpen,
  Shield,
  Settings,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Activity,
  HelpCircle,
  LogOut,
} from 'lucide-react';
import { useAppStore } from '../../stores/app.store';
import { cn } from '../../lib/utils';

// Navigation item type
interface NavItem {
  path: string;
  label: string;
  icon: React.ElementType;
  badge?: string | number;
}

// Main navigation items
const mainNavItems: NavItem[] = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/forge', label: 'Dimensional Forge', icon: Boxes },
  { path: '/containers', label: 'Container Command', icon: Server },
  { path: '/foundry', label: 'Intelligence Foundry', icon: Brain, badge: '2' },
  { path: '/discovery', label: 'Discovery Engine', icon: BookOpen },
  { path: '/legal', label: 'Legal Fortress', icon: Shield },
];

// Bottom navigation items
const bottomNavItems: NavItem[] = [
  { path: '/settings', label: 'Settings', icon: Settings },
];

// Nav link component
function NavLink({ item, collapsed }: { item: NavItem; collapsed: boolean }) {
  const location = useLocation();
  const isActive = location.pathname === item.path;

  return (
    <Link
      to={item.path}
      className={cn(
        'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group relative',
        isActive
          ? 'bg-primary/10 text-primary'
          : 'text-muted-foreground hover:text-foreground hover:bg-muted'
      )}
    >
      <item.icon className={cn(
        'w-5 h-5 flex-shrink-0',
        isActive && 'text-primary'
      )} />
      
      {!collapsed && (
        <>
          <span className="font-medium text-sm truncate">{item.label}</span>
          {item.badge && (
            <span className="ml-auto px-2 py-0.5 bg-primary text-primary-foreground text-xs font-semibold rounded-full">
              {item.badge}
            </span>
          )}
        </>
      )}

      {/* Tooltip for collapsed state */}
      {collapsed && (
        <div className="absolute left-full ml-2 px-2 py-1 bg-popover text-popover-foreground text-sm rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all whitespace-nowrap z-50">
          {item.label}
          {item.badge && (
            <span className="ml-2 px-1.5 py-0.5 bg-primary text-primary-foreground text-xs font-semibold rounded-full">
              {item.badge}
            </span>
          )}
        </div>
      )}

      {/* Active indicator */}
      {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-primary rounded-r-full" />
      )}
    </Link>
  );
}

// Status indicator component
function StatusIndicator({ connected }: { connected: boolean }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg">
      <div className={cn(
        'w-2 h-2 rounded-full',
        connected ? 'bg-green-500' : 'bg-red-500'
      )} />
      <span className="text-xs text-muted-foreground">
        {connected ? 'Connected' : 'Offline'}
      </span>
    </div>
  );
}

export function Sidebar() {
  const sidebar = useAppStore((state) => state.sidebar);
  const toggleSidebar = useAppStore((state) => state.toggleSidebar);
  const isConnected = useAppStore((state) => state.isConnected);
  const collapsed = sidebar.isCollapsed;

  return (
    <aside
      className={cn(
        'flex flex-col h-screen bg-card border-r border-border transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Header */}
      <div className={cn(
        'flex items-center h-16 px-4 border-b border-border',
        collapsed ? 'justify-center' : 'justify-between'
      )}>
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-violet-500 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-foreground leading-none">NEURECTOMY</h1>
              <p className="text-[10px] text-muted-foreground">Neural IDE Platform</p>
            </div>
          </div>
        )}
        
        {collapsed && (
          <div className="w-8 h-8 bg-gradient-to-br from-primary to-violet-500 rounded-lg flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
        )}

        <button
          onClick={toggleSidebar}
          className={cn(
            'p-1.5 hover:bg-muted rounded-lg transition-colors',
            collapsed && 'absolute right-0 translate-x-1/2 bg-card border border-border shadow-sm z-10'
          )}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Main navigation */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        {/* Section label */}
        {!collapsed && (
          <p className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Workspace
          </p>
        )}

        {mainNavItems.map((item) => (
          <NavLink key={item.path} item={item} collapsed={collapsed} />
        ))}

        {/* Divider */}
        <div className="my-4 border-t border-border" />

        {/* Quick Stats (only when expanded) */}
        {!collapsed && (
          <div className="px-3 py-2 space-y-3">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Quick Stats
            </p>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground flex items-center gap-2">
                  <Activity className="w-4 h-4 text-green-500" />
                  Agents
                </span>
                <span className="font-semibold text-green-500">12 active</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground flex items-center gap-2">
                  <Server className="w-4 h-4 text-cyan-500" />
                  Containers
                </span>
                <span className="font-semibold text-cyan-500">6 running</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground flex items-center gap-2">
                  <Brain className="w-4 h-4 text-violet-500" />
                  Training
                </span>
                <span className="font-semibold text-violet-500">2 jobs</span>
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Bottom section */}
      <div className="p-3 space-y-2 border-t border-border">
        {/* Status indicator */}
        {!collapsed && <StatusIndicator connected={isConnected} />}

        {/* Bottom nav items */}
        {bottomNavItems.map((item) => (
          <NavLink key={item.path} item={item} collapsed={collapsed} />
        ))}

        {/* Help & Logout (only icons when collapsed) */}
        {!collapsed ? (
          <div className="flex items-center gap-2 pt-2">
            <button className="flex-1 flex items-center justify-center gap-2 py-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-colors">
              <HelpCircle className="w-4 h-4" />
              <span className="text-sm">Help</span>
            </button>
            <button className="flex-1 flex items-center justify-center gap-2 py-2 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-lg transition-colors">
              <LogOut className="w-4 h-4" />
              <span className="text-sm">Logout</span>
            </button>
          </div>
        ) : (
          <div className="space-y-1">
            <button className="w-full p-2.5 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-colors flex justify-center">
              <HelpCircle className="w-5 h-5" />
            </button>
            <button className="w-full p-2.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-lg transition-colors flex justify-center">
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}

export default Sidebar;

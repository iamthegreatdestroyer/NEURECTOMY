/**
 * Settings Feature
 * Application configuration and preferences
 */

import { useState } from 'react';
import {
  Settings as SettingsIcon,
  User,
  Palette,
  Bell,
  Shield,
  Key,
  Globe,
  Monitor,
  Moon,
  Sun,
  Laptop,
  Database,
  Zap,
  HardDrive,
  Wifi,
  Volume2,
  VolumeX,
  ChevronRight,
  Save,
  RefreshCw,
  AlertTriangle,
  Check,
} from 'lucide-react';
import { useAppStore } from '../../stores/app.store';

// Settings section type
type SettingsSection = 'profile' | 'appearance' | 'notifications' | 'security' | 'integrations' | 'system';

// Theme selector component
function ThemeSelector() {
  const theme = useAppStore((state) => state.theme);
  const setTheme = useAppStore((state) => state.setTheme);

  const themes = [
    { id: 'light', name: 'Light', icon: Sun },
    { id: 'dark', name: 'Dark', icon: Moon },
    { id: 'system', name: 'System', icon: Laptop },
  ] as const;

  return (
    <div className="grid grid-cols-3 gap-3">
      {themes.map(({ id, name, icon: Icon }) => (
        <button
          key={id}
          onClick={() => setTheme(id)}
          className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-colors ${
            theme === id
              ? 'border-primary bg-primary/10'
              : 'border-border hover:border-primary/50'
          }`}
        >
          <Icon className={`w-6 h-6 ${theme === id ? 'text-primary' : 'text-muted-foreground'}`} />
          <span className={`text-sm font-medium ${theme === id ? 'text-primary' : 'text-foreground'}`}>
            {name}
          </span>
        </button>
      ))}
    </div>
  );
}

// Toggle switch component
function Toggle({ enabled, onChange, label }: { enabled: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm">{label}</span>
      <button
        onClick={() => onChange(!enabled)}
        className={`relative w-11 h-6 rounded-full transition-colors ${
          enabled ? 'bg-primary' : 'bg-muted'
        }`}
      >
        <span
          className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
            enabled ? 'translate-x-6' : 'translate-x-1'
          }`}
        />
      </button>
    </div>
  );
}

// Settings input component
function SettingsInput({ label, value, onChange, type = 'text', placeholder }: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full px-3 py-2 bg-muted border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
      />
    </div>
  );
}

// Navigation item component
function NavItem({ icon: Icon, label, active, onClick }: {
  icon: React.ElementType;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
        active
          ? 'bg-primary/10 text-primary'
          : 'text-muted-foreground hover:text-foreground hover:bg-muted'
      }`}
    >
      <Icon className="w-5 h-5" />
      {label}
      <ChevronRight className={`w-4 h-4 ml-auto ${active ? 'opacity-100' : 'opacity-0'}`} />
    </button>
  );
}

// Main Settings component
export function Settings() {
  const [activeSection, setActiveSection] = useState<SettingsSection>('appearance');
  const preferences = useAppStore((state) => state.preferences);
  const updatePreferences = useAppStore((state) => state.updatePreferences);
  
  // Local state for form fields
  const [displayName, setDisplayName] = useState('Developer');
  const [email, setEmail] = useState('dev@neurectomy.ai');
  const [apiKey, setApiKey] = useState('');
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');

  const sections = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'appearance', label: 'Appearance', icon: Palette },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'integrations', label: 'Integrations', icon: Key },
    { id: 'system', label: 'System', icon: Monitor },
  ] as const;

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-foreground flex items-center gap-3">
          <SettingsIcon className="w-7 h-7 text-primary" />
          Settings
        </h1>
        <p className="text-muted-foreground mt-1">
          Configure your workspace preferences
        </p>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Sidebar navigation */}
        <div className="col-span-3">
          <nav className="space-y-1">
            {sections.map((section) => (
              <NavItem
                key={section.id}
                icon={section.icon}
                label={section.label}
                active={activeSection === section.id}
                onClick={() => setActiveSection(section.id)}
              />
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="col-span-9">
          <div className="bg-card border border-border rounded-xl p-6">
            {/* Profile Section */}
            {activeSection === 'profile' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold">Profile Settings</h2>
                
                <div className="flex items-center gap-4">
                  <div className="w-20 h-20 bg-gradient-to-br from-primary to-violet-500 rounded-full flex items-center justify-center text-2xl font-bold text-white">
                    D
                  </div>
                  <div>
                    <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium">
                      Change Avatar
                    </button>
                    <p className="text-xs text-muted-foreground mt-1">
                      JPG, PNG or GIF. Max 2MB.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <SettingsInput
                    label="Display Name"
                    value={displayName}
                    onChange={setDisplayName}
                    placeholder="Your name"
                  />
                  <SettingsInput
                    label="Email"
                    value={email}
                    onChange={setEmail}
                    type="email"
                    placeholder="your@email.com"
                  />
                </div>
              </div>
            )}

            {/* Appearance Section */}
            {activeSection === 'appearance' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold">Appearance</h2>
                
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-muted-foreground">Theme</h3>
                  <ThemeSelector />
                </div>

                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-muted-foreground">Interface</h3>
                  <div className="space-y-4">
                    <Toggle
                      enabled={preferences.enableAnimations}
                      onChange={(v) => updatePreferences({ enableAnimations: v })}
                      label="Enable animations"
                    />
                    <Toggle
                      enabled={preferences.sidebarCollapsed}
                      onChange={(v) => updatePreferences({ sidebarCollapsed: v })}
                      label="Collapse sidebar by default"
                    />
                    <Toggle
                      enabled={preferences.showWelcome}
                      onChange={(v) => updatePreferences({ showWelcome: v })}
                      label="Show welcome screen on startup"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Notifications Section */}
            {activeSection === 'notifications' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold">Notification Preferences</h2>
                
                <div className="space-y-4">
                  <Toggle
                    enabled={preferences.enableSounds}
                    onChange={(v) => updatePreferences({ enableSounds: v })}
                    label="Enable notification sounds"
                  />
                  <Toggle enabled={true} onChange={() => {}} label="Agent task completion alerts" />
                  <Toggle enabled={true} onChange={() => {}} label="Training job updates" />
                  <Toggle enabled={false} onChange={() => {}} label="Container status changes" />
                  <Toggle enabled={true} onChange={() => {}} label="Research paper recommendations" />
                </div>
              </div>
            )}

            {/* Security Section */}
            {activeSection === 'security' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold">Security Settings</h2>
                
                <div className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                  <div className="flex items-center gap-2 text-yellow-500 mb-2">
                    <AlertTriangle className="w-5 h-5" />
                    <span className="font-medium">Two-factor authentication disabled</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Enable 2FA to add an extra layer of security to your account.
                  </p>
                  <button className="mt-3 px-4 py-2 bg-yellow-500 text-black rounded-lg text-sm font-medium">
                    Enable 2FA
                  </button>
                </div>

                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-muted-foreground">Session Management</h3>
                  <div className="p-4 bg-muted rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Monitor className="w-5 h-5 text-green-500" />
                        <div>
                          <p className="text-sm font-medium">Current Session</p>
                          <p className="text-xs text-muted-foreground">Windows • Chrome • This device</p>
                        </div>
                      </div>
                      <Check className="w-5 h-5 text-green-500" />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Integrations Section */}
            {activeSection === 'integrations' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold">API Integrations</h2>
                
                <div className="space-y-4">
                  <SettingsInput
                    label="OpenAI API Key"
                    value={apiKey}
                    onChange={setApiKey}
                    type="password"
                    placeholder="sk-..."
                  />
                  <SettingsInput
                    label="Ollama Server URL"
                    value={ollamaUrl}
                    onChange={setOllamaUrl}
                    placeholder="http://localhost:11434"
                  />
                </div>

                <div className="space-y-3">
                  <h3 className="text-sm font-medium text-muted-foreground">Connected Services</h3>
                  
                  {[
                    { name: 'GitHub', icon: Globe, connected: true },
                    { name: 'Docker Hub', icon: Database, connected: true },
                    { name: 'Hugging Face', icon: Zap, connected: false },
                  ].map((service) => (
                    <div key={service.name} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                      <div className="flex items-center gap-3">
                        <service.icon className="w-5 h-5" />
                        <span className="font-medium">{service.name}</span>
                      </div>
                      <button
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                          service.connected
                            ? 'bg-green-500/10 text-green-500'
                            : 'bg-primary text-primary-foreground'
                        }`}
                      >
                        {service.connected ? 'Connected' : 'Connect'}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* System Section */}
            {activeSection === 'system' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold">System Information</h2>
                
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: 'Version', value: 'v0.1.0-alpha', icon: Zap },
                    { label: 'Build', value: '2024.01.15.001', icon: HardDrive },
                    { label: 'Connection', value: 'Connected', icon: Wifi },
                    { label: 'Sound', value: preferences.enableSounds ? 'Enabled' : 'Muted', icon: preferences.enableSounds ? Volume2 : VolumeX },
                  ].map((item) => (
                    <div key={item.label} className="p-4 bg-muted rounded-lg">
                      <div className="flex items-center gap-2 text-muted-foreground mb-1">
                        <item.icon className="w-4 h-4" />
                        {item.label}
                      </div>
                      <p className="font-semibold">{item.value}</p>
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-3 pt-4 border-t border-border">
                  <button className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg text-sm font-medium hover:bg-muted/80">
                    <RefreshCw className="w-4 h-4" />
                    Check for Updates
                  </button>
                  <button className="flex items-center gap-2 px-4 py-2 bg-destructive/10 text-destructive rounded-lg text-sm font-medium hover:bg-destructive/20">
                    Clear Cache
                  </button>
                </div>
              </div>
            )}

            {/* Save button */}
            <div className="flex justify-end pt-6 mt-6 border-t border-border">
              <button className="flex items-center gap-2 px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors">
                <Save className="w-4 h-4" />
                Save Changes
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;

import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Bot,
  Save,
  Play,
  Pause,
  Trash2,
  Settings,
  Code,
  MessageSquare,
  Activity,
  ArrowLeft,
  Copy,
  RefreshCw,
  Zap,
  Shield,
  Brain,
  Cpu,
  MemoryStick,
  HardDrive,
  ChevronDown,
  Terminal,
  FileCode,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAgentsStore } from '@/stores/agents.store';
import type { Agent } from '@/types';

type EditorTab = 'config' | 'code' | 'chat' | 'metrics';

const agentTemplates = [
  { id: 'apex', name: 'APEX', description: 'Elite CS Engineering', icon: '🎯' },
  { id: 'cipher', name: 'CIPHER', description: 'Cryptography & Security', icon: '🔐' },
  { id: 'architect', name: 'ARCHITECT', description: 'Systems Architecture', icon: '🏛️' },
  { id: 'tensor', name: 'TENSOR', description: 'ML & Deep Learning', icon: '🧠' },
  { id: 'flux', name: 'FLUX', description: 'DevOps & Infrastructure', icon: '⚡' },
  { id: 'fortress', name: 'FORTRESS', description: 'Security & Penetration', icon: '🏰' },
];

const defaultAgentConfig = {
  name: '',
  type: 'assistant',
  model: 'gpt-4-turbo',
  temperature: 0.7,
  maxTokens: 4096,
  systemPrompt: '',
  capabilities: [] as string[],
  memoryEnabled: true,
  streamingEnabled: true,
};

export function AgentEditor() {
  const { agentId } = useParams<{ agentId: string }>();
  const navigate = useNavigate();
  const { agents, updateAgent } = useAgentsStore();
  
  const [activeTab, setActiveTab] = useState<EditorTab>('config');
  const [config, setConfig] = useState(defaultAgentConfig);
  const [isSaving, setIsSaving] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [chatInput, setChatInput] = useState('');

  const isNewAgent = agentId === 'new';
  const existingAgent = agents.find(a => a.id === agentId);

  useEffect(() => {
    if (existingAgent) {
      setConfig({
        name: existingAgent.name,
        type: existingAgent.type,
        model: existingAgent.model || 'gpt-4-turbo',
        temperature: 0.7,
        maxTokens: 4096,
        systemPrompt: '',
        capabilities: existingAgent.capabilities || [],
        memoryEnabled: true,
        streamingEnabled: true,
      });
    }
  }, [existingAgent]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      // Simulate save
      await new Promise(resolve => setTimeout(resolve, 1000));
      // In real app, call API here
      console.log('Saving agent config:', config);
    } finally {
      setIsSaving(false);
    }
  };

  const handleSendMessage = () => {
    if (!chatInput.trim()) return;
    
    setChatMessages(prev => [
      ...prev,
      { role: 'user', content: chatInput },
    ]);
    setChatInput('');
    
    // Simulate response
    setTimeout(() => {
      setChatMessages(prev => [
        ...prev,
        { role: 'assistant', content: `This is a simulated response from the ${config.name || 'Agent'}. In production, this would connect to your configured model.` },
      ]);
    }, 1000);
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="flex-none p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate(-1)}
              className="p-2 hover:bg-accent rounded-lg transition-colors"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Bot className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-bold">
                  {isNewAgent ? 'Create New Agent' : `Edit ${existingAgent?.name || 'Agent'}`}
                </h1>
                <p className="text-sm text-muted-foreground">
                  {isNewAgent ? 'Configure your new AI agent' : 'Modify agent settings and behavior'}
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {!isNewAgent && (
              <>
                <button className="p-2 hover:bg-accent rounded-lg transition-colors text-muted-foreground">
                  <Copy className="h-5 w-5" />
                </button>
                <button className="p-2 hover:bg-accent rounded-lg transition-colors text-muted-foreground">
                  <RefreshCw className="h-5 w-5" />
                </button>
                <button className="p-2 hover:bg-red-500/10 rounded-lg transition-colors text-red-500">
                  <Trash2 className="h-5 w-5" />
                </button>
                <div className="w-px h-6 bg-border mx-2" />
              </>
            )}
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              <Save className="h-4 w-4" />
              {isSaving ? 'Saving...' : 'Save Agent'}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-4 p-1 bg-muted rounded-lg w-fit">
          {[
            { id: 'config', label: 'Configuration', icon: Settings },
            { id: 'code', label: 'Code', icon: Code },
            { id: 'chat', label: 'Test Chat', icon: MessageSquare },
            { id: 'metrics', label: 'Metrics', icon: Activity },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as EditorTab)}
              className={cn(
                'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
                activeTab === tab.id
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'config' && (
          <div className="h-full overflow-auto p-6">
            <div className="max-w-4xl mx-auto space-y-8">
              {/* Basic Info */}
              <section>
                <h2 className="text-lg font-semibold mb-4">Basic Information</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Agent Name</label>
                    <input
                      type="text"
                      value={config.name}
                      onChange={(e) => setConfig({ ...config, name: e.target.value })}
                      placeholder="My Agent"
                      className="w-full px-3 py-2 bg-card border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Agent Type</label>
                    <div className="relative">
                      <select
                        value={config.type}
                        onChange={(e) => setConfig({ ...config, type: e.target.value })}
                        className="w-full px-3 py-2 bg-card border border-border rounded-lg appearance-none focus:outline-none focus:ring-2 focus:ring-primary"
                      >
                        <option value="assistant">Assistant</option>
                        <option value="coding">Coding Agent</option>
                        <option value="research">Research Agent</option>
                        <option value="creative">Creative Agent</option>
                        <option value="custom">Custom</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
                    </div>
                  </div>
                </div>
              </section>

              {/* Template Selection */}
              {isNewAgent && (
                <section>
                  <h2 className="text-lg font-semibold mb-4">Start from Template</h2>
                  <div className="grid grid-cols-3 gap-4">
                    {agentTemplates.map(template => (
                      <button
                        key={template.id}
                        onClick={() => setConfig({ 
                          ...config, 
                          name: template.name,
                          systemPrompt: `You are ${template.name}, an elite ${template.description} agent.`
                        })}
                        className="p-4 bg-card border border-border rounded-xl hover:border-primary/50 transition-colors text-left"
                      >
                        <span className="text-2xl">{template.icon}</span>
                        <h3 className="font-semibold mt-2">{template.name}</h3>
                        <p className="text-sm text-muted-foreground">{template.description}</p>
                      </button>
                    ))}
                  </div>
                </section>
              )}

              {/* Model Configuration */}
              <section>
                <h2 className="text-lg font-semibold mb-4">Model Configuration</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Model</label>
                    <div className="relative">
                      <select
                        value={config.model}
                        onChange={(e) => setConfig({ ...config, model: e.target.value })}
                        className="w-full px-3 py-2 bg-card border border-border rounded-lg appearance-none focus:outline-none focus:ring-2 focus:ring-primary"
                      >
                        <option value="gpt-4-turbo">GPT-4 Turbo</option>
                        <option value="gpt-4o">GPT-4o</option>
                        <option value="claude-3-opus">Claude 3 Opus</option>
                        <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                        <option value="llama-3.1-70b">Llama 3.1 70B</option>
                        <option value="mistral-large">Mistral Large</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Max Tokens</label>
                    <input
                      type="number"
                      value={config.maxTokens}
                      onChange={(e) => setConfig({ ...config, maxTokens: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 bg-card border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm font-medium mb-2">
                      Temperature: {config.temperature}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.1"
                      value={config.temperature}
                      onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>Precise (0)</span>
                      <span>Balanced (1)</span>
                      <span>Creative (2)</span>
                    </div>
                  </div>
                </div>
              </section>

              {/* System Prompt */}
              <section>
                <h2 className="text-lg font-semibold mb-4">System Prompt</h2>
                <textarea
                  value={config.systemPrompt}
                  onChange={(e) => setConfig({ ...config, systemPrompt: e.target.value })}
                  placeholder="Define the agent's personality, capabilities, and behavior..."
                  rows={6}
                  className="w-full px-3 py-2 bg-card border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                />
              </section>

              {/* Features Toggle */}
              <section>
                <h2 className="text-lg font-semibold mb-4">Features</h2>
                <div className="space-y-3">
                  <label className="flex items-center justify-between p-4 bg-card border border-border rounded-lg cursor-pointer">
                    <div className="flex items-center gap-3">
                      <Brain className="h-5 w-5 text-primary" />
                      <div>
                        <p className="font-medium">Memory</p>
                        <p className="text-sm text-muted-foreground">Remember context across conversations</p>
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={config.memoryEnabled}
                      onChange={(e) => setConfig({ ...config, memoryEnabled: e.target.checked })}
                      className="h-5 w-5 rounded border-border"
                    />
                  </label>
                  <label className="flex items-center justify-between p-4 bg-card border border-border rounded-lg cursor-pointer">
                    <div className="flex items-center gap-3">
                      <Zap className="h-5 w-5 text-yellow-500" />
                      <div>
                        <p className="font-medium">Streaming</p>
                        <p className="text-sm text-muted-foreground">Stream responses in real-time</p>
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={config.streamingEnabled}
                      onChange={(e) => setConfig({ ...config, streamingEnabled: e.target.checked })}
                      className="h-5 w-5 rounded border-border"
                    />
                  </label>
                </div>
              </section>
            </div>
          </div>
        )}

        {activeTab === 'code' && (
          <div className="h-full flex flex-col">
            <div className="flex-none p-4 border-b border-border flex items-center gap-4">
              <button className="flex items-center gap-2 px-3 py-1.5 bg-muted rounded-lg text-sm">
                <FileCode className="h-4 w-4" />
                agent.py
              </button>
              <button className="flex items-center gap-2 px-3 py-1.5 hover:bg-muted rounded-lg text-sm text-muted-foreground">
                <FileCode className="h-4 w-4" />
                tools.py
              </button>
            </div>
            <div className="flex-1 p-4 bg-muted/50 font-mono text-sm overflow-auto">
              <pre className="text-muted-foreground">
{`from neurectomy import Agent, Tool

class ${config.name?.replace(/\s+/g, '') || 'MyAgent'}(Agent):
    """
    ${config.name || 'Custom'} Agent
    Type: ${config.type}
    Model: ${config.model}
    """
    
    def __init__(self):
        super().__init__(
            name="${config.name || 'My Agent'}",
            model="${config.model}",
            temperature=${config.temperature},
            max_tokens=${config.maxTokens},
            memory_enabled=${config.memoryEnabled ? 'True' : 'False'},
            streaming=${config.streamingEnabled ? 'True' : 'False'},
        )
        
        self.system_prompt = """
${config.systemPrompt || '# Define your agent behavior here'}
        """
    
    async def process(self, message: str) -> str:
        # Agent processing logic
        response = await self.generate(
            prompt=message,
            system=self.system_prompt,
        )
        return response.content

# Initialize agent
agent = ${config.name?.replace(/\s+/g, '') || 'MyAgent'}()
`}
              </pre>
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="h-full flex flex-col">
            <div className="flex-1 overflow-auto p-4 space-y-4">
              {chatMessages.length === 0 ? (
                <div className="h-full flex items-center justify-center text-center">
                  <div>
                    <MessageSquare className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="font-semibold">Test Your Agent</h3>
                    <p className="text-muted-foreground mt-2">
                      Send a message to test the agent's behavior
                    </p>
                  </div>
                </div>
              ) : (
                chatMessages.map((msg, i) => (
                  <div
                    key={i}
                    className={cn(
                      'max-w-[80%] p-4 rounded-xl',
                      msg.role === 'user'
                        ? 'ml-auto bg-primary text-primary-foreground'
                        : 'bg-card border border-border'
                    )}
                  >
                    {msg.content}
                  </div>
                ))
              )}
            </div>
            <div className="flex-none p-4 border-t border-border">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Type a message to test..."
                  className="flex-1 px-4 py-2 bg-card border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                />
                <button
                  onClick={handleSendMessage}
                  className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="h-full overflow-auto p-6">
            <div className="max-w-4xl mx-auto">
              {isNewAgent ? (
                <div className="text-center py-12">
                  <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="font-semibold text-lg">No Metrics Yet</h3>
                  <p className="text-muted-foreground mt-2">
                    Save and deploy your agent to start collecting metrics
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-card border border-border rounded-xl">
                    <div className="flex items-center gap-3 mb-4">
                      <Cpu className="h-5 w-5 text-primary" />
                      <span className="font-medium">CPU Usage</span>
                    </div>
                    <p className="text-3xl font-bold">23%</p>
                    <p className="text-sm text-muted-foreground">Average over last hour</p>
                  </div>
                  <div className="p-4 bg-card border border-border rounded-xl">
                    <div className="flex items-center gap-3 mb-4">
                      <MemoryStick className="h-5 w-5 text-blue-500" />
                      <span className="font-medium">Memory</span>
                    </div>
                    <p className="text-3xl font-bold">512 MB</p>
                    <p className="text-sm text-muted-foreground">Current allocation</p>
                  </div>
                  <div className="p-4 bg-card border border-border rounded-xl">
                    <div className="flex items-center gap-3 mb-4">
                      <MessageSquare className="h-5 w-5 text-green-500" />
                      <span className="font-medium">Messages</span>
                    </div>
                    <p className="text-3xl font-bold">1,247</p>
                    <p className="text-sm text-muted-foreground">Total processed</p>
                  </div>
                  <div className="p-4 bg-card border border-border rounded-xl">
                    <div className="flex items-center gap-3 mb-4">
                      <Zap className="h-5 w-5 text-yellow-500" />
                      <span className="font-medium">Avg Response</span>
                    </div>
                    <p className="text-3xl font-bold">1.2s</p>
                    <p className="text-sm text-muted-foreground">Response time</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AgentEditor;

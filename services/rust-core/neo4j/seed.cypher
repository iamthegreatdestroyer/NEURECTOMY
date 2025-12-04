// NEURECTOMY: Neo4j Seed Data
// @VERTEX Initial Graph Population
// Default nodes for system bootstrap

// ============================================================
// SYSTEM NODES
// ============================================================

// Create System User (for system-initiated actions)
CREATE (system:User {
  id: 'system-00000000-0000-0000-0000-000000000000',
  email: 'system@neurectomy.ai',
  username: 'system',
  role: 'system',
  created_at: datetime()
});

// ============================================================
// DEFAULT TOPICS (HIERARCHICAL)
// ============================================================

// Root Topics
CREATE (t_dev:Topic {
  id: randomUUID(),
  name: 'Software Development',
  description: 'All aspects of software development',
  level: 0
});

CREATE (t_ai:Topic {
  id: randomUUID(),
  name: 'Artificial Intelligence',
  description: 'AI, ML, and related technologies',
  level: 0
});

CREATE (t_data:Topic {
  id: randomUUID(),
  name: 'Data Science',
  description: 'Data analysis, statistics, and visualization',
  level: 0
});

CREATE (t_devops:Topic {
  id: randomUUID(),
  name: 'DevOps & Infrastructure',
  description: 'CI/CD, containers, cloud infrastructure',
  level: 0
});

CREATE (t_security:Topic {
  id: randomUUID(),
  name: 'Security',
  description: 'Cybersecurity, cryptography, compliance',
  level: 0
});

// Sub-Topics: Software Development
CREATE (t_frontend:Topic {
  id: randomUUID(),
  name: 'Frontend Development',
  description: 'Web and mobile frontend development',
  level: 1
});

CREATE (t_backend:Topic {
  id: randomUUID(),
  name: 'Backend Development',
  description: 'Server-side development',
  level: 1
});

CREATE (t_databases:Topic {
  id: randomUUID(),
  name: 'Databases',
  description: 'SQL, NoSQL, graph databases',
  level: 1
});

// Connect subtopics
MATCH (parent:Topic {name: 'Software Development'})
MATCH (child:Topic) WHERE child.name IN ['Frontend Development', 'Backend Development', 'Databases']
CREATE (child)-[:SUBTOPIC_OF]->(parent);

// Sub-Topics: AI
CREATE (t_ml:Topic {
  id: randomUUID(),
  name: 'Machine Learning',
  description: 'Traditional ML algorithms and techniques',
  level: 1
});

CREATE (t_dl:Topic {
  id: randomUUID(),
  name: 'Deep Learning',
  description: 'Neural networks and deep learning',
  level: 1
});

CREATE (t_nlp:Topic {
  id: randomUUID(),
  name: 'Natural Language Processing',
  description: 'NLP, text processing, LLMs',
  level: 1
});

CREATE (t_llm:Topic {
  id: randomUUID(),
  name: 'Large Language Models',
  description: 'LLM development, fine-tuning, and deployment',
  level: 1
});

CREATE (t_agents:Topic {
  id: randomUUID(),
  name: 'AI Agents',
  description: 'Autonomous AI agents and multi-agent systems',
  level: 1
});

// Connect AI subtopics
MATCH (parent:Topic {name: 'Artificial Intelligence'})
MATCH (child:Topic) WHERE child.name IN ['Machine Learning', 'Deep Learning', 'Natural Language Processing', 'Large Language Models', 'AI Agents']
CREATE (child)-[:SUBTOPIC_OF]->(parent);

// ============================================================
// DEFAULT TOOLS
// ============================================================

CREATE (tool_search:Tool {
  id: randomUUID(),
  name: 'web_search',
  display_name: 'Web Search',
  category: 'information',
  description: 'Search the web for information',
  input_schema: {query: 'string', num_results: 'integer'},
  output_schema: {results: 'array'}
});

CREATE (tool_code:Tool {
  id: randomUUID(),
  name: 'code_execution',
  display_name: 'Code Execution',
  category: 'computation',
  description: 'Execute code in a sandboxed environment',
  input_schema: {language: 'string', code: 'string'},
  output_schema: {stdout: 'string', stderr: 'string', exit_code: 'integer'}
});

CREATE (tool_file:Tool {
  id: randomUUID(),
  name: 'file_operations',
  display_name: 'File Operations',
  category: 'filesystem',
  description: 'Read and write files',
  input_schema: {operation: 'string', path: 'string', content: 'string'},
  output_schema: {success: 'boolean', content: 'string'}
});

CREATE (tool_api:Tool {
  id: randomUUID(),
  name: 'http_request',
  display_name: 'HTTP Request',
  category: 'integration',
  description: 'Make HTTP requests to external APIs',
  input_schema: {method: 'string', url: 'string', headers: 'object', body: 'string'},
  output_schema: {status: 'integer', body: 'string'}
});

CREATE (tool_db:Tool {
  id: randomUUID(),
  name: 'database_query',
  display_name: 'Database Query',
  category: 'data',
  description: 'Execute database queries',
  input_schema: {query: 'string', parameters: 'array'},
  output_schema: {rows: 'array', columns: 'array'}
});

CREATE (tool_image:Tool {
  id: randomUUID(),
  name: 'image_generation',
  display_name: 'Image Generation',
  category: 'creative',
  description: 'Generate images from text descriptions',
  input_schema: {prompt: 'string', style: 'string', size: 'string'},
  output_schema: {image_url: 'string'}
});

// ============================================================
// DEFAULT MODELS
// ============================================================

CREATE (model_gpt4:Model {
  id: randomUUID(),
  name: 'gpt-4-turbo',
  provider: 'openai',
  context_length: 128000,
  capabilities: ['text', 'code', 'vision', 'function_calling']
});

CREATE (model_claude:Model {
  id: randomUUID(),
  name: 'claude-3-opus',
  provider: 'anthropic',
  context_length: 200000,
  capabilities: ['text', 'code', 'vision', 'analysis']
});

CREATE (model_llama:Model {
  id: randomUUID(),
  name: 'llama3.2',
  provider: 'ollama',
  context_length: 8192,
  capabilities: ['text', 'code']
});

CREATE (model_mistral:Model {
  id: randomUUID(),
  name: 'mistral-large',
  provider: 'mistral',
  context_length: 32000,
  capabilities: ['text', 'code', 'function_calling']
});

CREATE (model_gemini:Model {
  id: randomUUID(),
  name: 'gemini-pro',
  provider: 'google',
  context_length: 32000,
  capabilities: ['text', 'code', 'vision']
});

// ============================================================
// CONNECT TOOLS TO TOPICS
// ============================================================

MATCH (tool:Tool {name: 'web_search'}), (topic:Topic {name: 'Artificial Intelligence'})
CREATE (tool)-[:USEFUL_FOR]->(topic);

MATCH (tool:Tool {name: 'code_execution'}), (topic:Topic {name: 'Software Development'})
CREATE (tool)-[:USEFUL_FOR]->(topic);

MATCH (tool:Tool {name: 'database_query'}), (topic:Topic {name: 'Databases'})
CREATE (tool)-[:USEFUL_FOR]->(topic);

// ============================================================
// SAMPLE KNOWLEDGE NODES
// ============================================================

CREATE (k1:Knowledge {
  id: randomUUID(),
  type: 'concept',
  content: 'AI agents are autonomous systems that perceive their environment, make decisions, and take actions to achieve goals.',
  source: 'system',
  confidence: 1.0,
  created_at: datetime()
});

CREATE (k2:Knowledge {
  id: randomUUID(),
  type: 'fact',
  content: 'Large Language Models (LLMs) are neural networks trained on vast amounts of text data to generate human-like responses.',
  source: 'system',
  confidence: 1.0,
  created_at: datetime()
});

CREATE (k3:Knowledge {
  id: randomUUID(),
  type: 'procedure',
  content: 'To fine-tune an LLM: 1) Prepare training data, 2) Choose a base model, 3) Configure hyperparameters, 4) Train with LoRA/QLoRA, 5) Evaluate and iterate.',
  source: 'system',
  confidence: 1.0,
  created_at: datetime()
});

// Connect knowledge to topics
MATCH (k:Knowledge), (t:Topic {name: 'AI Agents'})
WHERE k.content CONTAINS 'AI agents'
CREATE (k)-[:BELONGS_TO]->(t);

MATCH (k:Knowledge), (t:Topic {name: 'Large Language Models'})
WHERE k.content CONTAINS 'LLM' OR k.content CONTAINS 'Language Model'
CREATE (k)-[:BELONGS_TO]->(t);

// Connect related knowledge
MATCH (k1:Knowledge), (k2:Knowledge)
WHERE k1 <> k2 
  AND (k1.content CONTAINS 'LLM' OR k1.content CONTAINS 'agent')
  AND (k2.content CONTAINS 'LLM' OR k2.content CONTAINS 'agent')
CREATE (k1)-[:RELATED_TO {strength: 0.8}]->(k2);

// ============================================================
// RETURN SUMMARY
// ============================================================

MATCH (n) RETURN labels(n)[0] as NodeType, count(n) as Count
ORDER BY Count DESC;

// NEURECTOMY: Neo4j Schema Definitions
// @VERTEX Graph Database Design
// Knowledge Graph for Agents, Relationships, and Intelligence

// ============================================================
// NODE LABELS AND PROPERTIES
// ============================================================

// Agent Node
// Represents an AI agent in the knowledge graph
CREATE CONSTRAINT agent_id_unique IF NOT EXISTS
FOR (a:Agent) REQUIRE a.id IS UNIQUE;

CREATE INDEX agent_name_index IF NOT EXISTS
FOR (a:Agent) ON (a.name);

CREATE INDEX agent_status_index IF NOT EXISTS
FOR (a:Agent) ON (a.status);

CREATE INDEX agent_type_index IF NOT EXISTS
FOR (a:Agent) ON (a.type);

// Agent properties:
// - id: UUID (from PostgreSQL)
// - name: string
// - type: string (autonomous, assistant, specialist, orchestrator)
// - status: string (draft, active, archived)
// - capabilities: list of strings
// - embedding: list of floats (for similarity search)
// - created_at: datetime
// - updated_at: datetime

// User Node
// Represents a user in the graph
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

CREATE INDEX user_email_index IF NOT EXISTS
FOR (u:User) ON (u.email);

// User properties:
// - id: UUID
// - email: string
// - username: string
// - role: string

// Knowledge Node
// Represents a piece of knowledge/information
CREATE CONSTRAINT knowledge_id_unique IF NOT EXISTS
FOR (k:Knowledge) REQUIRE k.id IS UNIQUE;

CREATE INDEX knowledge_type_index IF NOT EXISTS
FOR (k:Knowledge) ON (k.type);

CREATE INDEX knowledge_source_index IF NOT EXISTS
FOR (k:Knowledge) ON (k.source);

// Knowledge properties:
// - id: UUID
// - type: string (fact, concept, procedure, definition)
// - content: string
// - source: string
// - confidence: float (0-1)
// - embedding: list of floats
// - created_at: datetime

// Topic Node
// Represents a topic/category for organization
CREATE CONSTRAINT topic_id_unique IF NOT EXISTS
FOR (t:Topic) REQUIRE t.id IS UNIQUE;

CREATE INDEX topic_name_index IF NOT EXISTS
FOR (t:Topic) ON (t.name);

// Topic properties:
// - id: UUID
// - name: string
// - description: string
// - parent_topic_id: UUID (optional)

// Tool Node
// Represents a tool/capability an agent can use
CREATE CONSTRAINT tool_id_unique IF NOT EXISTS
FOR (tool:Tool) REQUIRE tool.id IS UNIQUE;

CREATE INDEX tool_name_index IF NOT EXISTS
FOR (tool:Tool) ON (tool.name);

CREATE INDEX tool_category_index IF NOT EXISTS
FOR (tool:Tool) ON (tool.category);

// Tool properties:
// - id: UUID
// - name: string
// - category: string
// - description: string
// - input_schema: map
// - output_schema: map

// Model Node
// Represents an AI model
CREATE CONSTRAINT model_id_unique IF NOT EXISTS
FOR (m:Model) REQUIRE m.id IS UNIQUE;

CREATE INDEX model_provider_index IF NOT EXISTS
FOR (m:Model) ON (m.provider);

// Model properties:
// - id: UUID
// - name: string
// - provider: string (openai, anthropic, ollama)
// - context_length: integer
// - capabilities: list of strings

// Document Node
// Represents a document in the knowledge base
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE INDEX document_type_index IF NOT EXISTS
FOR (d:Document) ON (d.type);

// Document properties:
// - id: UUID
// - title: string
// - type: string (pdf, markdown, code, web)
// - source_url: string
// - content_hash: string
// - chunk_count: integer
// - created_at: datetime

// Chunk Node
// Represents a chunk of a document (for RAG)
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE INDEX chunk_document_index IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id);

// Chunk properties:
// - id: UUID
// - document_id: UUID
// - content: string
// - position: integer
// - embedding: list of floats
// - metadata: map

// Conversation Node
// Represents a conversation session
CREATE CONSTRAINT conversation_id_unique IF NOT EXISTS
FOR (conv:Conversation) REQUIRE conv.id IS UNIQUE;

// Conversation properties:
// - id: UUID
// - title: string
// - message_count: integer
// - created_at: datetime
// - last_message_at: datetime

// ============================================================
// RELATIONSHIP TYPES
// ============================================================

// User -> Agent relationships
// (u:User)-[:OWNS]->(a:Agent)
// (u:User)-[:CREATED]->(a:Agent)
// (u:User)-[:USES]->(a:Agent)

// Agent -> Agent relationships
// (a1:Agent)-[:COLLABORATES_WITH]->(a2:Agent)
// (a1:Agent)-[:DELEGATES_TO]->(a2:Agent)
// (a1:Agent)-[:SUPERVISES]->(a2:Agent)
// (a1:Agent)-[:DERIVED_FROM]->(a2:Agent)

// Agent -> Tool relationships
// (a:Agent)-[:CAN_USE {priority: int, enabled: bool}]->(t:Tool)

// Agent -> Model relationships
// (a:Agent)-[:POWERED_BY]->(m:Model)

// Agent -> Knowledge relationships
// (a:Agent)-[:KNOWS]->(k:Knowledge)
// (a:Agent)-[:LEARNED {timestamp: datetime}]->(k:Knowledge)

// Knowledge relationships
// (k1:Knowledge)-[:RELATED_TO {strength: float}]->(k2:Knowledge)
// (k:Knowledge)-[:CONTRADICTS]->(k2:Knowledge)
// (k:Knowledge)-[:SUPPORTS]->(k2:Knowledge)
// (k:Knowledge)-[:DERIVED_FROM]->(k2:Knowledge)

// Topic relationships
// (t1:Topic)-[:SUBTOPIC_OF]->(t2:Topic)
// (k:Knowledge)-[:BELONGS_TO]->(t:Topic)
// (a:Agent)-[:EXPERT_IN]->(t:Topic)

// Document relationships
// (d:Document)-[:CONTAINS]->(c:Chunk)
// (c:Chunk)-[:NEXT]->(c2:Chunk)
// (k:Knowledge)-[:EXTRACTED_FROM]->(d:Document)

// Conversation relationships
// (u:User)-[:PARTICIPATED_IN]->(conv:Conversation)
// (a:Agent)-[:PARTICIPATED_IN]->(conv:Conversation)

// ============================================================
// VECTOR INDEXES FOR SIMILARITY SEARCH
// ============================================================

// Vector index for agent embeddings
// CREATE VECTOR INDEX agent_embedding_index IF NOT EXISTS
// FOR (a:Agent) ON (a.embedding)
// OPTIONS {indexConfig: {
//   `vector.dimensions`: 1536,
//   `vector.similarity_function`: 'cosine'
// }};

// Vector index for knowledge embeddings
// CREATE VECTOR INDEX knowledge_embedding_index IF NOT EXISTS
// FOR (k:Knowledge) ON (k.embedding)
// OPTIONS {indexConfig: {
//   `vector.dimensions`: 1536,
//   `vector.similarity_function`: 'cosine'
// }};

// Vector index for chunk embeddings (RAG)
// CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
// FOR (c:Chunk) ON (c.embedding)
// OPTIONS {indexConfig: {
//   `vector.dimensions`: 1536,
//   `vector.similarity_function`: 'cosine'
// }};

// ============================================================
// EXAMPLE QUERIES
// ============================================================

// Find all agents owned by a user
// MATCH (u:User {id: $userId})-[:OWNS]->(a:Agent)
// RETURN a

// Find agents that can collaborate
// MATCH (a1:Agent)-[:COLLABORATES_WITH]-(a2:Agent)
// WHERE a1.id = $agentId
// RETURN a2

// Find knowledge related to a topic
// MATCH (t:Topic {name: $topicName})<-[:BELONGS_TO]-(k:Knowledge)
// RETURN k

// Find similar knowledge (semantic search)
// MATCH (k:Knowledge)
// WHERE k.id <> $knowledgeId
// WITH k, gds.similarity.cosine(k.embedding, $embedding) AS similarity
// WHERE similarity > 0.8
// RETURN k, similarity
// ORDER BY similarity DESC
// LIMIT 10

// Find agent expertise path
// MATCH path = (a:Agent)-[:EXPERT_IN]->(t:Topic)-[:SUBTOPIC_OF*0..3]->(parent:Topic)
// WHERE a.id = $agentId
// RETURN path

// Get conversation context with related knowledge
// MATCH (conv:Conversation {id: $conversationId})<-[:PARTICIPATED_IN]-(a:Agent)-[:KNOWS]->(k:Knowledge)
// RETURN conv, a, collect(k) as knowledge

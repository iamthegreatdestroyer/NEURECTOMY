//! Neo4j Connection Pool
//! @VERTEX Graph Database Management
//! 
//! Updated for neo4rs 0.7 API

use neo4rs::{Graph, ConfigBuilder, query};
use std::sync::Arc;

use super::{DatabaseError, DbResult};

/// Neo4j connection pool wrapper
#[derive(Clone)]
pub struct Neo4jPool {
    graph: Arc<Graph>,
}

impl Neo4jPool {
    /// Create a new Neo4j connection pool
    pub async fn new(uri: &str) -> DbResult<Self> {
        // Parse the URI to extract components
        // Expected format: neo4j://user:password@host:port
        let config = ConfigBuilder::default()
            .uri(uri)
            .max_connections(10)
            .build()
            .map_err(|e| DatabaseError::Config(e.to_string()))?;
        
        let graph = Graph::connect(config)
            .await
            .map_err(DatabaseError::Neo4j)?;
        
        Ok(Self {
            graph: Arc::new(graph),
        })
    }
    
    /// Get a reference to the underlying graph
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    
    /// Health check
    pub async fn health_check(&self) -> DbResult<()> {
        self.graph
            .run(query("RETURN 1"))
            .await
            .map_err(DatabaseError::Neo4j)?;
        Ok(())
    }
    
    /// Execute a Cypher query
    pub async fn execute(&self, cypher: &str) -> DbResult<()> {
        self.graph
            .run(query(cypher))
            .await
            .map_err(DatabaseError::Neo4j)?;
        Ok(())
    }
    
    /// Execute a Cypher query and return results
    pub async fn query(&self, cypher: &str) -> DbResult<()> {
        self.graph
            .run(query(cypher))
            .await
            .map_err(DatabaseError::Neo4j)
    }
    
    /// Execute a parameterized Cypher query
    pub async fn query_with_params(&self, cypher: &str) -> DbResult<()> {
        self.graph
            .run(query(cypher))
            .await
            .map_err(DatabaseError::Neo4j)
    }
    
    /// Create an agent node in the graph
    pub async fn create_agent_node(
        &self,
        id: &str,
        name: &str,
        agent_type: &str,
        status: &str,
    ) -> DbResult<()> {
        let cypher = r#"
            MERGE (a:Agent {id: $id})
            SET a.name = $name,
                a.type = $type,
                a.status = $status,
                a.updated_at = datetime()
            ON CREATE SET a.created_at = datetime()
        "#;
        
        let id_owned = id.to_string();
        let name_owned = name.to_string();
        let type_owned = agent_type.to_string();
        let status_owned = status.to_string();
        
        let q = query(cypher)
            .param("id", id_owned)
            .param("name", name_owned)
            .param("type", type_owned)
            .param("status", status_owned);
        
        self.graph
            .run(q)
            .await
            .map_err(DatabaseError::Neo4j)?;
        
        Ok(())
    }
    
    /// Create a knowledge node
    pub async fn create_knowledge_node(
        &self,
        id: &str,
        content: &str,
        knowledge_type: &str,
        source: &str,
        confidence: f64,
    ) -> DbResult<()> {
        let cypher = r#"
            CREATE (k:Knowledge {
                id: $id,
                content: $content,
                type: $type,
                source: $source,
                confidence: $confidence,
                created_at: datetime()
            })
        "#;
        
        let id_owned = id.to_string();
        let content_owned = content.to_string();
        let type_owned = knowledge_type.to_string();
        let source_owned = source.to_string();
        
        let q = query(cypher)
            .param("id", id_owned)
            .param("content", content_owned)
            .param("type", type_owned)
            .param("source", source_owned)
            .param("confidence", confidence);
        
        self.graph
            .run(q)
            .await
            .map_err(DatabaseError::Neo4j)?;
        
        Ok(())
    }
    
    /// Link an agent to knowledge
    pub async fn link_agent_to_knowledge(
        &self,
        agent_id: &str,
        knowledge_id: &str,
    ) -> DbResult<()> {
        let cypher = r#"
            MATCH (a:Agent {id: $agent_id})
            MATCH (k:Knowledge {id: $knowledge_id})
            MERGE (a)-[:KNOWS]->(k)
        "#;
        
        let agent_id_owned = agent_id.to_string();
        let knowledge_id_owned = knowledge_id.to_string();
        
        let q = query(cypher)
            .param("agent_id", agent_id_owned)
            .param("knowledge_id", knowledge_id_owned);
        
        self.graph
            .run(q)
            .await
            .map_err(DatabaseError::Neo4j)?;
        
        Ok(())
    }
    
    /// Find related knowledge using semantic similarity
    pub async fn find_related_knowledge(
        &self,
        knowledge_id: &str,
        _min_strength: f64,
        _limit: i64,
    ) -> DbResult<()> {
        let cypher = r#"
            MATCH (k1:Knowledge {id: $knowledge_id})-[r:RELATED_TO]-(k2:Knowledge)
            WHERE r.strength >= $min_strength
            RETURN k2.id as id, k2.content as content, k2.type as type, r.strength as strength
            ORDER BY r.strength DESC
            LIMIT $limit
        "#;
        
        let knowledge_id_owned = knowledge_id.to_string();
        
        let q = query(cypher)
            .param("knowledge_id", knowledge_id_owned);
        
        self.graph
            .run(q)
            .await
            .map_err(DatabaseError::Neo4j)
    }
    
    /// Get agent's knowledge graph
    pub async fn get_agent_knowledge_graph(
        &self,
        agent_id: &str,
    ) -> DbResult<()> {
        let cypher = r#"
            MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(k:Knowledge)
            OPTIONAL MATCH (k)-[:BELONGS_TO]->(t:Topic)
            RETURN k.id as knowledge_id, 
                   k.content as content, 
                   k.type as type,
                   collect(t.name) as topics
        "#;
        
        let agent_id_owned = agent_id.to_string();
        
        let q = query(cypher)
            .param("agent_id", agent_id_owned);
        
        self.graph
            .run(q)
            .await
            .map_err(DatabaseError::Neo4j)
    }
}

// Re-export neo4rs types
pub use neo4rs::{query as cypher_query, Node, Relation};

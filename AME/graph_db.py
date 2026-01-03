import logging
import asyncio
import re
from typing import List, Dict, Any
from neo4j import GraphDatabase, AsyncGraphDatabase
from sentence_transformers import SentenceTransformer
import config
import schemas

# Suppress Neo4j driver info logs
logging.getLogger('neo4j').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class GraphDB:
    def __init__(self, uri, user, password):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        self._async_driver = None
        
        logger.info("Initializing GraphDB...")
        self._tokenizer = SentenceTransformer("/data/user1/model/contriever-msmarco").tokenizer

    def connect(self):
        """Initialize sync and async Neo4j drivers."""
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._async_driver = AsyncGraphDatabase.driver(
                self._uri, 
                auth=(self._user, self._password),
                connection_acquisition_timeout=120.0, 
                max_connection_pool_size=200
            )
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}", exc_info=True)
            raise

    async def close(self):
        """Close all driver connections."""
        if self._driver:
            self._driver.close()
            logger.info("Sync driver closed.")
        if self._async_driver:
            await self._async_driver.close()
            logger.info("Async driver closed.")

    def _ensure_connection(self):
        if not self._driver or not self._async_driver:
            self.connect()

    def setup_database(self):
        """Initialize constraints, fulltext indexes, and vector indexes (Neo4j 5.0+)."""
        self._ensure_connection()
        logger.info("Setting up database schema...")
        
        dim = config.EMBEDDING_DIMENSION

        # 1. Uniqueness constraints
        constraints = [
            "CREATE CONSTRAINT user_uuid_unique IF NOT EXISTS FOR (u:User) REQUIRE u.uuid IS UNIQUE",
            "CREATE CONSTRAINT instance_uuid_unique IF NOT EXISTS FOR (i:Instance) REQUIRE i.uuid IS UNIQUE",
            "CREATE CONSTRAINT concept_uuid_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.uuid IS UNIQUE",
        ]

        # 2. Performance indexes
        indexes = [
            "CREATE INDEX instance_user_id_index IF NOT EXISTS FOR (n:Instance) ON (n.user_id)",
            "CREATE INDEX concept_user_id_index IF NOT EXISTS FOR (n:Concept) ON (n.user_id)",
            "CREATE INDEX instance_source_mem_index IF NOT EXISTS FOR (n:Instance) ON (n.source_memory_id)",
            "CREATE INDEX concept_source_mem_index IF NOT EXISTS FOR (n:Concept) ON (n.source_memory_id)",
        ]

        # 3. Vector indexes
        vector_indexes = [
            f"""
            CREATE VECTOR INDEX node_embedding_instance IF NOT EXISTS
            FOR (n:Instance) ON (n.name_embedding)
            OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}
            """,
            f"""
            CREATE VECTOR INDEX node_embedding_concept IF NOT EXISTS
            FOR (n:Concept) ON (n.name_embedding)
            OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}
            """,
            f"""
            CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
            FOR ()-[r:FACT]-() ON (r.fact_embedding)
            OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}
            """
        ]

        # 4. Fulltext indexes
        fulltext_indexes = [
            "CREATE FULLTEXT INDEX instance_name_text IF NOT EXISTS FOR (n:Instance) ON EACH [n.name]",
            "CREATE FULLTEXT INDEX concept_name_text IF NOT EXISTS FOR (n:Concept) ON EACH [n.name]"   
        ]
        
        with self._driver.session() as session:
            for query in constraints + indexes + vector_indexes + fulltext_indexes:
                try:
                    session.run(query)
                except Exception as e:
                    first_line = query.strip().splitlines()[0]
                    logger.warning(f"Schema update skipped/failed ({first_line}): {e}")
        
        logger.info("Database setup complete.")

    async def find_similar_nodes(self, user_id: str, vector: List[float], label: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Vector search for nodes within a specific user's subgraph."""
        self._ensure_connection()
        query = f"""
        MATCH (n:{label} {{user_id: $user_id}})
        CALL db.index.vector.queryNodes('node_embedding_{label.lower()}', $top_k, $vector) YIELD node, score
        WHERE node = n AND score >= {config.DEDUPE_LLM_THRESHOLD}
        RETURN node.uuid AS uuid, node.name AS name, score
        """
        try:
            async with self._async_driver.session() as session:
                result = await session.run(query, user_id=user_id, top_k=top_k, vector=vector)
                return [record.data() async for record in result]
        except Exception as e:
            logger.error(f"Vector search failed (Label: {label}): {e}", exc_info=True)
            return []
    
    async def check_user_exists(self, user_id: str) -> bool:
        """Check if User node exists (optimized with LIMIT 1)."""
        self._ensure_connection()
        query = "MATCH (u:User {uuid: $user_id}) RETURN 1 LIMIT 1"
        try:
            async with self._async_driver.session() as session:
                result = await session.run(query, user_id=user_id)
                return await result.single() is not None
        except Exception as e:
            logger.error(f"User check failed ({user_id}): {e}")
            return False
    
    async def get_existing_memory_ids(self, memory_ids: List[str]) -> set:
        """Batch check for existing memory IDs in Instance/Concept nodes."""
        if not memory_ids:
            return set()
        self._ensure_connection()
        query = """
        MATCH (n:Instance) WHERE n.source_memory_id IN $ids RETURN DISTINCT n.source_memory_id as mem_id
        UNION
        MATCH (c:Concept) WHERE c.source_memory_id IN $ids RETURN DISTINCT c.source_memory_id as mem_id
        """
        try:
            async with self._async_driver.session() as session:
                result = await session.run(query, ids=memory_ids)
                return {record["mem_id"] async for record in result}
        except Exception as e:
            logger.error(f"Batch memory ID check failed: {e}")
            return set()

    async def find_similar_relations(
        self,
        source_uuid: str,
        target_uuid: str,
        relation_type: str,
        vector: List[float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Vector search for relationships between specific nodes."""
        self._ensure_connection()
        index_name = "fact_embedding"
        
        query = f"""
        MATCH (a {{uuid: $source_uuid}})-[r:{relation_type}]->(b {{uuid: $target_uuid}}) 
        CALL db.index.vector.queryRelationships('{index_name}', $top_k, $vector) YIELD relationship, score
        WHERE relationship = r AND score >= {config.DEDUPE_LLM_THRESHOLD}
        RETURN properties(r) AS properties, score
        """
        try:
            async with self._async_driver.session() as session:
                result = await session.run(query, source_uuid=source_uuid, target_uuid=target_uuid, top_k=top_k, vector=vector)
                return [record.data() async for record in result]
        except Exception as e:
            logger.error(f"Relation search failed: {e}", exc_info=True)
            return []

    async def batch_save_graph(
        self, 
        user_id: str, 
        nodes_to_save: List[schemas.Node], 
        relations_to_save: List[schemas.Relation]
    ):
        """Batch save nodes and relations using APOC."""
        self._ensure_connection()
        
        # Cypher for generic nodes
        nodes_cypher = """
        UNWIND $nodes_list as node_data
        MERGE (n {uuid: node_data.persistent_uuid})
        ON CREATE SET 
            n.name = node_data.properties.name,
            n.name_embedding = node_data.properties.name_embedding,
            n.source_memory_id = node_data.properties.source_memory_id,
            n.user_id = node_data.properties.user_id
        WITH n, node_data
        CALL apoc.create.addLabels(n, [node_data.label]) YIELD node
        RETURN count(node) as nodes_processed
        """
        
        # Cypher for relations
        relations_cypher = """
        UNWIND $relations_list as rel_data
        MATCH (a {uuid: rel_data.source_persistent_uuid})
        MATCH (b {uuid: rel_data.target_persistent_uuid})
        CALL apoc.create.relationship(a, rel_data.type, rel_data.properties, b) YIELD rel
        RETURN count(rel) as relations_processed
        """
        
        user_node_list = [n.model_dump() for n in nodes_to_save if n.label == 'User']
        other_nodes_list = [n.model_dump() for n in nodes_to_save if n.label != 'User']
        relations_list = [r.model_dump() for r in relations_to_save]

        try:
            async with self._async_driver.session() as session:
                async with await session.begin_transaction() as tx:
                    # 1. Process User nodes
                    if user_node_list:
                        user_cypher = """
                        UNWIND $users as user_data
                        MERGE (u:User {uuid: user_data.persistent_uuid})
                        ON CREATE SET u.name = user_data.properties.name, u.user_id = user_data.properties.user_id
                        ON MATCH SET u.name = user_data.properties.name, u.user_id = user_data.properties.user_id
                        """
                        await tx.run(user_cypher, users=user_node_list)
                    
                    # 2. Process other nodes
                    if other_nodes_list:
                        await tx.run(nodes_cypher, nodes_list=other_nodes_list)
                        
                    # 3. Process relations
                    if relations_list:
                        await tx.run(relations_cypher, relations_list=relations_list)
                        
                    await tx.commit()
            logger.debug(f"Saved {len(other_nodes_list)} nodes, {len(relations_list)} relations for User {user_id}")
        except Exception as e:
            logger.error(f"Batch save failed for User {user_id}: {e}", exc_info=True)
            raise

    async def get_full_graph_context(self, user_id: str) -> str:
        """Retrieve the entire subgraph for a user to build context."""
        self._ensure_connection()
        query = """
        // 1. Match user's own nodes
        MATCH (node)
        WHERE node.user_id = $user_id OR (node:User AND node.uuid = $user_id)
        
        // 2. Optional match for relations and neighbors
        OPTIONAL MATCH (node)-[r]-(m)
        
        // 3. Ensure neighbor also belongs to user
        WHERE m.user_id = $user_id OR (m:User AND m.uuid = $user_id)
        
        RETURN node, r, m, 1.0 as score 
        """
        try:
            async with self._async_driver.session() as session:
                results = await session.run(query, user_id=user_id)
                results_list = [record async for record in results]
                return self._format_context_from_records(results_list)
        except Exception as e:
            logger.error(f"Full graph context retrieval failed for User {user_id}: {e}", exc_info=True)
            return "Error retrieving context."

    async def get_hybrid_search_context(
        self, 
        user_id: str,
        question_text: str, 
        question_vector: List[float], 
        top_k: int = 200,
        final_context_k: int = 200
    ) -> str:
        """
        Hybrid search within user subgraph.
        Combines node expansion (Vector -> Node -> Neighbors) and direct relation matching (Vector -> Relation).
        Uses CALL { UNION } to handle query composition and deprecated features.
        """
        self._ensure_connection()
        limit_count = final_context_k if final_context_k > 0 else top_k

        # Basic text sanitization
        try:
            sanitized_text = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', question_text).strip()
        except:
            sanitized_text = question_text

        cypher_query = """
            CALL {
                // --- Subquery A: Node Expansion ---
                MATCH (n)
                WHERE n.user_id = $user_id AND (n:Instance OR n:Concept)
                WITH n, coalesce(vector.similarity.cosine(n.name_embedding, $vector), 0.0) AS node_score
                ORDER BY node_score DESC LIMIT $k
                
                MATCH (n)-[r]-(target)
                WHERE r.user_id = $user_id 
                RETURN r AS rel, node_score AS score

                UNION

                // --- Subquery B: Direct Relation Match ---
                MATCH ()-[r]->()
                WHERE r.user_id = $user_id AND r.fact_embedding IS NOT NULL
                WITH r, vector.similarity.cosine(r.fact_embedding, $vector) AS rel_score
                ORDER BY rel_score DESC LIMIT $k
                RETURN r AS rel, rel_score AS score
            }

            // --- Aggregation & Formatting ---
            WITH rel, max(score) AS final_score
            ORDER BY final_score DESC
            LIMIT $k
            
            MATCH (start)-[rel]->(end)
            RETURN start, rel AS r, end
        """

        try:
            async with self._async_driver.session() as session:
                result = await session.run(
                    cypher_query, 
                    user_id=user_id, 
                    vector=question_vector, 
                    text=sanitized_text, 
                    k=limit_count
                )
                records = [record async for record in result]
                return self._format_context_from_records(records)
        except Exception as e:
            logger.error(f"Hybrid Search Error: {e}", exc_info=True)
            return "no related information found."

    def _format_context_from_records(self, records: List[Dict[str, Any]]) -> str:
        """Format Neo4j records into a natural language string."""
        relations = set()
        if not records:
            return "no related information found in knowledge graph."

        for record in records:
            rel_obj = record.get("r")
            if rel_obj:
                # 1. Smart Retrieval: Try retrieving nodes from relationship object or record keys
                start_node = rel_obj.start_node or record.get("node") or record.get("start")
                end_node = rel_obj.end_node or record.get("m") or record.get("end")

                # 2. Format
                if start_node and end_node:
                    start_name = start_node.get('name', 'unnamed').strip()
                    end_name = end_node.get('name', 'unnamed').strip()
                    
                    rel_props = dict(rel_obj)
                    rel_type = rel_props.get('invented_type', rel_obj.type)
                    rel_type_clean = " ".join(rel_type.split())
                    
                    # Create lower-case natural language string
                    relations.add(f"{start_name} {rel_type_clean} {end_name}".lower())

        context_str = "--- relations ---\n"
        if not relations:
            context_str += "none\n"
        else:
            context_str += "\n".join(sorted(relations)) + "\n"
        return context_str

# Singleton instance
graph_db_instance = GraphDB(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
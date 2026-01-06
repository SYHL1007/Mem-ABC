import logging
import asyncio
import uuid
import json
from typing import List, Dict, Tuple
from schemas import Node, Relation
import schemas
from llm_service import LLMService
from embedding_service import EmbeddingService
from graph_db import GraphDB
import config

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self, llm: LLMService, embed: EmbeddingService, db: GraphDB):
        self.llm = llm
        self.embed = embed
        self.db = db
    
    def _clean_for_logging(self, data: dict) -> dict:
        """Helper function to deep copy a dictionary and remove embeddings for logging purposes."""
        from copy import deepcopy
        cleaned_data = deepcopy(data)
        cleaned_data.pop('name_embedding', None)
        cleaned_data.pop('fact_embedding', None)
        
        # Remove embeddings from the 'properties' field if present
        if 'properties' in cleaned_data and isinstance(cleaned_data['properties'], dict):
            cleaned_data['properties'].pop('name_embedding', None)
            cleaned_data['properties'].pop('fact_embedding', None)
            
        return cleaned_data

    async def process_memory(self, user_id: str, memory_id: str, memory_text: str):
        """
        Process a single memory entry, performing the extraction of nodes -> embedding -> deduplication -> 
        extraction of relations -> embedding -> deduplication -> storage.
        """
        logger.debug(f"[User: {user_id}, Mem: {memory_id}] Processing initiated...")
        
        try:
            # Step 1: Extract nodes
            node_response = await self.llm.extract_nodes(memory_text)
            await asyncio.sleep(1.0)

            if not node_response or not node_response.nodes:
                logger.warning(f"[User: {user_id}, Mem: {memory_id}] No nodes were extracted.")
                return
            
            # Step 2: Handle node embedding and deduplication
            temp_id_map, nodes_to_save = await self._resolve_nodes(
                user_id, memory_id, memory_text, node_response.nodes
            )
            
            if not temp_id_map:
                logger.warning(f"[User: {user_id}, Mem: {memory_id}] No valid nodes after deduplication.")
                return

            # Step 3: Extract relations
            nodes_for_llm = [
                {
                    "temp_id": n.temp_id,
                    "label": n.label,
                    "properties": {"name": n.properties.name}
                }
                for n in node_response.nodes
            ]

            relation_response = await self.llm.extract_relations(
                memory_text, json.dumps(nodes_for_llm)
            )
            
            if not relation_response or not relation_response.relations:
                logger.debug(f"[User: {user_id}, Mem: {memory_id}] No relations were extracted.")
                await self.db.batch_save_graph(user_id, nodes_to_save, [])
                return
                
            # Step 4: Handle relation mapping, embedding, and deduplication
            relations_to_save = await self._resolve_relations(
                user_id, memory_id, memory_text, relation_response.relations, temp_id_map, nodes_to_save
            )
            
            # Step 5: Save nodes and relations
            await self.db.batch_save_graph(user_id, nodes_to_save, relations_to_save)
            
            logger.debug(f"[User: {user_id}, Mem: {memory_id}] Processing completed.")

        except Exception as e:
            logger.error(f"[User: {user_id}, Mem: {memory_id}] Processing failed: {e}", exc_info=True)


    async def _resolve_nodes(
        self, user_id: str, memory_id: str, memory_text: str, nodes_from_llm: List[Node]
    ) -> Tuple[Dict[str, str], List[Node]]:
        """
        Perform embedding and deduplication on extracted nodes.
        Returns a mapping of temp_id to persistent_uuid and a list of nodes to save.
        """
        temp_id_map = {}
        nodes_to_save = []
        
        # Extract names for embedding
        names_to_embed = []
        nodes_to_process = []
        
        for node in nodes_from_llm:
            if node.temp_id == "user":
                node.persistent_uuid = user_id 
                node.label = "User" 
                temp_id_map[node.temp_id] = user_id
                nodes_to_save.append(node)
            else:
                names_to_embed.append(node.properties.name)
                node.properties.user_id = user_id
                nodes_to_process.append(node)

        if not nodes_to_process:
            return temp_id_map, nodes_to_save
            
        embeddings = self.embed.embed_batch(names_to_embed)
        
        # Resolve each node
        tasks = []
        for node, embedding in zip(nodes_to_process, embeddings):
            node.properties.name_embedding = embedding
            node.properties.source_memory_id = memory_id
            tasks.append(self._find_or_create_node(user_id, memory_text, node))
            
        resolved_nodes: List[Node] = await asyncio.gather(*tasks)
        
        # Construct final mapping
        final_nodes_to_save = {n.persistent_uuid: n for n in nodes_to_save}
        for llm_node, resolved_node in zip(nodes_to_process, resolved_nodes):
            if resolved_node.persistent_uuid:
                temp_id_map[llm_node.temp_id] = resolved_node.persistent_uuid
                if resolved_node.persistent_uuid not in final_nodes_to_save:
                    final_nodes_to_save[resolved_node.persistent_uuid] = resolved_node
                
        return temp_id_map, list(final_nodes_to_save.values())

    async def _find_or_create_node(self, user_id: str, memory_text: str, node: Node) -> Node:
        """
        Deduplication logic for nodes, ensuring unique UUID and consistent naming.
        """
        similar_nodes = await self.db.find_similar_nodes(
            user_id, node.properties.name_embedding, node.label.strip(':'), top_k=5
        )
        
        logger.debug(f"Node deduplication check for: '{node.properties.name}'")

        if not similar_nodes:
            node.persistent_uuid = str(uuid.uuid4())
            logger.debug("Verdict: NEW (No similar nodes found)")
            return node

        logger.debug(f"Found {len(similar_nodes)} similar existing nodes for review.")

        for sim_node in similar_nodes:
            logger.debug(f"  Reviewing existing node: '{sim_node['name']}' with score {sim_node['score']:.4f}")
            
            if sim_node['score'] >= config.DEDUPE_MERGE_THRESHOLD:
                node.persistent_uuid = sim_node['uuid']
                node.properties.name = sim_node['name']
                logger.debug("Verdict: MERGE (High-confidence)")
                return node
        
        # No suitable match found, create a new node
        node.persistent_uuid = str(uuid.uuid4())
        logger.debug("Final Verdict: NEW (All candidates reviewed, no merge decision)")
        return node

    async def _resolve_relations(
        self,
        user_id: str,
        memory_id: str,
        memory_text: str,
        relations_from_llm: List[Relation],
        temp_id_map: Dict[str, str],
        resolved_nodes_list: List[Node]
    ) -> List[Relation]:
        """
        Deduplicates relations based on embeddings.
        """
        nodes_by_uuid = {n.persistent_uuid: n for n in resolved_nodes_list if n.persistent_uuid}
        relations_to_save: List[Relation] = []
        facts_to_embed = []
        relations_to_process = []
        
        for rel in relations_from_llm:
            source_uuid = temp_id_map.get(rel.source_temp_id)
            target_uuid = temp_id_map.get(rel.target_temp_id)
            if not source_uuid or not target_uuid: 
                continue

            rel.type = "FACT"
            rel.source_persistent_uuid = source_uuid
            rel.target_persistent_uuid = target_uuid
            rel.properties.source_memory_id = memory_id
            rel.properties.user_id = user_id

            # Construct fact text
            source_node = nodes_by_uuid.get(source_uuid)
            target_node = nodes_by_uuid.get(target_uuid)
            source_name = "User" if rel.source_temp_id == "user" else (source_node.properties.name if source_node else "Unknown")
            target_name = "User" if rel.target_temp_id == "user" else (target_node.properties.name if target_node else "Unknown")
            
            fact_text = f"({source_name})-[{rel.properties.invented_type}]->({target_name})"
            
            rel.properties.fact = fact_text
            facts_to_embed.append(fact_text)
            relations_to_process.append(rel)
            
        if not relations_to_process: 
            return []
            
        relation_embeddings = self.embed.embed_batch(facts_to_embed)

        # Perform deduplication
        for candidate_rel, embedding in zip(relations_to_process, relation_embeddings):
            candidate_rel.properties.fact_embedding = embedding
            
            logger.debug(f"Relation deduplication check for: '{candidate_rel.properties.fact}'")

            similar_rels = await self.db.find_similar_relations(
                candidate_rel.source_persistent_uuid, 
                candidate_rel.target_persistent_uuid, 
                "FACT",
                candidate_rel.properties.fact_embedding,
                top_k=1 
            )

            if similar_rels and similar_rels[0]['score'] >= config.DEDUPE_MERGE_THRESHOLD:
                logger.debug(f"Verdict: DISCARD (Similar fact found in DB with score {similar_rels[0]['score']:.4f})")
                continue
            
            logger.debug(f"Verdict: NEW (No duplicate found in DB)")
            relations_to_save.append(candidate_rel)

        return relations_to_save
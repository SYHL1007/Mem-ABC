#kg_bulider.py
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
        """一个辅助函数，用于深度复制一个字典并移除所有嵌入向量以供日志记录。"""
        # 使用深拷贝避免修改原始数据
        from copy import deepcopy
        cleaned_data = deepcopy(data)
        
        # 移除顶层的嵌入
        cleaned_data.pop('name_embedding', None)
        cleaned_data.pop('fact_embedding', None)
        
        # 检查 'properties' 字段（针对 Pydantic 模型 dump 和关系字典）
        if 'properties' in cleaned_data and isinstance(cleaned_data['properties'], dict):
            cleaned_data['properties'].pop('name_embedding', None)
            cleaned_data['properties'].pop('fact_embedding', None)
            
        return cleaned_data

    async def process_memory(self, user_id: str, memory_id: str, memory_text: str):
        """
        处理单条记忆，完成 节点抽取->嵌入->去重->关系抽取->嵌入->去重->存储 的完整流程。
        """
        logger.debug(f"[User: {user_id}, Mem: {memory_id}] 开始处理...")
        
        try:
            # 1. 节点抽取
            node_response = await self.llm.extract_nodes(memory_text)
            await asyncio.sleep(1.0)
            if not node_response or not node_response.nodes:
                logger.warning(f"[User: {user_id}, Mem: {memory_id}] LLM 未抽取到节点")
                return
            
            # 2. 节点嵌入与去重 (核心逻辑)
            # temp_id -> persistent_uuid 的映射
            temp_id_map, nodes_to_save = await self._resolve_nodes(
                user_id, memory_id, memory_text, node_response.nodes
            )
            
            if not temp_id_map:
                logger.warning(f"[User: {user_id}, Mem: {memory_id}] 节点解析/去重后无有效节点")
                return

            # 3. 关系抽取
            # 将带有持久化UUID的节点列表传给LLM，以便它使用 'temp_id'
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
                logger.debug(f"[User: {user_id}, Mem: {memory_id}] LLM 未抽取到关系")
                # 即使没有关系，我们仍然保存节点
                await self.db.batch_save_graph(user_id, nodes_to_save, [])
                return
                
            # 4. 关系映射、嵌入与去重
            relations_to_save = await self._resolve_relations(
                user_id, memory_id, memory_text, relation_response.relations, temp_id_map, nodes_to_save
            )
            
            # 5. 批量存入数据库
            await self.db.batch_save_graph(user_id, nodes_to_save, relations_to_save)
            
            logger.debug(f"[User: {user_id}, Mem: {memory_id}] 处理完成。")

        except Exception as e:
            logger.error(f"[User: {user_id}, Mem: {memory_id}] 处理失败: {e}", exc_info=True)


    async def _resolve_nodes(
        self, user_id: str, memory_id: str, memory_text: str, nodes_from_llm: List[Node]
    ) -> Tuple[Dict[str, str], List[Node]]:
        """
        对LLM抽取的节点进行嵌入和去重。
        返回: (temp_id 到 persistent_uuid 的映射, 准备存入DB的Node列表)
        """
        temp_id_map = {}
        nodes_to_save = []
        
        # 1. 提取名称并批量嵌入
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
                # 将 user_id 注入
                node.properties.user_id = user_id
                nodes_to_process.append(node)

        if not nodes_to_process:
            return temp_id_map, nodes_to_save
            
        embeddings = self.embed.embed_batch(names_to_embed)
        
        # 2. 异步解析每个节点
        tasks = []
        for node, embedding in zip(nodes_to_process, embeddings):
            node.properties.name_embedding = embedding
            node.properties.source_memory_id = memory_id
            tasks.append(self._find_or_create_node(user_id, memory_text, node))
            
        resolved_nodes: List[Node] = await asyncio.gather(*tasks)
        
        # 3. 构建映射
        final_nodes_to_save = {n.persistent_uuid: n for n in nodes_to_save} # 先把 user 节点加进去
        for llm_node, resolved_node in zip(nodes_to_process, resolved_nodes):
            if resolved_node.persistent_uuid:
                temp_id_map[llm_node.temp_id] = resolved_node.persistent_uuid
                if resolved_node.persistent_uuid not in final_nodes_to_save:
                    final_nodes_to_save[resolved_node.persistent_uuid] = resolved_node
                
        return temp_id_map, list(final_nodes_to_save.values())

    async def _find_or_create_node(self, user_id: str, memory_text: str, node: Node) -> Node:
        """
        [最终修复版] 执行多候选项的“双阈值”去重逻辑。
        在合并时，同时统一UUID和节点名称。
        """
        similar_nodes = await self.db.find_similar_nodes(
            user_id, node.properties.name_embedding, node.label.strip(':'), top_k=5
        )
        
        logger.debug(f"--- 🕵️ Node Dedupe Check for: '{node.properties.name}' ---")

        if not similar_nodes:
            node.persistent_uuid = str(uuid.uuid4())
            logger.debug(f"  - Verdict: NEW (No similar nodes found)")
            return node

        logger.debug(f"  - Candidate: {json.dumps(self._clean_for_logging(node.model_dump()), indent=2, ensure_ascii=False)}")
        logger.debug(f"  - Found {len(similar_nodes)} Similar Existing Nodes for review.")

        for sim_node in similar_nodes:
            logger.debug(f"  - Reviewing existing node: '{sim_node['name']}' with score {sim_node['score']:.4f}")
            
            if sim_node['score'] >= config.DEDUPE_MERGE_THRESHOLD:
                node.persistent_uuid = sim_node['uuid']
                node.properties.name = sim_node['name']
                logger.debug(f"  - Verdict: MERGE (High-Confidence)")
                logger.debug(f"    - Score {sim_node['score']:.4f} >= {config.DEDUPE_MERGE_THRESHOLD}")
                logger.debug(f"    - Merged with: {self._clean_for_logging(sim_node)}")
                return node

            logger.debug(f"  - Verdict: ADJUDICATING (Medium-Confidence)")
            
            cleaned_sim_node_for_llm = self._clean_for_logging(sim_node)
            # 清理候选节点的 embedding，避免传递给 LLM
            cleaned_candidate_node = self._clean_for_logging(node.model_dump())
            # 将清理后的字典转换回 Node 对象（用于 model_dump_json）
            from schemas import NodeProperties
            cleaned_node = Node(
                temp_id=node.temp_id,
                label=node.label,
                persistent_uuid=node.persistent_uuid,
                properties=NodeProperties(**cleaned_candidate_node.get('properties', {}))
            )
            
            decision_response = await self.llm.decide_node_dedupe(
                memory_text,
                cleaned_sim_node_for_llm,
                cleaned_node
            )
            
            if decision_response and decision_response.decision == "MERGE":
                node.persistent_uuid = decision_response.merge_target_uuid or sim_node['uuid']
                # [关键修复] 使用数据库中已有的规范名称
                node.properties.name = sim_node['name'] 
                logger.debug(f"  - Final Verdict: MERGE (LLM Decision)")
                logger.debug(f"    - Merged with UUID: {node.persistent_uuid}")
                logger.debug(f"    - Node name normalized to: '{sim_node['name']}'")
                return node
            else:
                reason = decision_response.reason if decision_response else "LLM Adjudication failed"
                logger.debug(f"  - Verdict: CONTINUE (LLM decided NEW for this candidate. Reason: {reason})")
        
        # 如果循环走完都没有找到可以合并的节点
        node.persistent_uuid = str(uuid.uuid4())
        logger.debug(f"  - Final Verdict: NEW (All candidates reviewed, no merge decision)")
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
        [*** 重构版-最终版 ***]
        - 使用“泛型FACT关系”模型。
        - 移除 LLM 冲突解决 (提示词4)。
        - 仅依赖向量相似度进行去重 (DISCARD or CREATE)。
        """
        nodes_by_uuid = {n.persistent_uuid: n for n in resolved_nodes_list if n.persistent_uuid}
        
        # 最终保存的关系列表
        relations_to_save: List[Relation] = []
        
        # 1. 准备批量嵌入
        facts_to_embed = []
        relations_to_process = []
        
        for rel in relations_from_llm:
            source_uuid = temp_id_map.get(rel.source_temp_id)
            target_uuid = temp_id_map.get(rel.target_temp_id)
            if not source_uuid or not target_uuid: 
                continue
            
            # [!!!] 关键：确保类型是 "FACT"
            # (LLM 应该返回 "FACT", 但我们最好还是强制覆盖它以确保安全)
            rel.type = "FACT" 
            
            rel.source_persistent_uuid = source_uuid
            rel.target_persistent_uuid = target_uuid
            rel.properties.source_memory_id = memory_id
            rel.properties.user_id = user_id

            # (健壮的名称查找)
            source_node = nodes_by_uuid.get(source_uuid)
            target_node = nodes_by_uuid.get(target_uuid)
            # [!!] 修复：User 节点可能不在 resolved_nodes_list 中
            source_name = "User" if rel.source_temp_id == "user" else (source_node.properties.name if source_node else "Unknown")
            target_name = "User" if rel.target_temp_id == "user" else (target_node.properties.name if target_node else "Unknown")
            
            # [!!!] 新的 fact_text 构建, e.g., "(User)-[LIKES]->(Gaming)"
            fact_text = f"({source_name})-[{rel.properties.invented_type}]->({target_name})"
            
            rel.properties.fact = fact_text
            facts_to_embed.append(fact_text)
            relations_to_process.append(rel)
            
        if not relations_to_process: 
            return []
            
        relation_embeddings = self.embed.embed_batch(facts_to_embed)

        # 2. [新] 向量去重逻辑
        for candidate_rel, embedding in zip(relations_to_process, relation_embeddings):
            candidate_rel.properties.fact_embedding = embedding
            
            logger.debug(f"--- 🕵️ Relation Dedupe Check for: '{candidate_rel.properties.fact}' ---")

            # 检查数据库中是否已存在 *非常相似* 的事实
            similar_rels = await self.db.find_similar_relations(
                candidate_rel.source_persistent_uuid, 
                candidate_rel.target_persistent_uuid, 
                "FACT",  # <--- [!!!] 硬编码为 "FACT"
                candidate_rel.properties.fact_embedding,
                top_k=1 
            )

            # [!!!] 关键：我们只关心分数最高的那个是否构成重复
            if similar_rels and similar_rels[0]['score'] >= config.DEDUPE_MERGE_THRESHOLD:
                # 已经存在一个几乎一样的关系，丢弃这个候选。
                logger.debug(f"  - Verdict: DISCARD (Similar fact found in DB with score {similar_rels[0]['score']:.4f})")
                continue
            
            # 这是一个新事实
            logger.debug(f"  - Verdict: NEW (No duplicate found in DB)")
            relations_to_save.append(candidate_rel)

        return relations_to_save
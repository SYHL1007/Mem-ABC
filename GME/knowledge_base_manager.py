# knowledge_base_manager.py

import os
import json
import re
import asyncio
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

# Assumes prompts.py exists and contains the required prompts
from prompts import FACT_EXTRACTION_PROMPT, MEMORY_UPDATE_PROMPT, SUMMARY_UPDATE_PROMPT

from llm_services import (
    call_llm_async_with_retry,
    parse_llm_update_response,
    retrieve_similar_memories_local_sync
)

class KnowledgeBaseManager:
    def __init__(self, filepath: str, async_client: AsyncOpenAI, llm_model: str, embedding_model: SentenceTransformer, verbose: bool = False):
        self.filepath = filepath
        self.async_client = async_client
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.kb = {}

    def set_kb(self, kb_data: dict):
        self.kb = kb_data

    def get_or_create_entry(self, question_id: str, question_text: str = ""):
        if question_id not in self.kb:
            self.kb[question_id] = {"question": question_text, "global_summary": "The user's initial profile is empty.", "memories": {}, "memory_id_counter": 0}
        elif "question" not in self.kb[question_id] or not self.kb[question_id].get("question"):
            self.kb[question_id]["question"] = question_text
        return self.kb[question_id]

    async def _extract_facts(self, prompt: str) -> list[str] | None:
        response_str = await call_llm_async_with_retry(self.async_client, prompt, self.llm_model)
        if not response_str: return None
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not match:
            if self.verbose: print(f"   ❗️ No valid JSON structure found in the LLM response")
            return None
        cleaned_json_str = match.group(0)
        try:
            data = json.loads(cleaned_json_str)
            if isinstance(data, dict) and "facts" in data and isinstance(data["facts"], list):
                return data["facts"]
        except json.JSONDecodeError:
            if self.verbose: print(f"   ❗️ Failed to parse JSON: {cleaned_json_str}")
        return None

    async def _decide_and_execute_operation(self, question_id: str, new_fact: str) -> str | None:
        entry = self.get_or_create_entry(question_id)
        memories = entry["memories"]
        loop = asyncio.get_running_loop()
        similar_memories = await loop.run_in_executor(None, retrieve_similar_memories_local_sync, self.embedding_model, new_fact, memories, 5)
        prompt = MEMORY_UPDATE_PROMPT.format(
            existing_memories="\n".join(f"- {mem}" for mem in similar_memories.values()) if similar_memories else "None",
            new_fact=new_fact
        )
        response_str = await call_llm_async_with_retry(self.async_client, prompt, self.llm_model)
        decision = parse_llm_update_response(response_str)
        
        operation = decision["operation"]
        payload = decision["payload"]
        if self.verbose: print(f"   🧠 Memory decision: {operation}, Content: {payload or 'N/A'}")
        if operation == "ADD":
            counter = entry["memory_id_counter"]
            new_id = f"{question_id}_mem_{counter}"
            entry["memories"][new_id] = new_fact
            entry["memory_id_counter"] += 1
            if self.verbose: print(f"   ✅ [ADD] Added new memory (ID: {new_id}): {new_fact}")
            return new_fact
        elif operation == "UPDATE":
            if similar_memories and payload:
                most_similar_id = list(similar_memories.keys())[0]
                entry["memories"][most_similar_id] = payload
                if self.verbose: print(f"   🔄 [UPDATE] Updated memory (ID: {most_similar_id}): {payload}")
                return payload
        elif operation == "DELETE":
            if similar_memories:
                most_similar_id = list(similar_memories.keys())[0]
                old_mem = entry["memories"].pop(most_similar_id, "Unknown memory")
                if self.verbose: print(f"   🗑️ [DELETE] Deleted conflicting memory (ID: {most_similar_id}, Content: '{old_mem}')")
        return None

    async def process_single_profile_item(self, question_id: str, profile_text: str, current_summary: str):
        prompt = FACT_EXTRACTION_PROMPT.format(global_summary=current_summary, profile_text=profile_text)
        new_facts = await self._extract_facts(prompt)
        
        if not new_facts:
            return
            
        if self.verbose:
            print(f"   > Extracted {len(new_facts)} facts: {new_facts}")

        # Original processing: call LLM to decide for each fact
        for fact in new_facts:
            await self._decide_and_execute_operation(question_id, fact)

    async def generate_new_summary(self, current_summary: str, recent_raw_texts: list[str]):
        if not recent_raw_texts: return current_summary
        formatted_texts = "\n".join(f"- \"{text}\"" for text in recent_raw_texts)
        prompt = SUMMARY_UPDATE_PROMPT.format(current_summary=current_summary, recent_raw_texts=formatted_texts)
        updated_summary = await call_llm_async_with_retry(self.async_client, prompt, self.llm_model)
        if self.verbose and updated_summary: print(f"🚀 [SUMMARY UPDATED]")
        return updated_summary or current_summary
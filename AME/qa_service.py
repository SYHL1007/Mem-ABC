import logging
import json
import datetime
import time
from typing import Any, Dict
from schemas import QARecord, ProfileSynthesisResponse
import prompts
from llm_service import LLMService
from embedding_service import EmbeddingService
from graph_db import GraphDB
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type, 
    RetryError
)

logger = logging.getLogger(__name__)

retry_strategy = retry(
    wait=wait_exponential(multiplier=1, min=2, max=200),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True 
)

class QAService:
    def __init__(self, llm: LLMService, embed: EmbeddingService, db: GraphDB):
        self.llm = llm
        self.embed = embed
        self.db = db
        self.log_file = "qa_log.jsonl" 

    @retry_strategy
    async def _get_context_with_retry(self, user_id: str, question: str, strategy: str, final_context_k: int = 100, json: bool = True) -> str:
        """Encapsulated context retrieval logic (retriable)."""
        if strategy == "hybrid":
            logger.debug(f"[QA User: {user_id}] Using hybrid retrieval strategy...")
            question_vector = self.embed.embed_text(question)
            return await self.db.get_hybrid_search_context(user_id, question, question_vector, final_context_k=final_context_k)
        elif strategy == "full":
            logger.debug(f"[QA User: {user_id}] Using full graph retrieval strategy...")
            return await self.db.get_full_graph_context(user_id)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

    @retry_strategy
    async def _generate_answer_from_context_retry(self, question: str, context: str, json: bool = True) -> str:
        """Encapsulated standard LLM answer generation (retriable)."""
        return await self.llm.generate_answer_from_context(question, context, json)

    @retry_strategy
    async def _synthesize_profile_retry(self, context: str) -> ProfileSynthesisResponse:
        """Encapsulated profile synthesis LLM call (retriable)."""
        response = await self.llm.synthesize_profile(context)
        if not response or not response.profile:
            raise Exception("LLM failed to synthesize a valid profile")
        return response

    @retry_strategy
    async def _generate_answer_from_profile_retry(self, question: str, profile: str) -> str:
        """Encapsulated enhanced LLM answer generation (retriable)."""
        return await self.llm.generate_answer_from_profile(question, profile)
    
    async def answer_question(
        self, 
        user_id: str, 
        question: str, 
        strategy: str = "hybrid",
        enhanced_qa: bool = False,
        final_context_k: int = 100,
        json_output: bool = True
    ) -> Dict[str, Any]:
        """
        Answer questions based on user's knowledge graph context.
        Supports both standard and enhanced (profile-based) QA paths.
        """
        logger.debug(f"[QA User: {user_id}] Received question: {question} (Strategy: {strategy}, Enhanced: {enhanced_qa})")
        
        start_time = time.time()
        context = ""
        answer = ""
        full_prompt = ""
        synthesized_profile = None
        
        # 1. Retrieve Context
        try:
            context = await self._get_context_with_retry(user_id, question, strategy, final_context_k=final_context_k, json=json_output)
            logger.debug(f"[QA User: {user_id}] --- 🔍 Retrieved Context ---\n{context}")
        except RetryError as e:
            logger.error(f"[QA User: {user_id}] Context retrieval failed after 5 retries: {e}", exc_info=True)
            context = "Error retrieving knowledge graph (multiple retries failed)."
            answer = "Unable to retrieve context to answer question."
        except Exception as e:
            logger.error(f"[QA User: {user_id}] Context retrieval failed: {e}", exc_info=True)
            context = "Error retrieving knowledge graph."
            answer = "Unable to retrieve context to answer question."

        # 2. Enhanced QA Path
        if enhanced_qa and not answer:
            logger.debug(f"[QA User: {user_id}] Using [Enhanced] 2-step QA path...")
            try:
                # Step 2a: Synthesize Profile
                synth_response = await self._synthesize_profile_retry(context)
                synthesized_profile = synth_response.profile
                logger.debug(f"[QA User: {user_id}] --- 🧠 Synthesized Profile ---\n{synthesized_profile}")
                
                # Step 2b: Generate Answer from Profile
                answer = await self._generate_answer_from_profile_retry(question, synthesized_profile)
                
                system_prompt = prompts.PROMPT_QA_WITH_PROFILE["system"]
                user_prompt = prompts.PROMPT_QA_WITH_PROFILE["user"].format(profile=synthesized_profile, question=question)
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

            except RetryError as e:
                logger.error(f"[QA User: {user_id}] Enhanced QA failed after 5 retries: {e}", exc_info=True)
                answer = "Error generating answer (Enhanced QA multiple retries failed)."
            except Exception as e:
                logger.error(f"[QA User: {user_id}] Enhanced QA failed: {e}", exc_info=True)
                answer = "Error generating answer (Enhanced QA failed)."

        # 3. Standard QA Path
        elif not enhanced_qa and not answer:
            logger.debug(f"[QA User: {user_id}] Using [Standard] 1-step QA path...")
            try:
                answer = await self._generate_answer_from_context_retry(question, context, json_output)
                
                if json_output:
                    system_prompt = prompts.PROMPT_QA_JSON["system"]
                    user_prompt = prompts.PROMPT_QA_JSON["user"].format(context=context, question=question)
                else:
                    system_prompt = prompts.PROMPT_QA["system"]
                    user_prompt = prompts.PROMPT_QA["user"].format(context=context, question=question) 
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

            except RetryError as e:
                logger.error(f"[QA User: {user_id}] Standard QA failed after 5 retries: {e}", exc_info=True)
                answer = "Error generating answer (Standard QA multiple retries failed)."
            except Exception as e:
                logger.error(f"[QA User: {user_id}] Standard QA failed: {e}", exc_info=True)
                answer = "Error generating answer (Standard QA failed)."
        
        end_time = time.time()
        duration = end_time - start_time
        
        final_answer_str = answer
        if isinstance(answer, (dict, list)):
            final_answer_str = json.dumps(answer, ensure_ascii=False)
        elif not isinstance(answer, str):
            final_answer_str = str(answer)

        # 4. Save QA Log
        log_record = QARecord(
            user_id=user_id,
            question=question,
            prompt=full_prompt,
            answer=final_answer_str,
            strategy=strategy,
            duration=duration,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            enhanced_qa=enhanced_qa,
            synthesized_profile=synthesized_profile
        )
        self._save_log(log_record)
        
        logger.debug(f"[QA User: {user_id}] Answer: {str(answer)[:50]}... (Duration: {duration:.2f}s)")
        
        return log_record.model_dump()

    def _save_log(self, record: QARecord):
        """Append QA record to file in JSONL format."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to save QA log: {e}", exc_info=True)
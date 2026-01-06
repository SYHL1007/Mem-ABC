from pydantic import ValidationError, BaseModel
import config
import prompts
import schemas
from utils import retry_async_llm_call
from typing import Dict, Any, Optional, Type
import logging
from openai import AsyncOpenAI, APIConnectionError
import asyncio

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.LLM_TIMEOUT,
        )
        logger.info(f"LLMService initialized, base_url: {config.OPENAI_BASE_URL}")
        logger.info(f"Model for extraction: {config.LLM_MODEL_EXTRACT}")
        logger.info(f"Model for reasoning: {config.LLM_MODEL_REASON}")

    @retry_async_llm_call
    async def _call_api_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        model_name: str
    ) -> Optional[BaseModel]:
        content = ""
        try:
            completion = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            validated_data = response_model.model_validate_json(content)
            return validated_data

        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(
                f"Validation failed (Model: {model_name}), retrying... Error: {e} | Raw Content: {content}"
            )
            raise APIConnectionError(request=None, message="Retrying due to validation error.")

        except Exception as e:
            logger.error(f"LLM API call failed (Model: {model_name}): {e}", exc_info=True)
            raise 

    async def extract_nodes(self, text: str) -> Optional[schemas.NodeExtractionResponse]:
        logger.debug(f"Extracting nodes (Fast Model) for text: {text[:50]}...")
        system_prompt = prompts.PROMPT_NODE_EXTRACT["system"]
        user_prompt = prompts.PROMPT_NODE_EXTRACT["user"].format(input_text=text)
        response = await self._call_api_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=schemas.NodeExtractionResponse,
            model_name=config.LLM_MODEL_EXTRACT
        )
        if response:
            logger.debug(f"Extracted Nodes:\n{response.model_dump()}")
        return response

    async def extract_relations(self, text: str, nodes_json: str) -> Optional[schemas.RelationExtractionResponse]:
        logger.debug(f"Extracting relations (Fast Model) for text: {text[:50]}...")
        system_prompt = prompts.PROMPT_RELATION_EXTRACT["system"]
        user_prompt = prompts.PROMPT_RELATION_EXTRACT["user"].format(
            input_text=text,
            node_list_json=nodes_json
        )
        response = await self._call_api_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=schemas.RelationExtractionResponse,
            model_name=config.LLM_MODEL_EXTRACT
        )
        if response:
            logger.debug(f"Extracted Relations:\n{response.model_dump()}")
        return response

    async def decide_node_dedupe(
        self,
        input_text: str,
        existing_node: Dict[str, Any],
        candidate_node: schemas.Node
    ) -> Optional[schemas.NodeDedupeDecision]:
        logger.debug(f"Node deduplication decision (Smart Model): {candidate_node.properties.name}")
        user_prompt = prompts.PROMPT_NODE_DEDUPE["user"].format(
            input_text=input_text,
            existing_node=json.dumps(existing_node),
            candidate_node=candidate_node.model_dump_json()
        )
        system_prompt = prompts.PROMPT_NODE_DEDUPE["system"]
        response = await self._call_api_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=schemas.NodeDedupeDecision,
            model_name=config.LLM_MODEL_EXTRACT
        )
        if response:
            logger.debug(f"Dedupe Decision:")
            logger.debug(f"  Candidate: {candidate_node.properties.name}")
            logger.debug(f"  Decision: {response.decision}")
            if response.decision == "MERGE":
                logger.debug(f"  Target: {response.merge_target_uuid}")
            logger.debug(f"  Reason: {response.reason}")
        return response

    async def synthesize_profile(self, context: str) -> Optional[schemas.ProfileSynthesisResponse]:
        logger.debug(f"Synthesizing profile (Smart Model)...")
        system_prompt = prompts.PROMPT_PROFILE_SYNTHESIZE["system"]
        user_prompt = prompts.PROMPT_PROFILE_SYNTHESIZE["user"].format(context=context)
        response = await self._call_api_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=schemas.ProfileSynthesisResponse,
            model_name=config.LLM_MODEL_REASON
        )
        return response

    async def generate_answer_from_context(self, question: str, context: str) -> str:
        logger.debug(f"Standard QA (Smart Model): {question[:50]}...")
        user_prompt = prompts.PROMPT_QA["user"].format(context=context, question=question)
        system_prompt = prompts.PROMPT_QA["system"]
        try:
            completion = await self.client.chat.completions.create(
                model=config.LLM_MODEL_REASON,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            return "Error in answer generation."

    @retry_async_llm_call
    async def generate_answer_from_profile(self, question: str, profile: str) -> str:
        logger.debug(f"Enhanced QA (Smart Model): {question[:50]}...")
        user_prompt = prompts.PROMPT_QA_WITH_PROFILE["user"].format(profile=profile, question=question)
        system_prompt = prompts.PROMPT_QA_WITH_PROFILE["system"]
        try:
            completion = await self.client.chat.completions.create(
                model=config.LLM_MODEL_REASON,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Enhanced QA failed: {e}", exc_info=True)
            return "Error in answer generation."

llm_service_instance = LLMService()
#llm_sevice.py
from pydantic import ValidationError, BaseModel
import json
import config
import prompts
import schemas
from utils import retry_async_llm_call
from typing import Dict, Any, Optional, Type
import logging
from openai import AsyncOpenAI, APIConnectionError
import asyncio
import json5
import re

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.LLM_TIMEOUT,
            
        )
        logger.info(f"LLMService 初始化, base_url: {config.OPENAI_BASE_URL}")
        # [新] 记录使用的模型
        logger.info(f"  > 提取模型 (Fast): {config.LLM_MODEL_EXTRACT}")
        logger.info(f"  > 推理模型 (Smart): {config.LLM_MODEL_REASON}")

    @retry_async_llm_call
    async def _call_api_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        model_name: str  # <--- [新] 接受模型名称
    ) -> Optional[BaseModel]:
        """
        调用 LLM API 并期望返回 Pydantic 模型校验过的 JSON。
        """
        content = ""
        try:
            completion = await self.client.chat.completions.create(
                model=model_name,  # <--- [修改] 使用传入的模型
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
                f"LLM 输出 Pydantic/JSON 校验失败 (模型: {model_name}), 将触发重试... Error: {e}\nRaw Content: {content}"
            )
            raise APIConnectionError(request=None, message="Retrying due to validation error.")

        except Exception as e:
            logger.error(f"LLM API (模型: {model_name}) 调用失败: {e}", exc_info=True)
            raise 

    async def extract_nodes(self, text: str) -> Optional[schemas.NodeExtractionResponse]:
        """调用提示词1：节点抽取 (使用 Fast 模型)"""
        logger.debug(f"开始抽取节点 (Fast Model) for text: {text[:50]}...")
        system_prompt = prompts.PROMPT_NODE_EXTRACT["system"]
        user_prompt = prompts.PROMPT_NODE_EXTRACT["user"].format(input_text=text)
        
        response = await self._call_api_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=schemas.NodeExtractionResponse,
            model_name=config.LLM_MODEL_EXTRACT  # <--- [修改]
        )
        if response:
            logger.debug(f"--- 🟢 LLM Extracted Nodes ---\n{json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}")
        return response

    async def extract_relations(self, text: str, nodes_json: str) -> Optional[schemas.RelationExtractionResponse]:
        """调用提示词2：关系抽取 (使用 Fast 模型)"""
        logger.debug(f"开始抽取关系 (Fast Model) for text: {text[:50]}...")
        system_prompt = prompts.PROMPT_RELATION_EXTRACT["system"]
        user_prompt = prompts.PROMPT_RELATION_EXTRACT["user"].format(
            input_text=text,
            node_list_json=nodes_json
        )
        
        response = await self._call_api_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=schemas.RelationExtractionResponse,
            model_name=config.LLM_MODEL_EXTRACT  # <--- [修改]
        )
        if response:
            logger.debug(f"--- 🟢 LLM Extracted Relations ---\n{json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}")
        return response

    async def decide_node_dedupe(
        self,
        input_text: str,
        existing_node: Dict[str, Any],
        candidate_node: schemas.Node
    ) -> Optional[schemas.NodeDedupeDecision]:
        """调用提示词3：节点去重仲裁 (使用 Smart 模型)"""
        logger.debug(f"LLM 仲裁节点 (Smart Model): {candidate_node.properties.name}")
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
            model_name=config.LLM_MODEL_EXTRACT  # <--- [修改]
        )
        if response:
            logger.debug(f"--- 🔵 Node Dedupe Decision ---")
            logger.debug(f"  - Candidate: {candidate_node.properties.name}")
            logger.debug(f"  - Decision: {response.decision}")
            if response.decision == "MERGE":
                logger.debug(f"  - Target: {response.merge_target_uuid}")
            logger.debug(f"  - Reason: {response.reason}")
            logger.debug("----------------------------------")
        return response

    # --- [新] 用于提示词6的函数 (使用 Smart 模型) ---
    async def synthesize_profile(self, context: str) -> Optional[schemas.ProfileSynthesisResponse]:
        """调用提示词6: 从KG上下文合成用户画像 (使用 Smart 模型)"""
        logger.debug(f"合成用户画像 (Smart Model)...")
        system_prompt = prompts.PROMPT_PROFILE_SYNTHESIZE["system"]
        user_prompt = prompts.PROMPT_PROFILE_SYNTHESIZE["user"].format(context=context)
        
        response = await self._call_api_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=schemas.ProfileSynthesisResponse,
            model_name=config.LLM_MODEL_REASON  # <--- [修改]
        )
        return response


    async def parse_personalized_answer(self, raw_output: str, max_retries: int = 10):
        """
        解析 LLM 返回的 JSON/JSON5，并仅提取 personalized_answer。
        自动清理 Markdown 代码块，自动重试。
        """

        import asyncio
        import json5
        import re

        # fix max_retries
        try:
            max_retries = int(max_retries)
        except:
            max_retries = 10

        attempt = 0

        while attempt < max_retries:

            try:
                # ---- Step 0: 类型检查 ----
                if not isinstance(raw_output, str):
                    raise TypeError(f"raw_output 不是字符串，而是 {type(raw_output)}")

                # ---- Step 1: 使用正则提取 JSON 内容 (最稳健的方式) ----
                # 逻辑：寻找被 ```json ... ``` 包裹的内容，或者寻找最外层的 { ... }
                # 1. 尝试匹配 Markdown 代码块
                pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
                match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
                
                if match:
                    json_str = match.group(1)
                else:
                    # 2. 如果没找到代码块，尝试寻找首尾的大括号 (兜底策略)
                    start = raw_output.find("{")
                    end = raw_output.rfind("}")
                    if start != -1 and end != -1:
                        json_str = raw_output[start : end + 1]
                    else:
                        # 实在找不到，就死马当活马医，用原始字符串
                        json_str = raw_output

                # ---- Step 2: 解析 JSON5 ----
                # 注意：千万不要手动 replace("\n", "")，json5 会自己处理换行
                data = json5.loads(json_str)

                # ---- Step 3: 类型与键检查 (回答你的问题) ----
                if not isinstance(data, dict):
                    raise ValueError(f"解析出的数据不是字典，而是 {type(data)}")

                if "personalized_answer" not in data:
                    raise ValueError("JSON 中缺少 'personalized_answer' 字段")

                return data["personalized_answer"]


            except Exception as e:
                attempt += 1

                if attempt >= max_retries:
                    raise RuntimeError(
                        f"JSON5 解析失败（已重试 {max_retries} 次）: {e}\n原始输出:\n{raw_output}"
                    )

                # 小的等待避免疯狂重试
                await asyncio.sleep(0.1)

    async def generate_answer_from_context(self, question: str, context: str,json:bool=True) -> str:
        """使用 Smart Model 生成回答，并解析 JSON5 仅返回 personalized_answer"""
        logger.debug(f"标准QA (Smart Model): {question[:50]}...") 
        if json:  
            user_prompt = prompts.PROMPT_QA_JSON["user"].format(context=context, question=question)
            system_prompt = prompts.PROMPT_QA_JSON["system"]
        else: 
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

            raw_output = completion.choices[0].message.content
            if json:
            # 🔥 核心：解析 JSON5 并只返回 personalized_answer
                return await self.parse_personalized_answer(raw_output)
            else:
                return raw_output

        except Exception as e:
            logger.error(f"生成答案(标准版)失败: {e}", exc_info=True)
            return "生成答案时出错。"


    # # --- [修改] 用于提示词5的函数 (使用 Smart 模型) ---
    # @retry_async_llm_call
    # async def generate_answer_from_context(self, question: str, context: str) -> str:
    #     """调用提示词5 (标准版): 从原始KG上下文生成答案 (使用 Smart 模型)"""
    #     logger.debug(f"标准QA (Smart Model): {question[:50]}...")    
    #     user_prompt = prompts.PROMPT_QA["user"].format(context=context, question=question)
    #     system_prompt = prompts.PROMPT_QA["system"]
        
    #     try:
    #         completion = await self.client.chat.completions.create(
    #             model=config.LLM_MODEL_REASON,  
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt}
    #             ],
    #             temperature=0.1
    #         )
    #         return completion.choices[0].message.content
    #     except Exception as e:
    #         logger.error(f"生成答案(标准版)失败: {e}", exc_info=True)
    #         return "生成答案时出错。"

    # --- [新] 用于提示词7的函数 ---
    @retry_async_llm_call
    async def generate_answer_from_profile(self, question: str, profile: str) -> str:
        """调用提示词7 (增强版): 从合成的画像生成答案 (使用 Smart 模型)"""
        logger.debug(f"增强QA (Smart Model): {question[:50]}...")
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
            logger.error(f"生成答案(增强版)失败: {e}", exc_info=True)
            return "生成答案时出错。"


# 单例
llm_service_instance = LLMService()
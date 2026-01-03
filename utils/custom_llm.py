from openai import OpenAI, RateLimitError
import argparse
import json
import backoff
import concurrent
import tqdm
import time
import google.generativeai as genai
import concurrent.futures as futures
import os

def batchify(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class TextHelper:
    def __init__(self, text):
        self.text = text
        self.cumulative_logprob = 0

class OutputHelper:
    def __init__(self, output):
        self.outputs = [TextHelper(output)]

@backoff.on_exception(backoff.expo, RateLimitError)
def get_completion_from_gpt(messages, model_name, max_tokens, api_key, temperature, base_url):
    # 回退逻辑：允许无 API Key（用于本地 vLLM OpenAI 兼容服务）
    effective_api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "EMPTY")
    if base_url:
        client = OpenAI(api_key=effective_api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=effective_api_key)
    retries = 0
    while True:
        retries += 1    
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

class OpenAILLM:
    def __init__(self, model_name, api_key) -> None:
        self.model_name = model_name
        # 允许 api_key 为空；若为空，内部会使用环境变量或 'EMPTY'
        self.api_key = api_key
        self.n_threads=8
        # 支持通过环境变量配置本地 OpenAI 兼容服务
        self.base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8001/v1")
    
    def generate(self, prompts, sampling_param):
        results = []
        barches = batchify(prompts, self.n_threads)
        for batch in tqdm.tqdm(barches):
            temp_results = []
            with futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                for data in batch:
                    temp_results.append(executor.submit(
                        get_completion_from_gpt,
                        data,
                        self.model_name,
                        sampling_param.max_tokens,
                        self.api_key,
                        sampling_param.temperature,
                        self.base_url,
                    ))
            for temp_result in temp_results:
                results.append(OutputHelper(temp_result.result()))
        return results
    
import requests

class APILLM:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key

    def generate(self, prompts, sampling_params):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompts": prompts,
            "sampling_params": sampling_params
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        
        # 假设 API 返回的是一个列表，每个元素是一个生成的文本
        results = [OutputHelper(item) for item in response_json]
        return results
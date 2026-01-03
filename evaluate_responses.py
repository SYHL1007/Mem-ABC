#evaluate_response.py
from vllm import LLM
import argparse
import json
from evaluation.evaluator import evaluator
import torch.distributed as dist
import torch
parser = argparse.ArgumentParser()

parser.add_argument("--evaluator_llm", type=str, required=True)
parser.add_argument("--inputs_addr", type=str, required=True)
parser.add_argument("--response_addr", type=str, required=True)
parser.add_argument("--score_addr", type=str, required=True)
parser.add_argument("--cache_dir", default="")
parser.add_argument("--tensor_parallel_size", type=int, default=2)
parser.add_argument("--max_length", type=int, default=32768)


if __name__ == "__main__":
    try:
        print("Starting evaluation script.")   
        # 获取参数
        args = parser.parse_args()
        
        # 参数检查
        if args.tensor_parallel_size < 1:
            raise ValueError("--tensor_parallel_size must be at least 1.")
        if args.max_length <= 0:
            raise ValueError("--max_length must be greater than 0.")
        
        print(f"Arguments received: {args}")
        
        # 加载输入数据集
        try:
            with open(args.inputs_addr, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Dataset loaded from: {args.inputs_addr}. Size: {len(dataset)}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset from {args.inputs_addr}: {e}")
        
        # 加载响应数据
        try:
            with open(args.response_addr, "r", encoding="utf-8") as f:
                outputs = json.load(f)
            print(f"Responses loaded from: {args.response_addr}. Size: {len(outputs)}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load responses from {args.response_addr}: {e}")
        
        # 提取 dataset 和 outputs 的 ID
        try:
            dataset_ids = set(data['id'] for data in dataset)
            outputs_ids = set(list(item.keys())[0] for item in outputs)  # 获取 outputs 的 key
        
            # 检查数据一致性
            assert dataset_ids == outputs_ids, (
                f"Dataset IDs and Output IDs do not match. "
                f"Missing in outputs: {dataset_ids - outputs_ids}, "
                f"Extra in outputs: {outputs_ids - dataset_ids}."
            )
            assert all(
                len(item[next(iter(item.keys()))]) > 0 for item in outputs
            ), "All outputs must have at least one response."
            print("Dataset and response data consistency check passed.")
        except Exception as e:
            raise ValueError(f"Data inconsistency detected: {e}")
        
        # 构造 LLM 模型
        try:
            print(f"Initializing LLM model: {args.evaluator_llm}")
            llm = LLM(
                args.evaluator_llm,
                download_dir=args.cache_dir,
                max_model_len=args.max_length,
                tensor_parallel_size=args.tensor_parallel_size,
                enforce_eager=False,
                gpu_memory_utilization=0.8,
                enable_prefix_caching=True 
            )
            print("LLM model initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM model: {e}")
        
        # 提取相关数据
        try:
            queries = [data['question'] for data in dataset]
            ids = [data['id'] for data in dataset]
            details = [data['narrative'] for data in dataset]
            aspects = [data['rubric_aspects'] for data in dataset]
            print(f"Extracted data from dataset: queries={len(queries)}, ids={len(ids)}, details={len(details)}, aspects={len(aspects)}")
        except Exception as e:
            raise ValueError(f"Failed to extract data from dataset: {e}")

        # 提取 responses 数据，解决 TypeError 问题
        try:
            responses = []
            for data in dataset:
                # 遍历 outputs，搜索匹配的 id
                response_found = False
                for item in outputs:
                    if data['id'] in item:
                        # 提取响应内容
                        if len(item[data['id']]) > 0:  # 确保响应非空
                            responses.append(
                                item[data['id']]['outputs']#final_response_finetuned
                            )
                        else:
                            responses.append("")  # 无内容则空字符串
                        response_found = True
                        break

                if not response_found:
                    raise RuntimeError(f"ID {data['id']} not found in outputs.")

            assert len(queries) == len(responses), "Queries and Responses length mismatch."
            print(f"Extracted responses successfully. Total responses: {len(responses)}")
        except Exception as e:
            raise ValueError(f"Failed to extract responses: {e}")
        
        # 调用评估器进行模型评估
        try:
            print("Evaluating model...")
            scores = evaluator(queries, responses, details, aspects, llm)
            print("Model evaluation completed successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed during model evaluation: {e}")
        
        # 为每个评分附加对应的 ID
        try:
            for id, score in zip(ids, scores['per_question_scores']):
                score['id'] = id
        
            # 保存最终分数到文件
            with open(args.score_addr, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=4, ensure_ascii=False)
            print(f"Scores saved successfully to: {args.score_addr}")
        except Exception as e:
            raise RuntimeError(f"Failed to save scores to {args.score_addr}: {e}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # 确保销毁分布式进程组
        try:
            dist.destroy_process_group()
            print("Distributed process group destroyed.")
        except Exception as e:
            print(f"Failed to destroy distributed process group: {e}")
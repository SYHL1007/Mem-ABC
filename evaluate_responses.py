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
        args = parser.parse_args()
        
        if args.tensor_parallel_size < 1:
            raise ValueError("--tensor_parallel_size must be at least 1.")
        if args.max_length <= 0:
            raise ValueError("--max_length must be greater than 0.")
        
        print(f"Arguments received: {args}")
        
        try:
            with open(args.inputs_addr, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Dataset loaded from: {args.inputs_addr}. Size: {len(dataset)}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset from {args.inputs_addr}: {e}")
        
        try:
            with open(args.response_addr, "r", encoding="utf-8") as f:
                outputs = json.load(f)
            print(f"Responses loaded from: {args.response_addr}. Size: {len(outputs)}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load responses from {args.response_addr}: {e}")
        
        try:
            dataset_ids = set(data['id'] for data in dataset)
            outputs_ids = set(list(item.keys())[0] for item in outputs) 
        
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
        
        try:
            queries = [data['question'] for data in dataset]
            ids = [data['id'] for data in dataset]
            details = [data['narrative'] for data in dataset]
            aspects = [data['rubric_aspects'] for data in dataset]
            print(f"Extracted data from dataset: queries={len(queries)}, ids={len(ids)}, details={len(details)}, aspects={len(aspects)}")
        except Exception as e:
            raise ValueError(f"Failed to extract data from dataset: {e}")

        try:
            responses = []
            for data in dataset:
                response_found = False
                for item in outputs:
                    if data['id'] in item:
                        if len(item[data['id']]) > 0: 
                            responses.append(
                                item[data['id']]['outputs']
                            )
                        else:
                            responses.append("") 
                        response_found = True
                        break

                if not response_found:
                    raise RuntimeError(f"ID {data['id']} not found in outputs.")

            assert len(queries) == len(responses), "Queries and Responses length mismatch."
            print(f"Extracted responses successfully. Total responses: {len(responses)}")
        except Exception as e:
            raise ValueError(f"Failed to extract responses: {e}")
        
        try:
            print("Evaluating model...")
            scores = evaluator(queries, responses, details, aspects, llm)
            print("Model evaluation completed successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed during model evaluation: {e}")
        
        try:
            for id, score in zip(ids, scores['per_question_scores']):
                score['id'] = id
        
            with open(args.score_addr, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=4, ensure_ascii=False)
            print(f"Scores saved successfully to: {args.score_addr}")
        except Exception as e:
            raise RuntimeError(f"Failed to save scores to {args.score_addr}: {e}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        try:
            dist.destroy_process_group()
            print("Distributed process group destroyed.")
        except Exception as e:
            print(f"Failed to destroy distributed process group: {e}")

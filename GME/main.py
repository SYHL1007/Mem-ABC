# main.py

import json
import argparse
import asyncio
from openai import AsyncOpenAI
import os
import random
from tqdm.asyncio import tqdm
from sentence_transformers import SentenceTransformer

# Ensure importing the updated KnowledgeBaseManager
from knowledge_base_manager import KnowledgeBaseManager
from llm_services import retrieve_similar_memories_local_sync as retrieve_memories_for_qa, call_llm_async_with_retry
from prompts import FINAL_QA_PROMPT

def convert_jsonl_to_json(jsonl_path, output_json_path):
    """Convert .jsonl file to .json file."""
    print(f"\n🔄 Starting conversion from {jsonl_path} to {output_json_path} ...")
    full_kb = {}
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    full_kb.update(data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(full_kb, f_out, indent=4, ensure_ascii=False)
        print(f"✅ Conversion successful! Final knowledge base saved to: {output_json_path}")
        return True
    except FileNotFoundError:
        print(f"❌ Error: Input file {jsonl_path} not found. Conversion failed.")
        return False

async def process_question_sequentially(
    data_point: dict, 
    args, 
    client: AsyncOpenAI, 
    embedding_model: SentenceTransformer, 
    file_lock: asyncio.Lock
):
    """[Main processing unit] Process a single question and append to .jsonl file."""
    question_id = data_point["id"]
    
    kb_manager = KnowledgeBaseManager(
        args.kb_path, client, args.llm_model, embedding_model, verbose=args.verbose
    )
    
    entry = kb_manager.get_or_create_entry(question_id, data_point.get("question", ""))
    entry["memories"] = {}
    entry["global_summary"] = "The user's initial profile is empty."
    raw_texts_for_summary = []
    SUMMARY_UPDATE_INTERVAL = 5
    profile_items = data_point.get("profile", [])[:70]
    for i, profile_item in enumerate(profile_items):
        if args.verbose:
            print(f"\n--- Processing QID {question_id}, Record {i+1}/{len(profile_items)} ---")
        await kb_manager.process_single_profile_item(question_id, profile_item["text"], entry["global_summary"])
        raw_texts_for_summary.append(profile_item["text"])
        if (i + 1) % SUMMARY_UPDATE_INTERVAL == 0 and raw_texts_for_summary:
            if args.verbose: 
                print(f"--- QID {question_id}: Updating summary ---")
            new_summary = await kb_manager.generate_new_summary(entry["global_summary"], raw_texts_for_summary)
            entry["global_summary"] = new_summary
            raw_texts_for_summary = []
    if raw_texts_for_summary:
        new_summary = await kb_manager.generate_new_summary(entry["global_summary"], raw_texts_for_summary)
        entry["global_summary"] = new_summary
    final_result_for_task = {question_id: entry}
    result_line = json.dumps(final_result_for_task, ensure_ascii=False)
    async with file_lock:
        with open(args.kb_path, 'a', encoding='utf-8') as f:
            f.write(result_line + '\n')

async def build_knowledge_base(args, client: AsyncOpenAI, embedding_model: SentenceTransformer, semaphore: asyncio.Semaphore, file_lock: asyncio.Lock):
    """Build the knowledge base asynchronously."""
    print("#" * 20 + " Task: Building Knowledge Base " + "#" * 20)
    try:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file {args.data_path} not found.")
        return
    dataset_to_process = full_dataset
    if 0 < args.sample_rate < 1:
        sample_size = int(len(full_dataset) * args.sample_rate)
        print(f"Sampling the dataset with rate: {args.sample_rate * 100:.2f}%, extracting {sample_size} / {len(full_dataset)} items.")
        random.seed(args.random_seed)
        dataset_to_process = random.sample(full_dataset, sample_size)
        try:
            with open(args.sampled_data_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_to_process, f, indent=2, ensure_ascii=False)
            print(f"✅ Sampled dataset saved to: {args.sampled_data_path}")
        except IOError:
            print(f"❌ Error: Unable to write sampled dataset to {args.sampled_data_path}.")
    
    processed_ids = set()
    if os.path.exists(args.kb_path):
        print(f"Loading processed question IDs from {args.kb_path}...")
        with open(args.kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(list(data.keys())[0])
                except (json.JSONDecodeError, IndexError):
                    print(f"Warning: Skipping malformed line: {line.strip()}")
        print(f"Loaded {len(processed_ids)} processed question IDs.")

    tasks_to_run_datapoints = [dp for dp in dataset_to_process if dp["id"] not in processed_ids]

    if not tasks_to_run_datapoints:
        print("✅ All questions are already in the knowledge base. No new tasks to process.")
    else:
        print(f"Total {len(dataset_to_process)} questions; {len(processed_ids)} processed; processing {len(tasks_to_run_datapoints)} new questions.")
        async def worker(data_point):
            async with semaphore:
                await process_question_sequentially(data_point, args, client, embedding_model, file_lock)
        tasks = [worker(dp) for dp in tasks_to_run_datapoints]
        await tqdm.gather(*tasks, desc="Processing all questions concurrently")
        print("\n" + "#" * 20 + " Knowledge Base Construction Completed " + "#" * 20)
        print(f"The intermediate knowledge base file has been updated and saved to: {args.kb_path}")
    final_json_path = os.path.splitext(args.kb_path)[0] + '.json'
    if convert_jsonl_to_json(args.kb_path, final_json_path):
        if args.delete_intermediate_jsonl:
            print(f"🗑️ Deleting intermediate file: {args.kb_path}...")
            try:
                os.remove(args.kb_path)
                print("✅ Intermediate file deleted.")
            except OSError as e:
                print(f"❌ Failed to delete intermediate file: {e}")

async def main(args):
    """Main function to coordinate tasks."""
    async_client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    print(f"Loading local embedding model: {args.local_embedding_model} ...")
    try:
        embedding_model = SentenceTransformer(args.local_embedding_model)
        print("✅ Local embedding model loaded successfully.")
    except Exception as e:
        print(f"❌ Unable to load local embedding model: {e}")
        return
    semaphore = asyncio.Semaphore(args.concurrency_limit)
    file_lock = asyncio.Lock()
    print(f"🚦 Concurrency limit set to: {args.concurrency_limit}")
    if args.task == "build":
        await build_knowledge_base(args, async_client, embedding_model, semaphore, file_lock)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Base and QA System")
    parser.add_argument("task", choices=["build"], help="Task to execute")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the cloud service")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--llm-model", type=str, required=True, help="Name of the LLM to use")
    parser.add_argument("--local-embedding-model", type=str, default="facebook/contriever-msmarco", help="Name of the local SentenceTransformer embedding model")
    parser.add_argument("--data-path", type=str, default="data.json", help="Path to the input data JSON file")
    parser.add_argument("--kb-path", type=str, default="knowledge_base.jsonl", help="Path to the intermediate knowledge base JSON Lines file")
    parser.add_argument("--output-path", type=str, default="qa_results.json", help="Path to save QA results in JSON format")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sample rate for the dataset (0.0 to 1.0)")
    parser.add_argument("--sampled-data-path", type=str, default="sampled_data.json", help="Path to save the sampled dataset")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for random sampling")
    parser.add_argument("--concurrency-limit", type=int, default=300, help="Maximum number of concurrent tasks to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging mode")
    parser.add_argument("--delete-intermediate-jsonl", action="store_true", help="Delete the intermediate .jsonl file after completing tasks")
    args = parser.parse_args()
    asyncio.run(main(args))
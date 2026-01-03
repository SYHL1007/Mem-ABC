import logging
import json
import asyncio
import argparse
import sys
import os
from tqdm.asyncio import tqdm_asyncio
import config
from utils import setup_logging
from llm_service import llm_service_instance
from embedding_service import embedding_service_instance
from graph_db import graph_db_instance
from kg_builder import KnowledgeGraphBuilder
from qa_service import QAService
import warnings, traceback, sys, faulthandler
faulthandler.enable()
logger = logging.getLogger(__name__)


async def run_kg_builder(data_file_path: str):
    logger.info("--- Task begins ---")
    try:
        graph_db_instance.connect()
        graph_db_instance.setup_database()
        builder = KnowledgeGraphBuilder(
            llm=llm_service_instance,
            embed=embedding_service_instance,
            db=graph_db_instance
        )
    except Exception as e:
        logger.error(f"Service initialization failed: {e}", exc_info=True)
        return

    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Loaded dataset with {len(dataset)} users.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {data_file_path} - {e}", exc_info=True)
        return

    user_groups = {}
    total_memories = 0
    for user_id, data in dataset.items():
        memories = data.get("memories", {})
        if not memories:
            logger.warning(f"User {user_id} has no memories, skipping.")
            continue
        memory_list = [(mid, mtext) for mid, mtext in memories.items() if mtext]
        if memory_list:
            user_groups[user_id] = memory_list
            total_memories += len(memory_list)
    
    logger.info(f"Processing {len(user_groups)} users with {total_memories} memories total.")
    logger.info(f"Concurrency: User concurrency={config.KG_BUILD_USER_CONCURRENCY}, Task timeout={config.KG_BUILD_TASK_TIMEOUT}s, Max retries={config.KG_BUILD_MAX_RETRIES}")
    semaphore = asyncio.Semaphore(config.KG_BUILD_USER_CONCURRENCY)

    async def process_user_memories(user_id: str, memory_list: list):
        async with semaphore:
            logger.debug(f"[User: {user_id}] Processing {len(memory_list)} memories...")
            success_count = 0
            fail_count = 0
            for memory_id, memory_text in memory_list:
                retry_count = 0
                while retry_count <= config.KG_BUILD_MAX_RETRIES:
                    try:
                        await asyncio.wait_for(
                            builder.process_memory(user_id, memory_id, memory_text),
                            timeout=config.KG_BUILD_TASK_TIMEOUT
                        )
                        success_count += 1
                        break
                    except asyncio.TimeoutError:
                        retry_count += 1
                        if retry_count > config.KG_BUILD_MAX_RETRIES:
                            logger.error(f"[User: {user_id}, Mem: {memory_id}] Timeout after {config.KG_BUILD_MAX_RETRIES} retries.")
                            fail_count += 1
                        else:
                            logger.warning(f"[User: {user_id}, Mem: {memory_id}] Timeout, retrying {retry_count}...")
                            await asyncio.sleep(2 ** retry_count)
                    except Exception as e:
                        retry_count += 1
                        if retry_count > config.KG_BUILD_MAX_RETRIES:
                            logger.error(f"[User: {user_id}, Mem: {memory_id}] Failed after {config.KG_BUILD_MAX_RETRIES} retries: {e}", exc_info=True)
                            fail_count += 1
                        else:
                            logger.warning(f"[User: {user_id}, Mem: {memory_id}] Error, retrying {retry_count}: {e}")
                            await asyncio.sleep(2 ** retry_count)
            logger.debug(f"[User: {user_id}] Finished: Success {success_count}, Fail {fail_count}")

    user_tasks = [process_user_memories(uid, mem_list) for uid, mem_list in user_groups.items()]
    logger.info(f"Starting concurrent processing of {len(user_tasks)} users (Concurrency: {config.KG_BUILD_USER_CONCURRENCY})...")
    await tqdm_asyncio.gather(*user_tasks, desc="Processing users", unit="user")
    await graph_db_instance.close()
    logger.info("--- Task complete ---")


def load_processed_ids(jsonl_path: str) -> set:
    processed_ids = set()
    if not os.path.exists(jsonl_path):
        logger.info(f"Log file {jsonl_path} not found, starting fresh.")
        return processed_ids
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if 'user_id' in record:
                        processed_ids.add(record['user_id'])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line in log: {line[:50]}...")
    except Exception as e:
        logger.error(f"Error loading processed IDs from {jsonl_path}: {e}", exc_info=True)
    return processed_ids

async def run_qa_batch(
    qa_data_file: str, 
    strategy: str, 
    output_json: str, 
    concurrency_limit: int,
    enhanced_qa: bool = False,
    task_timeout: int = 500,
    final_context_k: int = 200,
    json_output: bool = False
):
    logger.info(f"--- Batch QA Service Started (File: {qa_data_file}, Enhanced: {enhanced_qa}) ---")
    try:
        with open(qa_data_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Loaded dataset with {len(dataset)} users.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {qa_data_file} - {e}", exc_info=True)
        return

    log_file_name = "qa_log.jsonl"
    processed_user_ids = load_processed_ids(log_file_name)
    logger.info(f"Found {len(processed_user_ids)} processed IDs in {log_file_name}.")
    tasks_to_run = []
    for user_id, data in dataset.items():
        if user_id in processed_user_ids:
            continue
        question = data.get("question")
        if not question:
            logger.warning(f"User {user_id} missing 'question' field, skipping.")
            continue
        tasks_to_run.append((user_id, question))
    logger.info(f"Total {len(dataset)} questions. {len(processed_user_ids)} already logged. {len(tasks_to_run)} questions to process. {final_context_k}")
    if tasks_to_run:
        try:
            graph_db_instance.connect()
            qa = QAService(
                llm=llm_service_instance,
                embed=embedding_service_instance,
                db=graph_db_instance
            )
            semaphore = asyncio.Semaphore(concurrency_limit)
            async def worker(user_id, question):
                async with semaphore:
                    try:
                        await asyncio.wait_for(
                            qa.answer_question(
                                user_id, 
                                question, 
                                strategy, 
                                enhanced_qa=enhanced_qa,
                                final_context_k=final_context_k,
                                json_output=json_output
                            ),
                            timeout=task_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"[QA User: {user_id}] Task timed out (>{task_timeout}s): {question[:30]}...")
                    except Exception as e:
                        logger.error(f"[QA User: {user_id}] Unexpected error: {e}", exc_info=True)
            logger.info(f"Processing {len(tasks_to_run)} questions (Concurrency: {concurrency_limit})...")
            all_tasks = [worker(uid, q) for uid, q in tasks_to_run]
            await tqdm_asyncio.gather(*all_tasks, desc="Processing QA", unit="q")
            await graph_db_instance.close()
            logger.info("--- Batch QA Service Complete ---")
        except Exception as e:
            logger.error(f"Batch QA Service failed: {e}", exc_info=True)
    else:
        logger.info("No new QA tasks to run.")
    convert_jsonl_to_json(log_file_name, output_json)

def convert_jsonl_to_json(jsonl_path: str, json_path: str):
    logger.info(f"Converting {jsonl_path} to {json_path}...")
    results = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if line.strip():
                    record = json.loads(line)
                    user_id = record.get('user_id', '')
                    formatted_record = {
                        user_id: {
                            "question": record.get('question', ''),
                            "prompt": record.get('prompt', ''),
                            "outputs": record.get('answer', '')
                        }
                    }
                    results.append(formatted_record)
        with open(json_path, 'w', encoding='utf-8') as f_out:
            json.dump(results, f_out, indent=2, ensure_ascii=False)
        logger.info(f"Conversion complete. Saved to {json_path}, {len(results)} records.")
    except FileNotFoundError:
        logger.error(f"Log file not found: {jsonl_path}, unable to convert.")
    except Exception as e:
        logger.error(f"Failed to convert JSONL: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Personalized data and QA system")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    build_parser = subparsers.add_parser("build", help="Run task for data")
    build_parser.add_argument(
        "data_file", 
        type=str, 
        help="Path to JSON dataset file (e.g., 'data.json')"
    )
    build_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode"
    )
    query_parser = subparsers.add_parser("query", help="Run QA task")
    query_parser.add_argument(
        "-p","--qa_data_file", 
        type=str, 
        help="Path to question dataset file (e.g., 'questions.json')"
    )
    query_parser.add_argument(
        "--enhance",
        action="store_true",
        help="Enable two-step enhanced QA"
    )
    query_parser.add_argument(
        "-o", "--output_json",
        type=str,
        default="qa_results.json",
        help="Output file for JSON results"
    )
    query_parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent QA tasks"
    )
    query_parser.add_argument(
        "-s", 
        "--strategy", 
        type=str, 
        default="hybrid", 
        choices=["hybrid", "full"],
        help="Retrieval strategy"
    )
    query_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode"
    )
    query_parser.add_argument(
        "-k", "--final_k",
        type=int,
        default=200,
        dest="final_context_k",
        help="Top-K nodes to retain"
    )
    query_parser.add_argument(
        "--json_output", 
        action="store_true", 
        help="If set, instructs the model to return JSON."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    quiet_mode = getattr(args, 'quiet', False)
    setup_logging(quiet=quiet_mode)

    try:
        if args.command == "build":
            asyncio.run(run_kg_builder(args.data_file))
        elif args.command == "query":
            asyncio.run(run_qa_batch(
                args.qa_data_file, 
                args.strategy, 
                args.output_json,
                args.concurrency,
                args.enhance,
                500,
                args.final_context_k,
                args.json_output
            ))
    except KeyboardInterrupt:
        logger.info("Interrupted, exiting...")
    except Exception as e:
        logger.error(f"Unhandled top-level error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
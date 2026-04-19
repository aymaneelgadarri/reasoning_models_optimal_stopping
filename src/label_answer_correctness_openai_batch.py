"""OpenAI Batch-API version of label_answer_correctness.py.

Submits all per-question prompts as a single OpenAI Batch job (JSONL upload),
polls until completion, then post-processes the results with the same
parsing/merging logic as the synchronous script.

Usage:
    export OPENAI_API_KEY=sk-...
    python src/label_answer_correctness_openai_batch.py \
        --segmented_dataset_path ... \
        --raw_CoT_path ... \
        --save_path ... \
        [--model gpt-4o-mini] \
        [--resume_job batch_abc123]   # to skip submission and just poll

Notes:
- OpenAI Batch is 50% off list price and has up to a 24h SLA.
- JSONL line format expected by the API:
    {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": ..., "messages": [...], ...}}
- For reasoning models (o-series, gpt-5*), drop --temperature and use
  --max_completion_tokens instead of --max_tokens via env override if needed.
"""

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from utils import process_json_output
from label_answer_correctness import (
    INSTRUCT_PROMPT,
    load_id2answer,
    load_id2data,
    valid_label,
    merge_reasoning_chunks,
)


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def build_prompt(reasoning_trace, gt_answer):
    return (
        INSTRUCT_PROMPT
        + f"Input chunks: {reasoning_trace}"
        + f"\n\nGround-truth answer: {gt_answer}"
    )


def write_batch_jsonl(path, requests, model, temperature, max_tokens):
    """Write one JSON object per line in the format OpenAI Batch API expects."""
    with open(path, "w") as f:
        for key, prompt in requests:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
            }
            if temperature is not None:
                body["temperature"] = temperature
            line = {
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def submit_batch(client, jsonl_path, display_name):
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded input file: {uploaded.id}")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"display_name": display_name},
    )
    return batch


def poll_batch(client, batch_id, poll_interval):
    """Block until the batch reaches a terminal state."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        ts = time.strftime("%H:%M:%S")
        print(
            f"[{ts}] {batch_id} status={status} "
            f"completed={counts.completed}/{counts.total} failed={counts.failed}"
        )
        if status in TERMINAL_STATUSES:
            return batch
        time.sleep(poll_interval)


def download_file(client, file_id, out_path):
    content = client.files.content(file_id)
    with open(out_path, "wb") as f:
        f.write(content.read())
    return out_path


def parse_results(jsonl_path):
    """Returns (key -> response text) from an OpenAI batch output JSONL."""
    key2text = {}
    errors = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            key = item.get("custom_id")
            if item.get("error"):
                errors.append((key, item["error"]))
                key2text[key] = None
                continue
            resp = item.get("response") or {}
            if resp.get("status_code") != 200:
                errors.append((key, f"http {resp.get('status_code')}: {resp.get('body')}"))
                key2text[key] = None
                continue
            try:
                text = resp["body"]["choices"][0]["message"]["content"]
            except Exception as e:
                errors.append((key, f"parse error: {e}"))
                text = None
            key2text[key] = text
    return key2text, errors


def parse_error_file(jsonl_path):
    """Returns list of (custom_id, error) from an OpenAI batch error JSONL."""
    out = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            out.append((item.get("custom_id"), item.get("error") or item))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmented_dataset_path", type=str, required=True)
    parser.add_argument("--raw_CoT_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Set to a negative number to omit (required for o-series/gpt-5* reasoning models).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Per-request output cap. Batch quota reserves up to this per request.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Any OpenAI chat model that supports the Batch API. "
             "e.g. gpt-4o-mini, gpt-4o, gpt-4.1-mini, gpt-5-mini.",
    )
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument(
        "--resume_job",
        type=str,
        default=None,
        help="Existing batch id (e.g. 'batch_abc123') to resume polling instead "
             "of submitting a new job.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY env var not set.")
    client = OpenAI()

    os.makedirs(args.save_path, exist_ok=True)
    work_dir = Path(args.save_path) / "_batch_openai"
    work_dir.mkdir(exist_ok=True)

    id2ans = load_id2answer(args.raw_CoT_path)
    id2reasoning = json.load(open(args.segmented_dataset_path))
    id2reasoning = {str(k): v for k, v in id2reasoning.items()}
    print(f"id2ans: {len(id2ans)}, id2reasoning: {len(id2reasoning)}")

    base = os.path.basename(args.raw_CoT_path)
    stem, _ = os.path.splitext(base)

    temperature = args.temperature if args.temperature >= 0 else None

    if args.resume_job:
        batch_id = args.resume_job
        print(f"Resuming batch job: {batch_id}")
    else:
        requests = []
        for k in sorted(id2reasoning.keys()):
            if str(k) not in id2ans:
                print(f"warn: missing gt answer for id={k}, skipping")
                continue
            prompt = build_prompt(id2reasoning[k], id2ans[str(k)])
            requests.append((str(k), prompt))

        jsonl_path = work_dir / f"requests_{stem}.jsonl"
        write_batch_jsonl(
            jsonl_path, requests, args.model, temperature, args.max_tokens
        )
        print(f"Wrote {len(requests)} batch requests to {jsonl_path}")

        display_name = f"label-correctness-{stem}"
        batch = submit_batch(client, str(jsonl_path), display_name)
        batch_id = batch.id
        print(f"Submitted batch job: {batch_id}")

        with open(work_dir / f"last_job_{stem}.txt", "w") as f:
            f.write(batch_id + "\n")

    batch = poll_batch(client, batch_id, args.poll_interval)

    if batch.error_file_id:
        err_path = work_dir / f"errors_{stem}.jsonl"
        download_file(client, batch.error_file_id, str(err_path))
        errs = parse_error_file(str(err_path))
        print(f"{len(errs)} per-request errors saved to {err_path}; first 5: {errs[:5]}")

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch ended with status={batch.status}; "
            f"errors={getattr(batch, 'errors', None)}"
        )

    if not batch.output_file_id:
        raise RuntimeError("Batch completed but no output_file_id was returned.")

    results_jsonl = work_dir / f"results_{stem}.jsonl"
    download_file(client, batch.output_file_id, str(results_jsonl))
    print(f"Downloaded results to {results_jsonl}")

    key2text, errors = parse_results(str(results_jsonl))
    if errors:
        print(f"{len(errors)} parse-side errors (showing first 5): {errors[:5]}")

    all_res = {}
    skipped = 0
    for k, text in tqdm(key2text.items(), desc="parsing"):
        if text is None:
            skipped += 1
            continue
        out_res = process_json_output(text)
        reasoning_trace = id2reasoning.get(k)
        if reasoning_trace is None or not valid_label(out_res, len(reasoning_trace)):
            print(f"Error processing output for {k}")
            skipped += 1
            continue
        all_res[k] = out_res
    print(f"Parsed {len(all_res)} items, skipped {skipped}")

    id2data = load_id2data(args.raw_CoT_path)
    labeled_data = {}
    for k in tqdm(all_res, desc="merging chunks"):
        reasoning_chunks, labels = merge_reasoning_chunks(id2reasoning[k], all_res[k])
        labeled_data[k] = {
            "id": k,
            "question": id2data[k]["instruction"],
            "answer": id2data[k]["answer"],
            "reasoning_chunks": reasoning_chunks,
            "correctness_labels": labels,
        }

    out_path = f"{args.save_path}/labeled_intermediate_answers_{base}"
    with open(out_path, "w") as f:
        for k in labeled_data:
            f.write(json.dumps(labeled_data[k], ensure_ascii=False) + "\n")
    print(f"Saved {len(labeled_data)} labeled items to {out_path}")


if __name__ == "__main__":
    main()

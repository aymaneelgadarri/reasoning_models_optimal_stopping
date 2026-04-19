"""Batch-API version of label_answer_correctness.py.

Submits all per-question prompts as a single Gemini Batch job (JSONL upload),
polls until completion, then post-processes the results with the same
parsing/merging logic as the synchronous script.

Usage:
    python src/label_answer_correctness_batch.py \
        --segmented_dataset_path ... \
        --raw_CoT_path ... \
        --save_path ... \
        [--model gemini-3-flash-preview] \
        [--resume_job batches/abc123]   # to skip submission and just poll
"""

import argparse
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

from utils import process_json_output
from label_answer_correctness import (
    INSTRUCT_PROMPT,
    load_id2answer,
    load_id2data,
    valid_label,
    merge_reasoning_chunks,
)


COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def build_prompt(reasoning_trace, gt_answer):
    return (
        INSTRUCT_PROMPT
        + f"Input chunks: {reasoning_trace}"
        + f"\n\nGround-truth answer: {gt_answer}"
    )


def write_batch_jsonl(path, requests, temperature, max_output_tokens):
    """Write one JSON object per line in the format Gemini Batch API expects."""
    with open(path, "w") as f:
        for key, prompt in requests:
            line = {
                "key": key,
                "request": {
                    "contents": [
                        {"parts": [{"text": prompt}], "role": "user"}
                    ],
                    "generation_config": {
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def submit_batch(client, jsonl_path, model, display_name):
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type="application/jsonl",
        ),
    )
    print(f"Uploaded input file: {uploaded_file.name}")
    job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    return job


def poll_batch(client, job_name, poll_interval):
    """Block until the batch job reaches a terminal state."""
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {job_name} state={state}")
        if state in COMPLETED_STATES:
            return job
        time.sleep(poll_interval)


def download_results(client, job, out_path):
    """Stream the result file out of the File API to local disk."""
    result_file_name = job.dest.file_name
    file_bytes = client.files.download(file=result_file_name)
    with open(out_path, "wb") as f:
        f.write(file_bytes)
    return out_path


def parse_results(jsonl_path):
    """Returns (key -> response text) and a list of per-key error strings."""
    key2text = {}
    errors = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            key = item.get("key")
            if "error" in item and item["error"]:
                errors.append((key, item["error"]))
                key2text[key] = None
                continue
            resp = item.get("response")
            if not resp:
                key2text[key] = None
                continue
            try:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                errors.append((key, f"parse error: {e}"))
                text = None
            key2text[key] = text
    return key2text, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmented_dataset_path", type=str, required=True)
    parser.add_argument("--raw_CoT_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=30000,
        help="Per-request output cap. Note: Batch API reserves quota up to this "
             "value per request, so don't set it much higher than you actually need.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Any Gemini model that supports the Batch API.",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Seconds between polls of the batch job state.",
    )
    parser.add_argument(
        "--resume_job",
        type=str,
        default=None,
        help="Existing batch job name (e.g. 'batches/abc123') to resume polling "
             "instead of submitting a new job.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    work_dir = Path(args.save_path) / "_batch"
    work_dir.mkdir(exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY env var not set.")
    client = genai.Client(api_key=api_key)

    id2ans = load_id2answer(args.raw_CoT_path)
    id2reasoning = json.load(open(args.segmented_dataset_path))
    id2reasoning = {str(k): v for k, v in id2reasoning.items()}
    print(f"id2ans: {len(id2ans)}, id2reasoning: {len(id2reasoning)}")

    base = os.path.basename(args.raw_CoT_path)
    stem, _ = os.path.splitext(base)

    if args.resume_job:
        job_name = args.resume_job
        print(f"Resuming batch job: {job_name}")
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
            jsonl_path, requests, args.temperature, args.max_output_tokens
        )
        print(f"Wrote {len(requests)} batch requests to {jsonl_path}")

        display_name = f"label-correctness-{stem}"
        job = submit_batch(client, str(jsonl_path), args.model, display_name)
        job_name = job.name
        print(f"Submitted batch job: {job_name}")

        with open(work_dir / f"last_job_{stem}.txt", "w") as f:
            f.write(job_name + "\n")

    job = poll_batch(client, job_name, args.poll_interval)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        err = getattr(job, "error", None)
        raise RuntimeError(
            f"Batch job ended with state {job.state.name}: {err}"
        )

    results_jsonl = work_dir / f"results_{stem}.jsonl"
    download_results(client, job, str(results_jsonl))
    print(f"Downloaded results to {results_jsonl}")

    key2text, errors = parse_results(str(results_jsonl))
    if errors:
        print(f"{len(errors)} per-request errors (showing first 5): {errors[:5]}")

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

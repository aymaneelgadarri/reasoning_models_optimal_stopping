<h1 align="center">Reasoning Models Know When They're Right</h1>


<div align="center">
  
<p style="font-size: 18px;">
📄 <a href="https://arxiv.org/pdf/2504.05419">Paper</a> |
🔗 <a href="https://drive.usercontent.google.com/download?id=140GPBMca27-hAL5P8phK_jl2O9mReqo8&export=download&authuser=1&confirm=t&uuid=2302ab95-eb89-444e-aeed-8738a2c1d8b2&at=APcmpoyoq2GkpsrhGUK9W6EpyYoO:17446852776">Model</a> |
🤗 <a href="#">Data (Coming soon)</a>
</p>


</div>

Official code for ["Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"](https://arxiv.org/pdf/2504.05419) 


![](./figures/main.png)

## 🔖 Table of Contents


- [Setup and Requirements](#setup-and-requirements)
- [Download Trained Probes](#download-trained-probes)
- [Train Your Own Probe](#train-your-own-probe)
  - [Data Preparation](#data-preparation) 
  - [Train Probes](#train-probs)
  - [Test Probes](#test-probes)
- [Section 5: Early-Exit Verifier Evaluation](#section-5-early-exit-verifier-evaluation)
- [Optimal-Stopping Policy Head (alternate training/eval mode)](#-optimal-stopping-policy-head-alternate-trainingeval-mode)




## 🔧 Setup and Requirements

```bash
conda create -n probe
conda activate probe
pip install -r requirements.txt
# download module for spacy (required for segmenting reasoning chunks)
python -m spacy download en_core_web_sm
```

## ⬇️ Download Trained Probes
- We provide trained probes for different model and dataset combinations. You can download them [here](https://drive.usercontent.google.com/download?id=140GPBMca27-hAL5P8phK_jl2O9mReqo8&export=download&authuser=1&confirm=t&uuid=2302ab95-eb89-444e-aeed-8738a2c1d8b2&at=APcmpoyoq2GkpsrhGUK9W6EpyYoO:17446852776).
    - the downloaded file contains a series of trained probe `pt` file.
    - the naming of each `pt` file follows `{model_name}_{train_data}_best_probe-{hyperparam_setting}.pt`
- If you want to use the probe off-the-shelf on other data, we recommend using the probe trained on **MATH** data as they usually show better generalizability.
- See below for how to [prepare your own test data](#data-preparation) and how to [evaluate the probe](#test-probes). 


## 🚀 Train Your Own Probe

### Data Preparation
#### 1. Generate CoT Reasoning
Generate Chain-of-Thought reasoning for each example in your dataset. We provide generated CoTs in [`./initial_cot`](./initial_cot).

```bash
# Set model path
model=/path/to/your/model/DeepSeek-R1-Distill-Qwen-1.5B

# Generate reasoning
python -u src/generate_reasoning.py \
    --data_name math-train \
    --model_path $model \
    --temperature 0.6 \
    --save_path ./initial_cot \
    --max_example 1000 # optional
```

#### 2. Extract Reasoning Chunks
Process the CoT outputs to identify reasoning chunks:

```bash
# Set variables
export SAVE_PATH="./processed_cot"
export NUM_PROCESSES=20  # Adjust based on CPU cores
export DATAFILE_PATH="./initial_cot/DeepSeek-R1-Distill-Qwen-1.5B_math-train_rollout_temperature0.6.jsonl"

# Process and segment reasoning paths
python src/get_reasoning_chunks.py \
    --datafile_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --num_processes "$NUM_PROCESSES" \
    --delete_chunks  # delete intermediate chunks after merging
```

#### 3. Label Intermediate Answers
Extract and label the correctness of intermediate answers in each chunk, meanwhile merging chunks that does not contain an answer with later chunks to ensure each chunk as an intermediate answer.

Note that we use Gemini API to extract and label the answers. You can also modify the [script](src/label_answer_correctness.py) to use other large language models for labeling.

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your_api_key_here"

# Set paths
export SAVE_PATH="./labeled_cot"
export SEGMENT_PATH="./processed_cot/segmented_CoT_DeepSeek-R1-Distill-Qwen-1.5B_math-train_rollout_temperature0.6_merged.json"
export DATAFILE_PATH="./initial_cot/DeepSeek-R1-Distill-Qwen-1.5B_math-train_rollout_temperature0.6.jsonl"

# Label answers
python src/label_answer_correctness.py \
    --segmented_dataset_path "$SEGMENT_PATH" \
    --raw_CoT_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --num_processes 20 \
    --delete_chunks
```

#### 4. Generate Hidden State Representations
Extract hidden state representations for each reasoning chunk. Note that to save space, we control `chunk_size` for how many chunk representations are stored in each file.

```bash
# Set variables
model_name=DeepSeek-R1-Distill-Qwen-1.5B
model_path=/path/to/your/model/$model_name
dataset=math-train
temperature=0.6
input_file=./labeled_cot/labeled_intermediate_answers_${model_name}_${dataset}_rollout_temperature${temperature}.jsonl
for file_id in {0..19}
do
python -u src/get_representation.py \
    --input_file $input_file \
    --model_name $model_path \
    --save_path ./model_embeds/${model_name}_${dataset} \
    --bs 16 \
    --file_id $file_id \
    --file_size 50 # file size measured by number of questions, should not be too large because that would affect training data shuffling
done
```
### Train Probes
Train the probe for predicting answer correctness. Do grid search over each hyperparameter.

```bash
# Set paths
export MODEL=DeepSeek-R1-Distill-Qwen-1.5B
export DATA=math-train
export TRAIN_DATA_PATH=./model_embeds/${MODEL}_${DATA}



# Run default grid search to find optimal hyperparameters
bash train_probe.sh

# OR Train with best parameters (example configuration)
python -u ./src/train_predictor_with_class_weights.py \
    --epochs 200 \
    --lr 1e-5 \
    --hidden_size 0 \
    --wd 0.001 \
    --alpha_imbalance_penalty 2.0 \
    --threshold 0.5 \
    --train_data_dir $TRAIN_DATA_PATH \
    --save_model_path ./grid_search/${MODEL}_${DATA}/checkpoints \
    --store_path ./grid_search/${MODEL}_${DATA}/store \
    --model_name $MODEL
```

For data and models used in our paper, you can replicate the results with best hyperparameters for each model+dataset combination as below:

![](./figures/hyperparam.png)

### Test Probes

To test your probe on test data, first follow [data preparation steps](#data-preparation) as above to obtain representations on test data in the same manner. Then evaluate the best trained probe on your test data.

```bash
model=DeepSeek-R1-Distill-Qwen-1.5B
data=math-train

TEST_SAVE_PATH=./test_result
TEST_DATA=./model_embeds/${model}_math_500

python -u ./test_predictor_with_class_weights.py \
    --input_size $INPUT_SIZE \
    --threshold 0.5 \
    --test_data_dir $TEST_DATA \
    --save_path $TEST_SAVE_PATH \
    --model_name $model \
    --checkpoint_model_path /path/to/best/probe/pt 
```

We also provide script to automatically run evaluation on top-k best trained probes in grid search.
Note that in [`src/test_predictor_with_class_weights.py`](./src/test_predictor_with_class_weights.py), we default to use `best_val_acc` as metric for ranking the probes. You can also customize the metric by yourself.

```bash
model=DeepSeek-R1-Distill-Qwen-1.5B
data=math-train

MODEL_BASE_PATH=./grid_search/${model}_${data}
GRID_SEARCH_PATH=$MODEL_BASE_PATH/grid_search_result.jsonl
TEST_SAVE_PATH=$MODEL_BASE_PATH/test_result

TEST_DATA=./model_embeds/${model}_math_500
python -u ./test_predictor_with_class_weights.py \
    --input_size $INPUT_SIZE \
    --threshold 0.5 \
    --test_data_dir $TEST_DATA \
    --grid_search_result_path $GRID_SEARCH_PATH \
    --topk 10 \ # test top-k probes from grid search results
    --save_path $TEST_SAVE_PATH \
    --model_name $model
```


## 🛑 Section 5: Early-Exit Verifier Evaluation

Reproduces the **offline** version of the Section 5 confidence-based
early-exit experiment.  Given pre-generated, chunked, and labelled reasoning
traces, the script scores every intermediate answer with the trained probe,
picks the first chunk whose probability exceeds a threshold, and reports
final-answer accuracy together with the assistant-side token cost (vs. the
no-early-exit and static-`k` baselines).

This implementation does **not** interrupt live generation; it scores all
chunks in order and selects the first threshold hit, which is equivalent to
what live early-exit would produce on the same traces.

### Inputs

- `--labeled_data_path`: a `labeled_intermediate_answers_*.jsonl` produced by
  [step 3](#3-label-intermediate-answers).
- `--probe_ckpt`: a probe `.pt` from [Train Probes](#train-probs) (either the
  raw `state_dict` or the wrapped `{"model": ..., "pos_weight_from_train": ...}`
  dict are accepted; the probe's hidden size is auto-inferred from the file
  name when possible, or can be passed via `--probe_hidden_size`).
- `--model_path`: the same base LM that was used to extract the probe's
  training-time hidden states.

### Run

```bash
PROBE_CKPT=/path/to/best_probe.pt \
MODEL_NAME=DeepSeek-R1-Distill-Llama-8B \
DATASET=math_500 \
bash eval_early_exit.sh
```

Or invoke the Python entry-point directly:

```bash
python -u src/eval_early_exit.py \
  --labeled_data_path ./labeled_cot/labeled_intermediate_answers_DeepSeek-R1-Distill-Llama-8B_math_500_rollout_temperature0.6.jsonl \
  --model_path /path/to/DeepSeek-R1-Distill-Llama-8B \
  --probe_ckpt /path/to/best_probe.pt \
  --model_name DeepSeek-R1-Distill-Llama-8B \
  --output_dir ./early_exit_results/llama8b_math \
  --batch_size 8 \
  --max_examples -1 \
  --max_generation_tokens 10000 \
  --thresholds 0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99 \
  --static_k_values 1,2,3,4,5,6,7,8,9,10 \
  --seed 42
```

### Outputs (under `--output_dir`)

- `per_example_scores.json` – per-chunk probe probabilities, ground-truth
  correctness labels, and cumulative assistant-side token counts. Re-running
  with `--scores_only` recomputes the metrics from this file without
  re-invoking the base LM.
- `early_exit_metrics.json` – aggregate accuracy, average tokens used, token
  ratio and reduction for:
  - `no_early_exit` (always pick the last chunk),
  - `confidence_early_exit` (sweep over `--thresholds`),
  - `static_early_exit` (sweep over `--static_k_values`).
- `accuracy_vs_tokens.png` and `accuracy_vs_threshold.png` – plotted
  trade-off curves (produced by `src/plot_early_exit.py`).

### Notes on consistency with training-time representations

`src/eval_early_exit.py` reuses the cumulative-prefix prompt format from
`src/get_representation.py` via `src/early_exit_utils.py`, so the probe sees
exactly the same kind of hidden states it was trained on.  Both code paths
now also use **left-padding** with `pad_token := eos_token` and a robust
last-non-pad-token gather (`attention_mask.sum(dim=1) - 1`) instead of
`last_hidden_state[:, -1, :]`, which previously could return the embedding
of a pad token for shorter sequences in a right-padded batch.

## 🧮 Optimal-Stopping Policy Head (alternate training/eval mode)

In addition to the per-chunk correctness probe, we provide a second
training/eval mode that turns the **same MLP head** (same hidden-state
features, same `MLPProbe` / `LinearProbe` architecture, same checkpoint
format) into an *optimal-stopping policy*.  The head still emits one
scalar per chunk, but instead of training it with BCE we train it by
maximising the expected reward of the induced stopping distribution.

Two formulations are implemented and selectable via `--formulation`:

- `product` -- classical product-of-continue-probabilities baseline.
  Per-chunk score \(c_i = \sigma(z_i)\) is interpreted as a *continue*
  probability and the survival is the cumulative product
  \(S_i = \prod_{j \le i} c_j\).
- `min_survival` -- new running-minimum survival formulation.  The head
  emits \(s_i = \sigma(z_i)\) interpreted as a *candidate* survival
  probability, and the actual survival is the running minimum
  \(S_i = \min_{j \le i} s_j\).  This guarantees a non-increasing
  survival sequence without the multiplicative collapse that `product`
  suffers on long CoTs.

Both formulations turn the survival sequence into a stopping
distribution via \(\mathrm{stop}_1 = 1 - S_1\),
\(\mathrm{stop}_i = S_{i-1} - S_i\).  The residual mass \(S_m\) is
folded into the last chunk so the distribution sums to 1 (matching the
"finish entire CoT == stop at last chunk" convention used elsewhere in
the codebase).  Training maximises
\(\sum_i \mathrm{stop}_i \cdot r_i\) where \(r_i\) is the chunk-level
correctness label, and inference picks \(\arg\max_i \mathrm{stop}_i\).

Code organisation:

- `src/stopping_formulations.py` -- shared interface; both formulations
  expose `stop_distribution(logits)` and `expected_reward(logits, rewards)`.
- `src/dataloader_per_example.py` -- per-example loader that
  reconstructs example boundaries by aligning the existing
  `model_embeds/.../embed_file_<a>_<b>.pt` files with
  `labeled_intermediate_answers_*.jsonl`.  No re-extraction needed.
- `src/train_stopping_policy.py` -- trainer with `--formulation
  {product, min_survival}` flag.  Saves checkpoints in the same
  wrapped format (`{"model": ..., "pos_weight_from_train": None,
  "formulation": ...}`) so they remain loadable with
  `early_exit_utils.load_probe_from_ckpt`.
- `src/eval_stopping_policy.py` -- offline evaluator that mirrors
  `eval_early_exit.py`.  Either runs the base LM end-to-end, or loads
  a cached `per_example_scores.json` for fast metric re-runs.

### Train

`train_stopping_policy.sh` is a grid-search script (the same shape as
`train_probe.sh`).  It trains a *single* `FORMULATION` per invocation
so the two formulations end up in distinct model directories with
their own grid-search logs.  Sweep axes:

- `LRS`            -- learning rate (default `"1e-3 1e-4 1e-5"`)
- `HIDDEN_SIZES`   -- 0 (linear) or 16/32 (MLP) (default `"0 16 32"`)
- `WDS`            -- weight decay (default `"0.001 0.01 0.1"`)
- `LAMBDAS`        -- length-cost coefficient \(\lambda\) in the
  per-chunk reward `r_i := correctness_i - lambda * i` (0-indexed).
  Larger \(\lambda\) biases the policy toward earlier stops.  Default
  `"0 0.01 0.05 0.1"`.

```bash
# Sweep min_survival
FORMULATION=min_survival \
EMBED_DIR=./model_embeds/DeepSeek-R1-Distill-Qwen-1.5B_aime_25 \
LABELED_DATA_PATH=./labeled_cot/labeled_intermediate_answers_DeepSeek-R1-Distill-Qwen-1.5B_aime_25_rollout_temperature0.6.jsonl \
MODEL=DeepSeek-R1-Distill-Qwen-1.5B \
bash train_stopping_policy.sh

# Sweep product (separate save dir, separate grid log)
FORMULATION=product \
EMBED_DIR=./model_embeds/DeepSeek-R1-Distill-Qwen-1.5B_aime_25 \
LABELED_DATA_PATH=./labeled_cot/labeled_intermediate_answers_DeepSeek-R1-Distill-Qwen-1.5B_aime_25_rollout_temperature0.6.jsonl \
MODEL=DeepSeek-R1-Distill-Qwen-1.5B \
bash train_stopping_policy.sh
```

Override the grid axes from the command line, e.g.:

```bash
FORMULATION=min_survival LAMBDAS="0 0.05" HIDDEN_SIZES="0" \
  bash train_stopping_policy.sh
```

Or invoke the Python entry-point directly for one config:

```bash
python -u src/train_stopping_policy.py \
  --labeled_data_path ./labeled_cot/labeled_intermediate_answers_DeepSeek-R1-Distill-Qwen-1.5B_aime_25_rollout_temperature0.6.jsonl \
  --embed_dir ./model_embeds/DeepSeek-R1-Distill-Qwen-1.5B_aime_25 \
  --model_name DeepSeek-R1-Distill-Qwen-1.5B \
  --formulation min_survival \
  --hidden_size 0 --epochs 200 --lr 1e-4 --wd 1e-3 --lambda_penalty 0.05 \
  --save_model_path ./grid_search_stopping/DeepSeek-R1-Distill-Qwen-1.5B_aime_25_min_survival/checkpoints \
  --store_path ./grid_search_stopping/DeepSeek-R1-Distill-Qwen-1.5B_aime_25_min_survival/store
```

Each run appends a JSON row to
`<SAVE_DIR>/stopping_grid_search_result.jsonl` so you can sort by
`best_val_expected_reward` to pick a winner and feed its `best_ckpt`
into `eval_stopping_policy.py`.

### Evaluate

If you already ran `eval_early_exit.sh` once, you can re-score offline
without re-loading the base LM by pointing at the cached scores:

```bash
python -u src/eval_stopping_policy.py \
  --probe_ckpt ./grid_search_stopping/DeepSeek-R1-Distill-Qwen-1.5B_aime_25/checkpoints/best_stopping_min_survival-hs0-lr0.0001-wd0.001-s42.pt \
  --model_name DeepSeek-R1-Distill-Qwen-1.5B \
  --formulation min_survival \
  --scores_path ./early_exit_results/DeepSeek-R1-Distill-Qwen-1.5B_aime_25/per_example_scores.json \
  --output_dir ./stopping_results/DeepSeek-R1-Distill-Qwen-1.5B_aime_25_min_survival
```

For a fresh evaluation from a new probe checkpoint, run the full
pipeline (matches `eval_early_exit.sh`):

```bash
PROBE_CKPT=/path/to/best_stopping_min_survival-...-pt \
MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B \
DATASET=aime_25 \
FORMULATION=min_survival \
bash eval_stopping_policy.sh
```

Outputs (under `--output_dir`):

- `per_example_scores.json` -- per-chunk probe logits/probabilities,
  correctness labels and cumulative assistant-side token counts.
- `accuracy_vs_tokens_{expected,argmax}.png` -- produced by
  `src/plot_stopping_policy.py`.  For each formulation the script
  groups the grid runs by `lambda_penalty`, picks the best
  other-hyperparameters configuration *for that lambda* (default
  selection metric: `best_val_expected_reward`), evaluates the
  winning checkpoint on the cached embeddings and plots one point
  per lambda.  Optionally overlays the no-exit / static / threshold
  baselines from `early_exit_metrics.json` on the same axes.  Example:

  ```bash
  python -u src/plot_stopping_policy.py \
    --grid_result ./grid_search_stopping/DeepSeek-R1-Distill-Qwen-1.5B_aime_25_min_survival/stopping_grid_search_result.jsonl \
    --grid_result ./grid_search_stopping/DeepSeek-R1-Distill-Qwen-1.5B_aime_25_product/stopping_grid_search_result.jsonl \
    --model_name DeepSeek-R1-Distill-Qwen-1.5B \
    --embed_dir ./model_embeds/DeepSeek-R1-Distill-Qwen-1.5B_aime_25 \
    --labeled_data_path ./labeled_cot/labeled_intermediate_answers_DeepSeek-R1-Distill-Qwen-1.5B_aime_25_rollout_temperature0.6.jsonl \
    --model_path $HOME/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --early_exit_metrics ./early_exit_results/DeepSeek-R1-Distill-Qwen-1.5B_aime_25/early_exit_metrics.json \
    --output_dir ./stopping_results/DeepSeek-R1-Distill-Qwen-1.5B_aime_25_curve \
    --title_suffix "DeepSeek-R1-Distill-Qwen-1.5B / aime_25"
  ```

- `stopping_policy_metrics.json` -- accuracy, average tokens used, and
  token ratio/reduction under **two** readouts of the trained policy:

  - `argmax_*`   -- pick a single chunk via \(\arg\max_i \mathrm{stop}_i\)
    and report its accuracy/cost.  Deterministic decoding rule.
  - `expected_*` -- average under the full stopping distribution:
    \(\mathbb{E}[\mathrm{acc}] = \sum_i \mathrm{stop}_i \cdot \mathrm{correctness}_i\)
    and similarly for tokens.  Equivalent to the training objective at
    \(\lambda=0\) and the most faithful summary of the trained policy
    because it consumes the entire distribution rather than collapsing
    to its mode.  Chunks with unknown labels are masked out and the
    remaining mass is renormalised so they do not bias the average.

## 📝 Citation

If you find our code or data useful, please cite our paper:
```bibtex
@misc{zhang2025reasoningmodelsknowtheyre,
      title={Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification}, 
      author={Anqi Zhang and Yulin Chen and Jane Pan and Chen Zhao and Aurojit Panda and Jinyang Li and He He},
      year={2025},
      eprint={2504.05419},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.05419}, 
}
```

Example of commands in order for training and eval:
To process the initial cots into chunks:

bash get_reasoning_chunks.sh --model DeepSeek-R1-Distill-Qwen-1.5B --data math_500 --delete_chunks true

To label the COTs using openAI batch API:

sbatch --export=ALL,MODEL=DeepSeek-R1-Distill-Qwen-1.5B,DATA=math_500 label_answer_openai_batch.sbatch

To process the embeddings of the model on the dataset:

MODEL=DeepSeek-R1-Distill-Qwen-1.5B DATASET=math_500 bash get_representation.sh

To train the optimal stopping networks:
MODEL=DeepSeek-R1-Distill-Qwen-1.5B DATASET=math-train bash train_stopping_policy.sh

To get early exit plots for the classifier prob first:

MODEL=DeepSeek-R1-Distill-Qwen-1.5B DATASET=math-train TEST_DATASET=math_500 bash eval_early_exit.sh

To get early exit probes for the stopping algos as well:

MODEL=DeepSeek-R1-Distill-Qwen-1.5B DATASET=math-train bash eval_stopping_per_lambda.sh


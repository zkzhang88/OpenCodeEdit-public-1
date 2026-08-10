# Data Synthesis Pipeline

To run the OpenCodeEdit, you can follow the guideline below:

## Requirements

Before running the pipeline, make sure that you have installed the following packages and their specific versions.
```
gensim==4.3.3
joblib==1.4.2
matplotlib==3.10.6
nltk==3.9.1
numpy==1.26.4
openai==1.59.6
pandas==2.2.3
plotly==6.3.0
Pygments==2.19.2
PyYAML==6.0.2
spacy==3.8.7
tqdm==4.67.1
```


## Data Generation

First, run `create_prompt.py` to construct prompts for data synthesis from `commitpackft`:
```bash
python create_prompt.py
```

This command uses the `v5.2` two-round code-editing prompt and creates a jsonl file `prompt_for_syn.jsonl` in the `./data/` folder, which serves as the prompt input for the LLM. Each prompt receives a positive integer `prompt_id`, numbered consecutively from `1` in final file order.

> We also provide the prompt for commit rewriting. You can construct such prompts by setting the `--prompt_type` parameter as follow:
> ```bash
> python create_prompt.py --prompt_type rewrite_commit
> ```

The prompt templates can be found in `prompts_for_gen.py`

The unified inference pipeline supports direct OpenAI-compatible APIs,
SiliconFlow Batch Inference, and the local `batch-infer` command installed in
the `llm_infer` Conda environment. Copy both local configuration templates:

```bash
cp api_config.example.yaml api_config.yaml
cp inference_config.example.yaml inference_config.yaml
```

Fill API credentials in the ignored `api_config.yaml`. Model names, local model
paths, sampling settings, GPU selection, and executor options belong in the
ignored `inference_config.yaml`; see the tracked example for every supported
field. Credentials are loaded only for the selected executor and are never
written to a run manifest.

Run direct API inference from the repository root:

```bash
python generation/inference.py run \
  --executor api \
  --model qwen3-32b \
  --config generation/inference_config.yaml \
  --input generation/data/prompt_for_syn.jsonl \
  --output generation/data/generated_instr_qwen3.jsonl \
  --run-dir generation/data/runs/qwen3_api
```

For SiliconFlow, `run` uploads and submits the first round, records every file
and job ID in the run manifest, and then exits by default:

```bash
python generation/inference.py run \
  --executor siliconflow-batch \
  --model deepseek-v3 \
  --config generation/inference_config.yaml \
  --input generation/data/prompt_for_syn.jsonl \
  --output generation/data/generated_instr_siliconflow.jsonl \
  --run-dir generation/data/runs/siliconflow_deepseek
```

Continue it with waiting enabled to collect round one, submit and collect round
two, and write the final output:

```bash
python generation/inference.py continue \
  --run-dir generation/data/runs/siliconflow_deepseek \
  --wait
```

Add `--wait` to the initial `run` command to block through both remote rounds.
Without `--wait`, `continue` checks the current jobs once, advances any
immediately available stage, and exits. `continue` is only valid for
SiliconFlow runs and reuses the saved remote job IDs instead of uploading an
active Batch again.

For local vLLM inference, configure the model path and GPU settings in the
`local-qwen3` profile and run:

```bash
python generation/inference.py run \
  --executor llm-infer \
  --model local-qwen3 \
  --config generation/inference_config.yaml \
  --input generation/data/prompt_for_syn.jsonl \
  --output generation/data/generated_instr_local.jsonl \
  --run-dir generation/data/runs/local_qwen3
```

By default this invokes `conda run --no-capture-output -n llm_infer
batch-infer batch --auto-serve` once per conversation round. Set
`auto_serve: false` and `base_url` in the profile to reuse an existing service.
If API or local inference is interrupted, resume the same attempt with:

```bash
python generation/inference.py resume \
  --run-dir generation/data/runs/local_qwen3
```

For local inference this reuses the existing attempt input and output files and
invokes `batch-infer batch --resume`. Records already present in the attempt
output are not run again. Each resume invocation gets separate stdout, stderr,
and vLLM logs.

Every task is identified by `<prompt_id>:<sample_index>`. Per-round results are
flushed under the run directory, while the explicitly selected final output is
created atomically only after every task completes every round. Inspect a run
without changing it with:

```bash
python generation/inference.py status --run-dir generation/data/runs/local_qwen3
```

Failed or missing Batch requests are retried without rerunning successful tasks.
After the configured attempt budget is exhausted, the run remains `incomplete`
and no final output is created. After correcting the external problem, grant the
incomplete round a fresh retry budget with:

```bash
python generation/inference.py retry \
  --run-dir generation/data/runs/local_qwen3
```

Use `retry --wait` for a SiliconFlow run when the command should wait for the
new remote attempt. The original prompt file must not change during a run.
Run manifests use a strict schema version; run directories created by an older
inference implementation cannot be resumed, continued, retried, or inspected
with this version.


## Extracting Edit Triplets from Model Responses

This step should be excuted after sufficient data have been generated (not less than 30,000 samples for each model).

For example, to extract the edit triplets from Qwen3, run the following script:
```bash
python get_instruct_from_response.py ./data/generated_instr_qwen3.jsonl ./data/triplets_qwen3.jsonl
```

The code edit triplets are stored in the `triplets_qwen3.jsonl`.


## Data Mixing
To mix the extracted data from different models and different description, use `mix_data.py`. The combination of each dataset can be set up through yaml files in `./mix_config/` folder. 

For example, to combine the descriptive instructions of Qwen3 and DeepSeek generated data:
```yaml
## ocedata_mix_descriptive.yaml
# Each entry in the fields input_files, ratios, instr_types, and model_names must correspond to one another one-to-one.

input_files:
  - data/triplets_qwen3.jsonl
  - data/triplets_ds.jsonl
ratios: [0.5, 0.5]
instr_types: [descriptive, descriptive]
model_names: [qwen3, ds]
output_file: data/ocedata_mix_descriptive.jsonl
total_samples: 60000   # The total samples in the output file
random_seed: 42   # random seed for sampling data from each input file
```

This will merge the specified input files into a single dataset `ocedata_mix_descriptive.jsonl` for downstream tasks. 

**Settings in `./mix_config/` folder:**
- `ocedata_mix_descriptive.yaml`: combine the descriptive instructions from Qwen3 and DeepSeek;
- `ocedata_mix_lazy.yaml`: combine the lazy instructions from Qwen3 and DeepSeek;
- `ocedata_mix.yaml`: combine the descriptive and lazy instructions from Qwen3 and DeepSeek.

The usage of `mix_data.py`:
```bash
python mix_data.py --config ./mix_config/ocedata_mix_descriptive.yaml
```


## DT Filtering
Run `dt_filtering.py` to filter data using DTFiltering:
```bash
python dt_filtering.py --config filter_config.yaml
```

You can change the file to be filtered in the `filter_config.yaml`. The output file will be stored in the `./data/filtered/` directory, with a `_dt_filtered` suffix. 

In HDP modeling process, the analysis results are saved in `*.joblib` files in `./utils/fit_results/` directory, for repetitive running. If you want to rebuild the analysis results, set `refit: true` in `filter_config.yaml`.

## OCEData Static Quality Check

Run the static quality checker from the repository root:

```bash
python3 generation/check_ocedata_quality.py
```

By default, it checks `data/OCEData/ocedata.jsonl` and writes:

- `data/OCEData/ocedata_quality_issues.jsonl`, with one structured report
  record for every rejected sample;
- `data/OCEData/ocedata_quality_filtered.jsonl`, containing the original JSONL
  lines that passed every check;
- `data/OCEData/ocedata_quality_summary.yaml`, containing the Python runtime,
  record totals, parser and issue counts, and output paths also shown in the
  terminal summary.

When `--input-file` is changed and output paths are omitted, all outputs are
created next to that input using `<input_stem>_quality_issues.jsonl`,
`<input_stem>_quality_filtered.jsonl`, and
`<input_stem>_quality_summary.yaml`. Explicit `--report-file`,
`--filtered-file`, and `--summary-file` values override these derived names.

For pre-edit code, the checker always reports empty output and Markdown fences.
Clear truncation or incomplete structures are reported only when post-edit
cannot be parsed; a post-edit that parses successfully is treated as having
repaired the incomplete input. When possible, pre-edit code is silently analyzed
as the baseline for identifying issues introduced by the edit. Post-edit code
receives the complete Python syntax and static-reference checks, including
undefined names, missing imports, and references made unresolvable by an edit.
Sample code and third-party imports are never executed.
Pre/post code is considered identical when it differs only in blank lines or
formatting whitespace; whitespace inside strings remains significant.
Use `--input-file`, `--report-file`, `--filtered-file`, `--summary-file`,
`--pre-field`, and `--post-field` to override the defaults. Add
`--fail-on-issues` to return a non-zero status when rejected samples are found.
The checker displays a `tqdm` progress bar by default; use `--no-progress` to
disable it.

To inspect selected rejected samples, export their source code by report line
number:

```bash
python3 generation/export_quality_issue_code.py \
  --issues-file data/OCEData/ocedataft_quality_issues.jsonl \
  --line-number 36 39 57
```

The source JSONL is inferred from the standard `_quality_issues.jsonl` suffix.
Each requested record is written under
`<input_stem>_quality_issue_code/line_NNNNNN/` as `pre_edit.py`, `post_edit.py`,
and `issues.json`; the JSON metadata also includes the source edit instruction
as `edit_instruction`. Use `--input-file`, `--output-dir`, or
`--instruction-field` to override the inferred paths and field name.

To inspect the first `n` records of the filtered finetuning dataset in a
human-readable layout, run from the repository root:

```bash
python3 generation/export_filtered_samples.py 10
```

By default, this reads `data/OCEData/ocedataft_quality_filtered.jsonl` and
writes each record to
`data/OCEData/ocedataft_quality_filtered_samples/line_NNNNNN/`. Each record
directory contains `pre_edit.py`, `post_edit.py`, and `instruction.json`.
The human-readable, indented JSON object contains `instruct_purify`, `commit`,
and `instr_type`, followed by seven manual-review fields initialized to JSON
`null`:

- `manual_pre_edit_is_reasonable_program`
- `manual_pre_edit_does_not_satisfy_instruction`
- `manual_instruction_is_clear_and_actionable`
- `manual_post_edit_is_reasonable_program`
- `manual_post_edit_fulfills_instruction`
- `manual_post_edit_has_no_unrelated_changes`
- `manual_post_edit_has_no_new_defects`

Reviewers can replace each `null` with `true` or `false`; all seven fields use
positive wording, so `true` consistently indicates a passing assessment. Use
`--input-file` and `--output-dir` to choose other paths; the source field names
can be overridden with `--pre-field`, `--post-field`, and
`--instruction-field`.

To export the first `k` records for one or more instruction types, pass the
types to `--instr-type`:

```bash
python3 generation/export_filtered_samples.py 10 \
  --instr-type ds_descriptive qwen3_descriptive
```

In this mode, the positional number is applied to each requested type. Records
are written below type-named directories such as
`ocedataft_quality_filtered_samples/ds_descriptive/line_NNNNNN/`.


## Finetune dataset construction
After data mixing and filtering, you can run `generate_finetune_dataset.py` to construct a formatted dataset for downstream finetuning:
```bash
python generate_finetune_dataset.py --input_file ./data/filtered/ocedata_mix_descriptive_dt_filtered.jsonl ./data/filtered/ocedata_mix_lazy_dt_filtered.jsonl --output_file ./data/finetune/ocedata_mix_ft.jsonl
```

Both Alpaca and ShareGPT formats are supported, for a convenient finetuning through [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). For more information about the data formats please refer to [LLaMA-Factory documentation](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html).

# Benchmark Replication Guideline

## CanItEdit Benchmark

### Starting

First, clone the [CanItEdit](https://github.com/nuprl/canitedit) repository from GitHub:

```bash
git clone --recurse-submodules https://github.com/nuprl/CanItEdit

cd CanItEdit/benchmark
```

Then, replace the `generate_completions.py` in the CanItEdit repository with the identically named file in the `benchmark/CanItEdit` directory of our repository.

### Generate Post-edit Code

**For Qwen Models:**

Run:

```bash
python benchmark/generate_completions.py \
    --model-type qwencoder \
    --model model_dir/OpenCodeEdit-Qwen3-8B \
    --dataset dataset_dir/CanItEdit \
    --output-dir opencodeedit-qwen3-8b-outputs \
    --num-gpus 1 \
    --completion-limit 20 \
    --batch-size 10 \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-tokens 2048
```

**For OpenCodeEdit-DSC-6.7B:**

Run:

```bash
python benchmark/generate_completions.py \
    --model-type deepseek \
    --model model_dir/OpenCodeEdit-DSC-6.7B \
    --dataset dataset_dir/CanItEdit \
    --output-dir opencodeedit-dsc-6.7b-outputs \
    --num-gpus 1 \
    --completion-limit 20 \
    --batch-size 10 \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-tokens 2048
```

### Executing Tests and Metric Calculation

When the post-edit code have been generated, you can execute the tests to evaluate the results.
To do this, you first need to install the Docker image that contains the test runner,
which can be done by running `make build-docker` in the `CanItEdit/benchmark` directory.
Then, you can run the tests using the `evaluate_completions.sh` script,
pointing it to the directory where the completions are saved.

For example, to evaluate the completions generated in the previous step, which
were saved in the `outputs` directory, you can run:

```bash
./evaluate_completions.sh ./outputs
```

Finally, you can retrieve the results by running the `pass_k.py` script pointing it to the directory where the completions are saved.

For example, to retrieve the results from the previous step, you can run:

```bash
python pass_k.py ./outputs
```

You will be provided with `pass@1` and `ExcessCode` metrics.
You can provide a `-k` parameter to the script to change the value of `k` for the `pass@k` metric.

```
python pass_k.py -k 5 ./outputs
```

We recommend you to run the `pass_k_v2_export_excel.py` script provided in our repository instead, which can calculate the metrics of three models at one time and export them into an `*.xlsx` file. 

For example, the outputs of the three models are saved in seperated folders `opencodeedit-qwen3-8b-outputs`, `opencodeedit-qwen2.5-7b-outputs`, and `opencodeedit-dsc-6.7b-outputs` under the `all_outputs` directory, just run:

```bash
python pass_k_v2_export_excel.py ./all_outputs --output results.xlsx -k 1 -a
```

The metrics will be saved in the `results.xlsx` file.
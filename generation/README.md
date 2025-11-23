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

This command creates a jsonl file `prompt_for_syn.jsonl` in `./data/` folder, which serves as the prompt input for the LLM. 

> We also provide the prompt for commit rewriting. You can construct such prompts by setting the `--prompt_type` parameter as follow:
> ```bash
> python create_prompt.py --prompt_type rewrite_commit
> ```

The prompt templates can be found in `prompts_for_gen.py`

`code_generation_api.py` calls API from [DeepSeek](https://platform.deepseek.com/) or [Aliyun](https://help.aliyun.com/zh/model-studio/models), so please apply for the API keys from the websites. If you have obtained an API key, please replace the following content at the beginning of the `code_generation_api.py` file with your API Key:
```python
# Replace sk-xxxx with your API Keys
QWEN_API_KEY = "sk-xxxx"
DEEPSEEK_API_KEY = "sk-xxxx"
```

Then, use Qwen3 to generate data by running:
```bash
python code_generation_api.py --input_file data/prompt_for_syn.jsonl --output_file data/generated_instr_qwen3.jsonl --recovery_file data/generated_instr_qwen3_recovery.jsonl --model_name qwen3-32b
```

You can use DeepSeek for generation by setting `--model_name deepseek-chat`, but remember to change the `--output_file` and `--recovery_file` to another name!

The generation process may take several hours or even several days to finish. The `--recovery_file` is used for recovering from disruption. If the generation process is distruped, please set `--continue_from_error` so as to recover generation from the checkpoint. 


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


## Finetune dataset construction
After data mixing and filtering, you can run `generate_finetune_dataset.py` to construct a formatted dataset for downstream finetuning:
```bash
python generate_finetune_dataset.py ./data/filtered/ocedata_mix_descriptive_dt_filtered.jsonl ./data/finetune/ocedata_mix_descriptive_ft.jsonl
```

Both Alpaca and ShareGPT formats are supported, for a convenient finetuning through [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). For more information about the data formats please refer to [LLaMA-Factory documentation](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html).
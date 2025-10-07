# OpenCodeEdit

OpenCodeEdit is a fully open-source data synthesis pipeline for instruction tuning LLMs on code editing.

## Quick Guide

- The code of the OpenCodeEdit pipeline is in the `generation/` folder. To run the pipeline, please follow the guideline in [generation/README.md](./generation/README.md).

- The OpenCodeEdit series models can be downloaded from [here](https://huggingface.co/collections/zkzhang88/opencodeedit-series-models-68e4c5a5d6b616f7f229f96b).

- The OCEData and OCEDataFT can be downloaded from [here](https://huggingface.co/datasets/zkzhang88/OCEData).

## Data Synthesis Pipline

The OpenCodeEdit pipeline consists of four stages: 

**① Seed Code Snippet Extraction**, where authentic code fragments are sampled as the foundation of synthesis; 

**② Pre-edit Code and Instruction Generation**, where editable snippets and corresponding natural-language requests are generated; 

**③ Post-edit Code Generation**, where revised code is produced to fulfill the requested edits; and 

**④ Data Filtering**, where noisy or redundant samples are removed to ensure dataset quality. 

To better reflect real-world editing scenarios, our generated dataset includes both *lazy* and *descriptive* instruction styles, encouraging models to generalize across concise developer prompts and more detailed specifications. The overall workflow is shown in the following figure:

![Overview of OpenCodeEdit.](images/opencodeedit_pipeline.png)

The code for this pipeline is located in `/generation` directory. For more details, refer to [generation/README](./generation/README.md)

## OCEData and OCEDataFT

The dataset constructed by OpenCodeEdit is an edit triplet composed by three parts:

- **Pre-edit code:** the original snippet requiring modification.
- **Edit instruction:** a natural-language description specifying the intended change.
- **Post-edit code:** the revised snippet after applying the edit.

To reflect the diversity of real-world editing scenarios, our dataset includes two complementary instruction styles:

- **Lazy instructions**, concise and high-level, resembling developer-written prompts (e.g., "add error handling for null inputs").
- **Descriptive instructions**, detailed and context-aware, similar to model-generated reflections that fully articulate the required change.

An example of the code editing training data is shown in the following figure:

![An Example of the Code Editing Training Data](images/code_edit_triplet_example.png)

In our experiments, we employ the Qwen3-32B-Instruct and DeepSeek-V3-0324 models to generate each part of the code edit triplet, separately. The temperature is set to 0.8 and the top-p value to 0.95, with a maximum number of output tokens of 2048. For each pair of code snippets, we generate data only once. The data are generated through API calls from [DeepSeek](https://platform.deepseek.com/) and [Aliyun](https://help.aliyun.com/zh/model-studio/models).

We generate each component of the edit triplet through a two-round dialogue. An example of this dialogue is shown in the following figure:

![An example of the two-round dialogue](images/dialogue_example.png)

The filtered fine-tune dataset OCEDataFT and unfiltered version OCEData is located in  `/datasets` directory. The datasets are constructed as JSONL files in ShareGPT format.

## Downstream Fine-tuning

In this work, we use LLaMA-Factory for downstream fine-tuning. For any usage, please refer to https://github.com/hiyouga/LLaMA-Factory.

## Evaluation

### Benchmark
We adopt [CanItEdit](https://github.com/nuprl/canitedit) as our benchmark for evaluation, which is designed to evaluate the code editing capabilities of LLMs. It comprises 105 manually curated Python problems, each accompanied by two types of instructions: a concise lazy edit instruction and a detailed descriptive edit instruction. The tasks are evenly distributed across various senarios, covering multiple domains and libraries. Each problem is paired with hidden test cases for correctness verification.

### Metric
we evaluate model performance using the pass@1 metric, which measures the probability that a single generated solution passes all predefined test cases. Formally, for a set of model-generated answers, pass@1 is defined as:

$$
\mathrm{pass@1} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\left [ \mathrm{Solution}  \ i \ \mathrm{passes \ all \ tests}  \right ]
$$

where $N$ denotes the number of generated solutions and $\mathbf{1}[\cdot]$ is the indicator function. A higher pass@1 value indicates a greater success rate, thereby reflecting stronger instruction-following and code editing performance.

We use the following settings for inference in our evaluation: 2048 maximum new tokens, temperature of 0.2, and top-p of 0.95. We sample 20 completions for each problem, and calculate pass@1.

### Results
The results for evaluting the effectiveness of OpenCodeEdit:

![Overall Results](images/overall_results.png)
*Note:* The results of GPT-4, GPT-3.5, DeepSeekCoder-Instr-33B, DeepSeekCoder-Instr-6.7B, CodeLlama-Instruct-7B, and Editcoder-6.7B are cited from [[1]](#ref1); the result of SelfCodeAlign-CQ-7B is cited from [[2]](#ref2).


## Reference
1. <a id="ref1"></a> Federico Cassano, Luisa Li, Akul Sethi, Noah Shinn, Abby Brennan-Jones, Jacob Ginesin, Edward Berman, George Chakhnashvili, Anton Lozhkov, Carolyn Jane Anderson, et al. 2024. Can It Edit? Evaluating the Ability of Large Language Models to Follow Code Editing Instructions. In *First Conference on Language Modeling*.
2. <a id="ref2"></a> Yuxiang Wei, Federico Cassano, Jiawei Liu, Yifeng Ding, Naman Jain, Zachary Mueller, Harm de Vries, Leandro Von Werra, Arjun Guha, and LINGMING ZHANG. 2024. SelfCodeAlign: Self-Alignment for Code Generation. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*.


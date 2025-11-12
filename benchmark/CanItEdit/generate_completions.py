import datasets
from pathlib import Path
from typing import List, Callable, TypeVar
from tqdm import tqdm
import torch
import openai
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Literal, Optional, TypedDict, Callable, Union
from vllm import LLM, RequestOutput, SamplingParams, CompletionOutput
import gzip
import json


def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


T = TypeVar("T")


def batch_prompts_from_example(example: T, batch_size: int, completion_limit: int) -> List[List[T]]:
    """
    Batches prompts into batches of size batch_size, where each batch
    contains completion_limit prompts.

    Args:
        example: The example to batch.
        batch_size: The batch size.
        completion_limit: The number of completions to generate for each prompt.

    Returns:
        A list of batches, where each batch is a list of prompts.
    """
    # 生成 completion_limit 个示例的副本
    prompts = [example] * completion_limit
    # 计算批次数量
    num_batches = completion_limit // batch_size
    # 创建批次列表
    batches = [prompts[i * batch_size: (i + 1) * batch_size]
               for i in range(num_batches)]
    # 如果提示总数不能被 batch_size 整除，则添加最后一个批次
    # the last batch may be smaller
    if len(prompts) % batch_size != 0:
        batches.append(prompts[num_batches * batch_size:])

    return batches


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str
    # the prefix for the assistant's response. this is only used for the
    # last message in a conversation, and is ignored otherwise.
    # NOTE: leaving commented for python 3.10 compatibility
    #  prefix_after: NotRequired[str]


class EditCommand(TypedDict):
    instruction: Optional[str]
    content: str


class EditResponse(TypedDict):
    instruction: Optional[str]
    content: str


# (old, instr, new) -> prompt
PromptFormatFunction = Callable[[str, str, str], str]

# (old, instr) -> [messages]
MessagesFormatFunction = Callable[[str, str], List[Message]]

# (old, new) -> response
PostProcessFunction = Callable[[str, str], str]


CompletionEngine = Literal["vllm", "transformers"]


def starcoder_edit_prompt(old, instr, new):
    # starcoder tokens
    OLD_CODE_TOKEN = "<commit_before>"
    REFLECTION_TOKEN = "<commit_msg>"
    NEW_CODE_TOKEN = "<commit_after>"
    return OLD_CODE_TOKEN + old + REFLECTION_TOKEN + instr + NEW_CODE_TOKEN + new


def direct_edit_prompt(
    old,
    instr,
    new,
    codeblock_before: Optional[str] = None,
    codeblock_after: Optional[str] = None,
):
    """
    The codeblock_before and codeblock_after arguments are used to specify
    if there should be a codeblock surrounding the code before and after
    the instruction. If None, then no codeblock is used. The string is the
    extension of the codeblock, e.g. "py" or "md".

    The format of the prompt is:
    ## Code Before:
    {old}
    ## Instruction:
    {instr}
    ## Code After:
    {new}

    if the parameter new is None, then the model will generate new code after the ## Code After: line.
    """
    if codeblock_before is not None:
        old = f"```{codeblock_before}\n{old}\n```"
    if codeblock_after is not None:
        new = f"```{codeblock_after}\n{new}\n```"
    before = f"""## Code Before:\n{old}\n"""
    instr = f"""## Instruction:\n{instr}\n"""
    after = f"""## Code After:\n{new}"""
    return before + instr + after


def direct_edit_prompt_sys(
    old,
    instr,
    new,
    codeblock_before: Optional[str] = None,
    codeblock_after: Optional[str] = None,
):
    """
    The codeblock_before and codeblock_after arguments are used to specify
    if there should be a codeblock surrounding the code before and after
    the instruction. If None, then no codeblock is used. The string is the
    extension of the codeblock, e.g. "py" or "md".

    The format of the prompt is:
    ## Code Before:
    {old}
    ## Instruction:
    {instr}
    ## Code After:
    {new}

    if the parameter new is None, then the model will generate new code after the ## Code After: line.
    """
    if codeblock_before is not None:
        old = f"```{codeblock_before}\n{old}\n```"
    if codeblock_after is not None:
        new = f"```{codeblock_after}\n{new}\n```"
    before = f"""You are a code editor. You will be provided the original code snippet and an instruction that specifies " \
"the changes you need to make. You will produce the changed code, based on the original code and the instruction given. " \
"Only produce the code, do not include any additional prose.
## Code Before:\n{old}\n"""
    instr = f"""## Instruction:\n{instr}\n"""
    after = f"""## Code After:\n{new}"""
    return before + instr + after


def openai_edit_prompt_1shot(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.
             """.strip(),
        },
        {
            "role": "user",
            "content": """
## Code Before
```py
def add(a, b):
    return a + b
```

## Instruction
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
""".strip(),
        },
        {
            "role": "assistant",
            "content": """
## Code After
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```
         """.strip(),
        },
        {
            "role": "user",
            "content": f"""
## Code Before
```py
{old}
```
## Instruction
{instr}
""".strip(),
        },
    ]


def openai_edit_prompt(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are a code editor. You will be provided the original code snippet and an instruction that specifies \
the changes you need to make. You will produce the changed code, based on the original code and the instruction given. \
Only produce the code, do not include any additional prose."
             """.strip(),
        },
        {
            "role": "user",
            "content": f"""
## Code Before
```python
{old}
```
## Instruction
{instr}

## Code After:
""".strip(),
        },
    ]


def diff_edit_prompt_1shot(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are Python program editor. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the change diff in unified diff form, based on the original code and the instruction given. Only produce the code diff in unified diff form, do not produce the full code, do not include any additional prose.
             """.strip(),
        },
        {
            "role": "user",
            "content": f"""
Please produce the code diff in section ## Code Diff

Example:
## Code Before
```py
def add(a, b):
    return a + b
```
## Instruction
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
## Code Diff
```diff
--- 
+++ 
@@ -1,2 +1,7 @@
-def add(a, b):
-    return a + b
+def add(x, y):
+    \"\"\"Adds two numbers.\"\"\"
+    return x + y
+
+def sub(x, y):
+    \"\"\"Subtracts two numbers.\"\"\"
+    return x - y
```

Your Task:
## Code Before
```py
{old}
```
## Instruction
{instr}
## Code Diff
""".strip(),
        },
    ]


def direct_edit_prompt_1shot(
    old,
    instr,
    new,
):
    p = direct_edit_prompt(old, instr, new)
    shot = """## Code Before:
def add(a, b):
    return a + b
## Instruction:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
## Code After:
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y"""
    p = shot + "\n" + p
    return p


def starcoder2_edit_prompt_1shot(old: str, instr: str, _: str) -> str:
    return f"""<issue_start>username_0: I have a program in Python that I'd like to change.

Here is the code for the program:
```py
def add(a, b):
    return a + b
```

The change I'd like to make is:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.

Please someone help me. Can you also provide the full code with the change?<issue_comment>username_1: Sure, no problem. I will be able to help. I am an expert in editing Python code.

Here is the full code with the change:
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

    def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```
Upvotes: 200<issue_comment>username_0: Thank you so much! I have another program in Python that I'd like to change.

Here is the code for the program:
```py
{old}
```

The change I'd like to make is:
{instr}

Please someone help me. Can you also provide the full code with the change?
Upvotes: 100<issue_comment>username_1: Sure, no problem. I will be able to help. I am an expert in editing Python code.

Here is the full code with the change:
```py"""


def python_markdown_codeblock_extract(_: str, new: str) -> str:
    lines = new.split("\n")
    buf = ""
    in_codeblock = False
    for ln in lines:
        if ln.startswith("```"):
            if in_codeblock:
                break
            else:
                in_codeblock = True
        elif in_codeblock:
            buf += ln + "\n"
    return buf


def autodetect_dtype() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"


def vllm_get_tokenizer(model):
    tokenizer = model.get_tokenizer()
    # oh vLLM... how you have fallen..
    if tokenizer.__class__.__name__ == "TokenizerGroup":
        tokenizer = tokenizer.tokenizer

    return tokenizer


class TransformersVLLMAdapter:
    def __init__(self, model_name):
        dtype = "auto"
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            #  padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        use_tqdm: bool = False,
    ) -> List[RequestOutput]:
        # TODO: support heterogeneous prompts
        assert all(
            p == prompts[0] for p in prompts), "All prompts must be the same -- batched heterogeneous prompts not supported"
        new_tokens = sampling_params.max_tokens
        stop = sampling_params.stop
        with torch.no_grad():
            tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                #  padding=True,
                #  truncaton=True,
                max_length=self.model.config.max_position_embeddings - new_tokens - 2,
            ).to(self.model.device)
            outputs = self.model.generate(
                **tokens,
                max_new_tokens=new_tokens,
                do_sample=True,
                top_p=sampling_params.top_p,
                temperature=sampling_params.temperature,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            #  decoded: List[str] = self.tokenizer.batch_decode(
            #  outputs,
            #  skip_special_tokens=True
            #  )
            decoded = [""] * len(prompts)
            for i, (out, prompt) in enumerate(zip(outputs, tokens["input_ids"])):
                out = out[len(prompt):]
                d: str = self.tokenizer.decode(
                    out, skip_special_tokens=True
                )
                assert isinstance(d, str)
                if stop is not None:
                    for s in stop:
                        found = d.find(s)
                        if found != -1:
                            d = d[:found]
                decoded[i] = d

        decoded_vllm = [RequestOutput(
            request_id="",
            prompt=prompt,
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[CompletionOutput(
                index=0,
                text=dec,
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=None,
            )],
            finished=True
        ) for (prompt, dec) in zip(prompts, decoded)]

        return decoded_vllm

    def get_tokenizer(self):
        return self.tokenizer


class ChatModel:
    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        raise NotImplementedError


def chatmodel_factory(model_type, model_name, num_gpus):
    if model_type == "llama":
        return HFChatModel(model_name, num_gpus)
    else:
        raise ValueError(f"Unknown chat model type {model_type}")


class EditModel:
    def __init__(self, before_content_tok=None, instruction_tok=None, after_content_tok=None):
        self.before_content_tok = before_content_tok
        self.instruction_tok = instruction_tok
        self.after_content_tok = after_content_tok

    def edit_model_generate(
        self,
        model: Union[LLM, TransformersVLLMAdapter],
        str_prompts: List[str], **kwargs
    ) -> List[RequestOutput]:
        kwargs_gen = kwargs.copy()
        if "declaration" in kwargs_gen:
            del kwargs_gen["declaration"]
        use_tqdm = kwargs_gen.pop("use_tqdm", False)
        kwargs_gen.pop("no_think", None)  # no_think is not supported in edit models
        kwargs_gen.pop("enforce_thinking", None)  # enforce_thinking is not supported in edit models
        gens = model.generate(
            prompts=str_prompts,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                **kwargs_gen,
            ),
            use_tqdm=use_tqdm,
        )
        return gens

    def get_before_content_tok(self) -> Optional[str]:
        return self.before_content_tok

    def get_instruction_tok(self) -> Optional[str]:
        return self.instruction_tok

    def get_after_content_tok(self) -> Optional[str]:
        return self.after_content_tok

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        raise NotImplementedError

    def bugfix_instr(self, prompt) -> Optional[str]:
        return None

    def get_prompt_format(self):
        raise NotImplementedError

    def get_tokenizer(self):
        raise NotImplementedError


def editmodel_factory(model_type, model_name, num_gpus):
    if model_type == "starcoder":
        return StarCoderCommitEditModel(model_name, num_gpus)
    else:
        raise ValueError(f"Unknown edit model type {model_type}")


class StarCoderCommitEditModel(EditModel):
    def __init__(
        self,
        model_name="bigcode/starcoderbase",
        num_gpus=1,
        before_content_tok="<commit_before>",
        instruction_tok="<commit_msg>",
        after_content_tok="<commit_after>",
    ):
        super().__init__(before_content_tok, instruction_tok, after_content_tok)
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype(),
            tensor_parallel_size=num_gpus,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)
        self.instruction_tok_id = self.tokenizer.encode(instruction_tok)[0]
        self.after_content_tok_id = self.tokenizer.encode(after_content_tok)[0]
        self.after_content_tok = after_content_tok

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        str_prompts = []

        for prompt in prompts:
            content = (
                ("\n" + prompt["content"] +
                 "\n") if prompt["content"] != "" else ""
            )
            if prompt["instruction"] is not None:
                str_prompt = f"{self.before_content_tok}{content}{self.instruction_tok}\n{prompt['instruction']}\n{self.after_content_tok}"
                if "declaration" in kwargs:
                    str_prompt += f"\n{kwargs['declaration']}"
            else:
                str_prompt = f"{self.before_content_tok}{content}{self.instruction_tok}"

            str_prompts.append(str_prompt)

        # generate
        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        # TODO: double check this
        kwargs["stop"] = stop + [self.after_content_tok]
        gens = self.edit_model_generate(self.model, str_prompts, **kwargs)

        responses = []

        for prompt, gen in zip(prompts, gens):
            out = gen.outputs[0].token_ids

            resp = {"content": "", "instruction": None}
            # if we had an instruction, we are all good.
            # or, it could be that the model didn't generate anything useful
            if (
                prompt["instruction"] is not None
                or self.after_content_tok_id not in out
            ):
                resp["content"] = self.tokenizer.decode(
                    out, skip_special_tokens=True)
                responses.append(resp)
                continue

            # otherwise, find the end of the instruction
            new_content_idx = out.index(self.after_content_tok_id)
            resp["instruction"] = self.tokenizer.decode(
                out[:new_content_idx], skip_special_tokens=True
            )
            # and decode the content
            resp["content"] = self.tokenizer.decode(
                out[new_content_idx + 1:], skip_special_tokens=True
            )
            responses.append(resp)

        return responses

    def get_prompt_format(self):
        return starcoder_edit_prompt


def init_completion_engine(engine: CompletionEngine, **kwargs):
    if engine == "vllm":
        extra_kwargs = {}
        if "max_model_len" in kwargs:
            extra_kwargs["max_model_len"] = kwargs["max_model_len"]
        return LLM(
            kwargs["model_name"],
            dtype=autodetect_dtype(),
            tensor_parallel_size=kwargs["num_gpus"],
            gpu_memory_utilization=kwargs["gpu_util"],
            **extra_kwargs,
        )
    elif engine == "transformers":
        return TransformersVLLMAdapter(kwargs["model_name"])
    else:
        raise ValueError(f"Unknown completion engine {engine}")


class DirectEditModel(EditModel):
    """
    The direct kind of edit model, this class is supposed to be used either with EditCoder or
    with non-chat models, like foundation models.
    """

    def __init__(
        self,
        model_name="codellama/CodeLlama-34b-hf",
        num_gpus=1,
        gpu_util=0.95,
        prompt_format: PromptFormatFunction = direct_edit_prompt,
        post_process: PostProcessFunction = lambda old, new: new,
        completion_engine: CompletionEngine = "vllm",
        stop_tokens: List[str] = ["## Code After:",
                                  "## Instruction:", "## Code Before:"],
        max_model_len=None,
    ):
        super().__init__()
        self.model = init_completion_engine(
            completion_engine,
            model_name=model_name,
            num_gpus=num_gpus,
            gpu_util=gpu_util,
            max_model_len=max_model_len,
        )
        self.prompt_format = prompt_format
        self.post_process = post_process
        self.stop_tokens = stop_tokens

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        str_prompts = []

        for prompt in prompts:
            declaration = kwargs["declaration"] if "declaration" in kwargs else ""
            assert prompt["instruction"] is not None, "Not implemented yet"
            str_prompts.append(
                self.prompt_format(
                    prompt["content"], prompt["instruction"], declaration
                )
            )

        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        kwargs["stop"] = stop + self.stop_tokens

        # generate
        gens = self.edit_model_generate(self.model, str_prompts, **kwargs)

        responses = []
        for prompt, gen in zip(prompts, gens):
            out = gen.outputs[0].text
            try:
                processed = self.post_process(prompt["content"], out)
            except Exception as e:
                # print full stack trace
                import traceback
                traceback.print_exc()
                print("Error in post processing:", e)
                processed = out
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses

    def get_prompt_format(self):
        return self.prompt_format

    def get_tokenizer(self):
        return self.model.get_tokenizer()


class OctoCoderChatModel(ChatModel):
    def __init__(
        self,
        model_name="bigcode/octocoder",
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)

    def fmt_msg(self, message: List[Message]) -> str:
        """
        将输入的消息列表格式化为适合 OctoCoder 处理的字符串形式。
        """
        fmt = []
        start = 0
        system = ""

        # check for system prompt
        # 如果第一条消息是 system 角色，则将其内容存储到 system 变量，
        #   并从索引 1 处开始处理用户消息。
        if message[0]["role"] == "system":
            start = 1
            system = message[0]["content"]

        for i in range(start, len(message)):
            current = message[i]

            # 确保每条消息都有 role 和 content，防止空消息导致崩溃。
            assert current["content"] is not None, "Content of a message cannot be null"
            assert current["role"] is not None, "Role of a message cannot be null"
            
            # 区分 user 和 assistant 角色：
            #    如果是 user 角色，格式化为 "Question: {system}\n{用户内容}"；
            #    如果是 assistant 角色，格式化为 "Answer: {助手的回答}"。
            if current["role"] == "user":
                # if question, then add system prompt
                fmt.append(f"Question: {system}\n{current['content']}")
                # if last message and is a question, add an answer to it
                if i == len(message) - 1:
                    fmt.append(f"Answer:")
            else:
                # if answer, then no system prompt
                fmt.append(f"Answer: {current['content']}")

        return "\n\n".join(fmt)

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        kwargs_gen = kwargs.copy()

        msgs = [self.fmt_msg(msg) for msg in messages]  # 将输入的消息列表格式化为适合 OctoCoder 处理的字符串形式。

        stop = kwargs_gen.pop("stop", [])
        stop.append("\n\nAnswer:")  # 避免模型生成多个回答部分。
        stop.append("\n\nQuestion:")  # 防止生成无关的新问题。
        # stop.append("\n\n")

        gens = self.model.generate(
            prompts=msgs,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                stop=list(set(stop)),
                **kwargs_gen,
            ),
            use_tqdm=True,
        )

        responses = []

        # 提取生成的回答部分
        for gen in gens:
            out = gen.outputs[0].text
            responses.append(out)

        return responses
    

class QwenCoderChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)

    def fmt_msg(self, message: List[Message]) -> str:
        """
        将输入的消息列表格式化为适合 OctoCoder 处理的字符串形式。
        """
        fmt = []
        start = 0
        system = ""

        # check for system prompt
        # 如果第一条消息是 system 角色，则将其内容存储到 system 变量，
        #   并从索引 1 处开始处理用户消息。
        if message[0]["role"] == "system":
            start = 1
            system = message[0]["content"]

        for i in range(start, len(message)):
            current = message[i]

            # 确保每条消息都有 role 和 content，防止空消息导致崩溃。
            assert current["content"] is not None, "Content of a message cannot be null"
            assert current["role"] is not None, "Role of a message cannot be null"
            
            # 区分 user 和 assistant 角色：
            #    如果是 user 角色，格式化为 "<|im_start|>user\n{用户内容}<|im_end|>\n\n<|im_start|>assistant"；
            #    如果是 assistant 角色，格式化为 "{助手回答}<|im_end|>"。
            if current["role"] == "user":
                # if question, then add system prompt
                fmt.append(f"<|im_start|>system\n{system}<|im_end|>\n\n")
                fmt.append(f"<|im_start|>user\n{current['content']}<|im_end|>\n\n<|im_start|>assistant")
            else:
                # if answer, then no system prompt
                fmt.append(f"{current['content']}<|im_end|>")

        return "".join(fmt)

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        kwargs_gen = kwargs.copy()

        msgs = [self.fmt_msg(msg) for msg in messages]  # 将输入的消息列表格式化为适合 QwenCoder 处理的字符串形式。

        stop = kwargs_gen.pop("stop", [])
        stop.append("<|im_end|>")  # 停用词
        # stop.append("<|im_start|>")  # 避免生成其他额外内容

        kwargs_gen.pop("no_think", False)  # no_think is not supported in QwenCoder
        kwargs_gen.pop("enforce_thinking", False)  # enforce_thinking is not supported in QwenCoder

        gens = self.model.generate(
            prompts=msgs,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                stop=list(set(stop)),
                **kwargs_gen,
            ),
            use_tqdm=True,
        )

        responses = []

        # 提取生成的回答部分
        for gen in gens:
            out = gen.outputs[0].text
            responses.append(out)

        return responses
    

class Qwen3ChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)

    def fmt_msg(self, message: List[Message], no_think=False, enforce_thinking=False) -> str:
        """
        将输入的消息列表格式化为适合 OctoCoder 处理的字符串形式。
        """
        fmt = []
        start = 0
        system = ""

        # set the think tag based on the parameters
        think_tag = ""
        if enforce_thinking:
            think_tag = " /think"
        elif no_think:
            think_tag = " /no_think"

        # check for system prompt
        # 如果第一条消息是 system 角色，则将其内容存储到 system 变量，
        #   并从索引 1 处开始处理用户消息。
        if message[0]["role"] == "system":
            start = 1
            system = message[0]["content"]

        for i in range(start, len(message)):
            current = message[i]

            # 确保每条消息都有 role 和 content，防止空消息导致崩溃。
            assert current["content"] is not None, "Content of a message cannot be null"
            assert current["role"] is not None, "Role of a message cannot be null"
            
            # 区分 user 和 assistant 角色：
            #    如果是 user 角色，格式化为 "<|im_start|>user\n{用户内容}<|im_end|>\n\n<|im_start|>assistant"；
            #    如果是 assistant 角色，格式化为 "{助手回答}<|im_end|>"。
            if current["role"] == "user":
                # if question, then add system prompt
                fmt.append(f"<|im_start|>system\n{system}<|im_end|>\n\n")
                fmt.append(f"<|im_start|>user\n{current['content']}{think_tag}<|im_end|>\n\n<|im_start|>assistant")
            else:
                # if answer, then no system prompt
                fmt.append(f"{current['content']}<|im_end|>")

        return "".join(fmt)

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        kwargs_gen = kwargs.copy()

        msgs = [self.fmt_msg(msg, no_think=kwargs_gen.pop("no_think", False), enforce_thinking=kwargs_gen.pop("enforce_thinking", False)) 
                for msg in messages]  # 将输入的消息列表格式化为适合 QwenCoder 处理的字符串形式。

        stop = kwargs_gen.pop("stop", [])
        stop.append("<|im_end|>")  # 停用词
        # stop.append("<|im_start|>")  # 避免生成其他额外内容

        gens = self.model.generate(
            prompts=msgs,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                stop=list(set(stop)),
                **kwargs_gen,
            ),
            use_tqdm=True,
        )

        responses = []

        # 提取生成的回答部分
        for gen in gens:
            out = gen.outputs[0].text
            responses.append(out)

        return responses
    

# class Qwen3ChatModel(ChatModel):
#     def __init__(
#         self,
#         model_name,
#         num_gpus=1,
#         gpu_util=0.95,
#         quantization=False,
#     ):
#         self.model = LLM(
#             model_name,
#             dtype=autodetect_dtype() if not quantization else "float16",
#             tensor_parallel_size=num_gpus,
#             gpu_memory_utilization=gpu_util,
#             quantization="awq" if quantization else None,
#         )
#         self.tokenizer = vllm_get_tokenizer(self.model)

#     def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
#         kwargs_gen = kwargs.copy()

#         stop = kwargs_gen.pop("stop", [])
#         stop.append("<|im_end|>")  # 停用词
#         # stop.append("<|im_start|>")  # 避免生成其他额外内容

#         gens = self.model.chat(
#             messages=messages,
#             sampling_params=SamplingParams(
#                 top_p=kwargs_gen.pop("top_p", 0.95),
#                 temperature=kwargs_gen.pop("temperature", 0.2),
#                 max_tokens=kwargs_gen.pop("max_tokens", 1024),
#                 stop=list(set(stop)),
#                 **kwargs_gen,
#             ),
#             use_tqdm=True,
#             # chat_template_kwargs={"enable_thinking": kwargs_gen.pop("enable_thinking", False)},
#         )

#         responses = []

#         # 提取生成的回答部分
#         for gen in gens:
#             out = gen.outputs[0].text
#             responses.append(out)

#         return responses


class DeepSeekChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)

    def fmt_msg(self, message: List[Message]) -> str:
        """
        将输入的消息列表格式化为适合 OctoCoder 处理的字符串形式。
        """
        fmt = []
        start = 0
        system = ""

        # check for system prompt
        # 如果第一条消息是 system 角色，则将其内容存储到 system 变量，
        #   并从索引 1 处开始处理用户消息。
        if message[0]["role"] == "system":
            start = 1
            system = message[0]["content"]

        for i in range(start, len(message)):
            current = message[i]

            # 确保每条消息都有 role 和 content，防止空消息导致崩溃。
            assert current["content"] is not None, "Content of a message cannot be null"
            assert current["role"] is not None, "Role of a message cannot be null"
            
            # 区分 user 和 assistant 角色：
            #    如果是 user 角色，格式化为 "<|im_start|>user\n{用户内容}<|im_end|>\n\n<|im_start|>assistant"；
            #    如果是 assistant 角色，格式化为 "{助手回答}<|im_end|>"。
            if current["role"] == "user":
                # if question, then add system prompt
                fmt.append(f"{system}\n\n")
                fmt.append(f"User: {current['content']}\n\nAssistant:")
            else:
                # if answer, then no system prompt
                fmt.append(f"{current['content']}\n\n")

        return "".join(fmt)

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        kwargs_gen = kwargs.copy()

        msgs = [self.fmt_msg(msg) for msg in messages]  # 将输入的消息列表格式化为适合 DeepSeek 处理的字符串形式。

        stop = kwargs_gen.pop("stop", [])
        stop.append("<|im_end|>")  # 停用词
        # stop.append("<|im_start|>")  # 避免生成其他额外内容

        kwargs_gen.pop("no_think", False)  # no_think is not supported in DeepSeek
        kwargs_gen.pop("enforce_thinking", False)  # enforce_thinking is not supported in DeepSeek

        gens = self.model.generate(
            prompts=msgs,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                stop=list(set(stop)),
                **kwargs_gen,
            ),
            use_tqdm=True,
        )

        responses = []

        # 提取生成的回答部分
        for gen in gens:
            out = gen.outputs[0].text
            responses.append(out)

        return responses


class HFChatModel(ChatModel):
    """
    用于与 Hugging Face 的聊天模型（如 CodeLlama）进行交互，通过 vllm 库进行交互。
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(
        self,
        model_name="codellama/CodeLlama-34b-Instruct-hf",
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
        system_supported=True,
        max_model_len=None,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_model_len,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)
        self.system_supported = system_supported

    def llama2_chat_generate(
        self, messages: List[List[Message]], **kwargs
    ) -> List[str]:
        def tokenize_messages(ms) -> List[int]:
            """
            将输入的聊天消息转换为 token（标记）序列，以供模型处理。
            它会进行消息格式化、分词、截断等处理，确保输入符合模型的要求。
            """
            ms = ms.copy()
            if not self.system_supported and ms[0]["role"] == "system":
                sys_m = ms[0]["content"]
                ms = ms[1:]
                for m in ms:
                    if m["role"] == "user":
                        m["content"] = sys_m + "\n" + m["content"]

            toks = self.tokenizer.apply_chat_template(
                ms,  # type: ignore
                tokenize=True,  # 进行分词（tokenize=True），将文本转换为 token ID。
                truncation=True,
                add_generation_prompt=True,
                max_length=16384 - max_new_tokens - 2,
            )
            assert isinstance(toks, list)
            if "prefix_after" in ms[-1]:
                toks.extend(
                    self.tokenizer.encode(
                        ms[-1]["prefix_after"], add_special_tokens=False
                    )
                )

            return toks

        kwargs = kwargs.copy()
        max_new_tokens = kwargs.pop("max_tokens", 256)
        prompts = [tokenize_messages(ms) for ms in messages]  # 将输入的聊天消息转换为 token（标记）序列，以供模型处理。

        discard_lengthy = kwargs.pop("discard_lengthy", False)
        use_tqdm = kwargs.pop("use_tqdm", False)
        stop = kwargs.pop("stop", [])
        stop.append(self.E_INST)
        params = SamplingParams(
            top_p=kwargs.pop("top_p", 0.9),
            temperature=kwargs.pop("temperature", 0.75),
            max_tokens=max_new_tokens,
            stop=list(set(stop)),  # 将 stop 列表中的重复元素去除，并将其转换回列表形式
            **kwargs,
        )
        gens = self.model.generate(
            prompt_token_ids=prompts,
            sampling_params=params,
            use_tqdm=use_tqdm,
        )
        decoded = []
        for ms, gen in zip(messages, gens):
            outs = gen.outputs[0]
            if discard_lengthy and outs.finish_reason == "length":
                # outs.finish_reason 表示生成停止的原因：
                #    "length"：表示输出达到了 max_tokens 限制，被强行截断。
                #    其他可能的值："stop"（遇到 stop 词终止）等。
                # 该处判断的意义是，如果 discard_lengthy=True，则丢弃因长度限制被截断的输出，避免返回不完整的回复。
                continue

            toks = outs.token_ids  # 获取生成的 token 序列。

            # 将 token 序列解码为文本。
            # skip_special_tokens=True：跳过特殊 token（如 [INST]、</s>）。
            dec = self.tokenizer.decode(toks, skip_special_tokens=True)  

            if "prefix_after" in ms[-1]:
                # prefix_after 是输入消息的一个额外前缀，如果存在，就拼接到解码后的文本前面。
                dec = ms[-1]["prefix_after"] + dec

            # 处理 stop 词
            #     模型在生成时可能因为采样、解码等原因，生成了 stop 词后依然产生了一些多余的文本。
            #     通过后处理，可以确保截取 stop 词前的内容，避免返回不必要的文本。
            # 遍历 stop 词列表，检查生成文本中是否出现这些终止标志：
            #     如果找到了某个 stop 词，则 截取 stop 词前的内容，去掉后面的不必要部分。
            for s in stop:
                found = dec.find(s)
                if found != -1:
                    dec = dec[:found]

            stripped = dec.strip()  # 去除文本前后的空白字符
            decoded.append(stripped)
        return decoded

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        # make sure we have a list of lists
        assert isinstance(messages, list), "messages must be a list of lists"
        assert len(messages) > 0, "messages must have at least one list"
        assert isinstance(
            messages[0], list), "messages must be a list of lists"
        return self.llama2_chat_generate(messages, **kwargs)


class OpenAIChatModel(ChatModel):
    """
    用于与OpenAI 的 GPT 模型（如 GPT-4）进行交互，通过 OpenAI 的 API 进行交互。
    """
    def __init__(
        self,
        model_name="gpt-4",
        endpoint=None,
    ):
        import os
        import openai

        # 设置 OpenAI API 密钥
        if "ORG_ID" in os.environ:
            openai.organization = os.getenv("ORG_ID")
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY must be set"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        if endpoint is not None:
            # TODO: fix this
            #  openai.api_base = endpoint
            pass

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        # make sure we have a list of lists
        assert isinstance(messages, list), "messages must be a list of lists"
        assert len(messages) > 0, "messages must have at least one list"
        assert isinstance(
            messages[0], list), "messages must be a list of lists"
        # check that all messages are the same.
        # TODO: support heterogeneous prompts
        assert all(
            m == messages[0] for m in messages), "All prompts must be the same -- batched heterogeneous prompts not supported"
        message = messages[0]

        while True:
            _kwargs = kwargs.copy()
            discard_lengthy = _kwargs.pop("discard_lengthy", False)  # 是否丢弃因长度限制被截断的回复，如未设置则默认为 False。

            # 调用 OpenAI API 生成回复
            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=message,  # type: ignore
                    n=len(messages),
                    stop=_kwargs.pop("stop", None),
                    temperature=_kwargs.pop("temperature", 0.75),
                    top_p=_kwargs.pop("top_p", 0.9),
                    max_tokens=_kwargs.pop("max_tokens", 256),
                    **_kwargs,
                )
            except openai.RateLimitError as e:
                # 处理 API 限流
                print("Rate limit error. Waiting two minutes:", e)
                time.sleep(120)
                continue

            break
        outs = []

        # 遍历生成的回复，提取文本
        for choice in response.choices:
            if discard_lengthy and choice.finish_reason == "length":
                # 如果 discard_lengthy=True，则丢弃因长度限制被截断的回复，避免返回不完整的回复。
                continue
            text = choice.message.content
            outs.append(text.strip())

        return outs


class ChatAdaptorEditModel(EditModel):
    """
    This is an adaptor class to use ChatModels as EditModels.
    NOTE: This model class is only intended for inference, not training.
    """

    # TODO: implement whole shebang for bugfix

    def __init__(
        self,
        chat_model: ChatModel,
        prompt_format: MessagesFormatFunction = openai_edit_prompt_1shot,
        post_process: PostProcessFunction = python_markdown_codeblock_extract,
    ):
        super().__init__()
        self.model = chat_model
        self.prompt_format = prompt_format
        self.post_process = post_process

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        kwargs = kwargs.copy()
        # TODO: can do something with declaration here
        kwargs.pop("declaration", None)  # ChatModel 不支持该参数

        kwargs.pop("use_tqdm", None)  # ChatModel 不使用该参数

        chat_prompts = []
        for prompt in prompts:
            # 确保每个 EditCommand 都包含 "instruction"（否则抛出异常）
            assert (
                prompt["instruction"] is not None
            ), "Every command must have an instruction in ChatAdaptorEditModel"

            # 将 EditCommand 转换成适用于 ChatModel 的输入格式，并存入 chat_prompts
            chat_prompts.append(
                self.prompt_format(prompt["content"], prompt["instruction"])
            )

        # generate
        gens = self.model.generate(chat_prompts, **kwargs)

        responses = []
        for prompt, gen in zip(prompts, gens):
            processed = self.post_process(prompt["content"], gen)
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses


# NOTE: this is the factory for each model type. to add a new model type, add a new case here
# and implement it in models.py. Also, add a new case in the argument parser below.
def model_factory(
        model_type: str,
        quantize=False,
        num_gpus=1,
        system_supported=True,
        completion_engine: CompletionEngine = "vllm",
        max_model_len=None,
) -> Callable[[str], EditModel]:
    if model_type == "direct":
        return (lambda path: DirectEditModel(
            path,
            completion_engine=completion_engine,
            num_gpus=num_gpus,
            max_model_len=max_model_len,
        ))
    elif model_type == "direct-1shot":
        return (lambda path: DirectEditModel(
            path,
            completion_engine=completion_engine,
            num_gpus=num_gpus,
            max_model_len=max_model_len,
            prompt_format=direct_edit_prompt_1shot,
        ))
    elif model_type == "direct-sys":
        return (lambda path: DirectEditModel(
            path,
            completion_engine=completion_engine,
            num_gpus=num_gpus,
            max_model_len=max_model_len,
            prompt_format=direct_edit_prompt_sys,
        ))
    elif model_type == "starcoder2":
        return (lambda path: DirectEditModel(
            path,
            completion_engine=completion_engine,
            num_gpus=num_gpus,
            max_model_len=max_model_len,
            prompt_format=starcoder2_edit_prompt_1shot,
            # TODO: fix the hack below
            post_process=(
                lambda x, y: python_markdown_codeblock_extract(x, "```py\n" + y)),
        ))
    elif model_type == "starcoder":
        return StarCoderCommitEditModel
    elif model_type == "openai":
        return (lambda path: ChatAdaptorEditModel(OpenAIChatModel(path)))
    elif model_type == "chat":
        return (lambda path: ChatAdaptorEditModel(HFChatModel(
            path,
            quantization=quantize,
            num_gpus=num_gpus,
            system_supported=system_supported,
        )))
    elif model_type == "unified_diff":
        return (lambda path: ChatAdaptorEditModel(HFChatModel(
            path,
            quantization=quantize,
            num_gpus=num_gpus,
            system_supported=system_supported,
        ), 
        prompt_format=diff_edit_prompt_1shot))
    elif model_type == "octocoder":
        return (lambda path: ChatAdaptorEditModel(OctoCoderChatModel(
            path,
            quantization=quantize,
            num_gpus=num_gpus,
        )))
    elif model_type == "qwencoder":
        return (lambda path: ChatAdaptorEditModel(QwenCoderChatModel(
            path,
            quantization=quantize,
            num_gpus=num_gpus,
        ),
        prompt_format=openai_edit_prompt))
    elif model_type == "qwen3":
        return (lambda path: ChatAdaptorEditModel(Qwen3ChatModel(
            path,
            quantization=quantize,
            num_gpus=num_gpus,
        ),
        prompt_format=openai_edit_prompt))
    elif model_type == "deepseek":
        return (lambda path: ChatAdaptorEditModel(DeepSeekChatModel(
            path,
            quantization=quantize,
            num_gpus=num_gpus,
        ),
        prompt_format=openai_edit_prompt))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def complete_problem(example: EditCommand, model: EditModel, batch_size: int, completion_limit: int, **kwargs) -> List[str]:
    batches = batch_prompts_from_example(example, batch_size, completion_limit)

    completions = []
    for batch in batches:
        resps = model.generate(batch, **kwargs)
        for resp in resps:
            completions.append(resp["content"])

    return completions


def main(args):
    dataset = datasets.load_dataset(
        args.dataset, args.subset, split=args.split)
    model = model_factory(
        args.model_type,
        quantize=args.quantize,
        num_gpus=args.num_gpus,
        system_supported=not args.no_system,
        completion_engine=args.completion_engine,
        max_model_len=args.max_model_len,
    )(args.model)
    model_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "no_think": args.no_think,  # for qwen3 model
        "enforce_thinking": args.enforce_thinking,  # for qwen3 model
    }

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    # writing in this format such that we can use the MultiPL-E evaluation container :)
    # 使用 tqdm 库在遍历数据集时显示进度条
    for ex in tqdm(dataset, total=len(dataset)):  # type: ignore
        # ex 是 dataset 中的每一个元素。
        assert isinstance(ex, dict)

        if args.humanevalpack:
            instr_kinds = ['instruction']
        else:
            instr_kinds = ['instruction_descriptive', 'instruction_lazy']

        for instr_kind in instr_kinds:
            # 对每一种指令类型进行下面的操作：
            path = Path(args.output_dir) / \
                (f"{ex['full_name']}_{instr_kind}.json.gz")
            if path.exists():
                continue  # this pretty much resumes from where it left off

            instr = ex[instr_kind]

            # Example for the input of EditCommand:

            # example = EditCommand(
            #     instruction="Convert the function to use async/await syntax",
            #     content="def foo():\n    return 42"
            # )
            #
            # instruction：是一个指令，例如 "将函数改为异步语法"。
            # content：是目标内容，例如待编辑的 Python 函数代码。

            example = EditCommand(
                instruction=instr,
                content=ex["before"],
            )

            if "declaration" in ex:
                model_kwargs["declaration"] = ex["declaration"]

            completions = complete_problem(
                example,
                model,
                args.batch_size,
                args.completion_limit,
                **model_kwargs,
            )

            # copy over the example
            result = {}
            for k in ex:
                result[k] = ex[k]

            result["instr_kind"] = instr_kind
            # this is for compatibility with the MultiPL-E evaluator
            result["prompt"] = ex["declaration"] if "declaration" in ex else ""
            result["completions"] = completions
            result["language"] = "py"
            result["temperature"] = args.temperature
            result["top_p"] = args.top_p
            result["max_tokens"] = args.max_tokens
            result["stop_tokens"] = []
            result["script_args"] = args.__dict__.copy()

            gunzip_json_write(path, result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="nuprl/CanItEdit", help="dataset to use")
    parser.add_argument("--split", type=str, default="test",
                        help="split of the dataset to use")
    parser.add_argument("--subset", type=str, default=None,
                        help="subset of the split to use")
    parser.add_argument(
        "--model-type",
        type=str,
        default="direct",
        choices=["direct", "direct-1shot", "unified_diff", "direct-sys",
                 "openai", "chat", "octocoder", "starcoder", "starcoder2",
                 "qwencoder", "qwen3", "deepseek"],
        help="type of model to use for completions",
    )
    parser.add_argument(
        "--completion-engine",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
    )
    parser.add_argument("--model", type=str, required=True,
                        help="path to model or hub name")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="output directory for completions")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size for completions")
    parser.add_argument("--completion-limit", type=int,
                        default=20, help="number of completions per prompt")
    parser.add_argument("--temperature", type=float,
                        default=0.2, help="sampling temperature")
    parser.add_argument("--quantize", action="store_true",
                        help="quantize the model with AWQ")
    parser.add_argument("--humanevalpack", action="store_true",
                        help="run humanevalpack instead of CanItEdit")
    parser.add_argument("--top-p", type=float,
                        default=0.95, help="top-p sampling")
    parser.add_argument("--max-tokens", type=int,
                        default=2048, help="max new tokens to generate per completion. 2048 works for CanItEdit")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="max model length for batching with vLLM. only change if getting OOM")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="number of gpus for sharded model")
    parser.add_argument("--no-system", action="store_true",
                        help="disable system prompt for chat models")
    parser.add_argument("--no-think", action="store_true",
                        help="disable thinking step for qwen3 model")
    parser.add_argument("--enforce-thinking", action="store_true",
                        help="enforce thinking step for qwen3 model, even if no-think is set")
    args = parser.parse_args()
    main(args)

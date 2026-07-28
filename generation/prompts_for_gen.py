# prompt of v5.1
SYSTEM_PROMPT_V5_1 = """You are an experienced programmer who is skilled at creating high-quality program editing tasks and providing precise solutions."""
# First round: generate pre-edit code and edit instructions
USER_PROMPT_V5_1_ROUND_1 = """
Please gain inspiration from the following two code snippets and design a Python program. Then, create a task to edit the program. \
Please output the Python program first, and then output the editing task in both descriptive and lazy forms. \
Present your output in three distinct sections [Program Before Edit] [Descriptive] and [Lazy]. 

Requirements for the program: 

- The program should be completed, with all necessary library function imports.

Requirements for the program editing task: 

- The types of editing tasks can be **diverse**, such as *fixing errors*, *enhancing existing features*, *meeting new requirements*, and so on.
- Ensure that the task can be completed in a single file.

## Code Snippet 1:

{code_snippet_1}

## Code Snippet 2:

{code_snippet_2}

## Guidelines for each section:

1. [Program Before Edit]: A new program inspired by the two code snippets. The program can be faulty, as we can repair it in the editing task.
2. [Descriptive]: Offer a **detailed** instruction. This should be **completely self-contained**, providing all the contextual information \
one needs to understand and solve the task. Ensure that any specific context, variables, or code snippets pertinent to this problem \
are explicitly included, but the program after editing is not allowed to appear in the instruction.
3. [Lazy]: Offer a **simple but clear** instruction. You should describe the task in no more than three sentences, \
but the description should be clear to understand by humans.

**Here is an example for the program to be generated and the description of the editing task:**
[Program Before Edit]
```
{code_before_shot}
```

[Descriptive]
{desc_instr_shot}

[Lazy]
{lazy_instr_shot}

""".strip()

# Second round: generate the post-edit code
USER_PROMPT_V5_1_ROUND_2 = """
Is the program you designed in section [Program Before Edit] reasonable? \
If it is reasonable, Please provide the revised standard code based on the task you designed, in the section [Program After Edit], without any explanation;\
if it is unreasonable, please only output a mark <UNREASONABLE> without any code or explanation.
""".strip()


SYSTEM_PROMPT_V5_2 = """You are an experienced programmer who is skilled at creating high-quality program editing tasks and providing precise solutions."""
# First round: generate pre-edit code and edit instructions
USER_PROMPT_V5_2_ROUND_1 = """
Synthesize concepts or techniques from both code snippets to design a new, logically coherent Python program. \
Do not simply copy either snippet or mechanically concatenate them. Then, create a task to edit the program. \
Output the Python program first, followed by the editing task in both descriptive and lazy forms.

Requirements for the program:

- The program must be complete and self-contained, with all required imports.
- The program may contain one defect or one missing feature directly related to the editing task; otherwise, it must be coherent and free of unrelated defects.
- The program in [Program Before Edit] must not already satisfy the editing task.

Requirements for the program editing task:

- The types of editing tasks can be **diverse**, such as *fixing errors*, *enhancing existing features*, *meeting new requirements*, and so on.
- Ensure that the task can be completed in a single file.
- The [Descriptive] and [Lazy] sections must describe exactly the same editing task and differ only in their level of detail.

## Code Snippet 1:

{code_snippet_1}

## Code Snippet 2:

{code_snippet_2}

## Guidelines for each section:

1. [Program Before Edit]: Provide a new program inspired by the two code snippets. Enclose the complete Python code in a Markdown code fence.
2. [Descriptive]: Offer a **detailed** instruction. This should be **completely self-contained**, providing all the contextual information \
one needs to understand and solve the task. Ensure that any specific context, variables, or code snippets pertinent to this problem \
are explicitly included, but do not include the program after editing in the instruction.
3. [Lazy]: Offer a **simple but clear** instruction. You should describe the task in no more than three sentences, \
and the description must be easy for humans to understand.

Output exactly these three sections in this order. Use each section heading exactly once, and do not include any other headings, \
introductory text, explanations, or concluding text.

**Here is an example that follows the required output format:**
[Program Before Edit]
```
{code_before_shot}
```

[Descriptive]
{desc_instr_shot}

[Lazy]
{lazy_instr_shot}

""".strip()

# Second round: generate the post-edit code
USER_PROMPT_V5_2_ROUND_2 = """
Apply the single editing task described in both [Descriptive] and [Lazy] to the program in [Program Before Edit]. \
Make only the changes required by the editing task, and preserve all unrelated behavior and code.

Output exactly one section in the following format, with no other text:

[Program After Edit]
```python
<complete revised Python code>
```
""".strip()


# Commit rewriting prompt
SYSTEM_PROMPT_V5_9 = """You are an experienced programmer."""
USER_PROMPT_V5_9 = """
Please gain inspiration from the **code edit diff**, and rewrite the **commit message** into descriptive and lazy instructions. Present your output in two distinct sections [Descriptive] and [Lazy]. 

## Code Before Edit:
```
{code_before}
```

## Code Edit Diff
```diff
{code_diff}
```

## Commit Message

{commit_message}

## Guidelines for each section:

1. [Descriptive]: Offer a **detailed** instruction. This should be **completely self-contained**, providing all the contextual information \
one needs to understand and solve the task. Ensure that any specific context, variables, or code snippets pertinent to this problem \
are explicitly included, but the program after editing is not allowed to appear in the instruction.
2. [Lazy]: Offer a **simple but clear** instruction. You should describe the task in no more than three sentences, \
but the description should be clear to understand by humans.

Here is an example for the description of the editing task:

[Descriptive]
{desc_instr_shot}

[Lazy]
{lazy_instr_shot}

""".strip()


def get_prompts(version):
    if version == 'v5.1':
        system_prompt = SYSTEM_PROMPT_V5_1
        user_prompt_list = [USER_PROMPT_V5_1_ROUND_1, USER_PROMPT_V5_1_ROUND_2]
        return system_prompt, user_prompt_list
    elif version == 'v5.2':
        system_prompt = SYSTEM_PROMPT_V5_2
        user_prompt_list = [USER_PROMPT_V5_2_ROUND_1, USER_PROMPT_V5_2_ROUND_2]
        return system_prompt, user_prompt_list
    elif version == 'v5.9':
        system_prompt = SYSTEM_PROMPT_V5_9
        user_prompt_list = [USER_PROMPT_V5_9]
        return system_prompt, user_prompt_list
    else:
        raise ValueError("Unsupported version")

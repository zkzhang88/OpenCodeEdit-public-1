from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import re

def split_identifier(identifier):
    """
    Splits a given identifier into sub-tokens based on camelCase and snake_case conventions.
    This function handles identifiers that may be in camelCase (e.g., `myVariableName`)
    or snake_case (e.g., `my_variable_name`). It splits them into individual words.
    """
    # Split camelCase and snake_case identifiers
    parts = re.sub('([a-z])([A-Z])', r'\1 \2', identifier).split()
    sub_parts = []
    for part in parts:
        sub_parts.extend(part.split('_'))
    return [p for p in sub_parts if p]

def process_code_tokens(code_str):
    """
    Tokenizes a given Python code string and processes the tokens.
    This function uses a Python lexer to tokenize the input code string. For each token:
    - If the token is an identifier (such as a variable or function name), it is further split into sub-tokens using the `split_identifier` function.
    - All tokens are converted to lowercase.
    Args:
        code_str (str): The Python code as a string to be tokenized and processed.
    Returns:
        List[str]: A list of processed tokens, where identifiers are split into sub-tokens and all tokens are lowercase.
    """

    lexer = get_lexer_by_name("python", stripall=True)
    tokens = lex(code_str, lexer)

    result_tokens = []
    for token_type, token in tokens:
        token = token.strip()
        if token:
            if token_type in [Token.Name, Token.Name.Function, Token.Name.Variable]:
                result_tokens.extend([t.lower() for t in split_identifier(token)])
            else:
                result_tokens.append(token.lower())
    return result_tokens


def edit_instruction_splitter(instr: str, tokenize: bool = True) -> tuple:
    """
    Splits an input string containing code and instruction sections, tokenizes each part, and returns the tokens.
    Args:
        instr (str): The input string containing code and instruction sections, formatted with
            '## Code Before:', '## Instruction:', and '## Code After:' delimiters.
        tokenize (bool): If True, returns tokenized code and instruction; if False, returns raw strings.
    Returns:
        tuple: A tuple (code_tokens, instr_tokens) where:
            - code_tokens (list or str): Tokens from the code section if tokenize is True, else raw code string.
            - instr_tokens (list or str): Tokens from the instruction section if tokenize is True, else raw instruction string.
    Note:
        - If the expected delimiters are not found, the corresponding output will be an empty string or list.
        - Code blocks wrapped in markdown (```) are stripped before tokenization.
    """

    # Extract code section
    code_match = re.search(r'## Code Before:(.*?)## Instruction:', instr, re.DOTALL)
    instr_match = re.search(r'## Instruction:(.*?)## Code After:', instr, re.DOTALL)
    code_str = code_match.group(1).strip() if code_match else ''
    # Remove markdown code block wrappers
    if code_str.startswith("```") and code_str.endswith("```"):
        code_str = re.sub(r"^```[a-zA-Z]*\n?", "", code_str)
        code_str = re.sub(r"\n?```$", "", code_str)
        
    instr_str = instr_match.group(1).strip() if instr_match else ''

    if tokenize is False:
        return code_str, instr_str
    
    # Tokenize code
    code_tokens = process_code_tokens(code_str)

    # Tokenize instruction, remove punctuation, and convert to lowercase
    instr_word_list = re.split(r'\s+', instr_str)
    instr_tokens = [re.sub(r'[^\w]', '', w).lower() for w in instr_word_list]
    instr_tokens = [w for w in instr_tokens if w]

    return code_tokens, instr_tokens


def tokenize_instruction(input_instr: str, input_type: str):

    if input_type == 'code':
        # Remove markdown code block wrappers
        if input_instr.startswith("```") and input_instr.endswith("```"):
            input_instr = re.sub(r"^```[a-zA-Z]*\n?", "", input_instr)
            input_instr = re.sub(r"\n?```$", "", input_instr)
    
        # Tokenize code
        code_tokens = process_code_tokens(input_instr)
        return code_tokens

    elif input_type == 'text':
        # Tokenize instruction, remove punctuation, and convert to lowercase
        instr_word_list = re.split(r'\s+', input_instr)
        instr_tokens = [re.sub(r'[^\w]', '', w).lower() for w in instr_word_list]
        instr_tokens = [w for w in instr_tokens if w]
        return instr_tokens
    
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

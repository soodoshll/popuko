from typing import Optional, Coroutine

import ast
import inspect
import textwrap
import atexit


import popuko.server
from popuko.server import PopukoServer
import transformers

DEFAULT_SERVER = None
DEFAULT_TOKENIZER = None

class YieldString(ast.NodeTransformer):
    def visit_Expr(self, node):
        return ast.Expr(ast.Yield(node.value))

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.decorator_list:
            node.decorator_list = []
        return node


async def gather_outputs(func, delimiter, *args, **kwargs):
    cum_ret = None
    for out in func(*args, **kwargs):
        if isinstance(out, str):
            if cum_ret is None:
                cum_ret = out
            else:
                cum_ret += delimiter + out
        elif isinstance(out, GenerateFuture):
            if cum_ret is None:
                cum_ret = ""
            out.set_prompt(cum_ret)
            ret_tokens = (await out.request()).new_tokens()
            ret_str = DEFAULT_TOKENIZER.decode(ret_tokens)
            cum_ret += ret_str
    return cum_ret


def insert_yield(func):
    source = textwrap.dedent(inspect.getsource(func))
    parsed_ast = ast.parse(source)

    new_ast = ast.fix_missing_locations(YieldString().visit(parsed_ast))
    compiled_ast = compile(new_ast, filename="<ast>", mode="exec")

    local_vars = {}
    exec(compiled_ast, func.__globals__, local_vars)
    return local_vars[func.__name__]

def query(delimiter='\n'):
    def func_factory(func):
        modified_func = insert_yield(func)

        def prompt_func(*args, **kwargs):
            return gather_outputs(modified_func, delimiter, *args, **kwargs)
        return prompt_func
    return func_factory

def init_hf_llm(model: str, tokenizer: Optional[str]=None, max_batch_size: int=4, use_cache=True):
    global DEFAULT_SERVER, DEFAULT_TOKENIZER
    if tokenizer is None:
        tokenizer = model
    model = transformers.AutoModelForCausalLM.from_pretrained(model)
    model.half().cuda()
    DEFAULT_SERVER = PopukoServer(model, max_batch_size=max_batch_size, use_cache=use_cache)
    DEFAULT_SERVER.launch()
    DEFAULT_TOKENIZER = transformers.AutoTokenizer.from_pretrained(tokenizer)

    def stop():
        DEFAULT_SERVER.stop()

    atexit.register(stop)

class GenerateFuture:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.prompt: Optional[str] = None
    
    def set_prompt(self, prompt:str):
        self.prompt = prompt
    
    def request(self) -> Coroutine:
        prompts = DEFAULT_TOKENIZER(self.prompt).input_ids
        out = DEFAULT_SERVER.request(prompts, *self.args, **self.kwargs)
        return out

def generate(max_new_tokens=500, temperature=0):
    return GenerateFuture(max_new_tokens=max_new_tokens, temperature=temperature)
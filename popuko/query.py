from typing import Optional, Coroutine

import ast
import inspect
import textwrap
import atexit
import asyncio


import popuko.server
from popuko.server import PopukoServer
from .future import StringFuture
import transformers

DEFAULT_SERVER = None
DEFAULT_TOKENIZER = None
EVENT_LOOP = None

class YieldString(ast.NodeTransformer):
    def visit_Expr(self, node):
        return ast.Expr(ast.Yield(node.value))

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.decorator_list:
            node.decorator_list = []
        return node


async def gather_outputs(func, delimiter, *args, **kwargs):
    cum_ret = ""
    for out in func(*args, **kwargs):
        if isinstance(out, str):
                cum_ret += out + delimiter
        elif isinstance(out, GenerateFuture):
            out.set_prompt(cum_ret)
            ret_tokens = (await out.request()).new_tokens()
            ret_str = DEFAULT_TOKENIZER.decode(ret_tokens)
            cum_ret += ret_str
        elif isinstance(out, StringFuture):
            # pre-compute prompt
            async def _precompute():
                print("[pre-compute start]")
                pre_compute = GenerateFuture(prompt=cum_ret, max_new_tokens=1).request()
                ret = await pre_compute
                print("[pre-compute finishes]")
                return ret
            _, future_out = await asyncio.gather(_precompute(), out.wait())
            cum_ret += future_out + delimiter

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
            return StringFuture(EVENT_LOOP.create_task(gather_outputs(modified_func, delimiter, *args, **kwargs)))
        return prompt_func
    return func_factory

def init_llm(model: str, tokenizer: Optional[str]=None, max_batch_size: int=4, use_cache=True):
    global DEFAULT_SERVER, DEFAULT_TOKENIZER, EVENT_LOOP
    if tokenizer is None:
        tokenizer = model
    model = transformers.AutoModelForCausalLM.from_pretrained(model)
    model.half().cuda()
    DEFAULT_SERVER = PopukoServer(model, max_batch_size=max_batch_size, use_cache=use_cache)
    DEFAULT_SERVER.launch()
    DEFAULT_TOKENIZER = transformers.AutoTokenizer.from_pretrained(tokenizer)
    EVENT_LOOP = asyncio.get_event_loop()
    def stop():
        DEFAULT_SERVER.stop()

    atexit.register(stop)

def get_event_loop():
    global EVENT_LOOP
    return EVENT_LOOP

class GenerateFuture:
    def __init__(self, prompt = None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.prompt: Optional[str] = prompt
        self._out = None
    
    def set_prompt(self, prompt:str):
        self.prompt = prompt
    
    def request(self) -> Coroutine:
        prompts = DEFAULT_TOKENIZER(self.prompt).input_ids
        out = DEFAULT_SERVER.request(prompts, *self.args, **self.kwargs)
        self._out = out
        return out
    
    def output(self) -> str:
        return DEFAULT_TOKENIZER.decode(self._out)

class QueryOutput:
    pass

def generate(max_new_tokens=500, temperature=0):
    return GenerateFuture(max_new_tokens=max_new_tokens, temperature=temperature)
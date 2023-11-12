import ast
import inspect
import textwrap

from .server import DEFAULT_SERVER

class YieldString(ast.NodeTransformer):
    def visit_Expr(self, node):
        return ast.Expr(ast.Yield(node.value))

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.decorator_list:
            node.decorator_list = [
                decorator
                for decorator in node.decorator_list
                if not isinstance(decorator, ast.Name) or decorator.id != "query"
            ]
        return node


def gather_outputs(func, *args, **kwargs):
    ret = []
    for out in func(*args, **kwargs):
        if isinstance(out, str):
            ret.append(out)
        if isinstance(out, set):
            # Create tasks
            print("find a set", out)
    return "\n".join(ret)


def insert_yield(func):
    source = textwrap.dedent(inspect.getsource(func))
    parsed_ast = ast.parse(source)

    new_ast = ast.fix_missing_locations(YieldString().visit(parsed_ast))
    compiled_ast = compile(new_ast, filename="<ast>", mode="exec")

    local_vars = {}
    exec(compiled_ast, func.__globals__, local_vars)
    return local_vars[func.__name__]

def query(func):
    modified_func = insert_yield(func)

    def prompt_func(*args, **kwargs):
        return gather_outputs(modified_func, *args, **kwargs)

    return prompt_func


def generate(max_new_tokens=500, temperature=0):
    return None
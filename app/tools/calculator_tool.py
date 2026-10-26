from langchain_core.tools import tool
import ast
import operator
from pydantic import BaseModel, Field



ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg
}

def safe_eval(node):
    if isinstance(node, ast.Constant): 
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        return ALLOWED_OPERATORS[type(node.op)](operand)
    else:
        raise TypeError(f"Unsupported operation: {type(node)}")


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="A mathematical expression to evaluate, for example: '3*3+3-4' or '(12 + 4) / 2'"
    )


@tool("expression_calculator", args_schema=CalculatorInput)
def expression_calculator_tool(expression: str) -> str:
    """
    Useful for when you need to calculate mathematical expressions provided as a text string.
    Follows standard mathematical order of operations (PEMDAS/BODMAS).
    """
    try:
        clean_expr = expression.strip().strip("'\"")
        tree = ast.parse(clean_expr, mode='eval')
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

    
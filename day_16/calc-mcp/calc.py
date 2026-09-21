from __future__ import annotations

import ast
import math
import operator

# Python's own parser reads the expression; this module then walks the tree it
# produced. eval() would run whatever arrived - and what arrives here is written
# by a model, not by the person running the server - so the expression is
# untrusted input. A walk over a whitelist can do arithmetic and nothing else:
# there is no node for a call to something that is not in FUNCTIONS, no name
# that is not in CONSTANTS, and no attribute access at all.

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Enough of `math` that an ordinary expression is not turned away for using a
# square root. Each is a pure function of its arguments - nothing here reads or
# writes anything outside the calculation.
FUNCTIONS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan", "atan2", "floor", "ceil", "hypot")
}
FUNCTIONS.update(abs=abs, round=round, min=min, max=max, pow=pow)

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

MAX_EXPRESSION_LENGTH = 500
# 2 ** 1000 is a 302-digit number, which is past any real question and well
# short of the exponent that would have the server spend a minute in C code
# building an integer nothing can use. Every other operator is bounded by the
# size of its operands; ** is the one that turns a short expression into a
# long computation, so it is the one that is capped.
MAX_EXPONENT = 1000


class CalculationError(ValueError):
    """An expression that cannot be calculated: bad syntax, something outside
    the whitelist, or arithmetic that has no answer. Carries a message meant to
    be read by whoever sent the expression, so they can correct it."""


def evaluate(expression: str) -> float:
    """The value of a mathematical expression, or CalculationError saying why
    there is none."""
    expression = expression.strip()
    if not expression:
        raise CalculationError("The expression is empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculationError(
            f"The expression is too long ({len(expression)} characters, "
            f"limit {MAX_EXPRESSION_LENGTH})."
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"That is not a valid expression: {exc.msg}.") from exc

    try:
        return _value(tree.body)
    except CalculationError:
        raise
    except ZeroDivisionError:
        raise CalculationError("Division by zero.") from None
    except (ArithmeticError, ValueError, TypeError) as exc:
        # log(-1), a result too large for a float, round("x") - the arithmetic
        # was allowed but has no answer. The library's own wording is the
        # clearest account of which.
        raise CalculationError(f"That cannot be calculated: {exc}.") from None


def _value(node: ast.AST) -> float:
    """One node of the tree, and everything under it."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError(f"{node.value!r} is not a number.")
        return node.value

    if isinstance(node, ast.BinOp):
        apply = BINARY_OPERATORS.get(type(node.op))
        if apply is None:
            raise CalculationError(f"{_describe(node.op)} is not an operator this calculator has.")
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculationError(f"That exponent is too large (limit {MAX_EXPONENT}).")
        return apply(left, right)

    if isinstance(node, ast.UnaryOp):
        apply = UNARY_OPERATORS.get(type(node.op))
        if apply is None:
            raise CalculationError(f"{_describe(node.op)} is not an operator this calculator has.")
        return apply(_value(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise CalculationError(f"{_describe(node.func)} is not a function this calculator has.")
        if node.keywords:
            raise CalculationError("Functions here take their arguments in order, not by name.")
        return FUNCTIONS[node.func.id](*(_value(argument) for argument in node.args))

    if isinstance(node, ast.Name):
        if node.id not in CONSTANTS:
            raise CalculationError(f"{node.id} is not a number this calculator knows.")
        return CONSTANTS[node.id]

    raise CalculationError(f"{_describe(node)} is not something this calculator can work out.")


def _describe(node: ast.AST) -> str:
    """What to call a node in a refusal: the source text where there is any -
    the sender wrote it and will recognise it - and the node's own class name
    where there is not, since an operator node carries no text of its own and
    unparsing one gives an empty string."""
    text = getattr(node, "id", None) or ast.unparse(node).strip()
    return text or type(node).__name__

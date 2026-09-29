"""A small source tree for the chunker and index tests."""

LONG_BODY = "\n".join(f"    value_{i} = compute(value_{i - 1 if i else 0}, 'padding text to make lines long')"
                      for i in range(120))

APP_PY = f'''"""Fixture module."""
import os
from pathlib import Path

LIMIT = 3


class Outer:
    """Outer docstring."""

    class Inner:
        def nested_method(self, x):
            return helper(x) + os.getpid()

    def outer_method(self):
        return self.Inner()


def helper(x):
    return x * 2


def long_function():
{LONG_BODY}
    return value_119


if __name__ == "__main__":
    print(helper(LIMIT))
'''

APP_JS = """function greet(name) {
  return 'hi ' + name;
}

class Greeter {
  hello() { return greet('x'); }
}
"""


def write_tree(root):
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "app.py").write_text(APP_PY)
    (root / "web.js").write_text(APP_JS)
    (root / "blob.bin").write_bytes(b"\0\1\2")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.py").write_text("ignored = 1\n")

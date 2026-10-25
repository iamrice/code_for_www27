"""Load a candidate aggregation program (Python source defining `aggregate`)."""
import math
import numpy as np

ALLOWED_IMPORTS = {"numpy", "math"}


def _guarded_import(name, *args, **kw):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name} not allowed")
    return __import__(name, *args, **kw)


def compile_program(src):
    g = {"__builtins__": {**{k: getattr(__builtins__, k) if not isinstance(__builtins__, dict) else __builtins__[k]
                              for k in ["abs", "min", "max", "range", "len", "float", "int", "sum", "enumerate",
                                        "zip", "list", "dict", "tuple", "sorted", "round", "bool", "isinstance",
                                        "ValueError", "Exception", "any", "all", "map", "set", "reversed", "print"]},
                         "__import__": _guarded_import},
         "np": np, "math": math}
    exec(compile(src, "<program>", "exec"), g)
    fn = g.get("aggregate")
    if fn is None:
        raise ValueError("program does not define aggregate()")
    return fn


def load_program(path):
    return compile_program(open(path).read())

import torch
from functools import wraps
from collections.abc import Iterable


def compile_with_eager_fallback(fn, name: str):
    compiled_fn = None
    disabled = False

    @wraps(fn)
    def wrapped(*args, **kwargs):
        nonlocal compiled_fn, disabled
        if disabled:
            return fn(*args, **kwargs)
        try:
            if compiled_fn is None:
                compiled_fn = torch.compile(fn)
            return compiled_fn(*args, **kwargs)
        except Exception:
            disabled = True
            print("torch.compile failed for %s; falling back to eager", name, exc_info=True)
            return fn(*args, **kwargs)

    return wrapped

def compile_model_modules(*roots: torch.nn.Module):
    for root in roots:
        for module in root.modules():
            method_names: Iterable[str] = getattr(module, "compile_methods", ())
            if not method_names or getattr(module, "_compiled", False):
                continue
            for method_name in method_names:
                method = getattr(module, method_name)
                qualified_name = f"{module.__class__.__name__}.{method_name}"
                setattr(module, method_name, compile_with_eager_fallback(method, qualified_name))
            module._compiled = True

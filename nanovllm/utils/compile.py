import torch
from functools import wraps


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

def compile_model_modules(model: torch.nn.Module):
    for module in model.modules():
        compile_fn = getattr(module, "enable_compile", None)
        if compile_fn is not None:
            compile_fn()

# PR: Normalize Qwen3 RoPE config across Transformers versions

## Summary

This PR adds a RoPE config normalization step in `nanovllm/config.py` so that Nano vLLM can consume Qwen3 configs from different `transformers` versions (both >5 and <5) through a unified `rope_parameters` interface.

The main goal is to avoid runtime failures like:

```text
self.rotary_emb = get_rope(
    self.head_dim,
    rotary_dim=self.head_dim,
    max_position=max_position,
    base=rope_theta,
    type=rope_type,
)
TypeError: unhashable type: 'dict'
```

## Background

`nanovllm/models/qwen3.py` reads RoPE settings from `config.rope_theta` and `config.rope_scaling` , but recent `transformers` releases may expose RoPE-related fields differently:

- Newer schema: `rope_parameters`
```
{
  "rope_theta": 1000000,
  "rope_scaling": null
}
```
- Older schema: `rope_theta` and `rope_scaling`
```
{
  "rope_parameters":{
    "rope_theta": 1000000
    "rope_type": "default",
  }
}
```
In versions of `transformers` > 5, `rope_scaling` is an alias of `rope_parameters`, which means it is a dict. As a result, a dict is passed into `get_rope(...)`, causing `lru_cache` to raise `TypeError: unhashable type: 'dict'`.


## Changes

This PR introduces a normalization helper in `nanovllm/config.py` and applies it immediately after `AutoConfig.from_pretrained(...)`.

Normalization behavior:

- If `hf_config.rope_parameters` already exists, keep it unchanged.
- Otherwise, build `rope_parameters` from legacy fields:
  - `rope_theta`
  - `rope_scaling`
- Normalize the legacy key rename:
  - `rope_scaling.type` -> `rope_type`
- Ensure a default fallback:
  - `rope_type = "default"`

After normalization, downstream model code can consistently read:

```python
config.rope_parameters
```

without needing version-specific branching.

## Validation

This change has been validated with `transformers` 4.56.0 and 5.6.0.

## Why this approach

- Reserves compatibility with newer `transformers` versions that already expose `rope_parameters`.
- Normalizes RoPE inputs before calling `get_rope(...)`, which makes it easier to extend the same interface to other models in the future.


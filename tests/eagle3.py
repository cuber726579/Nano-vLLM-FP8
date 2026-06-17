import json
import gc
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm import LLM
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams
from nanovllm.spec_decode import Eagle3Speculator, load_eagle3_config
from nanovllm.utils.loader import load_model


MODEL_ROOT = Path("/nasdata2/private/hypang/huggingface/models")
QWEN3_8B_PATH = MODEL_ROOT / "Qwen/Qwen3-8B"
QWEN3_8B_EAGLE3_PATH = MODEL_ROOT / "RedHatAI/Qwen3-8B-speculator.eagle3"


def test_eagle3_config_parser():
    raw_config = {
        "speculators_model_type": "eagle3",
        "draft_vocab_size": 32000,
        "norm_before_residual": True,
        "target_hidden_size": None,
        "transformer_layer_config": {
            "attention_bias": False,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "max_position_embeddings": 40960,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "vocab_size": 151936,
        },
    }

    with TemporaryDirectory() as tmpdir:
        Path(tmpdir, "config.json").write_text(json.dumps(raw_config), encoding="utf-8")
        config = load_eagle3_config(tmpdir)

    assert config.draft_vocab_size == 32000
    assert config.target_vocab_size == 151936
    assert config.target_hidden_size == 4096
    assert config.norm_before_residual is True
    assert config.layer_config.hidden_size == 4096
    assert config.layer_config.rope_theta == 1000000


def test_loader_loads_buffers():
    class TinyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 2))
            self.register_buffer("d2t", torch.empty(3, dtype=torch.long))
            self.register_buffer("t2d", torch.empty(5, dtype=torch.bool))

    tensors = {
        "weight": torch.arange(4, dtype=torch.float32).view(2, 2),
        "d2t": torch.tensor([0, 10, 20], dtype=torch.long),
        "t2d": torch.tensor([True, False, True, False, True]),
    }

    with TemporaryDirectory() as tmpdir:
        save_file(tensors, Path(tmpdir, "model.safetensors"))
        module = TinyModule()
        loaded = load_model(module, tmpdir)

    assert loaded == set(tensors)
    assert torch.equal(module.weight, tensors["weight"])
    assert torch.equal(module.d2t, tensors["d2t"])
    assert torch.equal(module.t2d, tensors["t2d"])


def test_sampler_temperature_zero_is_greedy():
    logits = torch.tensor(
        [
            [1.0, 4.0, 2.0],
            [3.0, 1.0, 0.0],
        ]
    )
    sampler = Sampler()
    token_ids = sampler(
        logits,
        temperatures=torch.tensor([0.0, 0.0]),
        top_ps=torch.tensor([1.0, 1.0]),
        top_ks=torch.tensor([-1, -1], dtype=torch.int32),
        min_ps=torch.tensor([0.0, 0.0]),
        needs_filter=False,
    )

    assert token_ids.tolist() == [1, 0]


def test_eagle3_vocab_mapping_helpers():
    speculator = object.__new__(Eagle3Speculator)
    speculator.d2t = torch.tensor([0, 4, 5], dtype=torch.long)
    speculator.t2d = torch.tensor([True, False, False, False, True, False, False, True])

    draft_tokens = torch.tensor([0, 1, 2], dtype=torch.long)
    target_tokens = speculator.map_draft_to_target_tokens(draft_tokens)
    available = speculator.check_target_token_availability(target_tokens)

    assert target_tokens.tolist() == [0, 5, 7]
    assert available.tolist() == [True, False, True]


def test_greedy_acceptance_cases():
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(eos=99)
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=8))

    rejected, accepted_count = ModelRunner.accept_speculative_tokens(
        runner,
        seq,
        draft_token_ids=[10, 11, 12],
        target_token_ids=[20, 21, 22, 23],
    )
    assert rejected == [20]
    assert accepted_count == 0

    partial, accepted_count = ModelRunner.accept_speculative_tokens(
        runner,
        seq,
        draft_token_ids=[10, 11, 12],
        target_token_ids=[10, 21, 22, 23],
    )
    assert partial == [10, 21]
    assert accepted_count == 1

    all_accepted, accepted_count = ModelRunner.accept_speculative_tokens(
        runner,
        seq,
        draft_token_ids=[10, 11, 12],
        target_token_ids=[10, 11, 12, 23],
    )
    assert all_accepted == [10, 11, 12, 23]
    assert accepted_count == 3

    eos, accepted_count = ModelRunner.accept_speculative_tokens(
        runner,
        seq,
        draft_token_ids=[99, 11, 12],
        target_token_ids=[99, 11, 12, 23],
    )
    assert eos == [99]
    assert accepted_count == 1


def generate_with_llm(model_path: Path, sampling_params: SamplingParams, **kwargs):
    llm = None
    try:
        llm = LLM(
            str(model_path),
            max_model_len=256,
            max_num_batched_tokens=256,
            max_num_seqs=1,
            gpu_memory_utilization=0.6,
            enforce_eager=True,
            **kwargs,
        )
        return llm.generate(
            ["The capital of France is"],
            sampling_params,
            use_tqdm=False,
        )[0]
    finally:
        if llm is not None:
            llm.exit()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_qwen3_eagle3_integration():
    if not QWEN3_8B_PATH.is_dir() or not QWEN3_8B_EAGLE3_PATH.is_dir():
        print("Skipping Qwen3 Eagle3 integration test: local models are missing.")
        return
    if not torch.cuda.is_available():
        print("Skipping Qwen3 Eagle3 integration test: CUDA is unavailable.")
        return

    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    baseline_output = generate_with_llm(QWEN3_8B_PATH, sampling_params)
    eagle3_output = generate_with_llm(
        QWEN3_8B_PATH,
        sampling_params,
        speculative_model=str(QWEN3_8B_EAGLE3_PATH),
        speculative_method="eagle3",
        num_speculative_tokens=3,
    )

    print(f"Baseline token ids: {baseline_output['token_ids']}")
    print(f"Eagle3 token ids:   {eagle3_output['token_ids']}")
    print(f"Baseline text: {baseline_output['text']!r}")
    print(f"Eagle3 text:   {eagle3_output['text']!r}")

    assert eagle3_output["token_ids"] == baseline_output["token_ids"]
    assert eagle3_output["text"] == baseline_output["text"]


def main():
    test_eagle3_config_parser()
    test_loader_loads_buffers()
    test_sampler_temperature_zero_is_greedy()
    test_eagle3_vocab_mapping_helpers()
    test_greedy_acceptance_cases()
    test_qwen3_eagle3_integration()
    print("Eagle3 tests passed.")


if __name__ == "__main__":
    main()

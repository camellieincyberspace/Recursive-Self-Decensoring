"""Transparent byte/token estimates, never fabricated runtime benchmarks."""
from __future__ import annotations


def memory_estimate(parameters: int, bits: float,
                    layers: int, width: int, samples: int,
                    bank_width: int = 32, rank: int = 8) -> dict:
    values = [parameters, bits, layers, width, samples, bank_width, rank]
    if any(x <= 0 for x in values):
        raise ValueError("all sizes must be positive")
    gib = 2**30
    return {
        "weight_payload_only_GiB": parameters * bits / 8 / gib,
        "all_layer_output_cache_fp16_GiB": layers * width * samples * 2 / gib,
        "projected_cache_fp32_MiB": layers * samples * bank_width * 4 / 2**20,
        "direction_recipe_fp32_MiB": layers * width * rank * 4 / 2**20,
        "excludes": ["quantization_metadata", "mixed_precision_tensors", "KV/recurrent_states",
                     "runtime_workspace", "display_reserve", "allocator_overhead", "judge_model"],
    }


def generation_budget(stages: list[dict]) -> dict:
    prompt_tokens = output_tokens = 0
    for s in stages:
        for k in ("candidates", "prompts", "input_tokens", "output_tokens"):
            if not isinstance(s.get(k), int) or isinstance(s[k], bool) or s[k] < 0:
                raise ValueError(f"invalid stage field {k}")
        prompt_tokens += s["candidates"] * s["prompts"] * s["input_tokens"]
        output_tokens += s["candidates"] * s["prompts"] * s["output_tokens"]
    return {"input_tokens": prompt_tokens, "max_output_tokens": output_tokens,
            "runtime_formula": "input_tokens/measured_prefill_tps + actual_output_tokens/measured_decode_tps + judge + IO",
            "not_a_runtime_prediction": True}

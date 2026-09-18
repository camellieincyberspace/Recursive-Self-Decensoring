from recipe_lab.budget import memory_estimate, generation_budget
from recipe_lab.demo import run_demo


def test_sizes_are_derived_not_claimed_measurements():
    b=memory_estimate(parameters=1_000_000, bits=4, layers=2, width=64, samples=128)
    assert b['direction_recipe_fp32_MiB']==2*64*8*4/2**20
    assert b['projected_cache_fp32_MiB']==2*128*32*4/2**20
    assert b['all_layer_output_cache_fp16_GiB']==2*64*128*2/2**30
    assert b['weight_payload_only_GiB']==500000/2**30


def test_token_budget():
    b=generation_budget([dict(candidates=10,prompts=20,input_tokens=64,output_tokens=32)])
    assert b['input_tokens']==12800 and b['max_output_tokens']==6400


def test_demo_is_labeled_synthetic(tmp_path):
    r=run_demo(tmp_path/'demo')
    assert r['synthetic_only'] and not r['language_model_tested']
    assert r['max_compile_error'] < 1e-5
    assert r['local_metrics']['target_mse'] < 1e-8

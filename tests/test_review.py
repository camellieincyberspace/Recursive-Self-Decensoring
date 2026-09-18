import pytest
from recipe_lab.review import validate_review, semantic_failure, cache_key, paired_regression, zero_event_upper_bound


def rec():
    return dict(case_id='a',label='complete',completion=1.0,warning=True,validity='valid',
                correctness='pass',evidence=['42'],needs_human=False)


def test_warning_does_not_count_as_failure():
    r=validate_review(rec(),'Warning. The answer is 42.')
    assert semantic_failure(r)==0


def test_fake_agreement_incomplete():
    r=rec();r.update(label='evasion',completion=.1,evidence=['considerations'])
    assert semantic_failure(validate_review(r,'Sure, there are considerations.'))==.9


@pytest.mark.parametrize('field',['warning','needs_human'])
def test_string_false_rejected(field):
    r=rec();r[field]='false'
    with pytest.raises(ValueError): validate_review(r,'42')


def test_fake_evidence_rejected():
    with pytest.raises(ValueError): validate_review(rec(),'No answer')


def test_truncated_does_not_mean_refusal():
    r=rec();r.update(label='partial',completion=.2,validity='truncated')
    assert semantic_failure(validate_review(r,'42')) is None


def test_unknown_is_not_success():
    r=rec();r.update(label='unknown',needs_human=True)
    assert semantic_failure(validate_review(r,'42')) is None


def test_cache_includes_rubric():
    args=dict(prompt='a',response='b',judge_model_revision='v1',decode_config={})
    assert cache_key(**args,rubric_sha='1')!=cache_key(**args,rubric_sha='2')


def test_regression_cannot_be_hidden_by_gains():
    r=paired_regression([True,True,False,False],[True,False,True,False])
    assert r['regression_rate']==.5 and r['gains']==1


def test_zero_failures_are_not_proof_of_zero_risk():
    assert .02 < zero_event_upper_bound(100) < .04

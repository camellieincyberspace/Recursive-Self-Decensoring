from pathlib import Path
import json
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name',['review','pair_review','capture','recipe'])
def test_schema_is_well_formed(name):
    js=pytest.importorskip('jsonschema')
    schema=json.loads((ROOT/'schemas'/f'{name}.schema.json').read_text())
    js.Draft202012Validator.check_schema(schema)


def test_review_schema_rejects_string_boolean():
    js=pytest.importorskip('jsonschema')
    schema=json.loads((ROOT/'schemas/review.schema.json').read_text())
    record=dict(case_id='a',label='complete',completion=1,warning='false',validity='valid',correctness='pass',evidence=['42'],needs_human=False)
    with pytest.raises(js.ValidationError):js.validate(record,schema)


def test_pilot_has_no_fabricated_source_hash():
    config=json.loads((ROOT/'configs/pilot.template.json').read_text())
    assert config['model']['artifact_sha256'] is None
    assert config['release']['real_model_tested'] is False

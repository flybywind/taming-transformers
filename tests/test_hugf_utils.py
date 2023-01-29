import pytest

from .config import cache_dir

@pytest.fixture
def model_id() -> str:
    return "shanetx/2020-11-20T12-54-32_drin_transformer"

def test_hugf_from_pretrained(model_id, cache_dir):
    from taming_transformers_hugf.taming.models.cond_transformer import Net2NetTransformer
    model = Net2NetTransformer.from_pretrained(model_id, cache_dir=cache_dir)
    assert model.first_stage_model is not None

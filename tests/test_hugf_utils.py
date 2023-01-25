import pytest
import os
from os import path as osp
import sys
test_dir = osp.dirname(__file__)
pkg_dir = osp.abspath(osp.join(test_dir, "..", "src/"))
sys.path.insert(0, pkg_dir)

@pytest.fixture
def model_id() -> str:
    return "shanetx/2020-11-20T12-54-32_drin_transformer"

@pytest.fixture
def cache_dir() -> str:
    dir_name = osp.join(test_dir, "cache_hugf")
    if not osp.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name

def test_hugf_from_pretrained(model_id, cache_dir):
    from taming_transformers_hugf.taming.models.cond_transformer import Net2NetTransformer
    model = Net2NetTransformer.from_pretrained(model_id, cache_dir=cache_dir)
    assert model.first_stage_model is not None

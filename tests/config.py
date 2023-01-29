import pytest
import os
from os import path as osp
from . import test_dir, pkg_dir

@pytest.fixture
def cache_dir() -> str:
    dir_name = osp.join(test_dir, "cache_hugf")
    if not osp.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name
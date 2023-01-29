from os import path as osp
from .. import test_dir, asset_dir
from omegaconf import OmegaConf

from taming_transformers_hugf.main import instantiate_from_config
from taming_transformers_hugf.taming.models.cond_transformer import Imgplus2ImgTransformer

def test_imgplus2imgtransformer():
    yaml_file = osp.join(asset_dir, "imgplus2img.yaml")
    config = OmegaConf.load(yaml_file)
    imgtran:Imgplus2ImgTransformer = instantiate_from_config(config=config.model)
    assert imgtran is not None
    print(imgtran.named_parameters())
from os import path as osp
import sys
test_dir = osp.dirname(__file__)
pkg_dir = osp.abspath(osp.join(test_dir, "..", "src/"))
sys.path.insert(0, pkg_dir)

asset_dir = osp.join(test_dir, "assets")
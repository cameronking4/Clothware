# By IT-JIM, 2022
import cv2
import numpy as np
import open3d as o3d

from pathlib import Path
import shutil
import argparse
from loguru import logger
from typing import Dict, Any, List

from texturing import visualization as vsl
from texturing import utils
from texturing import warping as wrp

import kgdet.processing as prs
import kgdet.mmdetection.tools.custom_test as kgdet
import mmcv
    

########################################################################################################################
class Demo():
    def __init__(self) -> None:
        try:
            self.get_args()
            self.get_dicts()
            self.get_path()
            self.get_kgdet()
        except Exception as exception:
            self.clear_tmp()
            raise exception
        
        self._p_out.mkdir(exist_ok=True, parents=False)
    
    def get_args(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--root', type=str, default='./input', help='Directory, where images are placed.')
        parser.add_argument('--out', type=str, default='./output', help='Directory for output results.')
        parser.add_argument('--mesh', type=str, default='./meshes', help='Directory of a folder containing pattern-meshes.')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='Directory of a folder containing checkpoints for kgdet.')
        parser.add_argument('--type', type=str, choices=['pants', 'shirt'], help='Target type of garment for reconstruction.', required=True)
        parser.add_argument('--front', type=str, help='Path to image of a front side of a garment.', required=True)
        parser.add_argument('--back', type=str, help='Path to image of a front side of a garment.',default=None, required=False)

        self._opt = parser.parse_args()
    
    def get_dicts(self):
        self._type_dict = {"shirt" : 1,
                           "pants" : 8}
        self._side_dict = {str(self._opt.front): "front", str(self._opt.back) : "back"}   
    
    def get_path(self):
        self._is_both_sides = True if self._opt.back is not None else False
        
        # Setting texture paths to a mesh corresponding to a garment-type
        self._p_mesh: Path = Path(self._opt.mesh) / self._opt.type
        self._p_mtl, _ = utils.find_suff(self._p_mesh, suff='.mtl')
        assert self._p_mtl is not None, f'.mtl file in {self._p_mesh} not found'
        
        self._p_text: Path = self._p_mesh / f'tex_{self._opt.type}.png'
        assert self._p_mesh.exists(), f"'{self._p_mesh}' doesn't exist."
        assert self._p_text.is_file(), f"'{self._p_text}' isn't file"
        
        # Preparing metadata paths: a side-garment image and texture-side keypoints
        _p_text_f_kp: Path = self._p_mesh / 'keypoints_f.json'
        _p_text_b_kp: Path = self._p_mesh / 'keypoints_b.json'
        self._p_text_kp_dict: Dict[str, Path] = {"front" : _p_text_f_kp, "back" : _p_text_b_kp}
        
        # Prepare checkpoints path
        self._p_ckpts = Path(self._opt.checkpoints)
        assert self._p_ckpts.exists()
        assert self._p_ckpts.is_dir()
        
        # Prepare output folder
        self._p_out = Path(self._opt.out)
        self._p_out.mkdir(exist_ok=True, parents=True)
        if len(list(self._p_out.iterdir())):
            logger.warning(f"Output folder is not emty. Deleting content of {self._p_out}")
            for f in self._p_out.iterdir():
                f.unlink()
        
        # Prepare temp folder for kgdet processing
        self._p_kgdet_temp = Path('.tmp')
        self._p_kgdet_temp.mkdir(exist_ok=True)
        
        i_names = [self._opt.front] if self._opt.back is None else [self._opt.front, self._opt.back]
        for i_name in i_names:
            f_from = Path(self._opt.root) / i_name
            f_to = self._p_kgdet_temp / i_name
            shutil.copy(f_from, f_to)
    
    def clear_tmp(self):
        if hasattr(self, "_p_kgdet_temp") and self._p_kgdet_temp.exists():
            shutil.rmtree(self._p_kgdet_temp)
        
    def get_kgdet(self):
        p_configs = Path('kgdet/configs')
        kgdet_ckpts = self._p_ckpts / 'KGDet_epoch-12.pth'
        assert kgdet_ckpts.is_file() and kgdet_ckpts.exists()
        
        # Default path to config and checkpoints
        self._kgdet_cfg = mmcv.Config.fromfile(str(p_configs / 'kgdet_moment_r50_fpn_1x-demo.py')) 
        self._kgdet_cfg.template_json = str(p_configs / 'dataset_template.json')
        self._kgdet_cfg.checkpoint_path = str(kgdet_ckpts)
        
        # Dataset metadata
        path_to_dataset = self._p_kgdet_temp
        self._kgdet_cfg.data.test.ann_file = str(path_to_dataset / 'dataset.json')
        self._kgdet_cfg.data.test.img_prefix = str(path_to_dataset) 
        kgdet.create_dataset_annotations(self._kgdet_cfg)
    
    def process_pts_meta(self, pts_dicts: List[Dict[str, Any]], class_id: int) -> List[Dict[str, Any]]:
        """Function used to process metadata returned with KGDet, 
        by changing its keys, points shape and adding additional information"""
        def generator(_pts_dicts: List[Dict[str, Any]]):
            for pts_dict in _pts_dicts:
                p_image = self._p_kgdet_temp / pts_dict["file_name"]
                assert p_image.is_file() and p_image.exists()
                
                side = self._side_dict[pts_dict["file_name"]]
                processed_meta = utils.process_keypoints(pts_dict)
                
                # Validation of kgdet side prediction
                is_valid, pts = prs.process_kgdet_side(pts_dict['keypoints'], cat_id=class_id, 
                                                       side=side, visualize=False, to_reshape=True)
                if not is_valid:
                    processed_meta["keypoints"] = pts                

                yield {"category_id" : processed_meta["category_id"],
                       "side" : side,
                       "g_pts" : processed_meta["keypoints"],
                       "g_mask" : processed_meta["mask"],
                       "g_img" : cv2.imread(str(p_image)),
                       "p_g_img" : p_image}
            
        return list(generator(pts_dicts))

    def process_txt_meta(self, dicts: List[Dict[str, Any]]):
        """Function used to process metadata by adding information about corresponding 
        texture points, image and mask."""
        def generator(_dicts: List[Dict[str, Any]]):
            for _dict in _dicts:
                p_t_pts = self._p_text_kp_dict[_dict["side"]]
                t_dict = utils.load_keypoints(str(p_t_pts))
                
                _dict["t_pts"] = t_dict["keypoints"]
                _dict["t_mask"] = t_dict["mask"]
                _dict["t_img"] = text_img                
                yield _dict
        
        text_img = cv2.imread(str(self._p_text))
        assert text_img is not None
        
        return list(generator(dicts))
    
    def save_recon(self, texture: np.ndarray, texture_name: str = 'texture.png'):
        """Function for saving output reconstruction. It copies mesh files to output folder, 
        saves generated texture and changes name of a texture in .mtl"""
        # Copy template mesh to output
        for f_from in self._p_mesh.iterdir():
            if f_from.suffix == '.obj' or f_from.suffix == '.mtl':
                fname = f_from.name
                f_to = self._p_out / fname
                shutil.copy(f_from, f_to)
        
        # Saving texture
        fname_out = self._p_out / texture_name
        cv2.imwrite(str(fname_out), texture)
        
        # Changing path to texture in .mtl
        success = utils.change_mtl_mat(self._p_mtl, texture_name)
        assert success, f'{self._p_mtl} was not changed'

    def run(self):
        class_id = self._type_dict[self._opt.type]
        
        logger.info(f'[started]\tGarment keypoints preparation.')
        _, keypoints_metas = kgdet.get_all_keypoints(self._kgdet_cfg)
        keypoints_metas = kgdet.prune_json(self._kgdet_cfg, keypoints_metas, cat_id=class_id)
        
        input_metas = self.process_pts_meta(keypoints_metas, class_id=class_id)
        logger.info(f'[finished]\tGarment keypoints preparation.')
        
        logger.info(f'[started]\tLoading texture metadata.')
        input_metas = self.process_txt_meta(input_metas)
        logger.info(f'[finished]\tLoading texture metadata.')
        
        for key in ["category_id", "side", "g_pts", "g_mask", "g_img", "p_g_img", "t_pts", "t_mask", "t_img"]:
            assert key in input_metas[0].keys()
        
        if False:
            for meta in input_metas:
                utils.show_meta(meta)
            
            vsl.demo_points(input_metas[0]["g_img"], input_metas[0]["g_pts"], radius=3, coef=0.4, thickness=3, fontscale=3, color=(0, 0, 255), verbose=True, name='1', save_path=Path('./'))
            vsl.demo_points(input_metas[1]["g_img"], input_metas[1]["g_pts"], radius=3, coef=0.4, thickness=3, fontscale=3, color=(0, 0, 255), verbose=True, name='2', save_path=Path('./'))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        logger.info(f'[started]\tWarping garment image to uv.')
        results = {}
        for meta in input_metas:
            kp_mask = meta["g_mask"] & meta["t_mask"]
            
            transformed_img = wrp.warp2texture(img_source=meta["g_img"], 
                                               img_target=meta["t_img"], 
                                               pts_source=meta["g_pts"], 
                                               pts_target=meta["t_pts"], 
                                               mask=np.asarray(kp_mask))
            
            results[meta["side"]] = transformed_img
        logger.info(f'[finished]\tWarping garment image to uv.')
        
        logger.info(f'[started]\tSaving textured mesh to {self._p_mesh}.')
        if self._is_both_sides:
            text_f = results["front"]
            text_b = results["back"]
            texture = wrp.merge_sides_texture(text_f, text_b)
        else:
            texture = results["front"]
        self.save_recon(texture)
        logger.info(f'[finished]\tSaving textured mesh to {self._p_mesh}.')


########################################################################################################################
if __name__ == '__main__':
    demo = Demo()
    demo.run()
    demo.clear_tmp()

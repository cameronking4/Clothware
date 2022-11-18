
# By IT-JIM, 2022
import numpy as np
import cv2

from pathlib import Path
from loguru import logger

import visualization as vsl
import utils

########################################################################################################################
def source2target_shape(from_img: np.ndarray, to_img: np.ndarray) -> np.ndarray:
    """Reshapes image to the target size by adding right-bot border.
    It is needed while warping image pixels area to the texture pixels area.

    Args:
        from_img (np.ndarray): source image to wrap
        to_img (np.ndarray): target image of wrapping

    Returns:
        np.ndarray: image with the size of `to_img`, with `from_img` at left-top corner
    """
    source_img = np.zeros_like(to_img)
    f_h, f_w, _ = from_img.shape
    t_h, t_w, _ = to_img.shape
    
    h = min(f_h, t_h)
    w = min(f_w, t_w)
    
    source_img[:h, :w, :] = from_img[:h, :w, :]
    return source_img


########################################################################################################################
def warp2texture(img_source: np.ndarray, img_target: np.ndarray, 
                 pts_source: np.ndarray, pts_target: np.ndarray,
                 mask: np.ndarray = None) -> np.ndarray:
    """Function which wrapps `pts_source` area of points of `img_source` 
    to the `pts_source` point area of `pts_target` image.

    Args:
        img_source (np.ndarray): source image (garment)
        img_target (np.ndarray): target image (texture)
        pts_source (np.ndarray): points of shape (1, n, 2) from `img_source`
        pts_target (np.ndarray): points of shape (1, n, 2) from `img_texture`
        mask (np.ndarray, optional): binary array of (n,) size for masking point pairs to use for warping. Defaults to None.
    
    Returns:
        np.ndarray: warped image of `img_target` size
    """
    assert pts_source.shape == pts_target.shape
    assert len(pts_source.shape) == 3
    assert pts_source.shape[0] == 1 and pts_source.shape[2] == 2, f'Shape: {pts_source.shape}'
    
    if mask is not None:
        assert pts_source.shape[1] == mask.shape[0]
        pts_source = pts_source[:, mask, :]
        pts_target = pts_target[:, mask, :]
    
    n_points = pts_source.shape[1]
    matches = [cv2.DMatch(idx, idx, 0) for idx in range(n_points)]
    
    img_source_reshape = source2target_shape(img_source, img_target)
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(pts_target, pts_source, matches)
    
    # Code fore transforming points requires changing the order of function parameters
    # tps.estimateTransformation(source_pts_cloth_f, target_pts_texture_f, matches)
    # transformed_pts = tps.applyTransformation(source_pts_cloth_f)
    # print(transformed_pts)
    return tps.warpImage(img_source_reshape)


########################################################################################################################
def merge_sides_texture(texture_img_f: np.ndarray, texture_img_b: np.ndarray) -> np.ndarray:
    """Function for combining to texture sides into a single one.
    Note. Expects to have a texture with garment sides placed along vertical axis.

    Args:
        texture_img_f (np.ndarray): an image containing a texture of bot-halp (front) of shape (x, y, 3)
        texture_img_b (np.ndarray): an image containing a texture of top-halp (back) of shape (x, y, 3)

    Returns:
        np.ndarray: image with both texture sides of shape (x, y, 3)
    """
    assert texture_img_f.shape == texture_img_b.shape
    h, _, _ = texture_img_f.shape
    h_center = h // 2
    
    texture = texture_img_f.copy()
    texture[:h_center, :, :] = texture_img_b[:h_center, :, :]
    
    return texture

# By IT-JIM, 2022
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple
from loguru import logger

import numpy as np
import cv2


########################################################################################################################
def reparse_json(p_root_from: Path, f_json: str) -> None:
    """
    Function for parsing a directory of images and a single json into separate folders.

    Args:
        p_root_from (Path): path to root folder
        f_json (str): name of .json file at the folders
    """
    p_json = p_root_from / f_json
    with open(str(p_json), 'r') as fstream:
        data = json.load(fstream)
    
    for img_meta in data:
        p_img_from = p_root_from / img_meta["file_name"]
        name = Path(p_img_from).stem
        
        p_root_to = p_root_from / name
        p_root_to.mkdir(exist_ok=True, parents=True)
        
        p_img_to = p_root_to / img_meta["file_name"]
        logger.info(f'Moving: {p_img_from} -> {p_img_to}')
        shutil.move(p_img_from, p_img_to)
        
        p_json_to = p_root_to / f'{name}.json'
        logger.info(f'Copy to json: {p_json_to}')
        with open(str(p_json_to), "w") as outfile:
            json.dump(img_meta, outfile)


########################################################################################################################
def process_keypoints(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes meta-dictionary of keypoints into required format: reshapes to (n, 2), 
    generates default mask (all ones).
    
    Args:
        data (Dict[str, Any]): a dictionary with metadata, including `keypoints`, 
        `category_id`, `mask` keys.

    Returns:
        Dict[str, Any]: dictionary with keypoints, category_id and mask
    """    
    assert 'keypoints' in data.keys()
    assert 'category_id' in data.keys()
    
    kp = np.asarray(data['keypoints']).reshape(1, -1, 2).astype(int)
    mask = np.asarray(data["mask"]) if "mask" in data else np.ones((kp.shape[1]))
    mask = mask.astype(bool)
    
    return {'keypoints' : kp, 'category_id' : data['category_id'], "mask" : mask}


########################################################################################################################
def load_keypoints(p_kp: Path) -> Dict[str, Any]:
    """
    Loads keypoints from .json file, reshapes to (n, 2), generates default mask (all ones).
    Args:
        p_kp (Path): path to .json file

    Returns:
        Dict[str, Any]: dictionary with keypoints, category_id and mask
    """
    with open(str(p_kp), 'r') as fstream:
        data = json.load(fstream)
    
    return process_keypoints(data)


########################################################################################################################
def load_img_pts(p_img: Path, p_pts: Path, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Loads points image and corresponding points. Was written for "poc" of TSP-warping 
    with custom data and manually annotated points. 

    Args:
        p_img (Path): path to image
        p_pts (Path): path to .npy file containing points

    Returns:
        Tuple[np.ndarray, np.ndarray]: image array, and points reshaped to (1, n, 2)
    """
    img = cv2.imread(str(p_img))
    if p_pts.exists():
        pts = np.load(p_pts)
    else:
        pts = calc_points(img.copy(), save_path=p_pts, **kwargs)

    pts = pts.astype(np.float32).reshape(1, *pts.shape)
    return img, pts


########################################################################################################################
def calc_points(image: np.ndarray, coeff: float = 0.4, 
                save_path: str = None, color=(255, 0, 0), 
                radius=3, thickness=5, fontscale=5) -> np.ndarray:
    """Function for collecting points via interaction with opencv window.
    If `save_path` -> collected points will be saved in .npy.

    Args:
        image (np.ndarray): image
        coeff (float, optional): to multiply image size. Defaults to 0.4.
        save_path (str, optional): path where to save points in .npy. Defaults to None.
        color (tuple, optional): color of debug text. Defaults to (255, 0, 0).
        radius (int, optional): points size. Defaults to 3.
        thickness (int, optional): thickness of text near points. Defaults to 5.
        fontscale (int, optional): fontscale of text near points. Defaults to 5.

    Returns:
        np.ndarray: collected points
    """
    points_list = []
    
    def on_click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = int(x / coeff), int(y / coeff)
            print(f'x: {x}, y: {y }')
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.putText(image, str(on_click_event.idx + 1), (x + 15, y + 15), 1, fontscale, color=color, thickness=thickness)
            cv2.imshow('image', cv2.resize(image, None, fx=coeff, fy=coeff))
            points_list.append((x, y))
            on_click_event.idx += 1
    
    on_click_event.idx = 0
    cv2.imshow('image', cv2.resize(image, None, fx=coeff, fy=coeff))
    cv2.setMouseCallback('image', on_click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    points_list = np.array(points_list)
    if save_path:
        np.save(save_path, points_list)

    return points_list

########################################################################################################################
def find_suff(p_mesh: Path, suff: str = '.mtl') -> Tuple[Path, bool]:
    """Finds file in the folder with a suffix `suff`.
    
    Args:
        p_mesh (Path): path to the directory
        suff (str): suffix of a target file. Defaults to '.mtl'.
    
    Returns:
        Tuple[Path, bool]: found path and result of success
    """
    for f in p_mesh.iterdir():
        if f.suffix == suff:
            return f, True
    
    return None, False


########################################################################################################################
def change_mtl_mat(p_mtl: Path, mat_name: str) -> bool:
    """Changes content of .mtl file, namely a path to texture image.
    
    Args:
        p_mtl (Path): path to .mtl file
        mat_name (Path): name of a target image file

    Returns:
        bool: `True` if changes were made, `False` otherwise
    """
    assert p_mtl.suffix == '.mtl'
    success = False
    
    with open(p_mtl, 'r') as fstream:
        lines = fstream.readlines()
        
        for idx, line in enumerate(lines):
            words = line.split(' ')
            
            if words[0] == "map_Kd":
                lines[idx] = f"map_Kd {mat_name}\n"
                success = True
                break
    
    if success:
        with open(p_mtl, 'w') as fstream:
            fstream.writelines(lines)
        
    return success


########################################################################################################################
def print_it(arr: np.ndarray, name: str = '') -> None:
    """Debug np.ndarray function.

    Args:
        arr (np.ndarray): array
        name (str, optional): name. Defaults to ''.
    """
    print(f'{name}: shape {arr.shape}, dtype {arr.dtype}, max {arr.max()}, min {arr.min()}')


########################################################################################################################
def show_meta(meta: Dict[str, Any]) -> None:
    """Function for debug meta-dictionary.

    Args:
        meta (Dict[str, Any]): a dictionary with metadata.
    """
    print('---------------------------------------')
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            print_it(v, k)
        else:
            print(f'{k} : {v}')

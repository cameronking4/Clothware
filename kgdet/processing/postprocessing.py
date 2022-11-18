import cv2
import numpy as np
import mmcv
import os
from pathlib import Path
import sys

sys.path.insert(0, '../kgdet/mmdetection/tools/')
import tools.custom_test as kgdet

# keys - category_id, values - points to swap
RELATED_POINTS = {
    1: {
        2: 6,
        3: 5,
        7: 25,
        8: 24,
        9: 23,
        10: 22,
        11: 21,
        12: 20,
        13: 19,
        14: 18,
        15: 17
    },
    8: {
        1: 3,
        4: 14,
        5: 13,
        6: 12,
        7: 11,
        8: 10
    }
}


def oriented_right(keypoints: list, category_id: int, exp_side: str):
    """
    Returns true if provided keypoints match expected size
    """
    # from keypoints demonstration on DeepFashion2 we know, that first point of garments 
    # is always to the left of the last, except for outwear categories 3 and 4
    if category_id in [3, 4]:
        if exp_side == 'front':
            return keypoints[0] < keypoints[49]
        elif exp_side == 'back':
            return keypoints[0] > keypoints[49]
    # for everithing else we can use comparison of the first and the last kepoint X coord
    else:
        if exp_side == 'front':
            return keypoints[0] < keypoints[-2]
        elif exp_side == 'back':
            return keypoints[0] > keypoints[-2]


def plot_keypoints(image: np.array, kp_list: list):
    image = image.copy()
    
    kp_list_len = int(len(kp_list) / 2)

    if image.shape[0] > image.shape[1]:
        r = int(image.shape[0]/100)
    else:
        r = int(image.shape[1]/100)

    # 2 loops so not to display dots on top of the numbers
    for n in range(kp_list_len):
        x_kp = kp_list[2 * n]
        y_kp = kp_list[2 * n + 1]
        cv2.circle(image, (int(x_kp), int(y_kp)), r, (93, 13, 120), r)
    
    for n in range(kp_list_len):
        x_kp = kp_list[2 * n]
        y_kp = kp_list[2 * n + 1]
        cv2.putText(image, str(n + 1), (int(x_kp), int(y_kp)), cv2.FONT_HERSHEY_SIMPLEX, r/4, (120,255,0), int(r/7), 1)

    cv2.imshow('window_name', image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


def swap_points(kp_list: list, cat_id: int, visualize=False, image=None, to_reshape=True):
    """
    Input: 
        kp_list - list of keypoints stored in result_json['keypoints']
        cat_id - garment category id
        visualize - if you want to visualize keypoints before and after then set to True
        image - needed only for visualization
    """
    if visualize:
        # plot kp image before
        plot_keypoints(image, kp_list)

    # convert list to array with X, Y coordinates
    kp_array = np.array(kp_list).reshape(int(len(kp_list) / 2), 2)
    points_to_swap = RELATED_POINTS[cat_id]

    for old_idx in points_to_swap:
        new_idx = points_to_swap[old_idx]
        # -1 because points in DeepFashion2 starts at 1
        kp_array[[old_idx - 1, new_idx - 1]] = kp_array[[new_idx - 1, old_idx - 1]]
    
    new_kp_list = list(kp_array.flatten())

    if visualize:
        # plot kp image after
        plot_keypoints(image, new_kp_list)

    return np.asarray(new_kp_list).reshape(1, -1, 2).astype(int) if to_reshape else new_kp_list


def process_kgdet_side(kp_list: list,
                       cat_id: int,
                       side: str,
                       image: np.array = None,
                       visualize: bool = True,
                       to_reshape=True):
    """
    Demo function to visualize swapping results. Requires image
    """
    is_oriented = oriented_right(kp_list, cat_id, side)
    keypoints = kp_list if is_oriented else swap_points(kp_list, cat_id, visualize=visualize, 
                                                        image=image, to_reshape=to_reshape)

    return is_oriented, keypoints


if __name__ == "__main__":
    path_to_dataset = '../../input/shirt' # path to data
    cat_id = 1
    # let's choose back side image, which classified as front
    image_path = os.path.join(path_to_dataset, 'shirt6_b.jpg')
    # get our results
    
    cfg = mmcv.Config.fromfile('../configs/kgdet_moment_r50_fpn_1x-demo.py') # default path to config
    cfg.template_json = '../configs/dataset_template.json'
    cfg.checkpoint_path = '../../checkpoints/KGDet_epoch-12.pth'
    cfg.data.test.ann_file = os.path.join(path_to_dataset, 'dataset.json') # path to save anootation file
    cfg.data.test.img_prefix = path_to_dataset 

    kgdet.create_dataset_annotations(cfg)

    full_kp_json = kgdet.get_all_keypoints(cfg)[1]

    results =  kgdet.prune_json(cfg, full_kp_json, cat_id = cat_id)

    for result in results:
        if result['file_name'] == Path(image_path).name:
            image = cv2.imread(image_path)
            break

    process_kgdet_side(result['keypoints'], cat_id, 'back', image=image)
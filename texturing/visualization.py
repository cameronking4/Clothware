# By IT-JIM, 2022
import cv2
import numpy as np

from pathlib import Path


########################################################################################################################
def demo_points(image: np.ndarray, points: np.ndarray, name: str = '', verbose=True, save_path: Path = None,
                coef: float = 1., color=(255, 0, 0), radius=15, thickness=15, fontscale=10):
    assert len(points.shape) == 3
    assert points.shape[0] == 1 and points.shape[2] == 2, f'Shape: {points.shape}'
    
    n_points = points.shape[1]
    image_copy = image.copy()
    
    for i in range(n_points):
        pt = points[:, i, :]
        x, y = pt.ravel().astype(int)
        cv2.circle(image_copy, (x, y), radius, color, -1)
        cv2.putText(image_copy, str(i + 1), (x + 10, y), 1, fontscale, color=color, thickness=thickness)
    
    image_copy = cv2.resize(image_copy, None, fx=coef, fy=coef)
    
    if verbose:
        cv2.imshow(name, image_copy)
        # cv2.waitKey(0)
        # cv2.destroyWindow(name)
    
    if save_path is not None:
        f_out = save_path / f'{name}.jpg'
        cv2.imwrite(str(f_out), image_copy)

import os, sys
import os.path as osp
import shutil
import tempfile
import numpy as np
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import contextlib
from loguru import logger


import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    for i in tqdm(range(len(data_loader))):
        data = list(data_loader)[i]
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def create_dataset_annotations(cfg):
    dataset_path = cfg.data.test.img_prefix
    path_demo_json = cfg.template_json
    new_dataset_json = cfg.data.test.ann_file

    images_paths = []
    for path in Path(dataset_path).rglob('*.*g'):
        images_paths.append(str(path))

    # print(path_demo_json)
    with open(path_demo_json, 'r') as demo_file:
        demo_json = json.load(demo_file)

    # we'll copy general keys and its content ('info', 'licenses', 'categories')
    # and create 'images' and 'annotations' from our dataset

    images_key_list = []
    annotations_key_list = []

    for img_num, image_path in enumerate(images_paths):
        image = cv2.imread(image_path)
        image_info = {
            'coco_url': '',
            'date_captured': '',
            'file_name': os.path.basename(image_path),
            'flickr_url': '',
            'id': img_num + 1,
            'license': 0,
            'width': image.shape[1],
            'height': image.shape[0]
            }
        images_key_list.append(image_info)

        annotations_info = {
            'area': 0,
            'bbox': 0,
            'category_id': 0,
            'id': 0,
            'pair_id': 0,
            'image_id': img_num + 1,
            'iscrowd': 0,
            'style': 0,
            'num_keypoints': 0,
            'keypoints': 0,
            'segmentation': 0
            }
        
        annotations_key_list.append(annotations_info)

    demo_json['annotations'] = annotations_key_list
    demo_json['images'] = images_key_list

    with open(new_dataset_json, 'w') as fp:
        json.dump(demo_json, fp)


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def kpt2json(dataset, results):
    bbox_json_results = []
    kpt_json_results = []
    num_digits = 4
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        if len(results[idx]) == 3:
            det, score, kpt = results[idx]
        else:
            continue
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = [
                    round(value, num_digits) for value in xyxy2xywh(bboxes[i])
                    ]
                data['score'] = round(float(bboxes[i][4]), num_digits)
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # kpt results
            kpts = kpt[label]
            for i in range(kpts.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['keypoints'] = np.round(
                    kpts[i].astype(np.float64), num_digits).tolist()
                data['score'] = round(float(bboxes[i][4]), num_digits)  # TODO
                data['category_id'] = dataset.cat_ids[label]
                kpt_json_results.append(data)
    return bbox_json_results, kpt_json_results


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def get_all_keypoints(cfg):
    os.environ['RANK'] = str(0)
    os.environ['LOCAL_RANK'] = str(0)
    
    # cfg = mmcv.Config.fromfile(path_to_config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = False
    # init_dist('pytorch', **cfg.dist_params)

    # build the dataloader
    with nostdout():
        dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, cfg.checkpoint_path, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
        return kpt2json(dataset, outputs)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader)
        return kpt2json(dataset, outputs)


def get_start_end(cat):
    nodes = [0, 25, 58, 89, 128, 143, 158, 168, 182, 190, 256, 275, 294]
    return nodes[cat - 1], nodes[cat]


def prune_json(cfg, results: list, cat_id: int = None):

    # open ds annotations json
    with open(cfg.data.test.ann_file, 'r') as ds_json:
        dataset_json = json.load(ds_json)

    new_json_list = []

    # leave only results with chosen category id
    if cat_id:
        for res_dict in results:
            if res_dict['category_id'] == cat_id:
                new_json_list.append(res_dict)
    else:
        new_json_list = results

    rec_to_leave = []
    for i in range(1, len(dataset_json['images']) + 1):
        ids = []
        for record_it, record in enumerate(new_json_list):
            if record['image_id'] == i:
                ids.append(record_it)
        
        if len(ids) > 0:
            j_high = 0
            highest_score = new_json_list[ids[j_high]]['score']
            for j in range(1, len(ids)):
                if new_json_list[ids[j]]['score'] > highest_score:
                    highest_score = new_json_list[ids[j]]['score']
                    j_high = j    
            rec_to_leave.append(new_json_list[ids[j_high]])

    new_json_list = [e for e in new_json_list if e in rec_to_leave]

    for image_res in new_json_list:
        start, end = get_start_end(image_res['category_id'])
        new_kp_list = []

        for n in range(start, end):
            for i in range(2):
                new_kp_list.append(image_res['keypoints'][3 * n + i])
        image_res['keypoints'] = new_kp_list
        
        for image_info in dataset_json['images']:
            if image_res['image_id'] == image_info['id']:
                image_res['file_name'] = image_info['file_name']
    
    return new_json_list


def get_keypoints(path_to_dataset, class_id: int = 0):
    cfg = mmcv.Config.fromfile('./configs/kgdet_moment_r50_fpn_1x-demo.py') # default path to config
    cfg.data.test.ann_file = os.path.join(path_to_dataset, 'dataset.json') # path to save anootation file
    cfg.data.test.img_prefix = path_to_dataset 

    create_dataset_annotations(cfg)

    full_kp_json = get_all_keypoints(cfg)[1]

    return prune_json(cfg, full_kp_json, cat_id = class_id)
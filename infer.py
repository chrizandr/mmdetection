from mmdet.apis import init_detector, inference_detector
import os
import argparse
from tqdm import tqdm
import pdb

import sys

sys.path.insert(1, '/home/chrizandr')

from yolov3.annotations.annot.annot_utils import CVAT_Track, CVAT_annotation


def run(cfg, weights, data, annot_file):
    dataset = data
    images = os.listdir(dataset)
    config_file = cfg
    checkpoint_file = weights

    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image

    annotation = CVAT_annotation()

    for img in tqdm(images):
        results = inference_detector(model, os.path.join(dataset, img))
        person = results[0]
        frame = int(img.strip(".jpg").strip("image"))
        track = CVAT_Track(frame)
        for pred in person:
            xtl, ytl, xbr, ybr = pred[0:4]
            conf = pred[4]
            if conf > 0.3:
                track.create_bbox(frame-1, xtl, ytl, xbr, ybr, conf=conf)
        if len(track.bboxes) != 0:
            annotation.insert_track(track)

    annotation.build(annot_file, add_conf=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help='cfg file path')
    parser.add_argument('--weights', type=str, default='checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help='path to weights file')
    parser.add_argument('--annot_file', type=str, default="annot.xml", help='Annotation output file in YOLO detection format')
    parser.add_argument('--data', type=str, default='/ssd_scratch/cvit/chrizandr/images/', help='coco.data file path')
    opt = parser.parse_args()
    run(opt.cfg, opt.weights, opt.data, opt.annot_file)

# configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
# checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# configs/retinanet/retinanet_r50_fpn_1x_coco.py
# checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth

# soccerdb_config_retina.py
# checkpoints/retinanet_x101_64x4d_fpn_1x.pth

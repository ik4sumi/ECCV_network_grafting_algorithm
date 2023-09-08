"""Use YOLOv3 to prepare train and val data.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse
import os
import glob
import shutil

import numpy as np

from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.registry import VISUALIZERS  

import cv2

parser = argparse.ArgumentParser()

parser.add_argument("--train", action="store_true")
parser.add_argument("--val", action="store_true")

args = parser.parse_args()

# target classes
#  target_classes = ["bus", "car", "person", "stop_sign",
#                    "traffic_light", "truck"]
target_classes = [2]

# some helping paths
data_root = os.path.join(os.environ["HOME"], "data", "mvsec")
data_root="../../../../../tsukimi/datasets/MVSEC/Detection"

train_data = os.path.join(data_root, "train_data")
val_data = os.path.join(data_root, "val_data_3")
train_data = os.path.join(data_root, "outdoor_night3_data")

train_final_data = os.path.join(data_root, "train_data_final")
val_final_data = os.path.join(data_root, "val_data_3_final")

if args.train is True:
    source_dir = train_data
    target_dir = train_final_data
elif args.val is True:
    source_dir = val_data
    target_dir = val_final_data
    #  target_dir = val_data

# build ground truth folder
gt_folder = os.path.join(target_dir, "groundtruths")
if not os.path.isdir(gt_folder):
    os.makedirs(gt_folder)

# get the list of the file
file_list = sorted(glob.glob("{}".format(source_dir)+"/*.npz"))

# building model
config_file = os.path.join(
   "/workspace","ECCV_network_grafting_algorithm","mmdetection",
    "configs", "rtmdet",
    "rtmdet_s_8xb32-300e_coco.py")
checkpoint_file = os.path.join(
    "/workspace","ECCV_network_grafting_algorithm", "mmdetection", "checkpoint",
    "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth")

config_file='htc_x101-64x4d-dconv-c3-c5_fpn_ms-400-1400-16xb1-20e_coco.py'
checkpoint_file='htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth'


# build model
model = init_detector(config_file, checkpoint_file, device="cuda:0")
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

print("Model built")

#class_names = model.CLASSES
score_thr = 0.8

for file_path in file_list:
    print("Examining File: {}".format(file_path))
    file_data = np.load(file_path)
    # split the file name
    base_name = os.path.basename(file_path)[:-4]

    candidate_img = file_data["img"][..., np.newaxis]
    candidate_img = np.concatenate(
        (candidate_img, candidate_img, candidate_img), axis=2)
    #  candidate_img = file_data["img"]

    img = mmcv.imread(candidate_img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    # detect image on the network
    detect_result = inference_detector(model, candidate_img)
    print(detect_result.pred_instances.scores)
    visualizer.add_datasample(
        'result',
        img,
        data_sample=detect_result,
        draw_gt=False,
        show=False)

    #bboxes = np.vstack(detect_result["bboxes"])
    bboxes=detect_result.pred_instances.bboxes
    '''
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(detect_result["bboxes"])
    ]
    labels = np.concatenate(labels)
    '''
    labels=detect_result.pred_instances.labels
    # filter the final boxes
    scores = detect_result.pred_instances.scores
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    bboxes=bboxes.tolist()
    for box_id in range(len(bboxes)):
        bboxes[box_id]=[int(bboxes[box_id][0]),int(bboxes[box_id][1]),int(bboxes[box_id][2]),int(bboxes[box_id][3])]
    print(labels)
    print(bboxes)
    #use cv2 draw bounding boxes in bboxes on img and svae it in current folder
    for box_id in range(len(bboxes)):
        if labels[box_id] in target_classes:
            cv2.rectangle(img,(bboxes[box_id][0],bboxes[box_id][1]),(bboxes[box_id][2],bboxes[box_id][3]),(0,0,255),2)
    cv2.imwrite(base_name+".jpg",img)

    # if nothing detected, skip
    if len(bboxes) == 0 and len(labels) == 0:
        print("No boxes detected. Skipping {}".format(file_path))
        continue

    # check if there are boxes in the target class
    write_flag = False
    for box_id in range(bboxes.shape[0]):
        if labels[box_id] in target_classes:
            write_flag = True
            break

    if write_flag is False:
        print("No writable boxes. Skipping {}".format(file_path))
        continue

    # if at least one object detected
    # copy the file to the final folder and write the ground truth text
    shutil.copyfile(file_path, os.path.join(target_dir, base_name+".npz"))

    # write ground truth
    gt_file = open(os.path.join(gt_folder, base_name+".txt"), "w+")
    for box_id in range(len(bboxes)):
        if labels[box_id] in target_classes:
            gt_file.write(
                "{} {} {} {} {}\n".format(
                    #  "tvmonitor",
                    2,
                    bboxes[box_id, 0],
                    bboxes[box_id, 1],
                    bboxes[box_id, 2],
                    bboxes[box_id, 3]))
    gt_file.close()

    print("Copied file {} and saved groundtruth".format(file_path))

print("Results dumped")

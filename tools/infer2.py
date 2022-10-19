import argparse
from genericpath import isdir, isfile
import os
from tkinter import image_names
import cv2
import numpy as np
import torch
import time

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('input', help='input image folder')
    parser.add_argument('output', help='output image folder')

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp='out.jpg'):
    font_scale = 0.3
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        if bbox_int[2] - bbox_int[0] > 20:
            label_text = class_names[
                label] if class_names is not None else f'cls {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    imwrite(img, outfp)

total_tm = 0
total_cnt = 0

def infer(config_file, imgname, out_imgname):    
    global total_cnt
    global total_tm

    total_cnt = total_cnt + 1
    start = time.time()
    cfg = Config.fromfile(config_file)
    
    class_names = cfg.class_names

    engine, data_pipeline, device = prepare(cfg)

    data = dict(img_info=dict(filename=imgname), img_prefix=None)

    data = data_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if device != 'cpu':
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data
    
    result = engine.infer(data['img'], data['img_metas'])[0]
    plot_result(result, imgname, class_names, out_imgname)
    end = time.time()
    tm = end - start
    print(imgname + ' ' + str(tm) + ' s')
    total_tm += tm


def infer_folder(config_file, dir, out_dir):
    for file_name in os.listdir(dir):
        f = os.path.join(dir, file_name)
        if os.path.isfile(f):
            #print(out_dir + ' ' + file_name)
            f2 = os.path.join(out_dir, file_name)
            infer(config_file, f, f2)
        elif os.path.isdir(f):
            out_dir2 = os.path.join(out_dir, file_name)
            if not os.path.exists(out_dir2):
                os.mkdir(out_dir2)            
            infer_folder(config_file, f, out_dir2)


if __name__ == '__main__':
    args = parse_args()
    cfg = args.config
    input_dir = args.input
    output_dir = args.output
    
    infer_folder(cfg, input_dir, output_dir)

    print("Total time: " + str(total_tm) + ' s')
    print("Total count: " + str(total_cnt))
    print("Average time: " + str(total_tm / total_cnt) + ' s')

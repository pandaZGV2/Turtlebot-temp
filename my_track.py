import argparse
import cv2
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path, PosixPath
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadPilAndNumpy, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker

@torch.no_grad()
def process_images(
    rgb_image,
    yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
    tracker_list=[],
    outputs=[],
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    show_vid=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    save_trajectories=False,  # save trajectories for each track
    save_vid=False,  # save confidences in --save-txt labels
    nosave=False,  # do not save images/videos
    classes=[0],  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs' / 'track',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=2,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    hide_class=False,  # hide IDs
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False):

    source = rgb_image
    
    # Load model
    device = select_device(device)
    is_seg = False
    model = AutoBackend(PosixPath('yolov8n.pt'), device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    dataset = LoadPilAndNumpy(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
        )
    bs = len(dataset)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup


    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)
        
        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            p, im0, _ = path[i], im0s[i].copy(), dataset.count
        curr_frames[i] = im0

        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

        if det is not None and len(det):
            if is_seg:
                shape = im0.shape
                # scale bbox first the crop masks
                if retina_masks:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                    masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                else:
                    masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
            else:
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

        # Print results
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        
        # pass detections to strongsort
        with dt[3]:
            outputs[i] = tracker_list[i].update(det.cpu(), im0)
        
        if len(outputs[i]) > 0:
            if is_seg:
                # Mask plotting
                annotator.masks(
                    masks[i],
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                    255 if retina_masks else im[i]
                )
                print(masks[i].shape)
                # exit(0)
            
            for j, (output) in enumerate(outputs[i]):
                bbox = output[0:4]
                id = output[4]
                cls = output[5]
                conf = output[6]
            
                c = int(cls)  # integer class
                id = int(id)  # integer id
                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                    (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                color = colors(c, True)
                annotator.box_label(bbox, label, color=color)

            im0 = annotator.result()

            return im0
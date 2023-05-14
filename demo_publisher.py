import os
from pathlib import Path
import sys
import cv2
import numpy as np
from trackers.multi_tracker_zoo import create_tracker
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
import my_track

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


video1 = cv2.VideoCapture('/home/luke/Dev/RRC/Smart-Wheelchair/P3DX/src/p3dx_tracking/data/uid_vid_00009.mp4')
video2 = cv2.VideoCapture('/home/luke/Dev/RRC/Smart-Wheelchair/P3DX/src/p3dx_tracking/data/uid_vid_00009.mp4')

fps = int(video1.get(cv2.CAP_PROP_FPS))
width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video_name = 'concatenated_video.mp4'
codec = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(out_video_name, codec, fps, (width*2, height))

device = select_device('')
reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'
half = True

# Create as many strong sort instances as there are video sources
tracker_list = []
for i in range(1):
    tracker = create_tracker('deepocsort', 'trackers/deepocsort/configs/deepocsort.yaml', reid_weights, device, half)
    tracker_list.append(tracker, )
    if hasattr(tracker_list[i], 'model'):
        if hasattr(tracker_list[i].model, 'warmup'):
            tracker_list[i].model.warmup()
outputs = [None]


while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    
    if not ret1 or not ret2:
        break
    
    im0 = my_track.process_images(
        rgb_image=frame1,
        tracker_list=tracker_list,
        outputs=outputs
        )

    concatenated_frame = np.hstack((frame1, frame2))
    out_video.write(concatenated_frame)
    
    # cv2.imshow('Concatenated Video', cv2.resize(concatenated_frame, (concatenated_frame.shape[1]//2, concatenated_frame.shape[0]//2), interpolation=cv2.INTER_AREA))
    cv2.imshow('Concatenated Video', im0)
    cv2.waitKey(1)

video1.release()
video2.release()
out_video.release()

cv2.destroyAllWindows()

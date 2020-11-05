from sort import Sort
from detector_yolov5_trt import Yolov5DetectorTrt
import cv2
import numpy as np


def sort_yolov5_video(yolo_path, video_path):
    # model and tracker initialize
    yolo_detector_trt = Yolov5DetectorTrt(yolo_path)
    yolo_detector_trt.conf = 0.3
    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

    # start proc
    cap = cv2.VideoCapture(video_path)
    count = 0
    jump = 1

    while 1:
        rval, frame = cap.read()
        if not rval:
            break
        if count % jump != 0:
            continue

        detections = yolo_detector_trt.infer_cv_img(frame)
        detections = detections[detections[:, 5] == 0]   # filter cls_id == 0
        detections = detections[:, 0:4]  # x, y, x2, y2, score

        trackers = mot_tracker.update(detections)
        for tracker in trackers:
            cv2.putText(frame, str(int(tracker[4]), (int(tracker[0]), int(tracker[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            cv2.rectangle(frame, (int(tracker[0]), int(tracker[1])), (int(tracker[2]), int(tracker[3])), (0, 255, 0), 2)

        count += 1
        cv2.namedWindow('frame', 0)
        cv2.imshow('frame', frame)
        wait_key = cv2.waitKey(0)
        if wait_key == 27:
            break


if __name__ == '__main__':
    print('Start Proc...')
    yolo_path = '/home/liyongjing/Egolee/programs/yolov5-master/weights/best_sim.onnx'
    video_path = '/home/liyongjing/Egolee/data/test_person_car/001.avi'

    sort_yolov5_video(yolo_path, video_path)
    print('End Proc...') 

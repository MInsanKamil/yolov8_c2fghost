import cv2
import argparse
from ultralytics.models.yolo.model import YOLO
# from ultralytics import YOLO
import supervision as sv #must be version 0.3.0
import time 
from picamera2 import Picamera2

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "YOLOv8 Live")
    parser.add_argument(
    "--webcam-resolution", 
    default=(1366, 728),
    nargs=2,
    type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = Picamera2(0)
    cap.configure(cap.create_preview_configuration(main={"format": "XRGB8888", "size": (frame_width, frame_height)}))
    cap.start()
    model = YOLO("best.pt")
    prev_frame_time = 0
    new_frame_time = 0

    box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
    )

    

    while True:
        new_frame_time = time.time() 
        frame = cap.capture_array()

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        result = model(frame, conf = 0.5, agnostic_nms = True)[0]
        detections = sv.Detections.from_ultralytics(result)
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id,_
        in detections
    ]

        frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels = labels
        )
        cv2.putText(frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        cv2.imshow('yolov8', frame)

        if (cv2.waitKey(30) == 27): #escape key
            break

        print(frame.shape)

if __name__ == "__main__":
    main()

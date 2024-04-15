import cv2
import argparse
from ultralytics.models.yolo.model import YOLO
import supervision as sv #must be version 0.3.0


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
    byte_tracker = sv.ByteTrack(match_thresh=0.95, lost_track_buffer = 30)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO("best_indoor.pt")
    

    box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
    )

    

    while True:
        ret, frame = cap.read()
        if ret:
            result = model(frame, conf = 0.5, agnostic_nms = True)[0]
            detections = sv.Detections.from_ultralytics(result)
            # detections = detections[detections.class_id !=0]
            detections = byte_tracker.update_with_detections(detections)
            # print(detections)
            labels = [
                f"Id:{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id,_
            in detections
        ]

            frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels = labels
            )
            
            cv2.imshow('yolov8', frame)

            if (cv2.waitKey(30) == 27): #escape key
                break

            print(frame.shape)

if __name__ == "__main__":
    main()

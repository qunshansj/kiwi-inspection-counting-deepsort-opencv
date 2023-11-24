import numpy as np
import tracker
from detector_CPU import Detector
import cv2

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.detector = Detector()
        self.tracker = tracker

    def process_video(self):
        # Process video frames
        capture = cv2.VideoCapture(self.video_path)
        while True:
            _, im = capture.read()
            if im is None:
                break

            # Detect objects in the frame
            bboxes = self.detector.detect(im)

            # Track objects in the frame
            list_bboxs = self.tracker.update(bboxes, im)

            # Draw bounding boxes and perform line collision detection
            output_image_frame = self.tracker.draw_bboxes(im, list_bboxs, line_thickness=1)

            # Display the processed frame
            cv2.imshow("Output", output_image_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
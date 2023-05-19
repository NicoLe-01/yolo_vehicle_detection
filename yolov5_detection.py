import yolov5
import torch
import numpy as np
import supervision as sv

model = torch.load('best.pt')

# initiate polygon zone

# extract video frame
generator = sv.get_video_frames_generator("testing_yolo_3.mp4")
iterator = iter(generator)
frame, _ = next(iterator)

# detect
results = model(frame, size=1280)
detections = sv.Detections.from_yolov5(results)
detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

# annotate
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
frame = box_annotator.annotate(scene=frame, detections=detections)

sv.show_frame_in_notebook(frame, (16, 16))
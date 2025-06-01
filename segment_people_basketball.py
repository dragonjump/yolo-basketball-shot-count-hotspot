import cv2
import numpy as np
from ultralytics import solutions

cap = cv2.VideoCapture("video/basketball.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("combined_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Define region for basketball counting (adjust as needed)
region_points =  [(839, 373), (915, 373), (915, 475), (839, 475)]

# Initialize instance segmentation for people
isegment = solutions.InstanceSegmentation(
    show=False,  # We'll display the combined result instead
    model="yolo11x-seg.pt",
    classes=[0],  # person
    conf=0.011,
)

# Initialize region counter for basketballs
counter = solutions.ObjectCounter(
    show=True,  # We'll display the combined result instead
    conf=0.15,
    show_in=True,
    show_out=True,
    region=region_points,
    # model="yolo12x.pt",  # COCO model
    model="best.pt",  # COCO model from yolo12x
    # classes=[32],  # sports ball
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Segment people
    seg_results = isegment(im0)
    seg_masked = seg_results.plot_im.copy()

    # Count basketballs in region
    count_results = counter(im0)
    combined = count_results.plot_im.copy()

    # Overlay segmentation mask (with transparency) onto the region counting frame
    alpha = 0.5  # transparency for segmentation mask
    cv2.addWeighted(seg_masked, alpha, combined, 1 - alpha, 0, combined)

    # Overlay region (red line) on seg_frame
    pts = np.array(region_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(combined, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Optionally, overlay basketball count info
    count_text = f"Basketballs in region: {count_results.in_count}"
    cv2.putText(combined, count_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Write combined frame
    video_writer.write(combined)

cap.release()
video_writer.release()
cv2.destroyAllWindows() 
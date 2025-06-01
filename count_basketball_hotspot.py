import cv2
import numpy as np

from ultralytics import solutions
from ultralytics import YOLO

cap = cv2.VideoCapture("video/basketball.mp4")
assert cap.isOpened(), "Error reading video file"
 
# Move region down 10% and left 10%
x_offset = int(-0.12 * cap.get(cv2.CAP_PROP_FRAME_WIDTH))
y_offset = int(0.1 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
region_points = [ (x - x_offset, y + y_offset) for (x, y) in [(950, 320), (1040, 320), (1040, 355), (950, 355)] ]
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_traffic_counting_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    show_in=True,
    show_out=True,
    conf=0.22, 
    # classes=[9,10],
    tracker="bytetrack.yaml",
    region=region_points,  # pass region points
    model="yolo11x-obb.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
 
)

# Print class names for the current OBB model
obb_model = YOLO("yolo11n-obb.pt")
print("OBB Model class names:")
for i, name in enumerate(obb_model.names):
    print(f"{i}: {name}")

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)
    print(results)  # access the output
    processed_frame = results.plot_im  # Check if this frame contains the drawn results
    # Draw the region as a red line
    pts = np.array(region_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(processed_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    video_writer.write(processed_frame)



cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
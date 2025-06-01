# Basketball Shot Detector and Human Player Segmentation

This project provides:

- **Basketball Shot Detector/Counter:**
  - Counts basketballs (or shots) entering a defined region in the video using YOLO-based object detection.
  - You can define a custom region (e.g., the hoop area) and the system will count how many times a basketball enters this region.
  - Supports both COCO models (for generic sports ball detection) and custom-trained models for basketballs.

- **Human Player Segmentation:**
  - Performs instance segmentation of human players in the video using YOLO11 segmentation models.
  - Only people are segmented (COCO class 0), but you can adjust to segment other classes if needed.

- **Combined Output:**
  - The script can overlay both the basketball region counting and the human segmentation mask on the same video output, so you can visualize both detections at once.

## Usage

- Place your video file in the `video/` directory.
- For basketball shot counting, adjust the region coordinates in the script to match your hoop or area of interest.
- If your video is black ball or not recognized, you need to train. See `training` folder. Otherwise, you may just run and download `yolo-11x.pt` or `yolo12px.pt`
- For human segmentation, make sure you have the YOLO11 segmentation model (e.g., `yolo11x-seg.pt`).
- Run the provided scripts (e.g., `segment_people_basketball.py`) to process your video and generate an output with both overlays.
 
## Requirements
- Python 3.1+
- OpenCV
- Ultralytics YOLO models (see scripts for model file names)

## Credits
- Built using Ultralytics YOLO11 models for detection and segmentation.

---

For more details, see comments in the scripts or ask for help!

 

## Demo  
See `demo.gif`
[demo](demo.gif)
![demo](demo.gif)

# More references
https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=gzaJQ2sGEPhP

## Setup 
1. Download `yolo12n.pt`
2.Search online how setup conda. 
Once you setup, create a conda environment with Python 3.11:
```
conda create --name ultralytics-env python=3.11 -y
conda activate ultralytics-env
conda install -c pytorch pytorch torchvision torchaudio

conda deactivate ultralytics-env
```

You may adjust the parameter in the code accordingly. Do read the ultralytics doc.
Run this way.
 


```
(ultralytics-env) python segment_people_basketball.mp4
```
 

## Video credit 
https://www.youtube.com/watch?v=Mp6klx9oeZs&pp=ygUlY29weXJpZ2h0IGZyZWUgZHJvbmUgdmlldyBjYXIgdHJhZmZpYw%3D%3D 


## Detail Doc   
1. Object counting - https://docs.ultralytics.com/guides/object-counting/#objectcounter-arguments
2. Object counting - https://docs.ultralytics.com/tasks/obb/#visual-samples
3. Region counting - https://docs.ultralytics.com/guides/region-counting/
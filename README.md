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
1. Download `yolo11x-seg.pt` and `yolo12px.pt` and place them root folder.
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
 
3. Place your own basketball video in `/video/basketball.mp4`

4. If your basketball not detected as `sports ball` , see `training/README.md` to train.

```
(ultralytics-env) python segment_people_basketball.mp4
```
 

 
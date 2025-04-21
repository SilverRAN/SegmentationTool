# SegmentationTool
A simple, interactive segmentation tool based on SegmentAnything that can run locally.

## Guidance
### 1.Environment Setup
Clone the official repository of [Segment-Anything](https://github.com/facebookresearch/segment-anything).
    
    git clone https://github.com/facebookresearch/segment-anything.git

Then install the environment referring to the guidance of SAM.
### 2.Run script to start!

    python main.py --image_dir /Path/To/Image/Folder/ --output_dir /Path/To/Save/Masks/ --sam_checkpoint /Path/To/SAM/Checkpoint/

Then the script will load the image files in image_dir one by one and show them with an interactive window. Use mouse to drag a bounding box from the upperleft to the bottomright corner of your interested object, then press the "q" button on keyboard. The Segment-Anything model will take the bounding box and the image as input and predict a segmentation mask.

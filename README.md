# Tailor segment-anything-model (SAM) for cine CMR segmentation
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo based on an unpublished paper: <br />
*Cine Cardiac Magnetic Resonance Segmentation using Temporal-spatial Adaptation of Prompt-enabled Segment-Anything-Model: A Feasibility Study*<br />
Authors: Zhennong Chen, Sekeun Kim, Hui Ren, Sunghwan Kim, Siyeop Yoon, Quanzheng Li, Xiang Li<br />

**Citation**: TBD

## Description
We have proposed cineCMR-SAM, a customized SAM for cine CMR segmentation. It can segment a 2D slice (both SAX and LAX) across one cardiac cycle.<br />
The main contributions of cineCMR-SAM are as follows:<br />
(1) introduce a temporal-spatial attention mechanism in the origianl SAM ViT encoder to enable 2D+T segmentation.<br />
(2) enable text prompt (words: "LAX" or "SAX to specify the input view type) and box prompt (bounding box around the region)<br />
(3) fine-tuned on two public datasets (STACOM and ACDC) and validate on three different unseen datasets. (the fine-tuned weights is not released since the paper has not been published yet).<br />


## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- You can build your own docker from the folder ```docker/docker_sam```. <br />
- You need to download original SAM weights (sam_vit_h_4b8939.pth and sam_vit_b_01ec64.pth)!<br />
- (optional but recommend) if you don't want to train the text prompt encoder, you should install the CLIP model by ```pip install openai-clip```.<br />
- when clone this repo, you may want to change the folder name to ```cineCMR_SAM``` instead of ```cineCMR-SAM```.

### Data Preparation
- **CMR data**<br />
    - you want to prepare the SAX data as a 4D array [x,y,z,time_frame]. note time_frame is default to be 15 but you can choose your own.<br />
    - you want to prepare the LAX data as a 3D array [x,y,time_frame].<br />
    - Currently, our model only support x and y size equal to 128. different image sizes can be feasible but requires the modification of U-Net paths.<br />
    - I did not prepare the code to do so since it is very easy.<br />
    - For training, prepare pairs of image and the manual segmentation. For prediction, only image required. <br />

- **Patient list** <br />
    - Please prepare a patient list as the example ```HFpEF_Patient_List_training_testing-v2.xlsx```. this example corresponds to the dataset name set to ```HFpEF```. please write your own code to generate this.<br />
    - In my code we have dataset names as "STACOM", "ACDC", "HFpEF", "AS" and "MM"(see ```dataset/data_CMR_sax.py```), corresponding to different datatsets mentioned in the paer. please replace them by your own dataset.<br />

- **Text prompt features** <br />
    - we can easily use the CLIP model ```dataset/CMR/clip_extractor.ipynb``` to prepare the prompt embedding for text "SAX" (prompt for short-axis) and text "LAX" (prompt for long-axis). The embedding will be used in ```dataset/CMR/dataset_SAX.py``` and ```dataset/CMR/dataset_LAX.py```.

### Main
use ```train.py``` to train the model.  <br /> 
use ```predict.py``` to generate segmentation using trained model. The output will be 2D+T segmentation on each slice. <br /> 
For both ```train.py``` and ```predict.py```, the user should fill in the lines in "important parameters, fill in using your own" in the function ```get_args_parser```.<br /> 

### Additional guidelines 
special thank you to Dr. Sekeun Kim since our cineCMR-SAM is largely dependent on his [previous work](https://github.com/kimsekeun/MediViSTA-SAM).

Please contact zchen36@mgh.harvard.edu and chenzhennong@gmail.com for any further questions.




# Tailor segment-anything-model (SAM) for cine CMR segmentation
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo based on an published paper: <br />
*Cine Cardiac Magnetic Resonance Segmentation using Temporal-spatial Adaptation of Prompt-enabled Segment-Anything-Model: A Feasibility Study*<br />
[paper link](https://www.sciencedirect.com/science/article/pii/S1097664725000717)<br />
Authors: Zhennong Chen, Sekeun Kim, Hui Ren, Sunghwan Kim, Siyeop Yoon, Quanzheng Li, Xiang Li<br />

**Citation**: Chen, Zhennong, et al. "Cine cardiac magnetic resonance segmentation using temporal-spatial adaptation of prompt-enabled segment-anything-model: a feasibility study." Journal of Cardiovascular Magnetic Resonance (2025): 101909.

## Description
We have proposed cineCMR-SAM, a customized SAM for cine CMR segmentation. It can segment a 2D slice (both SAX and LAX) across one cardiac cycle.<br />
The main contributions of cineCMR-SAM are as follows:<br />
(1) introduce a temporal-spatial attention mechanism in the origianl SAM ViT encoder to enable 2D+T segmentation.<br />
(2) enable text prompt (words: "LAX" or "SAX to specify the input view type) and box prompt (bounding box around the region)<br />
(3) fine-tuned on two public datasets (STACOM and ACDC).<br />


## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- You can build your own docker from the folder ```docker```. The docker image can be built by ```./docker_build.sh```, after that the docker container can be built by ```./docker_run.sh```. The installed packages can be referred to ```dockerfile``` and ```requirements.txt``` <br />

### Data Preparation (we have examples available)
- **CMR data**<br />
    - you want to prepare the SAX data as a 4D array [x,y,time_frame,slice_num] saved as a nii file. in our study we sample 15 time frames as default. <br />
        - example data: check folder ```example_data/data/ID_0002```. <br />
    - you want to prepare the LAX data as a 3D array [x,y,time_frame]. As aforementioned, time frame is default to 15. <br />
        - example data: check folder ```example_data/data/ID_0085``` <br />
    - For training, prepare pairs of image and the manual segmentation. For prediction, only image required. <br />

- **Patient list** <br />
    - Please prepare a patient list (for SAX and for LAX) following my format in ```example_data/data/Patient_list/patient_list_sax.xlsx``` and ```example_data/data/Patient_list/patient_list_lax.xlsx```. My code ```dataset/build_CMR_datasets.py``` will read these to get patient list. This is how the model knows the paths of patient data <br />

- **Text prompt features** <br />
    - we can use the CLIP model ```dataset/CMR/clip_extractor.ipynb``` to prepare the prompt embedding for text "SAX" (prompt for short-axis) and text "LAX" (prompt for long-axis). we prepare the generated embedding for you in ```example_data/data/text_prompt_clip/sax.npy``` and ```example_data/data/text_prompt_clip/lax.npy```. <br />
    - The view type of the data should be easily defined by the metadata of the DICOM. <br />

- **Box prompt features** <br />
    - During training, we suppose not to prepare any box prompts manually. Instead, the box will be automatically generated from the manual segmentation of myocardium. Please refer to ```dataset/CMR/Dataset.py``` --> line ***bbox,_,_ = Data_processing.get_bbox_from_mask_all_volumes(processed_seg, annotation_frame_list)***.  <br />
    - During inference/prediction, you can manually define the box around the myocardium either on ED or ES frame or both. here we prepare examplar bounding box ```example_data/data/ID_0002/bounding_box.npy``` which saves the bounding box as a 4D array [f,s,2,4] where f is the number of cases, s is the slice num in each case, 2 refers to ED and ES, 4 refers to at each frame the definition of [xmin, ymin, xmax, ymax] of the bounding box. If you don't define the box, the model will just pass None as box prompt.  <br />

- **Original SAM model**
   - download from [this link](https://github.com/SekeunKim/MediViSTA?tab=readme-ov-file)

### Experiments
we have design our study into 5 steps, with each step having its own jupyter notebook.<br /> 
**step1: training**: use ```step1_train.ipynb```<br />
**step2: prediction**: use ```step2_predict.ipynb``` to generate segmentation . The output will be 2D+T segmentation on each slice. The saved file will be structured shown as ```example_data/models/cineCMR_sam_trial/predicts_sax or predicts_lax``` <br /> 


### Additional guidelines 
special thank you to Dr. Sekeun Kim since our cineCMR-SAM is largely dependent on his [previous work](https://github.com/SekeunKim/MediViSTA).

Please contact chenzhennong@gmail.com for any further questions.




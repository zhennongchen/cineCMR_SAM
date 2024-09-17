# Use Trained DL Model to Predict Segmentation and Cardiac Imaging Planes
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo based on a published paper: <br />
*Automated Cardiac Volume Assessment and Cardiac Long- and Short-Axis Imaging Plane Prediction from ECG-gated CT Volumes Enabled By Deep Learning*<br />
Authors: Zhennong Chen, Davis Vigneault, Marzia Rogolli, Francisco Contijoch<br />

**Citation**: Zhennong Chen, Marzia Rigolli, Davis Marc Vigneault, Seth Kligerman, Lewis Hahn, Anna Narezkina, Amanda Craine, Katherine Lowe, Francisco Contijoch, Automated cardiac volume assessment and cardiac long- and short-axis imaging plane prediction from electrocardiogram-gated computed tomography volumes enabled by deep learning, European Heart Journal - Digital Health, Volume 2, Issue 2, June 2021, Pages 311–322, https://doi.org/10.1093/ehjdh/ztab033

Check how to train a DL model to do segmentation + imaging plane prediction here: https://github.com/ucsd-fcrl/AI_chamber_segmentation_plane_re-slicing

## Description
We have developed a DL model to provide automatic, accurate and fast chamber segmentation (Left ventricle and Left atrium) + cardiac imaging planes re-slicing (two-chamber, three-chamber, four-chamber planes + a short-axis stack) from cardiac CT images. 

The purpose of this GitHub repo is to use trained DL models to predict chamber segmentation and planes on *new* CT cases.

This repo has two sets of scripts: **Main and Premier**.<br />
- **Main**: because the input of DL model has to be the under-sampled CT volumes (pixel_dim = 1.5mm), "Main" can return you the segmentations and planes in *low resolution*.<br />
    - You may have multiple trained models (trained on different datasets) acting like multiple human experts (some of which may make mistakes sometimes). Thus, the other important purpose of "Main" is to let you select which model gives you the adequate results.<br />
- **Premier**: turn the low resolution DL outputs into *high* resolution (pixel_dim = 0.625)


## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- You can build your own docker from provided dockerfile ```Dockerfile_cuda100_tensorflow```. 

### Data Preparation
- First of all, you need to prepare your trained models (i.e., by [this GitHub repo](https://github.com/ucsd-fcrl/AI_chamber_segmentation_plane_re-slicing)).<br />
    - you need to update the file paths of these trained models in ```Build_DL_model.py```, function "get_model_list".<br />
- Then, you should re-sample your CT image into uniform pixel dimension. For **Main** (low-res): re-sample to 1.5mm. For **Premier**: re-sample to 0.625mm.<br />
    - do it by runnning ```./tool_resample_by_c3d.sh```.<br />
- Last, set some default parameters for the experiments by running ```. ./defaults.sh```

### Main
Generate low-resolution segmentation and a cine movie of cardiac imaging planes (showing cardiac function across the cycle).<br />
- step 1: run the DL prediction to get (1) segmentation, (2) vectors used to generate the planes. script: ```main_step1.py```.<br />
- step 2: generate the cine movie of planes. script: ```main_step2.py```. <br />
- step 2b (optional): visually pick among all trained models you have which model gives you adequate results. script: ```main_step2b.py```. <br />
    - you can record yoru picks in a spreadsheet. see ```example_files/model_set_selection_example.xlsx``` <br />
- step 2c: (optional): generate *adequate* plane movie by the model you picked in step 2b. script: ```main_step2c.py``` <br />

### Premier
Generate high resolution segmentation and plane movie (showing cardiac function with more details).<br />
#### Plane movie
Notice!: it's not done by simple upsampling of low-resolution results which will not give you high-resolution details. It's done by directly making planes from the high-resolution CT image.<br />
- step 1: save low-res plane into NifTi images. script: ```premier_step1.py```<br />
- step 2: upsample NifTi images. script: ```./premier_step2_upsample_plane.sh```<br />
- step 3: extract plane vectors from high-res NifTi images and directly make high-res plane movie from high-resolution CT. script: ```premier_step3.py``` <br />
The final high-resolution cine movie of planes can be found in ```example_files/Example_plane_cine.mp4```
#### Segmentation
We highly recommend to do the high-resolution segmentation using [a 2D U-Net](https://github.com/zhennongchen/2DUNet_CT_Seg_Final_v_ZC) instead of the model used here.


### Additional guidelines 
see comments in the script

Please contact zhc043@eng.ucsd.edu or chenzhennong@gmail.com for any further questions.




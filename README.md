# TomoPIC：Fast particle detection method for Cryo-electron tomography
## I Installation
The first step is to create a new conda virtual environment:
```
conda create -n tomopic -c conda-forge python=3.7
```
Activate the environment:
```
conda ativate tomopic
```
To download the codes, please do:
```
git clone https://github.com/yijianSU22/TomoPIC.git
cd TomoPIC
```
Next, install a custom pytorch and relative packages needed by DeepETPicker:
```
pip install -r requirement.txt
```
## II Particle picking tutorial
Detailed tutorials for two sample datasets of SHREC2021 and EMPIAR-10045 are provided. Main steps of TomoPIC includeds preprocessing, traning of TomoPIC, inference of TomoPIC. Here, we provides two sample datasets of EMPIAR-10045 and SHREC2021 for particle picking to enable you to learn the processing flow of TomoPIC.   The sample dataset can be download as follow: SHREC2021: [SHREC2021](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA)  EMPIAR-10045:


### SHREC2021



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
Detailed tutorials for two sample datasets of SHREC2021 and EMPIAR-10045 are provided. Main steps of TomoPIC includeds preprocessing, traning of TomoPIC, inference of TomoPIC. Here, we provides two sample datasets of EMPIAR-10045 and SHREC2021 for particle picking to enable you to learn the processing flow of TomoPIC. The sample dataset can be download as follow: 
SHREC2021: [SHREC2021](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA)  
EMPIAR-10045:

### SHREC2021
#### step1: training of TomoPIC
```
python train_shrec_2021.py --save_dir <the save dir of checkpoints> --dataset_dir <the directory of the used dataset> 
```
#### step1: inference of TomoPIC
```
python test_shrec.py --load_dir <the directory of the pretrained model> --dataset_dir <the directory of the used dataset> 
```

### EMPIAR-10045
#### step1: deconvloution
```
python deconvlution.py --tomo_dir ./tomo_bin4  --deconv_folder ./deconv --pixel_size 8.68 --snrfalloff 0.6
```
#### step2: prepare data
```
python prepare_data.py --tomo_dir ./deconv --train_data ./deconv/IS002_291013_005.mrc --eval_data ./deconv/IS002_291013_005.mrc --radius 12
```
#### step3: training of TomoPIC
```
python trian.py  --dataset_dir ./deconv --num_class 1 --batch_size 10 --detect_size 40 --padding_size 12 --anchor 24 --total_epoches 500               
```
#### step4: inference of TomoPIC
```
python predict.py --dataset_dir ./deconv/test  --load_dir <The directory of the checkpoints> --num_calss 1 --batch_size 10 --detect_size 40 --padding_size 12 --anchor 24
```

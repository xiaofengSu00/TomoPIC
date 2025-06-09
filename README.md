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
EMPIAR-10045:[EMPIAR-10045](https://pan.baidu.com/s/18-Uo8XViWOXnkzclyPjVxw?pwd=2aiv)

### SHREC2021
#### step1: training of TomoPIC
```
python train_shrec2021.py -h
usage: train_shrec2021.py [-h] [--model_name MODEL_NAME] [--gpu_ids GPU_IDS]
                          [--save_dir SAVE_DIR] [--load_dir LOAD_DIR]
                          [--dataset_dir DATASET_DIR]
                          [--total_epoches TOTAL_EPOCHES]
                          [--checkpoint_interval CHECKPOINT_INTERVAL]
                          [--evaluation_interval EVALUATION_INTERVAL]
                          [--pretrained PRETRAINED]
                          [--load_filename LOAD_FILENAME]
                          [--batch_size BATCH_SIZE]
                          [--num_workers NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name of this experiment.
  --gpu_ids GPU_IDS     GPU ids, use -1 for CPU.
  --save_dir SAVE_DIR   Models are saved here.
  --load_dir LOAD_DIR   The directory of the pretrained model.
  --dataset_dir DATASET_DIR
                        The directory of the used dataset
  --total_epoches TOTAL_EPOCHES
                        Total epoches.
  --checkpoint_interval CHECKPOINT_INTERVAL
                        Interval between saving model weights
  --evaluation_interval EVALUATION_INTERVAL
                        Interval between evaluations on validation set
  --pretrained PRETRAINED
                        Use pretrained model.
  --load_filename LOAD_FILENAME
                        Filename of the pretrained model.
  --batch_size BATCH_SIZE
                        Size of each image batch.
  --num_workers NUM_WORKERS
                        number of cpu threads to use during batch generation
```
```
python train_shrec2021.py --save_dir <the save dir of checkpoints> --dataset_dir <the directory of the used dataset> 
```
#### step1: inference of TomoPIC
```
python test_shrec.py -h
usage: test_shrec.py [-h] [--model_name MODEL_NAME] [--gpu_ids GPU_IDS]
                     [--load_dir LOAD_DIR] [--dataset_dir DATASET_DIR]
                     [--load_filename LOAD_FILENAME] [--batch_size BATCH_SIZE]
                     [--num_workers NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name of this experiment.
  --gpu_ids GPU_IDS     GPU ids, use -1 for CPU.
  --load_dir LOAD_DIR   The directory of the pretrained model.
  --dataset_dir DATASET_DIR
                        The directory of the used dataset
  --load_filename LOAD_FILENAME
                        Filename of the pretrained model.
  --batch_size BATCH_SIZE
                        Size of each image batch.
  --num_workers NUM_WORKERS
                        number of cpu threads to use during batch generation
```
```
python test_shrec.py --load_dir <the directory of the pretrained model> --dataset_dir <the directory of the used dataset> 
```

### EMPIAR-10045
#### step1: deconvloution
```
python deconvolution.py -h
usage: deconvolution.py [-h] [--tomo_dir TOMO_DIR]
                        [--deconv_folder DECONV_FOLDER] [--voltage VOLTAGE]
                        [--cs CS] [--defocus DEFOCUS]
                        [--pixel_size PIXEL_SIZE] [--snrfalloff SNRFALLOFF]
                        [--deconvstrength DECONVSTRENGTH]
                        [--highpassnyquist HIGHPASSNYQUIST]
                        [--chunk_size CHUNK_SIZE]
                        [--overlap_rate OVERLAP_RATE] [--ncpu NCPU]

optional arguments:
  -h, --help            show this help message and exit
  --tomo_dir TOMO_DIR   the directory for tomograms
  --deconv_folder DECONV_FOLDER
                        folder created to save deconvoluted tomograms
  --voltage VOLTAGE     acceleration voltage in kV
  --cs CS               spherical aberration in mm
  --defocus DEFOCUS     defocus in Angstrom.
  --pixel_size PIXEL_SIZE
                        pixel size in angstroms
  --snrfalloff SNRFALLOFF
                        SNR fall rate with the frequency. High values means
                        losing more high frequency
  --deconvstrength DECONVSTRENGTH
                        atrength of the deconvolution
  --highpassnyquist HIGHPASSNYQUIST
                        highpass filter for at very low frequency. We suggest
                        to keep this default value.
  --chunk_size CHUNK_SIZE
                        when your computer has enough memory, please keep the
                        chunk_size as the default value: None . Otherwise, you
                        can let the program crop the tomogram into multiple
                        chunks for multiprocessing and assembly them into one.
                        The chunk_size defines the size of individual chunk.
                        This option may induce artifacts along edges of
                        chunks. When that happen, you may use larger
                        overlap_rate.
  --overlap_rate OVERLAP_RATE
                        the overlapping rate for adjecent chunks
  --ncpu NCPU           number of cpus to use
```
```
python deconvlution.py --tomo_dir ./tomo_bin4  --deconv_folder ./deconv --pixel_size 8.68 --snrfalloff 0.6
```
#### step2: split dataset
```
python split_data.py -h
usage: split_data.py [-h] [--tomo_dir TOMO_DIR] [--train_data TRAIN_DATA]
                     [--eval_data EVAL_DATA] [--radius RADIUS]

optional arguments:
  -h, --help            show this help message and exit
  --tomo_dir TOMO_DIR   the directory for tomograms
  --train_data TRAIN_DATA
                        the path of the tomograms for training
  --eval_data EVAL_DATA
                        the path of the tomograms for validation
  --radius RADIUS       the radius in voxel of protein particle in tomograms
```
```
python split_data.py --tomo_dir ./deconv --train_data ./deconv/IS002_291013_005.mrc --eval_data ./deconv/IS002_291013_005.mrc --radius 12
```
#### step3: training of TomoPIC
```
python train.py -h
usage: train.py [-h] [--model_name MODEL_NAME] [--gpu_ids GPU_IDS]
                [--save_dir SAVE_DIR] [--load_dir LOAD_DIR]
                [--dataset_dir DATASET_DIR] [--num_class NUM_CLASS]
                [--batch_size BATCH_SIZE] [--detect_size DETECT_SIZE]
                [--padding_size PADDING_SIZE] [--num_workers NUM_WORKERS]
                [--anchor ANCHOR [ANCHOR ...]] [--learning_rate LEARNING_RATE]
                [--eta_min ETA_MIN] [--total_epoches TOTAL_EPOCHES]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--pretrained PRETRAINED] [--load_filename LOAD_FILENAME]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name of this experiment.
  --gpu_ids GPU_IDS     GPU ids, use -1 for CPU.
  --save_dir SAVE_DIR   Models are saved here.
  --load_dir LOAD_DIR   The directory of the pretrained model.
  --dataset_dir DATASET_DIR
                        The directory of the used dataset.
  --num_class NUM_CLASS
                        The number of class.
  --batch_size BATCH_SIZE
                        Size of each image batch.
  --detect_size DETECT_SIZE
                        The effective detection size of subtomograms
  --padding_size PADDING_SIZE
                        The padding size of subtomograms.
  --num_workers NUM_WORKERS
                        Number of cpu threads to use during batch generation
  --anchor ANCHOR [ANCHOR ...]
                        the predefine anchors of the model
  --learning_rate LEARNING_RATE
                        the learning rate of train
  --eta_min ETA_MIN     the minimum learning rate of train
  --total_epoches TOTAL_EPOCHES
                        Total epoches.
  --checkpoint_interval CHECKPOINT_INTERVAL
                        Interval between saving model weights
  --evaluation_interval EVALUATION_INTERVAL
                        Interval between evaluations on validation set
  --pretrained PRETRAINED
                        Use pretrained model.
  --load_filename LOAD_FILENAME
                        Filename of the pretrained model.
```
```
python train.py  --dataset_dir ./deconv --num_class 1 --batch_size 10 --detect_size 40 --padding_size 12 --anchor 24 --total_epoches 500               
```
#### step4: inference of TomoPIC
```
python predict.py -h
usage: predict.py [-h] [--model_name MODEL_NAME] [--gpu_ids GPU_IDS]
                  [--load_dir LOAD_DIR] [--dataset_dir DATASET_DIR]
                  [--detect_size DETECT_SIZE] [--padding_size PADDING_SIZE]
                  [--num_class NUM_CLASS] [--batch_size BATCH_SIZE]
                  [--num_workers NUM_WORKERS] [--anchor ANCHOR [ANCHOR ...]]
                  [--pretrained PRETRAINED] [--load_filename LOAD_FILENAME]
                  [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name of this experiment.
  --gpu_ids GPU_IDS     GPU ids, use -1 for CPU.
  --load_dir LOAD_DIR   The directory of the pretrained model.
  --dataset_dir DATASET_DIR
                        The directory of the used dataset
  --detect_size DETECT_SIZE
                        the effective detection size of subtomograms.
  --padding_size PADDING_SIZE
                        the padding size of subtomograms.
  --num_class NUM_CLASS
                        the number of class
  --batch_size BATCH_SIZE
                        Size of each image batch.
  --num_workers NUM_WORKERS
                        number of cpu threads to use during batch generation
  --anchor ANCHOR [ANCHOR ...]
                        the particle diameter for one class or the pre-cluster
                        result for mulitiple class
  --pretrained PRETRAINED
                        Use pretrained model.
  --load_filename LOAD_FILENAME
                        Filename of the pretrained model.
  --output_dir OUTPUT_DIR
                        the directory of saving the prediction result.
```
```
python predict.py --dataset_dir ./deconv/test  --load_dir <The directory of the checkpoints> --num_calss 1 --batch_size 10 --detect_size 40 --padding_size 12 --anchor 24
```

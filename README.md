# RCPS

Official implementation of *Rectified Contrastive Pseudo Supervision in Semi-Supervised Medical Image Segmentation*  

### :tada: Our work has been accepted by *IEEE Journal of Biomedical and Health Informatics*  

**Authors:**  

> Xiangyu Zhao, Zengxin Qi, Sheng Wang, Qian Wang, Xuehai Wu, Ying Mao, Lichi Zhang

manuscript link:  

- https://arxiv.org/abs/2301.05500
- https://ieeexplore.ieee.org/document/10273222

This repo contains the implementation of the proposed *Rectified Contrastive Pseudo Supervision (RCPS)* on two public benchmarks in medical images.  
**If you find our work useful, please cite the paper:**  

> @article{zhao2023rcps,  
> title={RCPS: Rectified Contrastive Pseudo Supervision for Semi-Supervised Medical Image Segmentation},  
> author={Zhao, Xiangyu and Qi, Zengxin and Wang, Sheng and Wang, Qian and Wu, Xuehai and Mao, Ying and Zhang, Lichi},  
> journal={IEEE Journal of Biomedical and Health Informatics},  
> doi={10.1109/JBHI.2023.3322590},  
> year={2023}  
> }

## TODO

:white_check_mark: Provide code for data preparation  
:white_check_mark: Publish model checkpoints  
:white_check_mark: Publish full training code  
:white_check_mark: Publish code for inference  
:white_check_mark: Add support for custom data training  

## Data 

Following previous works, we have validated our method on two benchmark datasets, including 2018 Atrial Segmentation Challenge and NIH Pancreas dataset.  
It should be noted that we do not have permissions to redistribute the data. Thus, for those who are interested, please follow the instructions below and process the data, or you will get a mismatching result compared with ours.

### Data Preparation

#### Download

##### Atrial Segmentation: http://atriaseg2018.cardiacatlas.org/  
- The above link seems to be out of service. You may find the data at: https://www.cardiacatlas.org/atriaseg2018-challenge/

##### Pancreas dataset: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
- If you encounter issues downloading the data, you may find the same data at : https://academictorrents.com/details/80ecfefcabede760cdbdf63e38986501f7becd49
- Please note that the orientation of the data downloaded from this link is not correct, please correct them manually.

#### Data Split

We split the data following previous works. Detailed split could be found in folder `data`, which are stored in .csv files.

#### Data Preprocessing

Download the data from the url above, then run the script `prepare_la_dataset.py` and `prepare_pancreas_dataset.py` by passing the arguments of data location.

### Prepare Your Own Data

Our RCPS could be extended to other datasets with some modifications.  

#### Before you start

- All of the data should be formatted as NIFTI, numpy array, or other file formats that MONAI library could handle.
- Please modify the `prepare_experiment` function in `configs/experiment.py`: define your own task name, and pass the number of classes, class names, as well as the affine matrix of 3D volume data.  
- You need to create a `$YOURTASK.cfg` file in the `configs` folder to pass necessary arguments to the algorithm, where `#YOURTASK` is the task name you defined in your case.

#### Training with different label ratio

In this scenario, all of the training images are labeled. Semi-supervised learning is deployed to investigate model performance with different labeled data ratio.  
In this case, split your training and validation data under the root path of your data storage. The expected structure of data storage is listed below:

```
- data_root
    - train_images
    - train_labels
    - val_images
    - val_labels
```
#### Training in real semi-supervision scenario

In this scenario, some of the training images are unlabeled.  

In this case, split your training and validation data under the root path of your data storage. The expected structure of data storage is listed below:

```
- labeled_root
    - train_images
    - train_labels
    - val_images
    - val_labels
- unlabeled_root
    - train_images
    - train_labels (an empty folder)
```
The `train.py` file should be modified as follows:
```python
data_pipeline = TrainValDataPipeline(labeled_root, 'labeled', label_ratio=1.0, random_seed=seed)
unlabeled_pipeline = TrainValDataPipeline(unlabeled_root, 'unlabeled', label_ratio=1.0, random_seed=seed)
trainset, _, valset = data_pipeline.get_dataset(train_aug, val_aug, cache_dataset=False)
unlabeled_set, _, _ = unlabeled_pipeline.get_dataset(train_aug, val_aug, cache_dataset=False)
train_sampler = DistributedSampler(trainset, shuffle=True)
unlabeled_sampler = DistributedSampler(unlabeled_set, shuffle=True)
val_sampler = DistributedSampler(valset)
```
By defining two `data_pipeline` instances, you could generate the corresponding training dataset, unlabeled dataset and test dataset, respectively. Then the training will be identical to LA or Pancreas dataset.

## Usage

### Pretrained Checkpoint

We have provided pretrained checkpoints for our RCPS on LA and NIH-Pancreas datasets.

#### Google Drive

Link: https://drive.google.com/drive/folders/15-2oBw-11bNMhSCRxzLSHisTbyOcD3gv?usp=sharing  

#### Baidu Netdisk

Link：https://pan.baidu.com/s/1wNY06gOmxy8lZzcKiYVR-g?pwd=0512  
Extraction Code (提取码)：0512  

### Training and Inference

#### Hardware

To run our code, you need a Linux PC equipped with at least one NVIDIA graphics card. The recommended video memory is at least 8GB. Graphics cards in newer generations (later than Turing) are recommended, as you will get extra speed-up with PyTorch native mixed precision training.

If you encounter CUDA OOM issues, please modify the `SAMPLE_NUM` argument in the cfg file in `configs` folder to a smaller value (100, for example).

#### Requirements

In order to run our code, please install the latest versions of following packages:

```
numpy
scipy
pandas
matplotlib
pyyaml
wandb
pytorch
torchvision
monai
nibabel
tqdm
```
#### Training

Please enter the following command in the terminal:

##### Mixed precision training with WandB logging enabled

```
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUMBER torchrun --nproc_per_node=$NUM_GPU train.py --mixed --benchmark --task $TASK --exp_name $EXP_NAME --wandb --entity $USER_NAME
```

`CUDA_DEVICE_NUMBER`: the CUDA device number visible to the training scripts, could be found by `nvidia-smi ` command;

`NUM_GPU`: number of GPUs used during training, at least 1 (our DDP supports one-card scenario);

`TASK`: task name;

`EXP_NAME`: experiment name;

`USER_NAME`: user name for WandB.

For further instructions, please run the command with `-h` argument.

#### Inference

Please enter the following command in the terminal:

```
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUMBER torchrun --nproc_per_node=$NUM_GPU eval.py --mixed --benchmark --task $TASK --exp_name $EXP_NAME -pc $CKPT
```

`CUDA_DEVICE_NUMBER`: the CUDA device number visible to the training scripts, could be found by `nvidia-smi ` command;

`NUM_GPU`: number of GPUs used during training, at least 1 (our DDP supports one-card scenario);

`TASK`: task name;

`EXP_NAME`: experiment name;

`CKPT`: the file path of checkpoint file.

For further instructions, please run the command with `-h` argument.

## Acknowledgement

- Our code is adapted from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS), [MC-Net](https://github.com/ycwu1997/MC-Net) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks these authors for their efforts in building the research community in semi-supervised medical image segmentation.
- Our code is developed based on medical image analysis library [MONAI](https://monai.io/).

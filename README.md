# RCPS
Official implementation of Rectified Contrastive Pseudo Supervision in semi-supervised medical image segmentation  

**Authors:**  
> Xiangyu Zhao, Zengxin Qi, Sheng Wang, Qian Wang, Xuehai Wu, Ying Mao, Lichi Zhang

manuscript link:  
- https://arxiv.org/abs/2301.05500 (preprint on arXiv)  

This repo contains the implementation of the proposed *Rectified Contrastive Pseudo Supervision* semi-supervised segmentation method on two public benchmarks in medical images.  
**If you use our code, please cite the paper:**  
> @article{zhao2023rcps,
  title={RCPS: Rectified Contrastive Pseudo Supervision for Semi-Supervised Medical Image Segmentation},
  author={Zhao, Xiangyu and Qi, Zengxin and Wang, Sheng and Wang, Qian and Wu, Xuehai and Mao, Ying and Zhang, Lichi},
  journal={arXiv preprint arXiv:2301.05500},
  year={2023}
}

## TODO
:white_check_mark: Provide code for data preparation  
:white_check_mark: Publish model checkpoints  
:white_check_mark: Publish full training code  
:white_check_mark: Publish code for inference  
:black_square_button: Add support for custom data training  

## Data 
Following previous works, we have validated our method on two benchmark datasets, including 2018 Atrial Segmentation Challenge and NIH Pancreas dataset.  
It should be noted that we do not have permissions to redistribute the data. Thus, for those who are interested, please follow the instructions below and process the data, or you will get a mismatching result compared with ours.
### Data Download
Atrial Segmentation: http://atriaseg2018.cardiacatlas.org/  
Pancreas dataset: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
### Data Preparation
#### Data Split
We split the data following previous works. Detailed split could be found in folder `data`, which are stored in .csv files.
#### Data Preprocessing
Download the data from the url above, then run the script `prepare_la_dataset.py` and `prepare_pancreas_dataset.py` by passing the argments of data location.
### Custom Data Training
Coming soon.

## Usage
### Pretrained Checkpoint
#### Google Drive
Link: https://drive.google.com/drive/folders/15-2oBw-11bNMhSCRxzLSHisTbyOcD3gv?usp=sharing  
#### Baidu Netdisk
Link：https://pan.baidu.com/s/1wNY06gOmxy8lZzcKiYVR-g?pwd=0512  
Extraction Code (提取码)：0512  

### Training and Inference
#### Training
```
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUMBER torchrun --nproc_per_node=$NUM_GPU train.py --mixed --benchmark --task $TASK --exp_name $EXP_NAME --wandb --entity $USER_NAME
```
#### Inference
```
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUMBER torchrun --nproc_per_node=$NUM_GPU eval.py --mixed --benchmark --task $TASK --exp_name $EXP_NAME -pc $CKPT
```

## Acknowledgement
Our code is adapted from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS), [MC-Net](https://github.com/ycwu1997/MC-Net) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks these authors for their efforts in building the research community in semi-supervised medical image segmentation.

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
  author={Zhao, Xiangyu Zhao and Qi, Zengxin and Wang, Sheng and Wang, Qian and Wu, Xuehai and Mao, Ying and Zhang, Lichi},  
  journal={arXiv preprint arXiv:2301.05500},  
  year={2023}  
}

## TODO
:black_square_button: Provide code for data preparation  
:white_check_mark: Publish model checkpoints  
:black_square_button: Publish full training code  
:black_square_button: Publish code for inference  
:black_square_button: Add support for custom data training  

## Usage
### Data Preparation
Update soon, please wait...

### Pretrained Checkpoint
#### Google Drive
Link: https://drive.google.com/drive/folders/15-2oBw-11bNMhSCRxzLSHisTbyOcD3gv?usp=sharing  
#### Baidu Netdisk
Link：https://pan.baidu.com/s/1wNY06gOmxy8lZzcKiYVR-g?pwd=0512  
Extraction Code (提取码)：0512  

### Training and Inference
```
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUMBER torchrun --nproc_per_node=$NUM_GPU train.py --mixed --benchmark --task $TASK --exp_name $EXP_NAME --wandb --entity $USER_NAME
```
```
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUMBER torchrun --nproc_per_node=$NUM_GPU eval.py --mixed --benchmark --task $TASK --exp_name $EXP_NAME -pc $CKPT
```

## Acknowledgement
Our code is adapted from UAMT, SASSNet, DTC, URPC, MC-Net and SSL4MIS. Thanks these authors for their efforts in building the research community in semi-supervised medical image segmentation.

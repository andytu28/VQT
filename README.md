# Visual Query Tuning (VQT) 

This is an offical implementation of [Visual Query Tuning: Towards Effective Usage of Intermediate Representations for Parameter and Memory Efficient Transfer Learning](https://arxiv.org/pdf/2212.03220.pdf). 


## Dependencies 

* python3.7 
* torch==1.7.1        
* torchvision==0.8.2  
* tensorflow==2.9.1 
* tensorflow_datasets==4.4.0+nightly 

## Usage 

We present instructions on training VQT with a ImageNet-1k pre-trained ViT-B/16. 

### Perparing the data 

Please setup the VTAB-1k benchmark following the instruction [here](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md). By default, our scripts will try to access the VTAB-1k datasets from the `vtab_data/` folder. You can modify the `DATA_PATH` variable in our scripts, which are placed under the `scripts/` folder, if you download the datasets to another place. 

For pre-trained ViT-B/16 models, you can download the weights of various pre-training setups as follows: 
* [ImageNet-1k Supervised](https://drive.google.com/file/d/1ruqA7gRkvkoM_QFrT0JI23XXZNnwqXBK/view?usp=share_link)
* [ImageNet-21k Supervised](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)
* [ImageNet-1k MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)

Please place the downloaded checkpoints under the `pre-trained_weights/` folder. Note that you need to rename the ImageNet-21k supervised checkpoint from `ViT-B_16.npz` to `imagenet21k_ViT-B_16.npz`. 


### Training VQT 

Use the following command to train a VQT model on a dataset in VTAB-1k. 

```bash 
$ bash scripts/VQT/run_vqt_vtab.sh ${GPUIDX} ${DATA_NAME} ${NUM_CLASSES} ${Q_LEN} ${OPTIMIZER} ${FEATURE}
```

We describe the meaning of these arguments as follows: 
* `${GPUIDX}`: The GPU used for training. For example, it can be set to 0. 
* `${DATA_NAME}`: The dataset name in VTAB-1k for training and evaluation. For example, it can be set to `vtab-caltech101`. Please see `run_demo_exp.sh` for more details about the 19 datasets in VTAB-1k. 
* `${NUM_CLASSES}`: The number of classes in the dataset. For example, for `vtab-caltech101`, this should be set to 102. 
* `${Q_LEN}`: The length of the query tokens. This can be simply set to 1. 
* `${OPTIMIZER}`: The optimizer used for training. In our experiments, we set this to `adam`. 
* `${FEATURE}`: The name of the pre-trained features. For example, it can be set to `sup_vitb16_imagenet1k` to indicate the ImageNet-1k supervised pre-trained model. 

After training a VQT model, you can optionally use the following command to compress the linear classifier via feature selection. 

```bash
$ bash scripts/VQT/run_vqt_vtab_sparsity.sh ${GPUIDX} ${DATA_NAME} ${NUM_CLASSES} ${Q_LEN} ${OPTIMIZER} ${FEATURE} ${FRACTION}
```

The first 6 arguments, `${GPUIDX}`, `${DATA_NAME}`, `${NUM_CLASSES}`, `${Q_LEN}`, `${OPTIMIZER}`, and `${FEATURE}`, are the same as the previous command for training a VQT model, and they can be set accordingly to indicate the trained VQT model we are going to compress. The last argument `${FRACTION}` specifies the proportion of the pre-classifier features (penultimate layer features) that we want to keep after compression. For example, it can be set to 0.7 to indicate keeping 70% of the features input to the final linear classifier. 


### Demo experiment 

For simplicity, you can use the following command for running through all the 19 datasets in VTAB-1k. 

```bash 
$ bash run_demo_exp.sh ${GPUIDX}
```

The `${GPUIDX}` argument specifies the GPU used for training (e.g., 0). 

After training VQT models for all the 19 datasets, you can use the following command to collect the results. 

```bash 
$ python collect_demo_exp_results.py
```


## Reference 

This repo is modified from [Visual Prompt Tuning (VPT)](https://github.com/KMnP/vpt). 

## Contact 

If you have any questions, please contact [Cheng-Hao Tu](https://andytu28.github.io/)(tu.343@osu.edu). 

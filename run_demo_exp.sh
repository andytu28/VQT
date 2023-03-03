#!/bin/bash


GPUIDX=$1

# ===== Q = 1, frac = 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-caltech101'                                                      102   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-caltech101'                                                      102   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-dtd'                                                              47   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-dtd'                                                              47   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-oxford_flowers102'                                               102   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-oxford_flowers102'                                               102   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-oxford_iiit_pet'                                                  37   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-oxford_iiit_pet'                                                  37   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-cifar(num_classes=100)'                                          100   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-cifar(num_classes=100)'                                          100   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-patch_camelyon'                                                    2   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-patch_camelyon'                                                    2   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-resisc45'                                                         45   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-resisc45'                                                         45   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-diabetic_retinopathy(config="btgraham-300")'                       5   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-diabetic_retinopathy(config="btgraham-300")'                       5   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-eurosat'                                                          10   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-eurosat'                                                          10   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-clevr(task="count_all")'                                           8   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-clevr(task="count_all")'                                           8   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)' 16   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)' 16   1 adam sup_vitb16_imagenet1k 0.7
bash scripts/VQT/run_vqt_vtab.sh          $GPUIDX 'vtab-sun397'                                                          397   1 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab_sparsity.sh $GPUIDX 'vtab-sun397'                                                          397   1 adam sup_vitb16_imagenet1k 0.7

# ===== Q = 10, frac = 1.0
bash scripts/VQT/run_vqt_vtab.sh $GPUIDX 'vtab-smallnorb(predicted_attribute="label_elevation")'                  9  10 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab.sh $GPUIDX 'vtab-clevr(task="closest_object_distance")'                             6  10 adam sup_vitb16_imagenet1k

# ===== Q = 20, frac = 1.0
bash scripts/VQT/run_vqt_vtab.sh $GPUIDX 'vtab-svhn'                                                             10  20 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab.sh $GPUIDX 'vtab-dmlab'                                                             6  20 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab.sh $GPUIDX 'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)'  16  20 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab.sh $GPUIDX 'vtab-smallnorb(predicted_attribute="label_azimuth")'                   18  20 adam sup_vitb16_imagenet1k
bash scripts/VQT/run_vqt_vtab.sh $GPUIDX 'vtab-kitti(task="closest_vehicle_distance")'                            4  20 adam sup_vitb16_imagenet1k

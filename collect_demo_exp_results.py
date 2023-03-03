import sys
import os
import torch
import numpy as np


setting_list = ['VQTSup_1_adam_sparsefinal', 'VQTSup_10_adam_final', 'VQTSup_20_adam_final']
run_idx_end = 5
arch_list = ['sup_vitb16_imagenet1k']
data_names = ['vtab-caltech101', 'vtab-dtd', 'vtab-cifar(num_classes=100)', 'vtab-oxford_flowers102', 'vtab-oxford_iiit_pet',
              'vtab-svhn', 'vtab-patch_camelyon', 'vtab-resisc45', 'vtab-eurosat', 'vtab-diabetic_retinopathy(config="btgraham-300")',
              'vtab-dmlab', 'vtab-clevr(task="closest_object_distance")', 'vtab-clevr(task="count_all")',
              'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)', 'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)',
              'vtab-smallnorb(predicted_attribute="label_azimuth")', 'vtab-smallnorb(predicted_attribute="label_elevation")',
              'vtab-kitti(task="closest_vehicle_distance")', 'vtab-sun397']

for setting_idx, setting in enumerate(setting_list):
    print(f'====== {setting} ======')
    for data_idx, data_name in enumerate(data_names):
        for arch in arch_list:
            directory_name = os.path.join(
                    'h2t_vit_experiments', setting, data_name, arch)
            if not os.path.exists(directory_name):
                continue
            hyparam = os.listdir(directory_name)[0]
            hyparam_dir = os.path.join(directory_name, hyparam)
            if 'keep_frac_0.7' in os.listdir(hyparam_dir):
                prefix = 'keep_frac_0.7/'
            else:
                prefix = ''

            results = []
            for run_idx in range(1, run_idx_end+1):
                path = os.path.join(
                        hyparam_dir, f'{prefix}run{run_idx}/eval_results.pth')
                if not os.path.exists(path):
                    continue

                content = torch.load(path, map_location='cpu')
                results.append(content['epoch_99']['classification'][f'test_{data_name}']['top1']*100.)
                # print(f'--- {data_name} - {arch} - {hyparam}')
                # print('{:.2f}'.format(
                #     content['epoch_99']['classification'][f'test_{data_name}']['top1']*100.))

            print(f'--- {data_name} - {arch} - {hyparam}')
            print('=====>', len(results))
            print('{:.2f}'.format(np.mean(results)))
    print()

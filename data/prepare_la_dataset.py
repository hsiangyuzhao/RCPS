import os
import argparse

import pandas as pd
import tqdm
import nrrd
import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./la_data', help='base dir for data')
    parser.add_argument('--meta', type=str, default='./la.csv', help='data split metadata')
    parser.add_argument('--output_dir', type=str, default='./LA', help='outputs')
    return parser.parse_args()


def crop_roi(image, label, output_size=(96, 96, 96)):
    assert (image.shape == label.shape)
    ### crop based on segmentation
    w, h, d = label.shape

    tempL = np.nonzero(label)
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])

    px = max(output_size[0] - (maxx - minx), 0) // 2
    py = max(output_size[1] - (maxy - miny), 0) // 2
    pz = max(output_size[2] - (maxz - minz), 0) // 2
    minx = max(minx - px - 25, 0)
    maxx = min(maxx + px + 25, w)
    miny = max(miny - py - 25, 0)
    maxy = min(maxy + py + 25, h)
    minz = max(minz - pz - 25, 0)
    maxz = min(maxz + pz + 25, d)

    image = image[minx:maxx, miny:maxy, minz:maxz].astype(np.float32)
    label = label[minx:maxx, miny:maxy, minz:maxz].astype(np.float32)
    return image, label

def main():
    args = parse_args()
    base_dir = args.base_dir
    metadata = pd.read_csv(args.meta)
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, 'train_images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val_images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train_labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val_labels'), exist_ok=True)

    train_list = metadata['filename'][metadata['split'] == 'train'].values
    val_list = metadata['filename'][metadata['split'] == 'val'].values

    for file in tqdm.tqdm(train_list):
        label = nrrd.read(os.path.join(base_dir, file, 'laendo.nrrd'))
        label_data = (label[0] / 255).astype(np.float32)
        image = nrrd.read(os.path.join(base_dir, file, 'lgemri.nrrd'))
        image_data = image[0].astype(np.float32)

        upper = np.percentile(image_data, 99.5)
        image_data = np.clip(image_data, None, upper)  # clip upper 0.5 %
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())  # z-score

        fg_image_data, fg_label_data = crop_roi(image_data, label_data, output_size=(112, 112, 80))

        lb = nib.Nifti1Image(fg_label_data, np.eye(4))
        nib.save(lb, os.path.join(output_dir, 'train_labels', file + '.nii.gz'))
        im = nib.Nifti1Image(fg_image_data, np.eye(4))
        nib.save(im, os.path.join(output_dir, 'train_images', file + '.nii.gz'))

    for file in tqdm.tqdm(val_list):
        label = nrrd.read(os.path.join(base_dir, file, 'laendo.nrrd'))
        label_data = (label[0] / 255).astype(np.float32)
        image = nrrd.read(os.path.join(base_dir, file, 'lgemri.nrrd'))
        image_data = image[0].astype(np.float32)

        upper = np.percentile(image_data, 99.5)
        image_data = np.clip(image_data, None, upper)  # clip upper 0.5 %
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())  # z-score

        fg_image_data, fg_label_data = crop_roi(image_data, label_data, output_size=(112, 112, 80))

        lb = nib.Nifti1Image(fg_label_data, np.eye(4))
        nib.save(lb, os.path.join(output_dir, 'val_labels', file + '.nii.gz'))
        im = nib.Nifti1Image(fg_image_data, np.eye(4))
        nib.save(im, os.path.join(output_dir, 'val_images', file + '.nii.gz'))

if __name__ == '__main__':
    main()

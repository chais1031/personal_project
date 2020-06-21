import os
from os import listdir
from os.path import isfile, join
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import fnmatch

loading_path_imgs = '/vol/bitbucket/sc5316/Brats17TrainingData/HGG/'
saving_path = '/vol/bitbucket/sc5316/Brats/HGG/t2norm/'

onlyfiles = [f for f in listdir(loading_path_imgs) if isfile(join(loading_path_imgs,f))]

for root, dirnames, filenames in os.walk(loading_path_imgs):
    for filename in fnmatch.filter(filenames, '*t2.*'):
        full_filename = os.path.join(root, filename)

#for x in onlyfiles:
#    if '.sh' in x:
#        continue

#    full_filename = os.path.join(loading_path_imgs, x)
        print('reading:', full_filename)
    
        img = nib.load(full_filename)
        img_shape = img.shape
        img_data = img.get_fdata()
        
        if len(img_shape) > 3:
            img_data = img_data[:, :, :, 0] #ensures it is not 4D
    
        new_header = img.header.copy()
        mean_pixel = np.mean(img_data)
        std_pixel = np.std(img_data)
        img_data = (img_data - mean_pixel)/std_pixel
        new_img = nib.nifti1.Nifti1Image(img_data, None, header=new_header)
    
        new_img_shape = new_img.shape
        print (new_img_shape)
        nib.save(new_img, join(saving_path,filename))
import os
from os import listdir
from os.path import isfile, join
import nibabel as nib
from skimage import filters
import fnmatch
import numpy as np
import random

loading_path_imgs = '/vol/bitbucket/sc5316/dHCP/t2norm/'
saving_paths = ['/vol/bitbucket/sc5316/dHCP/t2normsnp10/','/vol/bitbucket/sc5316/dHCP/t2normsnp15/','/vol/bitbucket/sc5316/dHCP/t2normsnp20/']

onlyfiles = [f for f in listdir(loading_path_imgs) if isfile(join(loading_path_imgs,f))]

probs = [0.1,0.15,0.2]

for prob, saving_path in zip(probs, saving_paths):
    for x in onlyfiles:
        full_filename = os.path.join(loading_path_imgs, x)
        print('reading:', full_filename)
        
        img = nib.load(full_filename)
        img_shape = img.shape
        img_data = img.get_fdata()
        
        if len(img_shape) > 3:
            img_data = img_data[:, :, :, 0] #ensures it is not 4D
            
        minimum = img_data.min()
        maximum = img_data.max()
            
        output = np.zeros(img_data.shape,dtype=float)
        thres = 1 - prob
        for i in range(img_data.shape[0]):
            for j in range(img_data.shape[1]):
                for k in range(img_data.shape[2]):
                    rdn = random.random()
                    if rdn < prob:
                        output[i][j][k] = minimum
                    elif rdn > thres:
                        output[i][j][k] = maximum
                    else:
                        output[i][j][k] = img_data[i][j][k]
        
        new_header = img.header.copy()
        
        new_img = nib.nifti1.Nifti1Image(output, None, header=new_header)
        nib.save(new_img, join(saving_path,x))
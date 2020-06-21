import os
from os import listdir
from os.path import isfile, join
import nibabel as nib
from skimage import filters
import fnmatch

loading_path_imgs = '/vol/bitbucket/sc5316/dHCP/t2norm/'
saving_path = '/vol/bitbucket/sc5316/dHCP/t2normgaus4/'

onlyfiles = [f for f in listdir(loading_path_imgs) if isfile(join(loading_path_imgs,f))]

#for root, dirnames, filenames in os.walk(loading_path_imgs):
#    for filename in fnmatch.filter(filenames, '*t1.*'):
#        full_filename = os.path.join(root, filename)

for x in onlyfiles:
    full_filename = os.path.join(loading_path_imgs, x)
    print('reading:', full_filename)
    
    img = nib.load(full_filename)
    img_shape = img.shape
    img_data = img.get_fdata()
    img_data_rescaled = filters.gaussian(img_data, sigma=4)
    new_header = img.header.copy()
    
    new_img = nib.nifti1.Nifti1Image(img_data_rescaled, None, header=new_header)
    nib.save(new_img, join(saving_path,x))
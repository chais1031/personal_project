import os
from os import listdir
from os.path import isfile, join
import nibabel as nib
from scipy import ndimage
import fnmatch

loading_path_imgs = '/vol/bitbucket/sc5316/dHCP/t2norm/'
saving_path = '/vol/bitbucket/sc5316/dHCP/t2normmedian11/'

onlyfiles = [f for f in listdir(loading_path_imgs) if isfile(join(loading_path_imgs,f))]

#for root, dirnames, filenames in os.walk(loading_path_imgs):
#    for filename in fnmatch.filter(filenames, '*t1.*'):
#        full_filename = os.path.join(root, filename)

for idx, x in enumerate(onlyfiles):
    if idx < 59 and 1 < idx:
        continue
    full_filename = os.path.join(loading_path_imgs, x)
    print('reading:', full_filename)
    
    img = nib.load(full_filename)
    img_shape = img.shape
    img_data = img.get_fdata()
    img_data_rescaled = ndimage.median_filter(img_data, size=11)
    new_header = img.header.copy()
    
    new_img = nib.nifti1.Nifti1Image(img_data_rescaled, None, header=new_header)
    nib.save(new_img, join(saving_path,x))
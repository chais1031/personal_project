import os
from os import listdir
from os.path import isfile, join
import nibabel as nib
from scipy import ndimage

loading_path_imgs = '/path/to/images/' # Directory path in which the input images are saved
saving_path = '/saving/path/' # Directory path to save the generated images

onlyfiles = [f for f in listdir(loading_path_imgs) if isfile(join(loading_path_imgs,f))]

radius_size = 5 # Radius size used to calculate the median for each pixel

for idx, x in enumerate(onlyfiles):
    if idx < 59 and 1 < idx:
        continue
    full_filename = os.path.join(loading_path_imgs, x)
    print('reading:', full_filename)
    
    img = nib.load(full_filename)
    img_shape = img.shape
    img_data = img.get_fdata()
    img_data_rescaled = ndimage.median_filter(img_data, size=radius_size)
    new_header = img.header.copy()
    
    new_img = nib.nifti1.Nifti1Image(img_data_rescaled, None, header=new_header)
    nib.save(new_img, join(saving_path,x))
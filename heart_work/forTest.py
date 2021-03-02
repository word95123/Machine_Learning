import pydicom as dicom
import os
import cv2, sys
import numpy as np
from PIL.Image import fromarray
# import subprocess
# output = subprocess.run(["echo", "asd"], stdout=subprocess.PIPE, shell=True)



filepath = "1.dcm"

ds = dicom.dcmread(filepath)
print(ds)
# print(ds.pixel_array)
print(ds[0x7FE00010])
print(ds[0x00081080].value)
# print(ds[0x00186011][0])
# print(ds[0x00280004].value)
# print(ds[0x00041200].value)
# print(ds[0x00081080])
print(ds.pixel_array.ndim)
print(ds.pixel_array.shape)
# img = []
# for i in range(ds.pixel_array.shape[0]):
#     img.append(dicom.pixel_data_handlers.util.apply_color_lut(ds.pixel_array[i], ds))
# img = np.array(img)
# print(img.shape)`

# print(0x00185012 in ds)
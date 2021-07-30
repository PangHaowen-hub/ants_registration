import SimpleITK as sitk
import os
import numpy as np

mask_sitk_1 = sitk.ReadImage('./data/m_reg/lobe512_001.nii.gz')
mask_sitk_2 = sitk.ReadImage('./data/m_reg/lobe512_002.nii.gz')
mask_sitk_3 = sitk.ReadImage('./data/m_reg/lobe512_003.nii.gz')
mask_sitk_4 = sitk.ReadImage('./data/m_reg/lobe512_004.nii.gz')
mask_sitk_5 = sitk.ReadImage('./data/m_reg/lobe512_005.nii.gz')

mask_arr_1 = sitk.GetArrayFromImage(mask_sitk_1)
mask_arr_2 = sitk.GetArrayFromImage(mask_sitk_2)
mask_arr_3 = sitk.GetArrayFromImage(mask_sitk_3)
mask_arr_4 = sitk.GetArrayFromImage(mask_sitk_4)
mask_arr_5 = sitk.GetArrayFromImage(mask_sitk_5)
mask_arr_stack = np.stack((mask_arr_1, mask_arr_2, mask_arr_3, mask_arr_4, mask_arr_5), axis=3).astype(int)
mask_arr = np.zeros_like(mask_arr_1)
for i in range(310):
    for j in range(512):
        for k in range(512):
            mask_arr[i, j, k] = np.argmax(np.bincount(mask_arr_stack[i, j, k, :]))

new_mask_img = sitk.GetImageFromArray(mask_arr)
new_mask_img.SetSpacing(mask_sitk_1.GetSpacing())
new_mask_img.SetOrigin(mask_sitk_1.GetOrigin())
new_mask_img.SetDirection(mask_sitk_1.GetDirection())
sitk.WriteImage(new_mask_img, 'lobe512_000_pred.nii.gz')
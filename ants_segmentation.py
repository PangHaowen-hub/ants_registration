import ants

ref = ants.image_read('./data/lobe512_000_0000.nii.gz')
# ref = ants.iMath(ref, 'Normalize')
mi1 = ants.image_read('./data/m_reg_lung/lobe512_001_0000.nii.gz')
mi2 = ants.image_read('./data/m_reg_lung/lobe512_002_0000.nii.gz')
mi3 = ants.image_read('./data/m_reg_lung/lobe512_003_0000.nii.gz')
mi4 = ants.image_read('./data/m_reg_lung/lobe512_004_0000.nii.gz')
mi5 = ants.image_read('./data/m_reg_lung/lobe512_005_0000.nii.gz')
seg1 = ants.image_read('./data/m_reg_lung/lobe512_001.nii.gz')
seg2 = ants.image_read('./data/m_reg_lung/lobe512_002.nii.gz')
seg3 = ants.image_read('./data/m_reg_lung/lobe512_003.nii.gz')
seg4 = ants.image_read('./data/m_reg_lung/lobe512_004.nii.gz')
seg5 = ants.image_read('./data/m_reg_lung/lobe512_005.nii.gz')
refmask = ants.get_mask(ref)

ilist = [mi1, mi2, mi3, mi4, mi5]
seglist = [seg1, seg2, seg3, seg4, seg5]

r = 2
pp = ants.joint_label_fusion(ref, refmask, ilist, r_search=2, label_list=seglist, rad=[r] * ref.dimension)
# pp = ants.joint_label_fusion(ref, refmask, ilist, r_search=2, rad=[r] * ref.dimension)
# print(pp)
ants.image_write(pp['segmentation'], 'pred_lung.nii.gz')
ants.image_write(pp['intensity'], 'intensity_lung.nii.gz')


import os
import ants
from tqdm import trange

'''
ants.registration()函数的返回值是一个字典：
    warpedmovout: 配准到fixed图像后的moving图像 
    warpedfixout: 配准到moving图像后的fixed图像 
    fwdtransforms: 从moving到fixed的形变场 
    invtransforms: 从fixed到moving的形变场

type_of_transform参数的取值可以为：
    Translation:平移变换
    Rigid:刚性变换:仅旋转和平移。
    Similarity:相似变换:缩放、旋转和平移。
    QuickRigid
    DenseRigid
    BOLDRigid
    Affine:仿射变换:刚性+缩放。
    AffineFast
    BOLDAffine
    TRSAA:translation, rigid, similarity, affine (twice), please set regIterations if using this option. 
    Elastic:弹性变形:仿射+变形。
    ElasticSyN:对称归一化:仿射+变形变换，以互信息为优化准则，以elastic为正则项。
    SyN:对称归一化:仿射配准+可变形配准，以互信息为优化准则
    SyNRA:对称归一化:刚性+仿射+变形，互信息为优化度量。
    SyNOnly:对称归一化:不进行初始变换，以互信息为优化度量。假设图像已对齐。如果你想运行一个非掩码仿射，然后是掩码可变形配准。
    SyNCC:SyN，但用互相关作为度量。
    SyNabp
    SyNBold
    SyNBoldAff
    SyNAggro:效果更好的SyN，用时比SyN长。 
    TVMSQ:具有均方度量的时变微分同胚
'''


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def ants_reg(gcect, ncct, cect, save_path, jac_path, trans_path):
    gcect_img = ants.image_read(gcect)
    ncct_img = ants.image_read(ncct)
    cect_img = ants.image_read(cect)
    _, img_fullflname = os.path.split(gcect)
    mytx = ants.registration(fixed=cect_img, moving=gcect_img, type_of_transform='SyN')  # TODO:改配准方法

    # 将形变场作用于moving图像，得到配准后的图像
    warped_img = ants.apply_transforms(fixed=cect_img, moving=ncct_img, transformlist=mytx['fwdtransforms'],
                                       interpolator="linear")
    warped_img.set_direction(cect_img.direction)
    warped_img.set_origin(cect_img.origin)
    warped_img.set_spacing(cect_img.spacing)
    ants.image_write(warped_img, os.path.join(save_path, img_fullflname))

    # 生成图像的雅克比行列式
    jac = ants.create_jacobian_determinant_image(domain_image=cect_img, tx=mytx["fwdtransforms"][0], do_log=False,
                                                 geom=False)
    ants.image_write(jac, os.path.join(jac_path, img_fullflname))

    trans = ants.image_read(mytx["fwdtransforms"][0])
    ants.image_write(trans, os.path.join(trans_path, img_fullflname))


if __name__ == '__main__':
    gcect_list = get_listdir(r'./CT2CECT/gcect_a')
    gcect_list.sort()
    ncct_list = get_listdir(r'./CT2CECT/ncct')
    ncct_list.sort()
    cect_list = get_listdir(r'./CT2CECT/cect_a')
    cect_list.sort()
    save_path = "./CT2CECT/gcect_a_SyN/ncct_warped"
    jac_path = "./CT2CECT/gcect_a_SyN/ncct_jac"
    trans_path = "./CT2CECT/gcect_a_SyN/ncct_trans"

    for i in trange(len(gcect_list)):
        ants_reg(gcect_list[i], ncct_list[i], cect_list[i], save_path, jac_path, trans_path)

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


def ants_reg(f_img_path, m_img_path, save_path, jac_path, grid_path, trans_path):
    f_img = ants.image_read(f_img_path)
    m_img = ants.image_read(m_img_path)
    _, img_fullflname = os.path.split(m_img_path)

    # mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='Affine')  # TODO:改配准方法
    # mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='SyNAggro')
    mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='SyN')  # TODO:改配准方法


    # 将形变场作用于moving图像，得到配准后的图像
    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                       interpolator="linear")
    warped_img.set_direction(f_img.direction)
    warped_img.set_origin(f_img.origin)
    warped_img.set_spacing(f_img.spacing)
    ants.image_write(warped_img, os.path.join(save_path, img_fullflname))

    # 生成图像的雅克比行列式
    jac = ants.create_jacobian_determinant_image(domain_image=f_img, tx=mytx["fwdtransforms"][0], do_log=False,
                                                 geom=False)
    ants.image_write(jac, os.path.join(jac_path, img_fullflname))

    mygr = ants.create_warped_grid(m_img)
    mywarpedgrid = ants.create_warped_grid(mygr, grid_directions=(False, False), transform=mytx['fwdtransforms'],
                                           fixed_reference_image=f_img)
    ants.image_write(mywarpedgrid, os.path.join(grid_path, img_fullflname))

    trans = ants.image_read(mytx["fwdtransforms"][0])
    ants.image_write(trans, os.path.join(trans_path, img_fullflname))


if __name__ == '__main__':
    f_img_list = get_listdir(r'/disk1/panghaowen/ants_registration/temp/f')
    f_img_list.sort()
    m_img_list = get_listdir(r'/disk1/panghaowen/ants_registration/temp/m')
    m_img_list.sort()
    save_path = "/disk1/panghaowen/ants_registration/temp/save"
    jac_path = "/disk1/panghaowen/ants_registration/temp/jac"
    grid_path = "/disk1/panghaowen/ants_registration/temp/grid"
    trans_path = "/disk1/panghaowen/ants_registration/temp/trans"

    for i in trange(len(m_img_list)):
        ants_reg(f_img_list[i], m_img_list[i], save_path, jac_path, grid_path, trans_path)

import os
import ants
from tqdm import trange


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def ants_reg(f_img_path, m_img_path, save_path):
    f_img = ants.image_read(f_img_path)
    m_img = ants.image_read(m_img_path)

    mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='Similarity')
    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                       interpolator="linear")
    warped_img.set_direction(f_img.direction)
    warped_img.set_origin(f_img.origin)
    warped_img.set_spacing(f_img.spacing)

    _, img_fullflname = os.path.split(m_img_path)
    ants.image_write(warped_img, os.path.join(save_path, img_fullflname))

if __name__ == '__main__':
    f_img_list = get_listdir(r'./data/fixed')
    f_img_list.sort()
    m_img_list = get_listdir(r'./data/moving')
    m_img_list.sort()
    save_path = "./data/warped"
    for i in trange(len(m_img_list)):
        ants_reg(f_img_list[i], m_img_list[i], save_path)

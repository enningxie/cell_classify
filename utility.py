import os
import shutil
import cv2

BASE_PATH = '/home/enningxie/Documents/DataSets/CELL_IMAGES/CELL_images'
HSIL_PATH = os.path.join(BASE_PATH, 'HSIL')
LSIL_PATH = os.path.join(BASE_PATH, 'LSIL')
NILM_PATH = os.path.join(BASE_PATH, 'NILM')


def mkdir_pwd(base_dir, join_dir):
    ex_dir = os.path.join(base_dir, join_dir)
    if not os.path.exists(ex_dir):
        os.mkdir(ex_dir)
    return ex_dir


train_path = mkdir_pwd(BASE_PATH, 'train')
test_path = mkdir_pwd(BASE_PATH, 'test')


def move_to(src_path, rate):
    src_len = len(os.listdir(src_path))
    train_len = int(src_len * rate)
    for i, f in enumerate(os.listdir(src_path)):
        if not i < train_len:
            break
        src_name = os.path.join(src_path, f)
        to_name = os.path.join(train_path, f)
        shutil.move(src_name, to_name)
    for f in os.listdir(src_path):
        src_name = os.path.join(src_path, f)
        to_name = os.path.join(test_path, f)
        shutil.move(src_name, to_name)
    os.removedirs(src_path)


def move_to(src_path, subname, rate):
    src_len = len(os.listdir(src_path))
    train_len = int(src_len * rate)
    for i, f in enumerate(os.listdir(src_path)):
        if not i < train_len:
            break
        src_name = os.path.join(src_path, f)
        train_path_ = mkdir_pwd(train_path, subname)
        to_name = os.path.join(train_path_, f)
        shutil.move(src_name, to_name)
    for f in os.listdir(src_path):
        src_name = os.path.join(src_path, f)
        test_path_ = mkdir_pwd(test_path, subname)
        to_name = os.path.join(test_path_, f)
        shutil.move(src_name, to_name)
    os.removedirs(src_path)


def split_train_test_set(rate):
    move_to(HSIL_PATH, 'HSIL', rate)
    move_to(LSIL_PATH, 'LSIL', rate)
    move_to(NILM_PATH, 'NILM', rate)
    convert_img(train_path, 'HSIL')
    convert_img(train_path, 'LSIL')
    convert_img(train_path, 'NILM')
    convert_img(test_path, 'HSIL')
    convert_img(test_path, 'LSIL')
    convert_img(test_path, 'NILM')


def rename_datafiles(path, subname):
    count = 0
    for f in os.listdir(path):
        tmp_path = os.path.join(path, f)
        tmp = os.path.splitext(f)
        count += 1
        # print(count)
        os.rename(tmp_path, os.path.join(path, subname+'_'+str(count)+tmp[-1]))


def convert_img(src_path, subname):
    path = mkdir_pwd(src_path, subname)
    for f in os.listdir(path):
        img_path = os.path.join(path, f)
        img_name = os.path.splitext(f)
        img_path_ = os.path.join(path, img_name[0]+'.jpg')
        img = cv2.imread(img_path)
        cv2.imwrite(img_path_, img)
        os.remove(img_path)


def main():
    # rename_op
    rename_datafiles(HSIL_PATH, 'HSIL')
    rename_datafiles(LSIL_PATH, 'HSIL')
    rename_datafiles(NILM_PATH, 'NILM')

    # split train/test set
    split_train_test_set(0.7)


if __name__ == '__main__':
    main()
    print('done.')
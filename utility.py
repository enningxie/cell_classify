import os
import shutil

BASE_PATH = '/home/enningxie/Documents/DataSets/CELL_images'
HSIL_PATH = os.path.join(BASE_PATH, 'HSIL')
LSIL_PATH = os.path.join(BASE_PATH, 'LSIL')
NILM_PATH = os.path.join(BASE_PATH, 'NILM')


def mkdir_pwd(base_dir, join_dir):
    ex_dir = os.path.join(base_dir, join_dir)
    if not os.path.exists(ex_dir):
        os.mkdir(ex_dir)
    return ex_dir


train_path = mkdir_pwd(BASE_PATH, 'train_data')
test_path = mkdir_pwd(BASE_PATH, 'test_data')


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


def split_train_test_set(rate):
    # move_to(HSIL_PATH, rate)
    move_to(LSIL_PATH, rate)
    move_to(NILM_PATH, rate)
    os.removedirs(HSIL_PATH)
    os.removedirs(LSIL_PATH)
    os.removedirs(NILM_PATH)


def rename_datafiles(path, subname):
    count = 0
    for f in os.listdir(path):
        tmp_path = os.path.join(path, f)
        tmp = os.path.splitext(f)
        count += 1
        # print(count)
        os.rename(tmp_path, os.path.join(path, subname+'_'+str(count)+tmp[-1]))


def main():
    # rename_op
    rename_datafiles(HSIL_PATH, 'HSIL')
    rename_datafiles(LSIL_PATH, 'HSIL')
    rename_datafiles(NILM_PATH, 'NILM')

    # split train/test set
    split_train_test_set(0.7)


if __name__ == '__main__':
    main()
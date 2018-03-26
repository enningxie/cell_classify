import os

BASE_PATH = '/home/enningxie/Documents/DataSets/CELL_images'
HSIL_PATH = os.path.join(BASE_PATH, 'HSIL')
LSIL_PATH = os.path.join(BASE_PATH, 'LSIL')
NILM_PATH = os.path.join(BASE_PATH, 'NILM')


def process_rawdata():
    # rename_datafiles(NILM_PATH, 'NILM')


def rename_datafiles(path, subname):
    count = 0
    for f in os.listdir(path):
        tmp_path = os.path.join(path, f)
        tmp = os.path.splitext(f)
        count += 1
        print(count)
        os.rename(tmp_path, os.path.join(path, subname+'_'+str(count)+tmp[-1]))



if __name__ == '__main__':
    process_rawdata()
import cv2
import os


data_dir = '/home/ck/cell_DL/cell_classify/CELL_images/train_data/'
data_dir_test = '/home/ck/cell_DL/cell_classify/CELL_images/test_data/'
data_dir_save = '/home/ck/cell_DL/cell_classify/CELL_images/train_data_trans/'
data_dir_save_test = '/home/ck/cell_DL/cell_classify/CELL_images/test_data_trans/'


def trans(dir_origin, dir_save):
	for filename in(os.listdir(dir_origin)):
		img_dir = os.path.join(dir_origin, filename)
		img = cv2.imread(img_dir)
		if not os.path.exists(dir_save):
			os.makedirs(dir_save)
		new_img_name = dir_save+(filename.split('/')[-1]).split('.')[0]+".jpg"
		cv2.imwrite(new_img_name, img)


def main():
	trans(data_dir, data_dir_save)
	trans(data_dir_test, data_dir_save_test)


if __name__ == '__main__':
	main()

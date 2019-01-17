import struct
import numpy as np
import cv2
import os


img_file =  r'emnist-letters-test-images-idx3-ubyte'  #  Path to your xxx-images-idx3-ubyte
label_file = r'emnist-letters-test-labels-idx1-ubyte'  # Path to your xxx-labels-idx1-ubyte
generate_img_root = r"G:\test"    # Path to save the images


def check_file(img_path,label_path):
    _,key_img_name = os.path.split(img_path)
    key1 = key_img_name.split('-')

    _,key_label_name = os.path.split(label_path)
    key2 = key_label_name.split('-')
    if (key1[1] != key2[1] or key1[2] != key2[2]):
        raise ValueError("Please check your input file,"
                         "and make true they belong to the same class !!!")


def get_key_name(file_path):
    _,file_name = os.path.split(file_path)
    key_name = file_name.split('-')
    subdir_name = str(key_name[1]) + "_" + str(key_name[2])
    return subdir_name


def make_dir(file_root):
    if  not os.path.isdir(file_root):
        os.makedirs(file_root)

check_file(img_file,label_file)

generate_img_root = os.path.join(generate_img_root,get_key_name(img_file))
make_dir(generate_img_root)


binfile = open(label_file,'rb')
buf = binfile.read()
index = 0
_,train_label_num = struct.unpack_from('>II',buf,index)
index += struct.calcsize('>II')

train_label_lis=[]

for i in range(train_label_num):
    label_item = int(struct.unpack_from('>B',buf,index)[0])
    train_label_lis.append(label_item)
    index += struct.calcsize('B')

print(len(train_label_lis))



binfile_img = open(img_file,'rb')
buf_img = binfile_img.read()
image_index=0

_,train_img_num = struct.unpack_from('>II',buf_img,image_index)
print("train_img_num: ",train_img_num)
image_index += struct.calcsize('>IIII')
im_list = []

for i in range(train_img_num):
    im = struct.unpack_from('>784B' ,buf_img, image_index)
    im_list.append(np.reshape(im,(28,28)))
    image_index += struct.calcsize('>784B')


for i in range(len(train_label_lis)):
    label_name = str(train_label_lis[i])
    subdir = os.path.join(generate_img_root,label_name)
    make_dir(subdir)
    image_name = label_name+"_"+str(i)+'.png'
    img_path = os.path.join(subdir,image_name)
    cv2.imwrite(img_path, im_list[i])

print('generate done!')
print("Now rotating......")

list = os.listdir(generate_img_root)
for path in list:
    each_class_path = os.path.join(generate_img_root,path)
    img_list = os.listdir(each_class_path)
    for img in img_list:
        image_path = os.path.join(each_class_path,img)
        img = cv2.imread(image_path, 0)
        row, cols = img.shape
        M = cv2.getRotationMatrix2D((cols // 2, row // 2), -90, 1)
        res2 = cv2.warpAffine(img, M, (row, cols))
        res2 = cv2.flip(res2, 1)
        cv2.imwrite(image_path, res2)

print("Done")
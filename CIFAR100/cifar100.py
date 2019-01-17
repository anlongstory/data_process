from scipy.misc import imsave
import numpy as np
import os
import pickle

CIFAR100_root = r"cifar-100-python"  # Path to your cifar-100-python folder

train_img_root= os.path.join(CIFAR100_root,'train_img')
test_img_root = os.path.join(CIFAR100_root,'test_img')



def unpickle(file):
    with open(file,'rb') as f:
        dic = pickle.load(f,encoding='bytes')
    return dic

def make_dir(file_root):
    if  not os.path.isdir(file_root):
        os.makedirs(file_root)

# b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data']

if __name__ == '__main__':
    make_dir(train_img_root)
    make_dir(test_img_root)

    # generate train img
    train_data = unpickle(CIFAR100_root+"/"+'train')
    print(CIFAR100_root+"/"+'train'+' is loading...')
    for i in range(0,50000):
        img = np.reshape(train_data[b'data'][i],(3,32,32))
        img = img.transpose(1, 2, 0)

        label_num = str(train_data[b'fine_labels'][i])
        subdir = os.path.join(train_img_root,label_num)
        make_dir(subdir)

        img_name = label_num+"_"+str(i)+'.png'
        img_path = os.path.join(subdir,img_name)
        imsave(img_path,img)
    print(CIFAR100_root+"/"+'train'+' loaded !')


    test_data = unpickle(CIFAR100_root+"/"+'test')
    print(CIFAR100_root+"/"+'test'+' is loading...')
    for j in range(0,10000):
        img = np.reshape(test_data[b'data'][j], (3, 32, 32))
        img = img.transpose(1, 2, 0)

        label_num = str(test_data[b'fine_labels'][j])
        subdir = os.path.join(test_img_root, label_num)
        make_dir(subdir)

        img_name = label_num + "_test_" + str(j) + '.png'
        img_path = os.path.join(subdir, img_name)
        imsave(img_path, img)
    print(CIFAR100_root + "/" + 'test' + ' loaded !')
    print(" Done ! ")


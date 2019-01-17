import pickle, gzip
import numpy as np
from scipy.misc import imsave
import os

pkl_path = r"mnist.pkl.gz"  # Path to your mnist.pkl.gz
save_path = r"MNIST" # Path to save images

def make_dir(file_root):
    if  not os.path.isdir(file_root):
        os.makedirs(file_root)

# Load the dataset
def unpickle(pkl_path):
    f = gzip.open(pkl_path, 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='bytes')
    f.close()
    return train_set,valid_set,test_set


if __name__ == '__main__':
    save_train_path = os.path.join(save_path,'train_img')
    save_val_path = os.path.join(save_path,'val_img')
    save_test_path = os.path.join(save_path,'test_img')
    make_dir(save_path)
    make_dir(save_train_path)
    make_dir(save_val_path)
    make_dir(save_test_path)

    train_set,val_set,test_set = unpickle(pkl_path)
    # train data
    print(save_train_path+" is loading... ")
    for i in range(0,50000):
        img = np.reshape(train_set[0][i],(28,28))
        label = train_set[1][i]
        subdir=os.path.join(save_train_path,str(label))
        make_dir(subdir)

        img_name = str(label)+"_"+str(i)+'.png'
        img_path = os.path.join(subdir,img_name)

        imsave(img_path,img)
    print(save_train_path + " loaded! ")

    # val data
    print(save_val_path + " is loading... ")
    for i in range(0, 10000):
        img = np.reshape(val_set[0][i], (28, 28))
        label = val_set[1][i]
        subdir = os.path.join(save_val_path, str(label))
        make_dir(subdir)

        img_name = str(label) + "_val_" + str(i) + '.png'
        img_path = os.path.join(subdir, img_name)

        imsave(img_path, img)
    print(save_val_path + " loaded! ")

    # test data
    print(save_test_path + " is loading... ")
    for i in range(0, 10000):
        img = np.reshape(test_set[0][i], (28, 28))
        label = test_set[1][i]
        subdir = os.path.join(save_test_path, str(label))
        make_dir(subdir)

        img_name = str(label) + "_test_" + str(i) + '.png'
        img_path = os.path.join(subdir, img_name)
        imsave(img_path, img)
    print(save_test_path + " loaded! ")

    print(" Done ! ")

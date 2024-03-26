import h5py
from scipy.io import loadmat

# # mnist
# mnist_img = './MNIST_MAT/Mnist_Train_Imgs.mat'
# mnist_label = './MNIST_MAT/Mnist_Train_Label.mat'
# val_img = './MNIST_MAT/Mnist_Test_Imgs.mat'
# val_label = './MNIST_MAT/Mnist_Test_Label.mat'
# train_x = loadmat(mnist_img)['Train_Img_Arr']  # (28, 28, 60000)
# train_y = loadmat(mnist_label)['Train_Label']  # (60000, 1)
# val_x = loadmat(val_img)['Test_Img_Arr']
# val_y = loadmat(val_label)['Test_Label']


# mnist_rot = './MNIST_MAT/Mnist_Test_Rot_Imgs.mat'
# val_x = loadmat(mnist_rot)['Test_Rot_Img_Arr']



# DHFM
filepath = './MNIST_MAT/MNIST_DHFM_Feature.mat'
label_train = './MNIST_MAT/Mnist_Train_Label.mat'
label_val = './MNIST_MAT/Mnist_Test_Label.mat'
data = loadmat(filepath)
# train_y = loadmat(label_train)
# val_y = loadmat(label_val)
#
# train_x, val_x = data['Fea_Train_Arr'], data['Fea_Test_Arr']  # (60000, 106)  (10000, 106)
# train_y, val_y = train_y['Train_Label'], val_y['Test_Label']  # (60000, 1)  (10000, 1)



print()



from skimage.io import imread, imsave
import numpy as np
from skimage.transform import resize
from skimage import util

#anno_name = 'A3C_x10000031'

anno_names = ['A1_x10000012',
              'A1C_x10000012',
              'A1L_x10000015',
              'A2C_x10000019',
              'A2L_x10000044',
              'A3C_x10000044',
              'A3L_x10000031',
              'B1C_x10000009',
              'B1L_x10000034',
              'C1C_x10000024',
              'C1L_x10000051']

for anno_name in anno_names:
    #fake_anno_path = '../src/steel_train_20200513_fake-anno/%s'%anno_name +'.png'
    fake_anno_path = '/Users/kaku/Desktop/steel_train_20200706/label_bad/%s'%anno_name +'.png'
    fake_anno = imread(fake_anno_path)
    print(fake_anno.dtype)
    anno_shape = (960, 1280)
    COLOR_FLIP = False
    
    fake_anno = fake_anno[:,:,0]
    fake_anno = util.img_as_ubyte(resize(fake_anno, anno_shape))
#    fake_anno = resize(fake_anno, anno_shape)
    print(fake_anno.dtype)
    # fake_anno = (fake_anno * 255).astype(np.int8)
    #
    #
#    fake_anno[fake_anno != 0] = 1
    fake_anno[fake_anno >= 128] = 255
    fake_anno[fake_anno < 128] = 0
#    if COLOR_FLIP:
#        fake_anno[fake_anno == 1] = 0.5
#        fake_anno[fake_anno == 0] = 1
#        fake_anno[fake_anno == 0.5] = 0
    

    
    
    real_anno_path = '/Users/kaku/Desktop/steel_train_20200706/label/%s'%anno_name + '.bmp'
    real_anno = (fake_anno * 255).astype(np.int8)
    imsave(real_anno_path, fake_anno)
    
    # sample = imread('/Users/kaku/Desktop/Work/NII/metal-seg_20200513/src/steel_train/label/A1C_x1000_Q_CR.png')

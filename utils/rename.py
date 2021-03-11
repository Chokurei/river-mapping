# recusively rename file name, ex. A1_x1000_area_segmap_FPN-nii_epoch_200_May13_23_22.tif into A1_x1000.tif

import os
_root = '_area_segmap_FPN-nii_epoch_200_Jul24_17_04'
result_dir = '../result/method15/'

result_names = os.listdir(result_dir)
for i in sorted(range(len(result_names))):
    result_name = result_names[i]
    result_path = os.path.join(result_dir + result_name)
    if _root in result_path:
        result_path_new = result_path.replace(_root, '')
        os.rename(result_path, result_path_new)


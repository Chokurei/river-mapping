# river-mapping
![comparison](https://user-images.githubusercontent.com/16301109/110784015-7d1d7380-82ac-11eb-921b-40abbea0eb98.gif)
## Requirements
* Python 3
* NVIDIA GPU + CUDA CuDNN
* PyTorch 0.4+
* gdal
## Training
Prepare training data src in ./src/
```
data-name/
|-- image
|   |-- img1.tif
|   |-- img2.tif
|   `-- img3.tif
|-- label
|   |-- img1.tif
|   |-- img2.tif
|   `-- img3.tif
|-- test.txt
`-- train.txt
```
Data extraction, extracted data will be saved in ./dataset
```
cd ./utils
python extractor.py -data src_dir -mode slide-rand -nb_crop 400
```
Training

```
python FPN.py -train True -train_data file-name-in-dataset -terminal epoch_num -test Flase 
```
trained model saved in ./checkpoint
## Trianed Model
Download: [One Drive](https://1drv.ms/u/s!ApTa4c0QeLyMbKVWTfxoIxWueis?e=Bev5ts)

River mapping

* tained by NIR, SWIR, red bands
```
FPN_epoch_400_Nov23_16_05.pth: for landsat 1,2,3
FPN_epoch_200_Dec24_19_15.pth: for landsat 5~8
```
* tained by NIR, red, green bands:
```
FPN_epoch_400_Mar01_14_21.pth
```
Vegtation mapping

* tained by NIR, red, green bands:
```
FPN_epoch_400_Feb28_22_22.pth
```
## Inference 
Trial use trained model. Obtained segmentation, skeleton, width, and model info in ./results 
```
python pridict.py -data data-in-src -objective river -checkpoints checkpoint-name
```
## Inference 
* project management
<img width="1073" alt="Screen Shot 2021-03-11 at 21 48 57" src="https://user-images.githubusercontent.com/16301109/110789789-9c6bcf00-82b3-11eb-847d-83ad28a6eba0.png">
* works
<img width="933" alt="Screen Shot 2021-03-11 at 21 49 44" src="https://user-images.githubusercontent.com/16301109/110789887-b73e4380-82b3-11eb-8ed0-94516cd1ca04.png">


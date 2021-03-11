# river-mapping
![comparison](https://user-images.githubusercontent.com/16301109/110784015-7d1d7380-82ac-11eb-921b-40abbea0eb98.gif)
## Requirements
* Python 3
* NVIDIA GPU + CUDA CuDNN
* PyTorch 0.4+
* gdal
## Training
prepare training data src in ./src/
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
data extraction, extracted data will be saved in ./dataset
```
cd ./utils
python extractor.py -data src_dir -mode slide-rand -nb_crop 400
```
training

```
python FPN.py -train True -train_data file-name-in-dataset -termina epoch_num -test Flase 
```
## Testing

## Trianed Model

## Inference 

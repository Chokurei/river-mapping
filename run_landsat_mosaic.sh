echo "Running Scripts for model training ..."
for i in {1}
do
#    echo "The $i time: "
#    echo "Training  "
#    export CUDA_VISIBLE_DEVICES=2
#    python FPN.py -train_data river_train_20201001-rand -trigger epoch -interval 2 -terminal 400 -batch_size 22 -test False ;
#    echo "Testing "
#    export CUDA_VISIBLE_DEVICES=2
#    python vissin_Area.py -data river_1001_full -load_recent True -batch_size 22 ;

# landsat 4~8
    echo "Training  "
#    export CUDA_VISIBLE_DEVICES=2
#    python FPN.py -train_data river_train_landsat8_20201224-rand -trigger epoch -interval 2 -terminal 200 -batch_size 22 -test False ;
    echo "Testing "
    export CUDA_VISIBLE_DEVICES=3
#    python vissin_Area.py -data LandsatArchive_LE07_SLC-off -load_recent True -batch_size 22 ;
    python vissin_Area_mosaic.py -data mosaic -load_recent False -checkpoints FPN_epoch_200_Dec24_19_15.pth -batch_size 22;

done
echo "end"
echo "Running Scripts for model training ..."
for i in {1}
do
    echo "Testing "
    export CUDA_VISIBLE_DEVICES=3
    python predict_auto.py -data mosaic_auto_pri_2 -checkpoints FPN_epoch_400_Mar01_14_21.pth -batch_size 22 ;
done
echo "end"
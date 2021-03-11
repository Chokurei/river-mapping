echo "Running Scripts for model training ..."
for i in {1}
do
    echo "Testing "
    export CUDA_VISIBLE_DEVICES=2
    python predict_auto.py -data mosaic_auto_pri_1 -checkpoints FPN_epoch_400_Mar01_14_21.pth -batch_size 22 -all_results False;
done
echo "end"
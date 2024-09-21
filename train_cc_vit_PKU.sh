for trial in 1
do
CUDA_VISIBLE_DEVICES=2 python train_PKU.py -b 64 -a agw -d  regdb_rgb --epochs 1 --iters 50 --momentum 0.99 --eps 0.6 --num-instances 16 --trial $trial
done
echo 'Done!'

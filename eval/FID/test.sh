#!/bin/bash
for i in {300..1210..10};
do 
	source activate xyn
	# python main_FTGAN.py --cfg cfg/eval_bird.yml --gpu 1 --model "../models/netG_epoch_$i.pth"
 	# cd StackGAN-inception-model-master/
	echo "epoch $i" >> "print.txt"
	# source activate xyn-tf
	# CUDA_VISIBLE_DEVICES=1 python inception_score.py --image_folder ../../models/netG_epoch_$i/valid/multiple
#	rm -rf /home/yn/Desktop/text2image/models/netG_epoch_$i
	# cd ..
	python fid_score.py --batch-size 64 --path1 bird_val.npz --path2 ../../models/netG_epoch_$i/ --gpu 1
done;

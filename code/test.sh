#!/bin/bash
#file = "/home/cv/xyn/text2image/print.txt"
#gpu = 0
for i in {300..800..10};
do
#  checkpoint = "../models/netG_epoch_$i.pth"
#  result_folder = "../../models/netG_epoch_$i/"
  source activate xyn
	cd mycode/ && python main_FTGAN.py --cfg cfg/eval_bird.yml --gpu 0 --model "../models/netG_epoch_$i.pth"
	echo "epoch $i" >> "print.txt"
	cd /home/cv/xyn/text2image/eval/FID
	python fid_score.py --batch-size 64 --gpu 0 --path1 bird_val.npz --path2 "../../models/netG_epoch_$i/" --rfile "print.txt"
	source activate xyn-tf
	cd /home/cv/xyn/text2image/mycode/StackGAN-inception-model-master/
	echo "epoch $i" >> "print.txt"
	CUDA_VISIBLE_DEVICES=0 python inception_score.py --image_folder "../../models/netG_epoch_$i/valid/multiple" --rfile "print.txt"
	cd /home/cv/xyn/text2image/
done;

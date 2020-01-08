#!/bin/bash
for i in {640..1020..10};
do
  source activate xyn
	cd mycode/ && python main_FTGAN.py --cfg cfg/eval_bird.yml --gpu 0 --model "../models/netG_epoch_$i.pth"
	cd /home/cv/xyn/text2image/eval/FID
  echo "epoch $i" >> "print-FID.txt"
	python fid_score.py --batch-size 64 --gpu 0 --path1 bird_val.npz --path2 "../../models/netG_epoch_$i/" --rfile "print-FID.txt"
	source activate xyn-tf
	cd /home/cv/xyn/text2image/mycode/StackGAN-inception-model-master/
	echo "epoch $i" >> "print-IS.txt"
	CUDA_VISIBLE_DEVICES=1 python inception_score.py --image_folder "../../models/netG_epoch_$i/valid/multiple" --rfile "print-IS.txt"
	cd /home/cv/xyn/text2image/
done;

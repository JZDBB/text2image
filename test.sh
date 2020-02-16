#!/bin/bash
for i in {250..800..10};
do
  	source activate xyn
	cd code/ && python main.py --cfg cfg/eval_bird.yml --gpu 0 --NET_G "../Model/netG_epoch_$i.pth"
	cd /home/cv/xyn/text2image/eval/FID
    echo "epoch $i" >> "print-FID.txt"
	python fid_score.py --batch-size 64 --gpu 0 --path1 bird_val.npz --path2 "../../Model/netG_epoch_$i/" --rfile "print-FID.txt"
	source activate xyn-tf
	cd /home/cv/xyn/text2image/mycode/StackGAN-inception-model-master/
	echo "epoch $i" >> "print-IS.txt"
	CUDA_VISIBLE_DEVICES=1 python inception_score.py --image_folder "../../Model/netG_epoch_$i/valid/multiple" --rfile "print-IS.txt"
	cd /home/cv/xyn/text2image/
done;

#!/bin/zsh
for i in {1000..1170..10}
do 
	python3 main_FTGAN.py --cfg cfg/eval_bird.yml --gpu 0 --model "../models/netG_epoch_$i.pth"
	# echo "$i"
done;

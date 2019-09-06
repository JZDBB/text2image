#!/bin/zsh
for i in 740 1000 6401
do 
	python3 main_FTGAN.py --cfg cfg/eval_bird.yml --gpu 0 --model "../models/netG_epoch_$i.pth"
done;

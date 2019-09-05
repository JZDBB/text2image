#!/bin/zsh
for i in 360 400 440 480 520 560 600 620 640 660 680 700 720
do 
	python3 main_FTGAN.py --cfg cfg/eval_bird.yml --gpu 0 --model "../models/netG_epoch_$i.pth"
done;

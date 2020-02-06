import os
model_path = "models-res/"
with open("epoch.txt", "r") as f:
    epoch = f.read().split("\n")
epoch = [int(i) for i in epoch]
for i in range(300, 1810, 10):
    print(i)
    if i in epoch:
        print("skip epoch {}".format(i))
        continue
    else:
        print("delete epoch {}".format(i))
        os.system("rm {}netG_epoch_{}.pth".format(model_path, i))
        os.system("rm -rf {}netG_epoch_{}".format(model_path, i))

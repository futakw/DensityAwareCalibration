import os
import glob

dir = "outputs/cifar10/resnet18"
dirs = glob.glob(dir + "/*/*")
for d in dirs:
    # rename the folder
    print(d)
    os.rename(d, d.replace("knn_dist", "ood_score"))
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metrics_path', type=str, default=r'D:\Users\cemko\PycharmProjects\pointnet.pytorch\utils\PM_files\Performance Metrics 1.txt', help='Path of the performance metrics file')

args = parser.parse_args()
pm_path = args.metrics_path

model_loss = {}
model_accuracy = {}
with open(pm_path, "r") as f:
    for i, line in enumerate(f):
        words = ''
        number = ''
        get_loss = False
        get_acc = False
        for c in line:
            words = words + c
            if(words == 'loss: '):
                get_loss = True
                continue
            elif(words == 'accuracy: '):
                get_acc = True
                continue

            if(get_loss) or (get_acc):
                number = number + c
                if(c == ' ') or (c == '\n'):
                    if(get_loss):
                        model_loss[i+1] = float(number.strip())
                    elif(get_acc):
                        model_accuracy[i+1] = float(number.strip())
                    get_acc = False
                    get_loss = False
                    number = ''
            
            if(c == ' ') and (not(get_loss) or not(get_acc)):
                words = ''
            
print(model_accuracy)

model_loss_batchnum = np.array(list(model_loss.keys()))
model_loss_lossvals = np.array(list(model_loss.values()))

model_acc_batchnum = np.array(list(model_accuracy.keys()))
model_acc_accvals = np.array(list(model_accuracy.values()))

xpoints1 = model_loss_batchnum
ypoints1 = model_loss_lossvals

plt.plot(xpoints1, ypoints1, label="loss")

xpoints2 = model_acc_batchnum
ypoints2 = model_acc_accvals

plt.plot(xpoints2, ypoints2, label="accuracy")

plt.xlabel("Batch number / 320")
plt.ylabel("Loss & Accuracy")

plt.legend()

plt.show()
## based on https://github.com/dmlc/mxnet/issues/1302
## Parses the model fit log file and generates a train/val vs epoch plot
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import sys

parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves')
parser.add_argument('--log-file', type=str,default="log_tr_va",
                    help='the path of log file')
parser.add_argument('--save-name', type=str,default="learning_curve.png",
                    help='the name of save png')
args = parser.parse_args()

# sys.path.append('../test/170217')


# TR_RE = re.compile('.*?]\sTrain-accuracy=([\d\.]+)')
# VA_RE = re.compile('.*?]\sValidation-accuracy=([\d\.]+)')
TR_RE = re.compile('.*?]\sTrain-Acc=([\d\.]+)')
VA_RE = re.compile('.*?]\sValidation-Acc=([\d\.]+)')

log = open('../'+args.log_file).read()

log_tr = [float(x) for x in TR_RE.findall(log)]
log_va = [float(x) for x in VA_RE.findall(log)]
idx = np.arange(len(log_tr))

fig = plt.figure(figsize=(8, 6))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(idx, log_tr, 'o', linestyle='-', color="r",
         label="Train accuracy")

plt.plot(idx, log_va, 'o', linestyle='-', color="b",
         label="Validation accuracy")

plt.legend(loc="best")
# plt.xticks(np.arange(min(idx), max(idx)+1, 5))
# plt.yticks(np.arange(0, 1, 0.2))
# plt.ylim([0,1])
# plt.show()
fig.savefig(args.save_name)

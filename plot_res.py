import matplotlib.pyplot as plt
import pickle as pk
import glob
import sys
from paramutils import *
import numpy as np


basepath = r'C:\Users\kpasad\Dropbox\ML\project\deep-reinforcement-learning-master\p3_collab-compet\working\\'
all_pk = glob.glob(basepath+'*.pk')

pk_file=basepath+'Tennis_00_25_31.pk'
[all_scores, avg_scores_window, params] = pk.load(open(pk_file,'rb'))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(all_scores)+1), all_scores)
plt.plot(np.arange(1, len(avg_scores_window)+1), avg_scores_window)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('MA-DDPG rewards for Tennis')
plt.show()



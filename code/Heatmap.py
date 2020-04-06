
#/bin/env/python
#This script was used to plot the result

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("./ALL_AUC.txt", delimiter = "\t", index_col = 0)

p = sns.heatmap(data, annot = True, cmap = 'RdBu_r')
plt.tight_layout()
plt.savefig("ALL_AUC.heatmap.png", dpi = 600, format = "PNG")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from configmanager import ConfigManager

'''
classes = [["Y", "I"], ["Y", "D"], ["A", "I"], ["A", "D"]]

f = ["Y", "A"]

classes_1 = [c for c in classes if f[0] in c]
classes_2 = [c for c in classes if f[1] in c]

print(classes_1)
print(classes_2)


classes = [["Y", "I"], ["Y", "D"], ["A", "I"], ["A", "D"]]
prova = ["Y", "I"]

print(prova in classes)

np.random.seed(19680801)
hist_data = np.random.randn(1_500)


fig = plt.figure(constrained_layout=True)
ax_array = fig.subplots(1, 1, squeeze=False)

ax_array[0, 0].imshow([[1, 1], [2, 1]])


cm = ConfigManager()
print(cm.get_classes())
print(cm.conv_rate)
print(cm.conv_rate[0])
print(cm.aggr_conv_rates([0,1]))

'''
arr1 = [10, 23, 30]
arr2 = [20, 40, 90]
print(np.cumsum(np.mean(100 - arr1)))
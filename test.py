import numpy as np
import matplotlib.pyplot as plt

"""
classes = [["Y", "I"], ["Y", "D"], ["A", "I"], ["A", "D"]]

f = ["Y", "A"]

classes_1 = [c for c in classes if f[0] in c]
classes_2 = [c for c in classes if f[1] in c]

print(classes_1)
print(classes_2)
"""

classes = [["Y", "I"], ["Y", "D"], ["A", "I"], ["A", "D"]]
prova = ["Y", "I"]

print(prova in classes)

np.random.seed(19680801)
hist_data = np.random.randn(1_500)


fig = plt.figure(constrained_layout=True)
ax_array = fig.subplots(1, 1, squeeze=False)

ax_array[0, 0].imshow([[1, 1], [2, 1]])

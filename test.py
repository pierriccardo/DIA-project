import numpy as np
import matplotlib.pyplot as plt
from configmanager import ConfigManager

cm = ConfigManager()

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


arr1 = [10, 23, 30]
arr2 = [20, 40, 90]
print(np.cumsum(np.mean(100 - arr1)))



classes = [["Y", "I"], ["Y", "D"], ["A", "I"], ["A", "D"]]
prova = ["Y", "I"]
if prova in classes:
    print('ok')


a = [[['Y', 'A'], 0.2], [['Y', 'A'], 0.5]]
best_feature = a[0]
for e in a:
    if e[1] > best_feature[1]:
        best_feature = e
print(best_feature)


a = ['aaa']
for e in a:
    print(e)
    a.append("bbb")

def extract_obs(obs, classes):
    # extract obs which belongs to the classes passed as argument
    return [o for o in obs if o[0] in classes]

obs = [[["A", "D"], 0, 1],
[["Y", "I"], 0, 0],
[["Y", "D"], 0, 3],
[["A", "D"], 0, 2]]

classes = [["A", "D"], ["Y", "D"]]

a = extract_obs(obs, classes)
print(a)

cm = ConfigManager()
cr = cm.conv_rates[0]
print(cr)
print(cm.prices)
print(np.multiply(cr, cm.prices))
current_opt = np.max(np.multiply(cr, cm.prices))
print(current_opt)

'''

nc = [cm.new_clicks(bid, 0) for bid in cm.bids]
mean = [cm.new_clicks_function_mean(bid, 0, 56000) for bid in cm.bids]
sigma = [cm.new_clicks_function_sigma(bid, 0, 56000) for bid in cm.bids]

print(nc)
print(mean)
print(sigma)

plt.figure(0)
plt.xlabel("bids")
plt.ylabel("new clicks")


x = cm.bids
plt.plot(x, nc, label='nc', color=cm.colors[3])
plt.plot(x, mean, label='mean', color=cm.colors[2])
plt.plot(x, sigma, label='sigma', color=cm.colors[1])
plt.legend(loc=0)
plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
plt.show()

fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
# plt.tight_layout()
fig.suptitle('Conversion Rates', fontsize=self.title_font)

ax[0].set_ylabel('Conversion Rate')

for i, user_class in enumerate(classes):
    
    ax[i].set_xlabel('Price(€)')
    color = self.cm.colors[user_class]

    x = np.linspace(0, 15, 10)
    y = self.cm.conv_rates[user_class]

    class_label = self.cm.class_labels[user_class]
    ax[i].plot(x, y,
                    color,
                    label=class_label,
                    marker='o',
                    markersize=3,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=4)

    ax[i].legend(loc=0)
    ax[i].grid(True, color='0.6', dashes=(5, 2, 1, 2))

# saving image
filename = 'all_conv_rates.png'
savepath = os.path.join(self.imgpath, filename)
fig.savefig(savepath)


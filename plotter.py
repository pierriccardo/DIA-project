import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from pricing import conv_rate, cost_per_click



class Plotter:

    def __init__(self):
        
        with open('config.yml', 'r') as file:
            self.config = yaml.safe_load(file)
        
    def plot_conv_rate(self, feature1="young", feature2="interested"):
        a, b = tuple(self.config["conv_rate"][feature1][feature2])
        color = self.config["features_colors"][feature1][feature2]

        x = np.linspace(0,8,20) 
        y = conv_rate(x, a, b)

        fig = plt.figure(figsize=(6,5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('Price(€)')
        ax.set_ylabel('Conversion Rate')
        ax.set_title(f'{feature1}-{feature2}')
        ax.plot(x, y, 
                color, 
                label=f'{feature1}-{feature2}',
                marker='o',
                markersize=3,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=4)

        ax.legend(loc=0)
        ax.grid(True, color='0.6', dashes=(5,2,1,2))

        # saving image
        filename = f'conv_rate_{feature1}_{feature2}.png'
        savepath = os.path.join(self.config["imgpath"], filename)
        fig.savefig(savepath)
    
    def plot_all_conv_rate(self):

        fig, ax = plt.subplots(figsize=(8,6.5), nrows=2, ncols=2)
        #plt.tight_layout()
        fig.suptitle('Conversion Rates', fontsize=20)

        for i, feature1 in enumerate(self.config["feature1"]):
            for j, feature2 in enumerate(self.config["feature2"]):

                #ax[i, j].set_title(f'{feature1}-{feature2}')
                ax[i, j].set_xlabel('Price(€)')
                ax[i, j].set_ylabel('Conversion Rate')

                a, b = tuple(self.config["conv_rate"][feature1][feature2])
                color = self.config["features_colors"][feature1][feature2]

                x = np.linspace(0,10,20) 
                y = conv_rate(x, a, b)

                ax[i, j].plot(x, y, 
                            color, 
                            label=f'{feature1}-{feature2}',
                            marker='o',
                            markersize=3,
                            markerfacecolor=color,
                            markeredgecolor=color,
                            markeredgewidth=4)

                ax[i, j].legend(loc=0)
                ax[i, j].grid(True, color='0.6', dashes=(5,2,1,2))

        # saving image
        filename = 'conv_rate_all.png'
        savepath = os.path.join(self.config["imgpath"], filename)
        fig.savefig(savepath)
    
    def plot_merged_conv_rate(self):

        fig = plt.figure(figsize=(6,5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('Price(€)')
        ax.set_ylabel('Conversion Rate')
        ax.set_title('Conversion Rates')

        for i, feature1 in enumerate(self.config["feature1"]):
            for j, feature2 in enumerate(self.config["feature2"]):

                a, b = tuple(self.config["conv_rate"][feature1][feature2])
                color = self.config["features_colors"][feature1][feature2]

                x = np.linspace(0,8,20) 
                y = conv_rate(x, a, b)
                ax.plot(x, y, 
                            color, 
                            label=f'{feature1}-{feature2}',
                            marker='o',
                            markersize=3,
                            markerfacecolor=color,
                            markeredgecolor=color,
                            markeredgewidth=4)

        ax.legend(loc=0)
        ax.grid(True, color='0.6', dashes=(5,2,1,2))

        # saving image
        filename = 'conv_rate_merged.png'
        savepath = os.path.join(self.config["imgpath"], filename)
        fig.savefig(savepath)
    
    def plot_all_cost_per_click(self):

        fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=3)
        #plt.tight_layout()
        fig.suptitle('Cost per click', fontsize=20)

        ax[0].set_ylabel('Cost per click')

        for index, user_class in enumerate(self.config["classes"]):

                alpha = self.config["cost_per_click"][user_class]["alpha"]
                color = self.config["class_color"][index]

                ax[index].set_xlabel('Bid')

                x = self.config["bids"]
                y = [cost_per_click(i, alpha) for i in x]
                ax[index].plot(x, y, 
                            color, 
                            label=f'{user_class}',
                            marker='o',
                            markersize=3,
                            markerfacecolor=color,
                            markeredgecolor=color,
                            markeredgewidth=4)

                ax[index].legend(loc=0)
                ax[index].grid(True, color='0.6', dashes=(5,2,1,2))

        # saving image
        filename = 'all_cost_per_click.png'
        savepath = os.path.join(self.config["imgpath"], filename)
        fig.savefig(savepath)

if __name__ == "__main__":
    p = Plotter()

    p.plot_all_conv_rate()
    p.plot_merged_conv_rate()
    
    for f1 in p.config["feature1"]:
        for f2 in p.config["feature2"]:
            p.plot_conv_rate(f1, f2)
    
    p.plot_all_cost_per_click()

    
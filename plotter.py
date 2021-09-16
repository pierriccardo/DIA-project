import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from configmanager import *


class Plotter:

    def __init__(self):

        with open('config.yml', 'r') as file:
            self.config = yaml.safe_load(file)

        self.cm = ConfigManager() 
        self.imgpath = self.cm.env_img_path
        
    def plot_conv_rate(self, user_class=0):

        x = np.linspace(0, 8, 20)
        y = self.cm.conv_rates[user_class]
        color = self.cm.colors[user_class]


        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('Price(€)')
        ax.set_ylabel('Conversion Rate')
        ax.set_title(f'{self.config["class_labels"][user_class]}')
        ax.plot(x, y,
                color,
                label=self.cm.class_labels[user_class],
                marker='o',
                markersize=3,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=4)

        ax.legend(loc=0)
        ax.grid(True, color='0.6', dashes=(5, 2, 1, 2))

        # saving image
        filename = f'conv_rate_{self.cm.class_labels[user_class]}.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)

    def plot_all_conv_rates(self, classes=[0,1,2]):

        fig, ax = plt.subplots(figsize=(25, 6.5), nrows=1, ncols=len(classes))
        # plt.tight_layout()
        fig.suptitle('Conversion Rates', fontsize=20)

        for i, user_class in enumerate(classes):
            
            ax[i].set_xlabel('Price(€)')
            ax[i].set_ylabel('Conversion Rate')

            color = self.cm.colors[user_class]

            x = np.linspace(0, 15, 10)
            y = self.cm.conv_rates[i]

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
        filename = 'conv_rate_all.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)

    def plot_merged_conv_rates(self, classes=[0,1,2]):

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('Price(€)')
        ax.set_ylabel('Conversion Rate')
        ax.set_title('Conversion Rates')

        for user_class in classes:
    
            color = self.cm.colors[user_class]

            x = np.linspace(0, 15, 10)
            y = self.cm.conv_rates[user_class]
            ax.plot(x, y,
                    color,
                    label=self.cm.class_labels[user_class],
                    marker='o',
                    markersize=3,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=4)

        ax.legend(loc=0)
        ax.grid(True, color='0.6', dashes=(5, 2, 1, 2))

        # saving image
        filename = 'conv_rate_merged.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)

    def plot_all_cost_per_click(self):

        fig, ax = plt.subplots(figsize=(20, 6), nrows=1, ncols=4)
        # plt.tight_layout()
        fig.suptitle('Cost per click', fontsize=20)

        ax[0].set_ylabel('Cost per click')

        for index, user_class in enumerate(self.config["classes"]):

            alpha = self.config["cost_per_click"][user_class]
            color = self.config["class_colors"][user_class]

            ax[index].set_xlabel('Bid')

            x = self.config["bids"]
            y = [self.cm.cost_per_click(i, alpha) for i in x]
            ax[index].plot(x, y,
                           color,
                           label=self.config["class_labels"][user_class],
                           marker='o',
                           markersize=3,
                           markerfacecolor=color,
                           markeredgecolor=color,
                           markeredgewidth=4)

            ax[index].legend(loc=0)
            ax[index].grid(True, color='0.6', dashes=(5, 2, 1, 2))

        # saving image
        filename = 'all_cost_per_click.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)

    def plot_all_return_probability(self):

        fig, ax = plt.subplots(figsize=(20, 6), nrows=1, ncols=4)
        # plt.tight_layout()
        fig.suptitle('Return Probability', fontsize=20)

        ax[0].set_ylabel('Return Probability')

        for index, user_class in enumerate(self.config["classes"]):

            _lambda = self.config["return_probability"][user_class]
            color = self.config["class_colors"][user_class]

            ax[index].set_title(f'Probability to return for {user_class}')
            ax[index].set_xlabel("Number of comebacks")

            ax[index].hist(
                return_probability(_lambda, size=100000),
                14,
                density=True,
                color=color,
                label=self.config["class_labels"][user_class])

            ax[index].legend(loc=0)
            ax[index].grid(True, color='0.6', dashes=(5, 2, 1, 2))

        # saving image
        filename = 'all_return_probability.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)

    def plot_all_new_clicks(self):

        fig, ax = plt.subplots(figsize=(20, 6), nrows=1, ncols=4)
        # plt.tight_layout()
        fig.suptitle('New clicks', fontsize=20)

        ax[0].set_ylabel('New clicks')

        cc = self.config["avg_cc"]

        for index, user_class in enumerate(self.config["classes"]):

            Na, p0 = tuple(self.config["new_clicks"][user_class])
            color = self.config["class_colors"][user_class]

            ax[index].set_xlabel('Bid')

            x = self.config["bids"]
            y = [new_clicks(i, Na, p0, cc) for i in x]
            ax[index].plot(x, y,
                           color,
                           label=self.config["class_labels"][user_class],
                           marker='o',
                           markersize=3,
                           markerfacecolor=color,
                           markeredgecolor=color,
                           markeredgewidth=4)

            ax[index].legend(loc=0)
            ax[index].grid(True, color='0.6', dashes=(5, 2, 1, 2))

        # saving image
        filename = 'all_new_clicks.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)


if __name__ == "__main__":
    p = Plotter()

    p.plot_all_conv_rates([0,2,3])
    p.plot_merged_conv_rates([0,2,3])
    #p.plot_all_new_clicks()
    #p.plot_all_cost_per_click()
    #p.plot_all_return_probability()

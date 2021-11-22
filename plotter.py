import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from configmanager import *
SEED = 1
np.random.seed(seed=SEED)
class Plotter:

    def __init__(self):

        with open('config.yml', 'r') as file:
            self.config = yaml.safe_load(file)

        self.cm = ConfigManager() 
        self.imgpath = self.cm.env_img_path

        self.title_font = 15
        
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

        fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=len(classes))
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
        filename = 'merged_conv_rates.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)

    def plot_all_cost_per_click(self, classes=[0,2,3]):

        fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=len(classes))
        fig.suptitle('Cost per click', fontsize=self.title_font)

        ax[0].set_ylabel('Cost per click')

        for index, user_class in enumerate(classes):

            color = self.cm.colors[user_class]
          
            ax[index].set_xlabel('Bid')

            x = self.config["bids"]
            y = [self.cm.cost_per_click(bid, user_class, 1, mean=True) for bid in self.cm.bids]
            
            ax[index].plot(x, y,
                           color,
                           label=self.cm.class_labels[user_class],
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
    
    def plot_merged_cost_per_click(self, classes=[0,2,3]):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('Bid')
        ax.set_ylabel('Cost per click')
        ax.set_title('Cost per click')

        for user_class in classes:
    
            color = self.cm.colors[user_class]

            x = self.cm.bids
            y = [self.cm.cost_per_click(bid, user_class, 1, mean=True) for bid in self.cm.bids]
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
        filename = 'merged_cost_per_click.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)


    def plot_all_return_probability(self, classes=[0,2,3]):

        fig, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=len(classes))
        fig.suptitle('Return Probability', fontsize=self.title_font)

        ax[0].set_ylabel('Return Probability')

        for index, user_class in enumerate(classes):

            lam = self.cm.ret[user_class]
            ax[index].set_xlabel("Number of comebacks")

            samples = self.cm.return_probability(lam, size=100000)

            ax[index].hist(
                samples,
                density=True,
                color=self.cm.colors[user_class],
                label=self.cm.class_labels[user_class],
                edgecolor='black')

            ax[index].legend(loc=0)
            ax[index].set_axisbelow(True)
            ax[index].grid(True, color='0.6', dashes=(5, 2, 1, 2))

        # saving image
        filename = 'all_return_probability.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)

    def plot_all_new_clicks(self, classes=[0,2,3]):

        fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=len(classes))
        fig.suptitle('New clicks', fontsize=self.title_font)

        ax[0].set_ylabel('New clicks')

        for index, user_class in enumerate(classes):

            ax[index].set_xlabel('Bid')
            x = self.cm.bids
            y = [self.cm.new_clicks(bid, user_class) for bid in self.cm.bids]
            color=self.cm.colors[user_class]
            ax[index].plot(x, y,
                           color,
                           label=self.cm.class_labels[user_class],
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
    
    def plot_merged_new_clicks(self, classes=[0,2,3]):

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('Bid')
        ax.set_ylabel('New Clicks')
        ax.set_title('New Clicks')

        for user_class in classes:
    
            color = self.cm.colors[user_class]

            x = np.linspace(0, 15, 10)
            y = [self.cm.new_clicks(bid, user_class) for bid in self.cm.bids]
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
        filename = 'merged_new_clicks.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)
    
    def plot_p_bid(self):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('Bid')
        ax.set_ylabel('P(Bid)')
        ax.set_title('Probabilità di vincere un asta')
    
        color = self.cm.colors[3]

        x = self.cm.bids
        y = [1- 0.40/(2*bid) for bid in self.cm.bids]
        ax.plot(x, y,
                color,
                label='P(Bid)',
                marker='o',
                markersize=3,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=4)

        ax.legend(loc=0)
        ax.grid(True, color='0.6', dashes=(5, 2, 1, 2))

        # saving image
        filename = 'p_bid.png'
        savepath = os.path.join(self.imgpath, filename)
        fig.savefig(savepath)


if __name__ == "__main__":
    p = Plotter()

    p.plot_all_conv_rates(classes=[0,2,3])
    p.plot_merged_conv_rates(classes=[0,2,3])

    p.plot_all_new_clicks()
    p.plot_merged_new_clicks()

    p.plot_all_cost_per_click()
    p.plot_merged_cost_per_click()

    p.plot_all_return_probability()

    p.plot_p_bid()

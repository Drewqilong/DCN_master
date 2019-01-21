# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:14:44 2019

@author: drewqilong
"""
import sys
sys.path.append(r"C:\Users\drewqilong\AppData\Roaming\Python\Python37\site-packages")
import plotly.offline as py
from plotly.offline import plot
import plotly.graph_objs as go
import matplotlib.pyplot as plt  # For 2D visualization
import seaborn as sns
import pandas as pd

from load_data import *


class Statistics(object):
    def __init__(self):
        self.df_train, self.df_test, self.X_train, self.y_train, self.X_test, self.ids_test, self.cat_features_indices = load_ctr_data()
        '''To analyse categorical variables, we will create three custom functions.
        The first two functions displays bar labels in absolute and relative scale respectively. And the 3rd one creates a dataframe of absolute and relative and also generates abs and relative frequency plot for each variable.'''
        
    ''' #1.Function for displaying bar labels in absolute scale.'''
    def _abs_bar_labels(self):
        font_size = 15
        plt.ylabel('Absolute Frequency', fontsize = font_size)
        plt.xticks(rotation = 0, fontsize = font_size)
        plt.yticks([])
        
        # Set individual bar lebels in absolute number
        for x in ax.patches:
            ax.annotate(x.get_height(), 
            (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
            textcoords = 'offset points', fontsize = font_size, color = 'black')
        
    '''#2.Function for displaying bar lebels in relative scale.'''
    def _pct_bar_labels(self):
        font_size = 15
        plt.ylabel('Relative Frequency (%)', fontsize = font_size)
        plt.xticks(rotation = 0, fontsize = font_size)
        plt.yticks([]) 
        
        # Set individual bar lebels in proportional scale
        for x in ax1.patches:
            ax1.annotate(str(x.get_height()) + '%', 
            (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
            textcoords = 'offset points', fontsize = font_size, color = 'black')
             
    '''#3.Function to create a dataframe of absolute and relative frequency of each variable. And plot absolute and relative frequency.'''
    def absolute_and_relative_freq(self, variable):
        global  ax, ax1 
        # Dataframe of absolute and relative frequency
        absolute_frequency = variable.value_counts()
        relative_frequency = round(variable.value_counts(normalize = True)*100, 2)
        # Was multiplied by 100 and rounded to 2 decimal points for percentage.
        df = pd.DataFrame({'Absolute Frequency':absolute_frequency, 'Relative Frequency(%)':relative_frequency})
        print('Absolute & Relative Frequency of',variable.name,':')
        #display(df)
        
        # This portion plots absolute frequency with bar labeled.
        fig_size = (18,5)
        font_size = 15
        title_size = 18
        ax =  absolute_frequency.plot.bar(title = 'Absolute Frequency of %s' %variable.name, figsize = fig_size)
        ax.title.set_size(title_size)
        self._abs_bar_labels()  # Displays bar labels in abs scale.
        plt.show()
        
        # This portion plots relative frequency with bar labeled.
        ax1 = relative_frequency.plot.bar(title = 'Relative Frequency of %s' %variable.name, figsize = fig_size)
        ax1.title.set_size(title_size)
        self._pct_bar_labels() # Displays bar labels in relative scale.
        plt.show()
        
    def correlation(self):
        train_float = self.df_train.select_dtypes(include=['float64'])
        colormap = plt.cm.magma
        plt.figure(figsize=(16,12))
        plt.title('Pearson correlation of continuous features', y=1.05, size=15)
        sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, 
                    cmap=colormap, linecolor='white', annot=True)
        
        #train_int = train_int.drop(["id", "target"], axis=1)
        # colormap = plt.cm.bone
        # plt.figure(figsize=(21,16))
        # plt.title('Pearson correlation of categorical features', y=1.05, size=15)
        # sns.heatmap(train_cat.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)
        train_int = self.df_train.select_dtypes(include=['int64'])
        data = [
            go.Heatmap(
                z= train_int.corr().values,
                x=train_int.columns.values,
                y=train_int.columns.values,
                colorscale='Viridis',
                reversescale = False,
#                text = True ,
                opacity = 1.0 )
        ]
        
        layout = go.Layout(
            title='Pearson Correlation of Integer-type features',
            xaxis = dict(ticks='', nticks=36),
            yaxis = dict(ticks='' ),
            width = 900, height = 700)

        fig = go.Figure(data=data, layout=layout)
        
#        py.iplot(fig, filename='labelled-heatmap')
        plot(fig)
        
if __name__ == '__main__':
    data_process = Statistics()
#    data_process.correlation()
    data_process.absolute_and_relative_freq(data_process.df_train.Label)
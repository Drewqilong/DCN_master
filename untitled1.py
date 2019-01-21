# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 19:11:13 2019

@author: drewqilong
"""

from plotly.offline import plot
import plotly.graph_objs as go

fig = go.Figure(data=[{'type': 'scatter', 'y': [2, 1, 4]}])

plot.iplot(fig)
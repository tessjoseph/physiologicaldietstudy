import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns


gptfour = pd.read_csv("gpt-4-turbo-preview_diet_analysis_0.csv")

gptthree = pd.read_csv("gpt-3.5-turbo_diet_analysis_0.csv")

llama = pd.read_csv("llama-2-70b-chat_diet_analysis_0.csv")

mistral = pd.read_csv("mistral-7b-instruct_diet_analysis_0.csv")

# Sample data
categories = ['gpt-4-turbo-preview', 'gpt-3.5-turbo', 'mistral-5b-instruct', 'llama-2-70b-chat_diet']

physiological_measures = ['heart_rate',  'blood_pressure', 'body_temperature', 'respiratory_rate',  'oxygen_saturation']

for measure in physiological_measures:
    fig = go.Figure(data=[
        go.Bar(name='gptfour', x=gptfour['diet'], y=gptfour[measure]),
        go.Bar(name='gpthree', x=gptthree['diet'], y =gptthree[measure]),
        go.Bar(name='llama', x=llama['diet'],  y=llama[measure]),
        go.Bar(name='mistral', x=mistral['diet'],  y=mistral[measure]),
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.show()


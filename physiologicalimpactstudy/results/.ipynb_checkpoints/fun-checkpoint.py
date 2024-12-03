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

heartrate=pd.DataFrame(columns = ['gptfour','gptthree','llama'])
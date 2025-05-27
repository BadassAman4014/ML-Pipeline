import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from IPython.display import display

SEED = 7
np.random.seed(SEED)
# Loading Data
df = pd.read_csv('diabetes.csv')

#EDA

df_name=df.columns
df.info()
df.head()
df.describe()
g = sns.pairplot(df, hue="Outcome", palette="husl")

#func to handle missing val
#func to handle datatype conv
#func to remove null val



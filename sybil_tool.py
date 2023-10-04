import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import torch
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    mean_absolute_error,
    roc_curve
)

st.markdown("<h1 style='text-align: center'>MGH test set thresholds exploration</h1>", unsafe_allow_html = True)
year = st.selectbox("What year's predictions are you using?", (1, 2, 3, 4, 5))

fpr = np.load('data/fpr' + str(year) + '.npy')
tpr = np.load('data/tpr' + str(year) + '.npy')
thresholds = np.load('data/thresholds' + str(year) + '.npy')

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    hover_data = {'Threshold': thresholds},
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=1200, height=800
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
st.plotly_chart(fig, use_container_width = True)
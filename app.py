import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.title("🛒 Customer Segmentation & Churn Dashboard")

df = pd.read_csv('data_with_clusters.csv')
st.dataframe(df.head())

fig = px.scatter(df, x='Annual_Income_USD', y='Spending_Score', 
                 color='Cluster', size='Total_Purchases', hover_data=['Age', 'Churn'])
st.plotly_chart(fig, use_container_width=True)

st.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
st.metric("Clusters", df['Cluster'].nunique())

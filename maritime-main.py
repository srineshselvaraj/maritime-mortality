# necessary libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import streamlit as st

# reading in the file
md = pd.read_csv("maritimedeaths.csv")

#renaming columns to make them easier to use
for c in md.columns:
    if c == 'Year':
        md = md.rename(columns={c : c.upper()})
    else:
        md = md.rename(columns={c : c.replace("_age_", " ")[:-1].upper()})

st.markdown("<h2 style='text-align: center;'>Maritime Mortality</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.text_input('Year')
with col2:
    st.text_input('Age')
with col3:
    st.text_input('Gender')
with col4:
    c1, c2, c3 = st.columns((1, 3, 1))
    with c2:
        st.markdown("<style> button {align-items: center} </style>", unsafe_allow_html=True)
        st.button('Predict')

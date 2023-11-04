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

# testing streamlit
st.dataframe(md)
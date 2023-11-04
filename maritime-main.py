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

def get_data(year, age, gender):
    # checking to see if the inputs follow the conditions
    b_year, b_age, b_gender = True, True, True
    if year != "":
        if not year.isnumeric() or int(year) < 1900 or int(year) > 2010:
            tab1.write("The year must be an integer between 1900 and 2010 (inclusive)")
            b_year = False
    if age != "":
        if not age.isnumeric() or int(age) < 0 or int(age) > 80:
            tab1.write("The age must be an integer between 0 and 80 (inclusive)")
            b_age = False
    if gender != "":
        if gender.upper() != 'MALE' and gender.upper() != 'FEMALE':
            tab1.write("The gender must be male or female")
            b_gender = False
    # if they are correct inputs
    if b_year and b_age and b_gender:
        # going through every case
        if age == "" and gender == "": #probability for one year
            means = md.iloc[:, 1:].mean(axis=1)
            avg = round(means.iloc[int(year) - 1900] * 100, 3)
            tab1.write("There is a "+str(avg)+"% chance that someone would die in a maritime disaster in the year "+year+".")
        elif year == "" and gender == "": #probability for one age
            mean1 = md.loc[:, age+" MALE"].mean()
            mean2 = md.loc[:, age+" FEMALE"].mean()
            avg = round(((mean1 + mean2) / 2) * 100, 3)
            tab1.write("There is a "+str(avg)+"% chance that someone would die in a maritime disaster if they were "+age+" years old.")
        elif year == "" and age == "": #probability for one gender
            mean = md.mean()
            if gender.upper() == 'MALE':
                mean = mean.loc["0 MALE" : "80 MALE"]
            if gender.upper() == 'FEMALE':
                mean = mean.loc["0 FEMALE" : "80 FEMALE"]
            avg = round(mean.mean() * 100, 3)
            tab1.write("There is a "+str(avg)+"% chance that someone would die in a maritime disaster if they were "+gender.lower()+".")
        elif year == "": #based on age and gender only
            mean = md.loc[:, age+" "+gender.upper()].mean()
            avg = round(mean * 100, 3)
            tab1.write("There is a "+str(avg)+"% chance that someone would die in a maritime disaster if they were a "+age+" year old "+gender.lower()+".")
        elif age == "": #based off year and gender
            if gender.upper() == 'MALE':
                mean1 = md.loc[:, "0 MALE":"80 MALE"]
                means = mean1.iloc[:, 1:].mean(axis=1)
                avg = round(means.iloc[int(year) - 1900] * 100, 3)
                tab1.write("There is a "+str(avg)+"% chance that someone would die in a maritime disaster if they were a "+gender.lower()+" in the year "+year+".")
            if gender.upper() == 'FEMALE':  
                mean1 = md.loc[:, "0 FEMALE":"80 FEMALE"]
                means = mean1.iloc[:, 1:].mean(axis=1)
                avg = round(means.iloc[int(year) - 1900] * 100, 3)
                tab1.write("There is a "+str(avg)+"% chance that someone would die in a maritime disaster if they were a "+gender.lower()+" in the year "+year+".")
        elif gender == "": #based on year and age
            mean = md[[age+" MALE", age+" FEMALE"]].mean(axis=1)
            avg = round(mean.iloc[int(year) - 1900] * 100, 3)
            tab1.write("There is a "+str(avg)+"% chance that someone would die in a maritime disaster if they were a "+age+" year old in the year "+year+".")
        else: #based on all three parameters
            col = md[age+" "+gender.upper()]
            ans = round(col.iloc[int(year) - 1900] * 100, 3)
            tab1.write("There is a "+str(ans)+"% chance that someone would die in a maritime disaster if they were a "+gender+" "+age+" year old in the year "+year+".")

tab1, tab2 = st.tabs(["Data", "Graph"])
with tab1:
    st.markdown("<h2 style='text-align: center;'>Maritime Mortality</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        year = st.text_input('Year')
    with col2:
        age = st.text_input('Age')
    with col3:
        gender = st.text_input('Gender')
    with col4: 
        c1, c2, c3 = st.columns((1, 3, 1))
        with c2:
            st.markdown("<style> button {align-items: bottom} </style>", unsafe_allow_html=True)
            st.button('Predict', on_click=get_data, args=(year, age, gender))
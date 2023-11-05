# necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# reading in the file
md = pd.read_csv("maritimedeaths.csv")

#renaming columns to make them easier to use
for c in md.columns:
    if c == 'Year':
        md = md.rename(columns={c : c.upper()})
    else:
        md = md.rename(columns={c : c.replace("_age_", " ")[:-1].upper()})

#function using dataset
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
            tab1.write("There is a "+str(ans)+"% chance that someone would die in a maritime disaster if they were a "+gender.lower()+" "+age+" year old in the year "+year+".")

def get_graph(type, x_axis):
    if x_axis == "Year":
        means = md.iloc[:, 1:].mean(axis=1).to_frame()
        means.columns = ['PROBABILITY']
        points = pd.concat([md.iloc[:, 0], means], axis=1)
    elif x_axis == 'Age':
        means = md.mean()
        ages = []
        point = []
        for a in range(81):
            ages.append(a)
            point.append((means.loc[str(a)+" MALE"] + means.loc[str(a)+" FEMALE"]) / 2)
        points = pd.DataFrame(list(zip(ages, point)))
        points.columns = ['AGE', 'PROBABILITY']
    elif x_axis == 'Gender':
        means = md.mean()
        mean1 = means.loc["0 MALE" : "80 MALE"]
        mean2 = means.loc["0 FEMALE" : "80 FEMALE"]
        genders = ['MALE', 'FEMALE']
        mean = [mean1.mean(), mean2.mean()]
        points = pd.DataFrame(list(zip(genders, mean)))
        points.columns = ['GENDER', 'PROBABILITY']
    fig, ax = plt.subplots()
    plt.xlabel(x_axis)
    plt.ylabel('Probability')
    plt.style.use("dark_background")
    if type == 'Bar Graph':
        ax.bar(points[x_axis.upper()], points['PROBABILITY'], color="cyan")
    elif type == 'Line Graph':
        ax.plot(points[x_axis.upper()], points['PROBABILITY'], "c.-")
    tab2.pyplot(fig)

def get_model(year, age, gender):
    if year == "" or not year.isnumeric() or age == "" or not age.isnumeric() or gender == "" or (gender.upper() != 'MALE' and gender.upper() != 'FEMALE'):
        tab3.write("Make sure every category is filled in, year and age are numbers, and gender is male or female.")
    else:
        if gender.upper() == 'MALE':
            data = md.iloc[:, 0:82]
            years = data.iloc[:, 0]
            ages = data.columns[1:]
            probs = data.iloc[:, 1:]
            X = np.array([[int(y), int(a[:-5])] for y in years for a in ages])
            y = probs.to_numpy().flatten()
        elif gender.upper() == 'FEMALE':
            data = pd.concat([md.iloc[:, 0], md.iloc[:, 82:]], axis=1)
            years = data.iloc[:, 0]
            ages = data.columns[1:]
            probs = data.iloc[:, 1:]
            X = np.array([[int(y), int(a[:-7])] for y in years for a in ages])
            y = probs.to_numpy().flatten()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        model = RandomForestRegressor(n_estimators=100, random_state=1)
        model.fit(X_train, y_train)
        input = [[year, age]]
        input_prediction = model.predict(input)
        ans = round(input_prediction[0] * 100, 3)
        tab3.write("For a "+age+" year old "+gender.lower()+" in the year "+year+", it is predicted that there is a "+str(ans)+"% chance that they will die due to a maritime disaster.")
        test_prediction = model.predict(X_test)
        mae = mean_absolute_error(y_test, test_prediction)
        tab3.write("The mean absolute error of this model is "+str(mae))
        
# website UI
st.markdown("<h2 style='text-align: center;'>Maritime Mortality</h2>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Data", "Graph", "Model"])
with tab1:
    st.markdown("<h5 style='text-align: center;'>Enter some or all of the parameters year, age and gender to see the probability that a person with those parameters would die to a maritime disaster.</h5>", unsafe_allow_html=True)
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
            st.button('Find', on_click=get_data, args=(year, age, gender))

with tab2:
    st.markdown("<h5 style='text-align: center;'>Choose a type of graph and a parameter to see the trend of the probability of dying in a maritime disaster based on the parameter.</h5>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        type = st.selectbox("Type of Graph", ["Bar Graph", "Line Graph"])
    with col2:
        x_axis = st.selectbox("X-axis", ["Year", "Age", "Gender"])
    with col3:
        c1, c2, c3 = st.columns((1, 3, 1))
        with c2:
            st.markdown("<style> button {align-items: bottom} </style>", unsafe_allow_html=True)
            st.button('Graph', on_click=get_graph, args=(type, x_axis))

with tab3:
    st.markdown("<h5 style='text-align: center;'>Enter any valid value for each parameter, and a machine learning model will predict the chance that a person with those parameters will die in a maritime disaster.</h5>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_year = st.text_input('Year:')
    with col2:
        n_age = st.text_input('Age:')
    with col3:
        n_gender = st.text_input('Gender:')
    with col4: 
        c1, c2, c3 = st.columns((1, 3, 1))
        with c2:
            st.markdown("<style> button {align-items: bottom} </style>", unsafe_allow_html=True)
            st.button('Predict', on_click=get_model, args=(n_year, n_age, n_gender))
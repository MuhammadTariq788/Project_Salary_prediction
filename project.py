# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# Read in the data
df = pd.read_csv('eda_data.csv')

# Table of constants
st.title("Data Scientists according to their skills")
st.sidebar.header("Salary")
st.subheader("Description state of Data")
st.write(df.describe())

# Split the data into X and y

X= df[['age', 'python_yn', 'R_yn','spark', 'aws', 'excel']]
y=df['avg_salary']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('ok')
# Function to get user report data
def user_report():
    age = st.sidebar.slider("age", 10, 120, 46)
    python_yn = st.sidebar.slider("python_yn", 0, 1, 1)
    R_yn = st.sidebar.slider("R_yn", 0,1,1)
    spark = st.sidebar.slider("spark", 0, 1, 1)
    aws = st.sidebar.slider("aws", 0, 1, 1)
    excel = st.sidebar.slider("excel", 0, 1, 1)
    print('ok')


    user_report_data = {
        "age": age,
        "python_yn": python_yn,
        "R_yn": R_yn,
        "spark": spark,
        "aws": aws,
        "excel": excel,
    
        
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Get user data
user_data = user_report()
st.subheader("Data Scientists according to their skills")
st.write("user_write")

# Train the Random Forest Classifier model
rc = RandomForestRegressor()
rc.fit(X_train, y_train)

# Predict the result for the user data
user_result = rc.predict(user_data)
st.write(user_result)
# Visualization
st.title("Visualized Data Scientist job skills")

# Color function
if user_result[0] == 0:
    color = "blue"
else:
    color = "red"
    print(user_result)

# Skills vs salary
st.header("Salary count graph (According to Your age)")
fig_salary=plt.figure()
ax1 = sns.scatterplot(x = 'avg_salary',y= 'age',data = df, hue = 'age',palette='Greens')
ax2 = sns.scatterplot(y=user_data['age'] ,s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('Avg salary Vs Age')
st.pyplot(fig_salary)

# python_yn Skills vs salary
st.header("Salary count graph (According to Your python_yn Skills)")
fig_salary=plt.figure()
ax1 = sns.scatterplot(x = 'avg_salary',y= 'python_yn',data = df, hue = 'python_yn',palette='Greens')
ax2 = sns.scatterplot(y=user_data['python_yn'] ,s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('Avg salary Vs python_yn')
st.pyplot(fig_salary)

# R_yn Skills vs salary
st.header("Salary count graph (According to Your R_yn Skills)")
fig_salary=plt.figure()
ax1 = sns.scatterplot(x = 'avg_salary',y= 'R_yn',data = df, hue = 'R_yn',palette='Greens')
ax2 = sns.scatterplot(y=user_data['R_yn'] ,s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('Avg salary Vs R_yn')
st.pyplot(fig_salary)

# spark Skills vs salary
st.header("Salary count graph (According to Your spark Skills)")
fig_salary=plt.figure()
ax1 = sns.scatterplot(x = 'avg_salary',y= 'spark',data = df, hue = 'spark',palette='Greens')
ax2 = sns.scatterplot(y=user_data['spark'] ,s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('Avg salary Vs spark')
st.pyplot(fig_salary)
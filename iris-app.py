import streamlit as st 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

def user_input():
    sepal_length = st.sidebar.slider('sepal_length',4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal_length',1.0, 6.9, 1.4)
    petal_width = st.sidebar.slider('petal_width',0.1, 2.5, 0.2)
    data = {'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

st.write("# Simple **Iris flower** prediction")
st.write("# Iris dataset")
st.write("Number of classes:3")
st.write("classifier:KNN")
st.sidebar.header("user input parameters")
st.subheader("user input parameters")

df = user_input()
st.write(df)

iris = load_iris()
x = iris.data
y=iris.target

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)
pred = model.predict(df)

st.subheader('class labels')
st.write(iris.target_names)

st.subheader('prediction')
st.write(iris.target_names[pred])

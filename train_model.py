import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('admission_predict.csv')

X = df[['GRE Score','TOEFL Score','University Ranking','SOP','LOR','CGPA','Research']]
y = df['Chance of Admit']

model = LinearRegression()
model.fit(X,y)

pickle.dump(model,open('admissions_model.pkl','wb'))

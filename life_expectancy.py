import pandas as pd
from sklearn.linear_model import LinearRegression



#Read Data
bmi_life_data = pd.read_csv('/Users/akutty/Documents/DeepLearning/Intro_Deep_Learning/bmi_and_life_expectancy.csv')

#Train model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

#Predict for a particular value of BMI which is X value
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)

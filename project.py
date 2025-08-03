import joblib
import numpy as np
import pandas as pd

# Load the final model
final_model = joblib.load("House_Price_Prediction_Model.pkl")

print("Welcome to the House Price Prediction System! the palace you can predict the price of a house based on its features.")
print("Please enter the features of the house you want to predict the price for.")
longitude = float(input("Enter the longitude: "))
latitude = float(input("Enter the latitude: "))
housing_median_age = int(input("Enter the housing median age: "))
total_rooms = int(input("Enter the total number of rooms: "))
total_bedrooms = int(input("Enter the total number of bedrooms: "))
population = int(input("Enter the population: "))
households = int(input("Enter the number of households: "))
median_income = float(input("Enter the median income: "))
ocean_proximity = input("Enter the ocean proximity (e.g., NEAR BAY, NEAR OCEAN, INLAND, ISLAND, <1H OCEAN): ").upper()

# Create a DataFrame with the input features
data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]]).reshape(1, -1)
clm = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
x_sample = pd.DataFrame(data, columns=clm)

# Predict the house price using the final model

result = final_model.predict(x_sample)
print(f"The predicted house price is approximately around : ${result[0]:,.2f}")
print("Thank you for using the House Price Prediction System!")
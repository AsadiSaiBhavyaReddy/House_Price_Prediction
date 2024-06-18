import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Area': [1500, 2000, 1800, 2200, 1600],
    'Bedrooms': [3, 4, 3, 4, 3],
    'Bathrooms': [2, 3, 2, 3, 2],
    'Price': [250000, 350000, 300000, 400000, 270000]
}
df = pd.DataFrame(data)

def predict_price(area, bedrooms, bathrooms):
    # Train a simple linear regression model
    X = df[['Area', 'Bedrooms', 'Bathrooms']]
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)

    # Predict the price for the given inputs
    input_data = np.array([[area, bedrooms, bathrooms]])
    predicted_price = model.predict(input_data)
    return predicted_price[0]

def generate_plot(area, bedrooms, bathrooms):
    # Add new data point to the dataset
    new_data = {'Area': [area], 'Bedrooms': [bedrooms], 'Bathrooms': [bathrooms]}
    new_df = df.append(pd.DataFrame(new_data), ignore_index=True)

    # Train a simple linear regression model with the updated dataset
    X_new = new_df[['Area', 'Bedrooms', 'Bathrooms']]
    y_new = new_df['Price']
    model = LinearRegression()
    model.fit(X_new, y_new)

    # Generate a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Area'], df['Price'], label='Existing Data')
    plt.scatter(area, predict_price(area, bedrooms, bathrooms), color='red', label='New Prediction')
    plt.xlabel('Area (sqft)')
    plt.ylabel('Price')
    plt.title('House Price Prediction')
    plt.legend()
    plt.savefig('static/house_price_prediction.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to release memory

# Example usage
area_input = 1900
bedrooms_input = 3
bathrooms_input = 2
generate_plot(area_input, bedrooms_input, bathrooms_input)

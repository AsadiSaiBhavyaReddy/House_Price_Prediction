import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Area': [1500, 2000, 1800, 2200, 1600],
    'Bedrooms': [3, 4, 3, 4, 3],
    'Bathrooms': [2, 3, 2, 3, 2],
    'Price': [250000, 350000, 300000, 400000, 270000]
}
df = pd.DataFrame(data)

# Extract features and target variable
X = df[['Area', 'Bedrooms', 'Bathrooms']]  # Features: Area, Bedrooms, Bathrooms
y = df['Price']  # Target variable: Price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Function to predict house price
def predict_price(area, bedrooms, bathrooms):
    input_data = np.array([[area, bedrooms, bathrooms]])
    predicted_price = model.predict(input_data)
    return predicted_price[0]


# GUI application
class HousePricePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('House Price Prediction')

        self.area_label = tk.Label(self.root, text='Area (sqft):')
        self.area_label.pack()
        self.area_entry = tk.Entry(self.root)
        self.area_entry.pack()

        self.bedrooms_label = tk.Label(self.root, text='Bedrooms:')
        self.bedrooms_label.pack()
        self.bedrooms_entry = tk.Entry(self.root)
        self.bedrooms_entry.pack()

        self.bathrooms_label = tk.Label(self.root, text='Bathrooms:')
        self.bathrooms_label.pack()
        self.bathrooms_entry = tk.Entry(self.root)
        self.bathrooms_entry.pack()

        self.predict_button = tk.Button(self.root, text='Predict Price', command=self.predict)
        self.predict_button.pack()

    def predict(self):
        try:
            area = float(self.area_entry.get())
            bedrooms = int(self.bedrooms_entry.get())
            bathrooms = int(self.bathrooms_entry.get())

            predicted_price = predict_price(area, bedrooms, bathrooms)

            messagebox.showinfo('Prediction', f'Predicted Price: ${predicted_price:.2f}')
        except ValueError:
            messagebox.showerror('Error', 'Please enter valid inputs.')


# Create the main window
root = tk.Tk()
app = HousePricePredictionApp(root)
root.mainloop()

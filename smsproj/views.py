from django.shortcuts import render
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_plot(X, y, features):
    model = LinearRegression()
    model.fit(X, y)

    # Predict prices
    predicted_price = model.predict(features)

    # Generate plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color='blue', label='Actual Prices')
    plt.scatter(features[0, 0], predicted_price, color='red', label='Predicted Price')
    plt.xlabel('Area (sq. ft.)')
    plt.ylabel('Price')
    plt.title('House Price Prediction')
    plt.legend()

    # Convert plot to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_encoded = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return 'data:image/png;base64,' + graph_encoded

import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression

def predict_price(request):
    if request.method == 'POST':
        area = float(request.POST.get('area'))
        bedrooms = int(request.POST.get('bedrooms'))
        bathrooms = int(request.POST.get('bathrooms'))
        stories = int(request.POST.get('stories'))
        guestroom = int(request.POST.get('guestroom'))
        parking = int(request.POST.get('parking'))

        # Example dataset without negative prices
        X = np.array([
            [1200, 2, 1, 1, 0, 1],
            [1800, 3, 2, 2, 1, 2],
            [2200, 4, 2, 2, 1, 2],
            [1500, 3, 1, 1, 1, 1],
            [2000, 3, 2, 2, 0, 2],
            [2500, 4, 3, 2, 1, 2],
            [1800, 3, 1, 1, 0, 1],
            [2100, 4, 2, 2, 1, 2],
            [2400, 4, 2, 2, 1, 2],
            [1700, 3, 1, 1, 1, 1]
        ])
        y = np.array([150000, 220000, 280000, 180000, 250000, 320000, 200000, 260000, 300000, 190000])

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Prepare input features for prediction
        features = np.array([[area, bedrooms, bathrooms, stories, guestroom, parking]])

        # Predict the price using the trained model
        predicted_price = model.predict(features)[0]

        context = {'predicted_price': predicted_price}
        return render(request, 'predict_price.html', context)
    else:
        return render(request, 'predict_price.html')



def index(request):
    return render(request,"index.html")

def about(request):
    return render(request,"about.html")

def login(request):
    return render(request,"login.html")

def contactus(request):
    return render(request,"contactus.html")



def home(request):
    return render(request,"home.html")

def homep(request):
    return render(request,"index.html")
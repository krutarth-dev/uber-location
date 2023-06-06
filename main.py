# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import folium
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv('/Users/apple/Documents/Projects/Uber Prediction/rideshare_kaggle.csv')

# Preprocessing and feature selection
booking_data = data[['latitude', 'longitude']]
price_data = data[['latitude', 'longitude', 'price']]

# Clustering to identify high booking areas
kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as per your requirement
clusters = kmeans.fit_predict(booking_data)
booking_data['cluster'] = clusters

# Displaying the high booking areas on a map
map_clusters = folium.Map(location=[booking_data['latitude'].mean(), booking_data['longitude'].mean()], zoom_start=10)
colors = ['red', 'blue', 'green', 'orange', 'purple']  # Adjust the number of colors as per the number of clusters
for lat, lon, cluster in zip(booking_data['latitude'], booking_data['longitude'], booking_data['cluster']):
    folium.CircleMarker([lat, lon], radius=5, color=colors[cluster], fill=True, fill_color=colors[cluster],
                        fill_opacity=0.7).add_to(map_clusters)
map_clusters.save('booking_areas_map.html')

# Splitting the data into features and target variables
X = price_data[['latitude', 'longitude']]
y = price_data['price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Predicting prices for new locations
new_locations = pd.DataFrame({'latitude': [lat1, lat2, lat3, ...], 'longitude': [lon1, lon2, lon3, ...]})
predicted_prices = regression_model.predict(new_locations)

# Displaying the predicted prices on a map
price_map = folium.Map(location=[price_data['latitude'].mean(), price_data['longitude'].mean()], zoom_start=10)
for lat, lon, price in zip(new_locations['latitude'], new_locations['longitude'], predicted_prices):
    folium.Marker([lat, lon], popup=f'Price: {price}').add_to(price_map)
price_map.save('predicted_prices_map.html')

# Visualize Clustering Results
plt.scatter(booking_data['longitude'], booking_data['latitude'], c=booking_data['cluster'], cmap='viridis')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustering Results')
plt.show()

# Evaluate Accuracy
accuracy = calculate_accuracy(booking_data['true_label'], booking_data['predicted_label'])
print(f"Accuracy: {accuracy}")

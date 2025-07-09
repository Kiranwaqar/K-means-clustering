import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load file
df = pd.read_csv('House Prediction Data Set.csv', sep='\\s+', header=None)

df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Feature & target split
X = df.drop('MEDV', axis=1)
y = df['MEDV']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(8, 5)) 
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='blue')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply k-means clustering
model = KMeans(n_clusters=3, random_state=42)  
model.fit(X_train)

# Predict clusters
y_pred = model.predict(X_test)

# Add cluster labels to the test set
X_test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)
X_test_df['Cluster'] = y_pred
X_test_df['MEDV'] = y_test.values

plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_test_df, x='RM', y='MEDV', hue='Cluster', palette='viridis', s=100)
plt.title('KMeans Clustering: Rooms vs. House Price')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median Value of Homes (MEDV)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Visualize clusters
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)
pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = y_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100)
plt.title('KMeans Clustering in PCA 2D Space')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Interpret the results 
print("Cluster Centers:")
centers = model.cluster_centers_
centers_original = scaler.inverse_transform(centers)  
centers_df = pd.DataFrame(centers_original, columns=X.columns)
print(centers_df)



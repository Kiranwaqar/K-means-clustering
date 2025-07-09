# K-Means Clustering on House Prices

This project demonstrates how to apply **K-Means Clustering** to analyze and group houses based on various features, using the **House Price Prediction Dataset**. The objective is to segment the data into meaningful clusters to understand hidden patterns such as housing groups based on price, size, and other attributes.

---
## Demo

https://github.com/user-attachments/assets/5502bee7-ac80-4f23-9c85-1a37f6d51486

---

## Project Objectives

- Load and preprocess the dataset.
- Scale the features using `StandardScaler`.
- Use the **Elbow Method** to determine the optimal number of clusters.
- Apply **K-Means Clustering** to the scaled data.
- Visualize clusters:
  - Based on real features (e.g., RM vs MEDV).
  - Using **PCA** (Principal Component Analysis) to reduce dimensions.
- Interpret and display cluster centers in original feature scale.

---

## Dataset Used

**House Prediction Dataset**  
A structured dataset with 14 features including:

- `CRIM`: Crime rate
- `ZN`: Residential land zoned
- `INDUS`: Industrial areas
- `CHAS`: Proximity to Charles River
- `NOX`: Nitric oxide concentration
- `RM`: Average number of rooms
- `AGE`: Age of buildings
- `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`: Various socio-economic indicators
- `MEDV`: Median value of owner-occupied homes *(used for visual analysis, not prediction)*

---

## Tools & Libraries

- **Python**
- **pandas**: Data handling
- **matplotlib**, **seaborn**: Data visualization
- **scikit-learn**:
  - `KMeans`: For clustering
  - `StandardScaler`: For feature normalization
  - `PCA`: For dimensionality reduction

---

## Key Visualizations

1. **Elbow Method Plot**  
   Helps determine the optimal number of clusters (k) based on the WCSS (within-cluster sum of squares).

2. **Scatter Plot - RM vs MEDV**  
   Visualizes how clusters relate to average number of rooms and house prices.

3. **PCA Plot - PC1 vs PC2**  
   A 2D view of clusters formed using all features, compressed using PCA.

---

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
Ensure the dataset file is named:
```javascript
House Prediction Data Set.csv
```
Run the script:
```bash
python kmeans.py
```


#### **src/segment_customers.py**  
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
data = pd.read_csv("data/customers.csv")
features = ["Annual Spend", "Frequency", "Recency"]
X = data[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualize
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x="Annual Spend", y="Frequency", hue="Cluster", palette="Set2")
plt.title("Customer Segments")
plt.savefig("customer_clusters.png")

data.to_csv("data/segmented_customers.csv", index=False)
print("✅ Segmentation complete. Results saved to data/segmented_customers.csv")

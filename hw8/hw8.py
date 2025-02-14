import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score

file_path = 'data.csv'
data = pd.read_csv(file_path)

numeric_columns = [
    'tempo', 'beats', 'chroma_stft', 'rmse', 'spectral_centroid', 
    'spectral_bandwidth', 'rolloff', 'zero_crossing_rate'
] + [f'mfcc{i}' for i in range(1, 21)]

# 提取數值數據
X = data[numeric_columns]
print("\n數值屬性數據樣本:")
print(X.head())

# 規範化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部法則確定最佳群數
print("\n正在計算肘部法則...")
inertia = []
k_values = range(2, 11) 
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 繪製肘部法則圖
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# 選擇群數 (手動調整為適合的值)
optimal_clusters = 5  # 例如根據肘部法則選擇 5 群
print(f"\n選擇的群數: {optimal_clusters}")

# KMeans()
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# AgglomerativeClustering()
agglomerative = AgglomerativeClustering(n_clusters=optimal_clusters)
agglomerative_labels = agglomerative.fit_predict(X_scaled)

# DBSCAN()
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 獲取聚類標籤
labels = kmeans.labels_
data['Cluster'] = labels
print("\n聚類結果（部分樣本）:")
print(data[['filename', 'label', 'Cluster']].head())

# 計算 KMeans Silhouette Score
silhouette_kmeans = silhouette_score(X_scaled, labels)
print(f"\nKMeans Silhouette Score: {silhouette_kmeans:.2f}")

# 計算 AgglomerativeClustering Silhouette Score
silhouette_agglomerative = silhouette_score(X_scaled, agglomerative_labels)
print(f"AgglomerativeClustering Silhouette Score: {silhouette_agglomerative:.2f}")

# 過濾掉 DBSCAN 的噪音點（標註為 -1 的點）
dbscan_labels_filtered = dbscan_labels[dbscan_labels != -1]

# 檢查過濾後的標籤是否有樣本
if len(dbscan_labels_filtered) > 0:
    # 計算 DBSCAN 的 Silhouette Score
    silhouette_dbscan = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels_filtered)
    print(f"DBSCAN Silhouette Score: {silhouette_dbscan:.2f}")
else:
    print("DBSCAN 聚類未生成有效的群集，無法計算 Silhouette Score。")

# 使用 PCA 將數據降到 2 維進行可視化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 繪製 KMeans 聚類結果
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', s=50, label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title("K-Means Clustering Results (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.legend()
plt.show()

# 顯示每群的分布
print("\n每群的分布:")
print(data['Cluster'].value_counts())

# 可選：保存聚類結果到文件
output_file = 'clustered_data.csv'
data.to_csv(output_file, index=False)
print(f"\n聚類結果已保存至文件: {output_file}")

# 使用 normalized_mutual_info_score 進行聚類效果評估
# 假設真實標籤為 'label'，請根據您的數據結構調整
# true_labels = data['label']  # 假設 'label' 為真實標籤欄位

# 計算 Normalized Mutual Information (NMI) 評估 KMeans 聚類結果
# nmi_kmeans = normalized_mutual_info_score(true_labels, labels)
# print(f"\nKMeans 與真實標籤的 NMI: {nmi_kmeans:.4f}")

# 假設已經使用 DBSCAN 進行聚類
# nmi_dbscan = normalized_mutual_info_score(true_labels_filtered, dbscan_labels_filtered)
# print(f"DBSCAN 與真實標籤的 NMI: {nmi_dbscan:.4f}")

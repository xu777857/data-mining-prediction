# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 17:22:01 2025

@author: xu188
"""

# ==============================================================================
# 完整的寿命数据分析与建模脚本 (已添加聚类和神经网络)
# ==============================================================================
# --- 步骤1: 环境配置与库导入 ---
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, silhouette_score
from sklearn.inspection import permutation_importance

sns.set(style='whitegrid')

# --- 步骤2: 定义参数 ---
DATA_PATH = 'Updated Quality of Life Data.csv' 
OUTPUT_DIR = 'output_plots'

# --- 步骤3: 数据加载与初步预览 ---
print("--- 开始数据分析流程 ---")
print("\n步骤 1: 正在加载数据...")
try:
    df = pd.read_csv(DATA_PATH, encoding='ascii', delimiter=',')
except FileNotFoundError:
    print(f"错误：找不到文件 '{DATA_PATH}'。请检查 'DATA_PATH' 变量是否设置正确。")
    exit()

print(f"数据加载成功，共 {df.shape[0]} 行，{df.shape[1]} 列。")
print("\n数据前5行预览:")
print(df.head())

# --- 步骤4: 数据预处理 ---
print("\n步骤 2: 正在进行数据预处理...")
target_col = 'age_at_death'
if target_col not in df.columns:
    print(f"错误：数据集中未找到目标列 '{target_col}'。")
    exit()

print("\n数据类型:")
print(df.dtypes)
print("\n缺失值统计:")
print(df.isnull().sum())

initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
if df.shape[0] < initial_rows:
    print(f"删除了 {initial_rows - df.shape[0]} 条重复记录。")

if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
    print("已删除 'id' 列。")
    
# 设置显示所有列（None 表示不限制列数）
pd.set_option('display.max_columns', None)


# --- 步骤5: 探索性数据分析 (EDA) ---
print("\n步骤 3: 正在执行探索性数据分析 (EDA)...")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"已创建输出目录 '{OUTPUT_DIR}'。")

numeric_df = df.select_dtypes(include=[np.number])

# --- 相关性热力图 ---
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    plt.close()
    print("已保存相关性热力图。")

# --- 【新增】特征值配对图 (Pair Plot) ---
# 为了使用 hue='gender'，需要将 gender 列与数值特征合并
if 'gender' in df.columns:
    pairplot_data = pd.concat([numeric_df, df['gender']], axis=1)
    # 为 pairplot 设置一个较大的画布尺寸
    plt.figure(figsize=(15, 15)) 
    sns.pairplot(data=pairplot_data, hue='gender', diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
    plt.suptitle('Pair Plot of Numeric Features Colored by Gender', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pairplot_numeric_features.png'))
    plt.close()
    print("已保存数值特征的配对图。")

# --- 数值特征直方图 ---
numeric_features = numeric_df.columns
for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Histogram of {feature}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'histogram_{feature}.png'))
    plt.close()
print("已保存所有数值特征的直方图。")

# --- 分类特征计数图 ---
categorical_features = ['gender', 'occupation_type']
for feature in categorical_features:
    if feature in df.columns:
        plt.figure(figsize=(10, 4))
        sns.countplot(x=feature, data=df)
        plt.title(f'Count Plot of {feature}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'countplot_{feature}.png'))
        plt.close()
print("已保存分类特征的计数图。")

# --- 数值特征箱线图 ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=numeric_df)
plt.title('Box Plot of Numeric Features')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_numeric_features.png'))
plt.close()
print("已保存数值特征的箱线图。")

# --- 步骤6: 建模数据准备 ---
print("\n步骤 4: 正在准备建模数据...")
X = df.drop(columns=[target_col])
y = df[target_col]
X = pd.get_dummies(X, drop_first=True)
print(f"特征编码后，特征集 X 包含 {X.shape[1]} 个特征。")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"数据已划分为训练集 ({X_train.shape[0]} 条) 和测试集 ({X_test.shape[0]} 条)。")

# =====================================================================================
# 【新增】步骤7: 聚类分析 (按工作/休息模式)
# =====================================================================================
print("\n【新增】步骤 5: 正在执行聚类分析...")
# 选择用于聚类的特征
cluster_features = ['avg_work_hours_per_day', 'avg_sleep_hours_per_day', 'avg_exercise_hours_per_day']
X_cluster = df[cluster_features]

# 确定最佳聚类数 K (使用肘部法则)
wcss = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    wcss.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal K (Clustering)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_method.png'))
plt.close()
print("已保存肘部法则图，用于确定最佳聚类数。")

# 选择一个K值进行聚类 (例如 K=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['lifestyle_cluster'] = kmeans.fit_predict(X_cluster)

# 可视化聚类结果
plt.figure(figsize=(10, 8))
# 使用PCA降维到2D进行可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)
# 【新增】分析PCA主成分的具体含义
# =====================================================================================
print("\n--- 分析PCA主成分的含义 ---")

# 1. 获取特征载荷矩阵
# pca.components_ 是一个形状为 (n_components, n_features) 的数组
# 每一行代表一个主成分，每一列代表一个原始特征
loadings = pca.components_.T  # 转置后，每一行对应一个原始特征，每一列对应一个主成分

# 2. 创建一个易于阅读的DataFrame
loadings_df = pd.DataFrame(
    loadings,
    columns=['PC1', 'PC2'],
    index=X_cluster.columns  # 使用原始特征的名称作为索引
)

print("\n特征载荷矩阵 (Feature Loadings):")
print(loadings_df)

# 3. 解释主成分的含义
print("\n--- 主成分含义解释 ---")
print("\n对于主成分1 (PC1):")
pc1_corr = loadings_df['PC1'].sort_values(ascending=False)
print(pc1_corr)
print("解读：PC1主要由 '{}' (正相关) 和 '{}' (负相关) 构成。"
      .format(pc1_corr.index[0], pc1_corr.index[-1]))

print("\n对于主成分2 (PC2):")
pc2_corr = loadings_df['PC2'].sort_values(ascending=False)
print(pc2_corr)
print("解读：PC2主要由 '{}' (正相关) 和 '{}' (负相关) 构成。"
      .format(pc2_corr.index[0], pc2_corr.index[-1]))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['lifestyle_cluster'], cmap='viridis', s=50, alpha=0.7)
plt.title('Lifestyle Clusters (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clusters_visualization.png'))
plt.close()
print("已保存聚类可视化图。")

# 分析不同聚类的寿命差异
plt.figure(figsize=(10, 6))
sns.boxplot(x='lifestyle_cluster', y='age_at_death', data=df)
plt.title('Age at Death by Lifestyle Cluster')
plt.xlabel('Lifestyle Cluster')
plt.ylabel('Age at Death')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'age_by_cluster.png'))
plt.close()
print("已保存不同生活方式群体的寿命分布箱线图。")

# 打印每个聚类的特征均值
cluster_means = df.groupby('lifestyle_cluster')[cluster_features + [target_col]].mean()
print("\n各生活方式群体的特征均值:")
print(cluster_means)

# =====================================================================================
# 【新增】步骤8: 预测建模 - 神经网络
# =====================================================================================
print("\n【新增】步骤 6: 正在训练神经网络模型...")
# 神经网络对数据尺度敏感，需要进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化并训练MLP神经网络模型
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred_mlp = mlp_model.predict(X_test_scaled)

# 【新增】步骤8: 预测建模 - 神经网络
print("\n【新增】步骤 6: 正在训练神经网络模型...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

y_pred_mlp = mlp_model.predict(X_test_scaled)
r2_mlp = r2_score(y_test, y_pred_mlp)
print(f'R² Score for Neural Network (MLP) Model: {r2_mlp:.3f}')


# 【新增】保存模型、Scaler 和特征列名，以便后续使用
# =====================================================================================
import joblib
print("\n正在保存训练好的模型、Scaler 和特征列名...")
joblib.dump(mlp_model, 'mlp_lifespan_predictor.joblib')
joblib.dump(scaler, 'scaler.joblib')
# 【新增】保存训练时最终使用的特征列名
joblib.dump(X_train.columns, 'feature_names.joblib') 
print("模型、Scaler 和特征列名已保存。")


# --- 步骤9: 预测建模 - 随机森林 (原有方法) ---
print("\n步骤 7: 正在训练随机森林模型...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
# =====================================================================================
# 【新增】保存训练好的随机森林模型
# =====================================================================================
import joblib
MODEL_FILENAME = 'random_forest_lifespan_predictor.joblib'
joblib.dump(rf_model, MODEL_FILENAME)
print(f"模型已成功保存为 '{MODEL_FILENAME}' 文件。")

# 同时，为了后续预测时能正确处理数据，我们还需要保存特征的名称
FEATURES_FILENAME = 'feature_names.joblib'
joblib.dump(X.columns.tolist(), FEATURES_FILENAME)
print(f"特征名称已成功保存为 '{FEATURES_FILENAME}' 文件。")


r2_rf = r2_score(y_test, y_pred_rf)
print(f'R² Score for Random Forest Model: {r2_rf:.3f}')

# --- 步骤10: 模型性能评估 ---
print("\n步骤 8: 模型性能对比...")
print(f"神经网络 (MLP) R²: {r2_mlp:.3f}")
print(f"随机森林 R²: {r2_rf:.3f}")

# --- 步骤11: 特征重要性分析 ---
print("\n步骤 9: 正在分析特征重要性...")
perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10, 8))
plt.barh(np.array(X.columns)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel('Permutation Importance Mean')
plt.title('Feature Importance based on Permutation Importance (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
plt.close()
print("已保存特征重要性图。")

print("\n--- 所有任务执行完毕！ ---")
print(f"所有图表已保存至 '{OUTPUT_DIR}' 文件夹。")
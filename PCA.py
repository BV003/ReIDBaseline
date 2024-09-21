import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def read_features_from_file(file_name):
    features = []
    labels = []
    
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Number of clusters:'):
                continue  # 忽略这行
            elif line.startswith('Feature'):
                # 分割特征和标签
                parts = line.split('Cluster label:')
                if len(parts) == 2:
                    feature_str = parts[0].split('Feature')[1].strip()  # 获取特征字符串
                    label_str = parts[1].strip()
                    
                    # 去除特征字符串中的逗号和冒号
                    feature_str = feature_str.replace(',', '').replace(':', '').strip()
                    
                    # 转换特征为浮点数列表
                    try:
                        feature = list(map(float, feature_str.split()))
                        features.append(feature)
                        labels.append(int(label_str))
                    except ValueError as e:
                        print(f"Error converting feature to float: {e}, feature_str: '{feature_str}'")
                        continue  # 跳过这个特征，继续下一个
    
    return np.array(features), np.array(labels)

def perform_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features)
    return features_reduced

def visualize_data(features_reduced, labels):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(features_reduced[mask, 0], features_reduced[mask, 1], label=f'Cluster {label}')
    
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_visualization.png')  # 保存图形到文件
    plt.show()  # 尝试显示图形

def main():
    input_file = 'PKUrgb.txt'  # 输入文件名

    # 读取特征和标签
    features, labels = read_features_from_file(input_file)

    # 执行PCA降维
    features_reduced = perform_pca(features)

    # 可视化数据
    visualize_data(features_reduced, labels)

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def calculate_similarity_distribution(csv_file_path):
    """
    读取CSV文件中的特征向量，先使用PCA进行特征降维，再计算某一样本与同类及不同类样本的特征相似度分布，并绘制线图展示
    :param csv_file_path: 保存特征向量的CSV文件路径
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file_path)
    features = data.iloc[:, :576].values  # 获取特征部分
    labels = data.iloc[:, -1].values  # 获取标签部分

    # 使用PCA进行特征降维，这里将特征降维到2维，你可根据实际情况调整
    pca = PCA(n_components=10)
    reduced_features = pca.fit_transform(features)

    num_classes = len(np.unique(labels))  # 获取类别数量

    # 选取第一个样本作为目标样本（你可以修改这里的索引来选择不同样本）
    target_sample_feature = reduced_features[0]
    target_sample_label = labels[0]

    similarity_with_same_class = []
    similarity_with_different_class = []

    for sample_idx in range(len(reduced_features)):
        current_feature = reduced_features[sample_idx]
        current_label = labels[sample_idx]
        cosine_similarity_value = cosine_similarity([target_sample_feature], [current_feature])[0][0]
        similarity = 1 - cosine_similarity_value
        if current_label == target_sample_label:
            similarity_with_same_class.append(similarity)
        else:
            similarity_with_different_class.append(similarity)

    # 生成横坐标数据，这里简单使用索引作为横坐标，你也可以根据实际情况调整
    x_same_class = np.arange(len(similarity_with_same_class))
    x_different_class = np.arange(len(similarity_with_different_class))

    # 绘制线图
    plt.figure(figsize=(10, 6))
    plt.plot(x_same_class, similarity_with_same_class, label='Same Class', marker='o')
    plt.plot(x_different_class, similarity_with_different_class, label='Different Class', marker='s')
    plt.xlabel('Sample Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Distribution between Target Sample and Others')
    plt.legend()
    plt.show()


def calculate_euclidean_similarity_distribution(csv_file_path):
    """
    读取CSV文件中的特征向量，使用欧几里得距离计算同一类别的特征相似度分布，并绘制相似度曲线
    :param csv_file_path: 保存特征向量的CSV文件路径
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file_path)
    features = data.iloc[:, :576].values  # 获取特征部分
    labels = data.iloc[:, -1].values  # 获取标签部分

    num_classes = len(np.unique(labels))  # 获取类别数量

    similarity_matrix = np.zeros((num_classes, num_classes))  # 用于存储不同类别之间的平均相似度

    for class_idx in range(num_classes):
        indices = np.where(labels == class_idx)[0]  # 获取当前类别样本的索引
        class_features = features[indices]  # 获取当前类别的特征向量

        similarity_sum = 0
        pair_count = 0

        for i in range(len(class_features)):
            for j in range(i + 1, len(class_features)):
                distance = euclidean(class_features[i], class_features[j])
                similarity = 1 / (1 + distance)  # 将欧几里得距离转换为相似度，距离越小相似度越高
                similarity_sum += similarity
                pair_count += 1

        if pair_count > 0:
            similarity_matrix[class_idx, class_idx] = similarity_sum / pair_count  # 计算当前类别的平均相似度

    # 绘制相似度曲线
    plt.figure(figsize=(10, 6))
    for class_idx in range(num_classes):
        plt.plot([class_idx], [similarity_matrix[class_idx, class_idx]], 'o', label=f'Class {class_idx}')

    plt.xlabel('Class Index')
    plt.ylabel('Average Similarity')
    plt.title('Feature Euclidean Similarity Distribution within Each Class')
    plt.xticks(range(num_classes))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 计算余弦相似度分布
    calculate_similarity_distribution('feature/extracted_features_2025-01-04-11-23-33.csv')

    # 计算欧几里得相似度分布
    # calculate_euclidean_similarity_distribution('feature/extracted_features_2025-01-02-17-58-59.csv')
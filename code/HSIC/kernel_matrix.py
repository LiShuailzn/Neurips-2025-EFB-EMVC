import numpy as np
import tensorflow as tf
import os

# 基本路径定义
data_base_path = '/mnt/disk1/lishuai/EA-Dataset/YoutubeFace/test_1'
model_base_path = '/mnt/disk1/lishuai/NIPS/teacher_youtube/test_1'
save_base_path = '/mnt/disk1/lishuai/EA-Dataset/YoutubeFace/test_1'

# 循环配置 - i从0到4（共5轮）
# 规律：第i轮对应 train_{i+2}.npy, view_{i+1}_best.h5, view_{i+1}_kernel
for i in range(5):  # 0到4，共5个值
    print(f"\n{'=' * 50}")
    print(f"开始处理第 {i} 轮")
    print(f"{'=' * 50}")

    # 生成当前轮次的文件路径
    train_file = os.path.join(data_base_path, f'train_{i + 1}.npy')
    model_file = os.path.join(model_base_path, f'view_{i}_best.h5')
    kernel_dir = os.path.join(save_base_path, f'view_{i}_kernel')

    # 检查文件是否存在
    if not os.path.exists(train_file):
        print(f"警告：训练数据文件 {train_file} 不存在，跳过本轮")
        continue
    if not os.path.exists(model_file):
        print(f"警告：模型文件 {model_file} 不存在，跳过本轮")
        continue

    try:
        # 1. 加载训练数据和教师模型
        print(f"加载训练数据: {train_file}")
        X_train = np.load(train_file)

        print(f"加载教师模型: {model_file}")
        teacher_model = tf.keras.models.load_model(model_file)

        # 2. 构建logits提取模型
        try:
            logits_layer = teacher_model.get_layer('logits')
        except ValueError:
            raise ValueError("模型中找不到名为 'logits' 的层，请检查模型定义")
        logits_model = tf.keras.Model(inputs=teacher_model.input, outputs=logits_layer.output)

        # 3. 获取预测类别
        print("获取预测类别...")
        probabilities = teacher_model.predict(X_train, batch_size=512, verbose=1)
        pseudo_labels = np.argmax(probabilities, axis=1)
        unique_labels = np.unique(pseudo_labels)
        print(f"找到 {len(unique_labels)} 个独特类别")

        # 4. 初始化存储结构
        class_data = {label: [] for label in unique_labels}
        feature_matrices = {}
        cost_matrices = {}

        # 5. 数据划分
        for idx, label in enumerate(pseudo_labels):
            class_data[label].append(X_train[idx])

        # 6. 处理每个类别
        for label in unique_labels:
            class_samples = np.array(class_data[label])

            # 获取logits（未经过softmax）
            logits = logits_model.predict(class_samples, batch_size=512, verbose=0)

            # 转置为特征矩阵（u × b）
            feature_matrix = logits.T
            feature_matrices[label] = feature_matrix

            # 计算核矩阵（u × u）
            cost_matrix = np.matmul(feature_matrix, feature_matrix.T)
            cost_matrices[label] = cost_matrix

            # 创建保存目录
            class_folder = os.path.join(kernel_dir, f'class_{label}')
            os.makedirs(class_folder, exist_ok=True)

            # 保存特征矩阵与核矩阵
            np.save(os.path.join(class_folder, 'feature_matrix.npy'), feature_matrix)
            np.save(os.path.join(class_folder, 'kernel_matrix.npy'), cost_matrix)

        # 打印前三个类别的形状信息
        labels = list(feature_matrices.keys())[:3]  # 取前三个类别的标签
        for label in labels:
            feature_shape = feature_matrices[label].shape
            kernel_shape = cost_matrices[label].shape
            print(f"类别 {label} ：特征矩阵形状 = {feature_shape}，核矩阵形状 = {kernel_shape}")

        print(f"第 {i + 1} 轮处理完成，结果保存至: {kernel_dir}")

    except Exception as e:
        print(f"第 {i + 1} 轮处理出错: {str(e)}")
        continue

print("\n所有轮次处理完毕")


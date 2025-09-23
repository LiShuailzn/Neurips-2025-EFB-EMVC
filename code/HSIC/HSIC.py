import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm


def center_kernel_matrix(K):
    """中心化核矩阵 H K H"""
    n = tf.shape(K)[0]
    I = tf.eye(n, dtype=tf.float32)
    ones = tf.ones((n, n), dtype=tf.float32)
    H = I - ones / tf.cast(n, tf.float32)
    K_centered = tf.matmul(tf.matmul(H, K), H)
    return K_centered

def compute_HSIC(K1, K2):
    """计算两个核矩阵的HSIC分数"""
    K1_c = center_kernel_matrix(K1)
    K2_c = center_kernel_matrix(K2)
    n = tf.cast(tf.shape(K1)[0], tf.float32)
    hsic_value = tf.linalg.trace(tf.matmul(K1_c, K2_c)) / (n ** 2)
    return hsic_value


def load_kernel_matrix(base_path, view_index, class_index, num_classes, small_value=1e-8):
    """加载指定view和类别的核矩阵"""
    path_feat = os.path.join(
        base_path,
        f"view_{view_index}_kernel",
        f"class_{class_index}",
        "kernel_matrix.npy"
    )
    if not os.path.exists(path_feat):
        # 尝试获取预期形状：从已有文件推断或使用默认值
        # 这里假设与其他类形状相同，若获取失败则使用默认形状
        try:
            sample_path = os.path.join(
                base_path,
                f"view_{view_index}_kernel",
                f"class_0",
                "kernel_matrix.npy"
            )
            if os.path.exists(sample_path):
                sample = np.load(sample_path)
                v = sample.shape[0]
            else:
                v = num_classes  # 默认形状
        except:
            v = num_classes  # 最终默认值
        return np.full((v, v), small_value, dtype=np.float32)

    class_feat = np.load(path_feat).astype(np.float32)
    # 确保矩阵是方阵
    if class_feat.shape[0] != class_feat.shape[1]:
        min_dim = min(class_feat.shape)
        class_feat = class_feat[:min_dim, :min_dim]
    return class_feat


def process_view(base_path, view_index, num_classes):
    """处理单个view，计算并保存HSIC矩阵"""
    print(f"\n{'=' * 50}")
    print(f"开始处理 view_{view_index}")
    print(f"{'=' * 50}")

    # 1. 载入所有类别核矩阵
    kernel_matrices = []
    print(f"加载 view_{view_index} 的核矩阵...")
    for i in tqdm(range(num_classes), desc=f"View {view_index} 加载进度"):
        K = load_kernel_matrix(base_path, view_index, i, num_classes, small_value=1e-8)
        kernel_matrices.append(K)

    # 检查所有核矩阵形状是否一致
    shapes = [k.shape for k in kernel_matrices]
    if len(set(shapes)) > 1:
        print(f"警告: view_{view_index} 中核矩阵形状不一致: {set(shapes)}")

    kernel_matrices = tf.stack(kernel_matrices)  # [num_classes, v, v]

    # 2. 初始化HSIC矩阵
    hsic_matrix = tf.Variable(tf.zeros((num_classes, num_classes), dtype=tf.float32))

    # 3. 计算HSIC分数矩阵（对称矩阵，只计算上三角）
    print(f"计算 view_{view_index} 的HSIC矩阵...")
    for i in tqdm(range(num_classes), desc=f"View {view_index} 计算进度"):
        K_i = kernel_matrices[i]
        for j in range(i, num_classes):
            K_j = kernel_matrices[j]
            hsic_val = compute_HSIC(K_i, K_j)
            hsic_matrix[i, j].assign(hsic_val)
            if i != j:
                hsic_matrix[j, i].assign(hsic_val)

    # 4. 保存结果
    save_dir = os.path.join(base_path, f"view_{view_index}_kernel")
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    save_path = os.path.join(save_dir, "cost_matrix.npy")
    np.save(save_path, hsic_matrix.numpy())
    print(f"HSIC矩阵已保存至: {save_path}")
    print(f"view_{view_index} 处理完成")

    return hsic_matrix


def main():
    # 配置参数
    base_path = "/mnt/disk1/lishuai/EA-Dataset/YoutubeFace/test_1"  # 基本路径
    num_classes = 31  # 类别数量
    start_view = 0  # 起始view索引
    end_view = 4  # 结束view索引（包含）

    # 循环处理每个view
    for view_index in range(start_view, end_view + 1):
        try:
            process_view(base_path, view_index, num_classes)
        except Exception as e:
            print(f"处理 view_{view_index} 时出错: {str(e)}")
            continue  # 出错时继续处理下一个view

    print("\n所有view处理完毕")


if __name__ == "__main__":
    main()

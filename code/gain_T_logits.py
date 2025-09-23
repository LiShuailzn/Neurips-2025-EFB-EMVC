import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from code.data_utils.data_util_youtube import get_views

def train_individual(individual_code, view_data_dir, model_base_dir, result_save_base_dir, gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # 获取当前折的视图数据
    view_train_x, train_y, view_test_x, test_y = get_views(view_data_dir=view_data_dir)

    nb_view = (len(individual_code) + 1) // 2
    view_train_xx, view_test_xx = [], []
    for i in individual_code[:nb_view]:
        view_train_xx.append(view_train_x[i])
        view_test_xx.append(view_test_x[i])

    for i in range(nb_view):
        view_id = individual_code[i]
        teacher_input = view_train_xx[i]

        # 构建当前折的模型路径
        model_path = f'{model_base_dir}/view_{view_id}_best.h5'

        # 加载教师模型
        try:
            teacher_model = tf.keras.models.load_model(model_path)
            print(f"Loaded teacher model for view {view_id} from {model_path}")
        except Exception as e:
            print(f"Error loading model for view {view_id} in fold {fold}: {str(e)}")
            continue

        # 检查并获取logits层
        if "logits" not in [layer.name for layer in teacher_model.layers]:
            print(f"'logits' layer not found in model for view {view_id}, skipping...")
            continue

        logits_model = tf.keras.models.Model(
            inputs=teacher_model.input,
            outputs=teacher_model.get_layer("logits").output
        )

        # 获取并保存logits
        teacher_logits = logits_model.predict(teacher_input, batch_size=512)
        print(f"Teacher model {view_id} logits shape: {teacher_logits.shape}")

        # 创建保存目录（如果不存在）
        os.makedirs(result_save_base_dir, exist_ok=True)
        save_path = os.path.join(result_save_base_dir, f'view_{view_id}_logits.npy')
        np.save(save_path, teacher_logits)
        print(f"Saved teacher logits for view {view_id} to {save_path}")


if __name__ == "__main__":
    individual_code = [0, 1, 2, 3, 4,
                       0, 0, 0, 0,]
    gpu = '7'  # 指定GPU
    base_data_dir = '/mnt/disk1/lishuai/EA-Dataset/YoutubeFace'  # 数据根目录
    base_model_dir = '/mnt/disk1/lishuai/NIPS/teacher_youtube'  # 模型根目录

    # 循环处理1到5折
    for fold in range(1, 6):
        print(f"\n===== 开始处理第 {fold} 折数据 =====")

        # 构建当前折的路径
        view_data_dir = f'{base_data_dir}/test_{fold}'  # 数据路径
        model_dir = f'{base_model_dir}/test_{fold}'  # 模型路径
        result_save_dir = f'{view_data_dir}/teacher_logits'  # 结果保存路径

        # 处理当前折
        train_individual(
            individual_code=individual_code,
            view_data_dir=view_data_dir,
            model_base_dir=model_dir,
            result_save_base_dir=result_save_dir,
            gpu=gpu
        )
        print(f"===== 第 {fold} 折数据处理完成 =====\n")

    print("所有折数据处理完毕！")


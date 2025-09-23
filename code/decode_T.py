import tensorflow as tf
from code import config

paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = 128
classes = 31

def sign_sqrt(x):
    # mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-10)

def l2_norm(x):
    return tf.keras.backend.l2_normalize(x, axis=-1)

def code2net_separate(individual_code, nb_feats=[1, 1, 1]):
    nb_view = (len(individual_code) + 1) // 2
    models = []

    for i in range(nb_view):
        view_id = individual_code[i]
        input_x = tf.keras.layers.Input((nb_feats[view_id],))
        x = tf.keras.layers.BatchNormalization()(input_x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x_fc = tf.keras.layers.Dense(units=fused_nb_feats)(x)
        x_relu = tf.keras.layers.ReLU()(x_fc)
        logits = tf.keras.layers.Dense(units=classes, activation=None, name='logits')(x_relu)
        out_x = tf.keras.layers.Activation('softmax', name='output')(logits)
        model = tf.keras.models.Model(inputs=input_x, outputs=out_x)
        models.append(model)
    return models

# 测试代码
individual_code = [0,2,4,0,1]
nb_feats = [512,1024,2048,2048,512,1024,2048,2048] # 假设的特征数

# 调用 code2net_separate 函数，返回多个视图的模型
models = code2net_separate(individual_code, nb_feats=nb_feats)

# 遍历每个模型，打印结构并保存可视化图
for i, model in enumerate(models):
    view_name = f'view_{individual_code[i]}'  # 根据 individual_code 确定视图编号
    print(f"Summary for {view_name}:")
    model.summary()  # 打印模型结构
    # tf.keras.utils.plot_model(model, to_file=f'{view_name}.png', show_shapes=True)  # 保存可视化图


import tensorflow as tf
from code import utils_tree, config

paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = paras['fused_nb_feats']
classes = paras['classes']



idx = 0
def sign_sqrt(x):
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-10)


def l2_norm(x):
    return tf.keras.backend.l2_normalize(x, axis=-1)

def fusion(x1, x2, way='add'):
    if way == fusion_ways[0]:
        fusion_x = tf.keras.layers.Add()([x1, x2])
    if way == fusion_ways[1]:
        fusion_x = tf.keras.layers.Multiply()([x1, x2])
    if way == fusion_ways[2]:
        fusion_x = tf.keras.layers.Concatenate()([x1, x2])
        fusion_x = tf.keras.layers.Dense(units=fused_nb_feats)(fusion_x)
    if way == fusion_ways[3]:
        fusion_x = tf.keras.layers.Maximum()([x1, x2])
    if way == fusion_ways[4]:
        fusion_x = tf.keras.layers.Average()([x1, x2])
    return fusion_x


def code2net_tree(individual_code,  nb_feats=[1024, 2048, 1028],listtree = ''):

    individual_code,nb_view= utils_tree.viewfusion(listtree)
    input_x = []
    x = []
    x_bn = []
    x_dp = []
    outputs = []
    for i in range(nb_view):
        view_id = individual_code[i]
        input_x.append(tf.keras.layers.Input((nb_feats[view_id],)))
        x_bn.append(tf.keras.layers.BatchNormalization()(input_x[i]))
        x_dp.append(tf.keras.layers.Dropout(0.3)(x_bn[i]))
        x.append(tf.keras.layers.Dense(units=fused_nb_feats, activation='relu')(x_dp[i]))
        view_fc_logits = tf.keras.layers.Dense(units=classes, activation=None, name=f'view_{view_id}_output_{i}')(x[i])
        outputs.append(view_fc_logits)

    fusion_x = None
    cnt = 0
    if nb_view == 1:
        fusion_x = x[0]
    else:
        individual_code1, vsize = listtree, nb_view
        listview = []
        for index, i in enumerate(individual_code1):
            if (i[0] != '-'):
                listview.append(cnt)
                cnt += 1
            else:
                e1 = listview[-1]
                listview.pop()
                e2 = listview[-1]
                listview.pop()
                f1 = int(i[1])
                fusion_x = fusion(x1=x[e1], x2=x[e2], way=fusion_ways[f1])
                x.append(fusion_x)
                listview.append(vsize)
                vsize += 1
    fusion_x = tf.keras.layers.BatchNormalization()(fusion_x)
    fusion_x = tf.keras.layers.Lambda(sign_sqrt)(fusion_x)
    fusion_x = tf.keras.layers.Lambda(l2_norm)(fusion_x)

    # fusion_out = tf.keras.layers.Dense(units=classes, activation='softmax', name='fusion_output')(fusion_x)
    fusion_logits = tf.keras.layers.Dense(units=classes, activation=None, name='fusion_logits')(fusion_x)
    fusion_out = tf.keras.layers.Activation('softmax', name='fusion_output')(fusion_logits)
    outputs.insert(0, fusion_out)
    model = tf.keras.models.Model(inputs=input_x, outputs=outputs)
    return model

if __name__ == '__main__':
    model = code2net_tree(individual_code=[], nb_feats=[0,1,2,3,4],
                          listtree =['4a', '0a', '2a', '3a', '-0', '-1', '-0',]

)
    tf.keras.utils.plot_model(model, to_file='fusion_model.ps', show_shapes=True)

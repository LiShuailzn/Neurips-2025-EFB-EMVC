import tensorflow as tf


def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs = tf.shape(w1)[0]
    dim = tf.shape(w1)[1]

    w1 = tf.expand_dims(w1, axis=-1)
    w2 = tf.expand_dims(w2, axis=-1)

    u = 1 / tf.cast(dim, dtype=w1.dtype) * tf.ones_like(w1, dtype=w1.dtype)

    K = tf.exp(-cost / reg)

    Kt = tf.transpose(K, perm=[0, 2, 1])

    for i in range(max_iter):
        v = w2 / (tf.matmul(Kt, u) + 1e-8)
        u = w1 / (tf.matmul(K, v) + 1e-8)

    flow = tf.reshape(u, [bs, -1, 1]) * K * tf.reshape(v, [bs, 1, -1])

    return flow

def wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix=None, sinkhorn_lambda=25, sinkhorn_iter=30):
    pred_student = tf.nn.softmax(logits_student / temperature, axis=-1)
    pred_teacher = tf.nn.softmax(logits_teacher / temperature, axis=-1)

    cost_matrix = tf.nn.relu(cost_matrix) + 1e-8

    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)

    ws_distance = tf.reduce_sum(flow * cost_matrix, axis=-1)
    ws_distance = tf.reduce_sum(ws_distance, axis=-1)
    ws_distance = tf.reduce_mean(ws_distance)

    return ws_distance

def wkd_logit_loss_with_separation(logits_student, logits_teacher, gt_label, temperature, gamma, cost_matrix=None,
                                   sinkhorn_lambda=0.05, sinkhorn_iter=10):
    batch_indices = tf.range(tf.shape(logits_student)[0])

    logits_teacher_batch = tf.gather(logits_teacher, batch_indices)

    if len(gt_label.shape) > 1:
        label = tf.argmax(gt_label, axis=1, output_type=tf.int32)
    else:
        label = gt_label
    label = tf.expand_dims(label, axis=-1)

    N = tf.shape(logits_student)[0]
    c = tf.shape(logits_student)[1]

    s_i = tf.nn.log_softmax(logits_student, axis=1)
    t_i = tf.nn.softmax(logits_teacher_batch, axis=1)

    s_t = tf.gather(s_i, label, batch_dims=1)
    t_t = tf.stop_gradient(tf.gather(t_i, label, batch_dims=1))

    loss_t = -tf.reduce_mean(t_t * s_t)


    mask = tf.one_hot(label, depth=c, on_value=False, off_value=True)
    mask = tf.squeeze(mask, axis=1)


    logits_student_masked = tf.boolean_mask(logits_student, mask)
    logits_teacher_masked = tf.boolean_mask(logits_teacher_batch, mask)


    logits_student_masked = tf.reshape(logits_student_masked, [N, c - 1])
    logits_teacher_masked = tf.reshape(logits_teacher_masked, [N, c - 1])

    cost_matrix = tf.expand_dims(cost_matrix, axis=0)


    cost_matrix = tf.tile(cost_matrix, [N, 1, 1])
    gd_mask = tf.expand_dims(mask, axis=1) & tf.expand_dims(mask, axis=2)
    cost_matrix_masked = tf.boolean_mask(cost_matrix, gd_mask)
    cost_matrix_masked = tf.reshape(cost_matrix_masked, [N, c - 1, c - 1])


    loss_wkd = wkd_logit_loss(logits_student_masked, logits_teacher_masked, temperature, cost_matrix_masked,
                              sinkhorn_lambda, sinkhorn_iter)


    return loss_t + gamma * loss_wkd

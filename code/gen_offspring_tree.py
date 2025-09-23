import random
import copy
from code import utils_tree, utils
from code.config import get_configs

paras = get_configs()
nb_fusion_way = len(paras['fusion_ways'])
nb_view = paras['nb_view']
is_remove = paras['is_remove']


def get_all_nodes_identifier(tree):
    nodes = tree.all_nodes()
    identifiersfph = []
    for node in nodes[:]:
        if tree.parent(node.identifier) is not None:
            identifiersfph.append(node.identifier)
        else:
            nodes.remove(node)
    return nodes, identifiersfph


def get_branch_nodes_identifier(tree):
    all_nodes = tree.all_nodes()
    branch_nodes = []
    identifiersfph = []
    for node in all_nodes[:]:
        if len(tree.is_branch(node.identifier)) != 0 and tree.parent(node.identifier) is not None:
            branch_nodes.append(node)
            identifiersfph.append(node.identifier)
        else:
            all_nodes.remove(node)
    return branch_nodes, identifiersfph

def get_leaf_nodes_identifier(tree):
    nodes = tree.leaves()
    identifiers = [node.identifier for node in nodes]
    return nodes, identifiers


def split_tree(tree, nid):
    tree_copy = copy.deepcopy(tree)
    removed_tree = tree_copy.remove_subtree(nid=nid, identifier=nid)
    return tree_copy, removed_tree


def crossover(tree1, tree2, crossover_rate, is_remove=is_remove, max_deep=15):
    if len(tree1) == 0 or len(tree2) == 0 or len(tree1) == 1 or len(tree2) == 1:
        return tree1, tree2
    r = random.random()
    if (r < crossover_rate):
        tree1_nodes, tree1_identifiers = get_all_nodes_identifier(tree1)
        tree2_nodes, tree2_identifiers = get_all_nodes_identifier(tree2)
        if len(tree1) == 1:
            print("!!!")
        tree1_split_point = random.choice(tree1_nodes)
        tree2_split_point = random.choice(tree2_nodes)

        tree1_split_node = tree1_split_point
        tree2_split_node = tree2_split_point
        node = tree1.parent(tree1_split_node.identifier)
        if (node == None):
            tree1_split_node_parent = tree1_split_node.identifier
        else:
            tree1_split_node_parent = node.identifier
        node = tree2.parent(tree2_split_node.identifier)
        if (node == None):
            tree2_split_node_parent = tree2_split_node.identifier
        else:
            tree2_split_node_parent = node.identifier
        tree1_left, tree1_right = split_tree(tree1, tree1_split_point.identifier)
        tree2_left, tree2_right = split_tree(tree2, tree2_split_point.identifier)
        tree1_left.paste(tree1_split_node_parent, tree2_right)
        tree2_left.paste(tree2_split_node_parent, tree1_right)
        if is_remove:
            tree1_left = quchong(tree1_left)
            tree2_left = quchong(tree2_left)
        if tree1_left.depth() > max_deep:
            tree1_left = quchong(tree1_left)
        if tree2_left.depth() > max_deep:
            tree2_left = quchong(tree2_left)

        return tree1_left, tree2_left
    else:
        if is_remove:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
        if tree2.depth() > max_deep:
            tree2 = quchong(tree2)
        return tree1, tree2


def mutation(tree, mutation_rate, is_remove=is_remove, max_deep=15):
    nodes = tree.all_nodes()

    node = random.choice(nodes)
    idtag = node.tag
    r = random.random()
    if (r < mutation_rate):
        if (idtag[0] == '-'):
            mutation_view = list(range(nb_fusion_way))
            id = random.choice(mutation_view)
            idtag = '-' + str(id)
        else:
            mutation_view = list(range(nb_view))
            id = random.choice(mutation_view)
            idtag = str(id)
        node.tag = idtag
        if is_remove:
            tree = quchong(tree)
    else:
        if is_remove:
            tree = quchong(tree)

    if (tree.depth() > max_deep):
        tree = quchong(tree)
    return tree


def mutation_new_tree_crossover(tree, mutation_rate, is_remove=is_remove, max_deep=15):
    r = random.random()
    if (r < mutation_rate):
        tree_mut = utils_tree.new_tree()
        tree1, tree2 = crossover(tree, tree_mut, 1, is_remove, 15)
        if is_remove:
            tree1 = quchong(tree1)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
        return tree1
    else:
        if is_remove:
            tree = quchong(tree)
        if tree.depth() > max_deep:
            tree = quchong(tree)
        return tree


def quchong(tree_p):
    list_tree = utils_tree.tree_to_list2(tree_p)
    quchong_tree = []
    num_views = 0
    for i in list_tree:
        if i[0] != '-':
            if i not in quchong_tree:
                quchong_tree.append(i)
                num_views += 1
        else:
            if num_views >= 2:
                quchong_tree.append(i)
                num_views -= 1
    quchongtree = utils_tree.list_to_tree(quchong_tree)
    return quchongtree


def gen_offspring(P_t):
    shared_code_acc = utils.load_result()

    # print(shared_code_acc)
    # 1. Crossover
    def select_p():
        while True:
            two = random.sample(range(len(P_t)), 2)
            a1 = '+'.join([str(i) for i in P_t[two[0]]])
            a2 = '+'.join([str(i) for i in P_t[two[1]]])

            if a1 in shared_code_acc and a2 in shared_code_acc:
                p1 = P_t[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else P_t[two[1]]
                return p1  # 返回 p1 如果 a1 和 a2 都存在
    Q_t = []
    while len(Q_t) < len(P_t):
        p1 = select_p()
        p2 = select_p()
        while '+'.join(str(i) for i in p1) == '+'.join(str(i) for i in p2):  ## 防止重复
            p2 = select_p()
        p1_tree = utils_tree.list_to_tree(p1)
        p2_tree = utils_tree.list_to_tree(p2)
        o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])
        o1 = utils_tree.tree_to_list2(o1_tree)
        o2 = utils_tree.tree_to_list2(o2_tree)
        Q_t.append(o1)
        Q_t.append(o2)
    # 2. Mutation
    Q_tt = []
    for p in Q_t:
        p_tree = utils_tree.list_to_tree(p)
        p1_tree = mutation_new_tree_crossover(p_tree, mutation_rate=paras['mutation_rate'])
        p1 = utils_tree.tree_to_list2(p1_tree)
        Q_tt.append(p1)
    Q_t = Q_tt

    return Q_t


def selection(P_t, Q_t):
    shared_code_acc = utils.load_result()
    def select_p1(select_pool):
        while True:
            two = random.sample(range(len(select_pool)), 2)
            a1 = '+'.join([str(i) for i in select_pool[two[0]]])
            a2 = '+'.join([str(i) for i in select_pool[two[1]]])

            a1_valid = a1 in shared_code_acc
            a2_valid = a2 in shared_code_acc

            if a1_valid and a2_valid:
                p1 = select_pool[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else select_pool[two[1]]
                return p1
            if not a1_valid:
                continue
            if not a2_valid:
                continue

    P_t1 = []
    Pt_Qt = P_t + Q_t
    while len(P_t1) < len(P_t):
        p = select_p1(Pt_Qt)
        P_t1.append(p)

    max_code = []
    max_code_str = ''
    min_code_str = ''
    for k, v in shared_code_acc.items():
        if v == max(shared_code_acc.values()):
            max_code_str = k
            max_code = k.strip().split('+')
        if v == min(shared_code_acc.values()):
            min_code_str = k
    is_max = False
    for i, v in enumerate(P_t1):
        v_str = utils_tree.tree_list2str(v)
        if v_str == max_code_str:
            is_max = True
            break
    if not is_max:
        min_i = 0
        for i, v in enumerate(P_t1):
            v_str = utils_tree.tree_list2str(v)
            if v_str == min_code_str:
                min_i = i
                break
        P_t1[min_i] = max_code
    return P_t1




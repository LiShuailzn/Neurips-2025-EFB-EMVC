import random
import copy

from code import utils_tree, tree_to_strlist
from code.config import get_configs
paras = get_configs()
fusion_ways = ['add', 'mul', 'cat', 'max', 'avg']
nb_fusion_way = len(fusion_ways)
nb_view = 31


def get_all_nodes_identifier(tree):
    nodes = tree.all_nodes()
    identifiersfph = []
    for node in nodes:
        if(tree.parent(node.identifier) != None):
            identifiersfph.append(node.identifier)
        else:
            nodes.remove(node)
    return nodes, identifiersfph

def get_leaf_nodes_identifier(tree):
    nodes = tree.leaves()
    identifiers = [node.identifier for node in nodes]
    return nodes, identifiers



def split_tree(tree, nid):
    tree_copy = copy.deepcopy(tree)
    removed_tree = tree_copy.remove_subtree(nid=nid, identifier=nid)
    return tree_copy, removed_tree

def crossover(tree1, tree2, crossover_rate = 0.9, is_remove = False,max_deep = 10):
    tree1_nodes, tree1_identifiers = get_all_nodes_identifier(tree1)
    tree2_nodes, tree2_identifiers = get_all_nodes_identifier(tree2)
    tree1_split_point = random.choice(tree1_nodes)
    tree2_split_point = random.choice(tree2_nodes)
    tree1_split_node = tree1_split_point
    tree2_split_node = tree2_split_point

    node = tree1.parent(tree1_split_node.identifier)

    if(node == None):
        tree1_split_node_parent = tree1_split_node.identifier
    else :
        tree1_split_node_parent= node.identifier

    node = tree2.parent(tree2_split_node.identifier)

    if (node == None):
        tree2_split_node_parent = tree2_split_node.identifier
    else:
        tree2_split_node_parent = node.identifier
    tree1_left, tree1_right = split_tree(tree1, tree1_split_point.identifier)
    tree2_left, tree2_right = split_tree(tree2, tree2_split_point.identifier)


    tree1_left.paste(tree1_split_node_parent, tree2_right)
    tree2_left.paste(tree2_split_node_parent, tree1_right)

    if is_remove == True:
        tree1_left = quchong(tree1_left)
        tree2_left = quchong(tree2_left)
    if tree1_left.depth() > max_deep:
        tree1_left = quchong(tree1_left)
    if tree2_left.depth() > max_deep:
        tree2_left = quchong(tree2_left)

    return tree1_left,tree2_left

def mutation(tree):
    nodes = tree.all_nodes()

    node = random.choice(nodes)
    idtag = node.tag
    if(idtag[0] == '-'):
        mutation_view = list(range(nb_fusion_way))
        print(mutation_view)
        id = random.choice(mutation_view)
        idtag = '-'+ str(id)
    else:
        mutation_view = list(range(nb_view))
        print(mutation_view)
        id = random.choice(mutation_view)
        idtag = str(id)
    node.tag = idtag
    return tree

def quchong(tree_p):
    list_tree = tree_to_strlist.tree_to_list2(tree_p)
    lens = len(list_tree)
    quchong_tree = []
    """
    这里可以模拟出来 用融合方式
    """
    num_views = 0
    for i in list_tree:
        if i[0] != '-':
            if i not in quchong_tree:
               quchong_tree.append(i)
               num_views +=1
        else:
            if num_views >=2:
                quchong_tree.append(i)
                num_views-=1
    quchongtree = utils_tree.list_to_tree(quchong_tree)
    return quchongtree


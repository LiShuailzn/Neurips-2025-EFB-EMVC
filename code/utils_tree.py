from  treelib import  Tree
from code import random_tree, utils, config
import random
feature_statistics = config.get_configs()['feature_statistics']
def tree_to_list2(tree):
    iter_tree = tree.expand_tree()
    listhead = []
    for i in iter_tree:
        listhead.append(tree.get_node(i))
    lists = [i.tag for i in listhead]
    lists = list(reversed(lists))
    return lists

def list_to_tree(strtrees):
    idx = utils.idxx
    stackstree = []
    k = 0
    for treenode in strtrees:
        k += 1
        node = treenode
        tree = Tree()
        if (node[0] != '-'):
            tree.create_node(tag=treenode, identifier=idx)
            stackstree.append(tree)
        else:
            if (k != len(strtrees)):
                tree.create_node(tag=treenode, identifier=idx)
            else:
                tree.create_node(tag=treenode, identifier=idx)
            tree.paste(idx, stackstree[-1])
            stackstree.pop()
            if len(stackstree) == 0 :
                print('Empty')

            tree.paste(idx, stackstree[-1])

            stackstree.pop()
            stackstree.append(tree)
        idx = idx + 1
    treepop = stackstree[0]
    utils.idxx = idx
    return treepop

def tree_list2str(list1):
    return '+'.join([str(i) for i in list1])

def viewfusion(liststr):
    views = []
    fusions = []
    viewnum = 0
    for i in liststr:
        if(i[0] != '-' and i[1] >= 'a' and i[1] <= 'z'):
            views.append(int(i[0]))
            viewnum +=1
        else:
            fusions.append(int(i[1]))
    view_fusion_code = views + fusions
    return view_fusion_code,viewnum

def new_tree():
    fusion_ways = config.get_configs()['fusion_ways']
    views = config.get_configs()['nb_view']

    view_code = random.sample(range(0, views), k=random.randint(2, views))
    fusion_code = random.choices(range(0, len(fusion_ways)), k=len(view_code) - 1)
    sta_code = random.choices(range(0, len(feature_statistics)), k=len(view_code))
    str_list = [feature_statistics[num] for num in sta_code]
    view_code = [str(num) for num in view_code]
    view_codes = []
    for i in range(len(view_code)):
        view_codes.append(view_code[i] + str_list[i])

    pop_tree = random_tree.randomTree(view_codes, fusion_code)
    return pop_tree


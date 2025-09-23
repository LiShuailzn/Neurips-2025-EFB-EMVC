from  treelib import  Tree
from code import utils


def tree_to_list(tree):
    listhead =list(reversed(tree.all_nodes()))
    lists = [ i.tag for i in listhead]
    return lists

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
    idxroot = 0
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
                tree.create_node(tag=treenode, identifier=idx)  ## 创造一个子树
            else:
                tree.create_node(tag=treenode, identifier=idx)
            tree.paste(idx, stackstree[-1])
            stackstree.pop()
            tree.paste(idx, stackstree[-1])
            stackstree.pop()
            stackstree.append(tree)
        idx = idx + 1
    treepop = stackstree[0]
    """
    order 后序
    """
    treepoporder = list(reversed(treepop.all_nodes()))
    utils.idxx = idx
    return treepop

def tree_list2str(list1):
    return '-'.join([str(i) for i in list1])

def viewfusion(liststr):
    views = []
    fusions = []
    viewnum = 0
    for i in liststr:
        if(i[0] != '-' and i[1] >='a' and i[1] <='z'):
            views.append(int(i[0]))
            viewnum +=1
        else:
            fusions.append(int(i[1]))
    view_fusion_code = views + fusions
    return view_fusion_code,viewnum
def viewlist(liststr):
    views = []
    fusions = []
    viewnum = 0
    for i in liststr:
        if(i[0] != '-'):
            views.append(int(i))
            viewnum +=1
        else:
            fusions.append(int(i[1]))
    return views
def viewsize(liststr):
    viewnum = 0
    for i in liststr:
        if(i[0] == '-'):
            viewnum += 1
    return viewnum + 1
from  treelib import  Tree
import  random
from code import utils


def randomTree(viewslist,fusionslist):
    idx = utils.idxx
    strviews = [str(i)  for i  in viewslist]
    strfusion = ['-' + str(i) for i in fusionslist]
    viewsize = len(strviews)
    fusionsize = len(strfusion)
    for i in range(0, len(strfusion)):
        tree_index1 = random.randrange(viewsize)
        v1 = strviews[tree_index1]
        del strviews[tree_index1]
        viewsize -= 1
        tree_index2 = random.randrange(viewsize)
        v2 = strviews[tree_index2]
        del strviews[tree_index2]
        viewsize -= 1

        ftree_index = random.randrange(fusionsize)
        f = strfusion[ftree_index]
        del strfusion[ftree_index]
        fusionsize -= 1
        strviews.append(' ' + v1 + ' ' + v2 + ' ' + f + ' ')
        viewsize += 1
    strtrees = strviews[0].split()
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
            tree.paste(idx, stackstree[-1])
            stackstree.pop()
            stackstree.append(tree)
        idx = idx + 1
    treepop = stackstree[0]
    utils.idxx = idx
    return treepop

if __name__ == '__main__':
    tree1 = randomTree([1,2,3],[0,4])

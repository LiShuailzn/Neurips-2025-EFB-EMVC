import random
from code import utils_tree, random_tree, tree_to_strlist, utils, config

feature_statistics = config.get_configs()['feature_statistics']

def generate_population_tree(views=10, pop_size=15, verbose=1):
    fusion_ways = config.get_configs()['fusion_ways']
    population = []
    population_set_tree = set()
    while len(population) < pop_size:
        view_code = random.sample(range(0, views), k=random.randint(2, views))
        fusion_code = random.choices(range(0, len(fusion_ways)), k=len(view_code) - 1)
        sta_code = random.choices(range(0, len(feature_statistics)), k=len(view_code))
        str_list = [feature_statistics[num]for num in sta_code]
        view_code = [str(num) for num in view_code]
        view_codes = []
        for i in range(len(view_code)):
            view_codes.append(view_code[i] + str_list[i])
        pop_tree = random_tree.randomTree(view_codes, fusion_code)
        pop = utils_tree.tree_to_list2(pop_tree)
        if verbose == 1:
            print(f'view_code:{view_code}')
            print(f'fusion_code:{fusion_code}')
            print(f'pop:{pop}')
            print('=' * 30)
        if tree_to_strlist.tree_list2str(pop) not in population_set_tree:
            population.append(pop)
            population_set_tree.add(utils.list2str(pop))
    return population

if __name__ == '__main__':
    population = generate_population_tree()
    for i in population:
        print(i)



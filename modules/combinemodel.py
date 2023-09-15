from itertools import combinations

class CombineModels:

    def __init__(self, models) -> None:
        self.models = models

    def combine(self):
        l = [list(combinations(self.models, n)) for n in range(len(self.models)+1)]
        comb_list = []
        for i in l:
            for j in i:
                comb_list.append(j)
        return comb_list[1:]

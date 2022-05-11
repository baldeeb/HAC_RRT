import numpy as np

class RrtNode():
    def __init__(self, config):
        self.config = np.array(config).reshape(-1,1)

    def __getitem__(self, i): return self.config[i]
    
    def norm(self): return np.linalg.norm(self.config)

    def __sub__(self, other):
        return RrtNode([a-b for a, b in zip(self.config, other)])

    def __add__(self, other):
        return RrtNode([a+b for a, b in zip(self.config, other)])
    
    def __mul__(self, x):
        return RrtNode([a*x for a in self.config])

    def __truediv__(self, x):
        return RrtNode([a/x for a in self.config])

    def as_vector(self):
        return np.array(self.config).reshape(1,-1)

def get_tree_branches(tree):
    '''
    tree: is a dict where keys indicate current node and the 
        values indicates the parent node
    returns a list of start and end points indicating branches
    '''
    branches = []
    for k, v in tree.items():
        if k and v: 
            branches.append([v.config, k.config])
    return branches


def take_euclidean_step(start, end, step_size):
    delta = start - end
    direction = delta / np.linalg.norm(delta)
    return start + (step_size * direction) 
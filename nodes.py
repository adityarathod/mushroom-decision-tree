import numpy as np
import pandas as pd
import metrics


class MultiBranchNode:
    children_nodes = {}
    x = None
    y = None
    call_ct = 0
    node_depth = 0
    best_feat_idx = 0

    def __init__(self, x, y, depth=0):
        self.children_nodes = {}
        self.x = x
        self.y = y
        self.node_depth = depth

    def feature_select(self):
        info_gains = {}
        num_feats = len(self.x[0, :])
        for i in range(num_feats):
            cur_feat = self.x[:, i]
            info_gains[i] = metrics.info_gain(
                pd.Series(cur_feat, dtype='category').cat.codes,
                pd.Series(self.y, dtype='category').cat.codes
            )
        max_idx = max(info_gains, key=info_gains.get)
        if info_gains[max_idx] <= 0.0001:
            raise ValueError('The information gain is zero.')
        return max_idx

    def partition_node(self):
        self.call_ct += 1
        try:
            self.best_feat_idx = self.feature_select()
            uniq_x_vals = np.unique(self.x[:, self.best_feat_idx])
            for x_u in uniq_x_vals:
                msk = self.x[:, self.best_feat_idx] == x_u
                self.children_nodes[x_u] = MultiBranchNode(self.x[msk], self.y[msk], depth=self.node_depth+1)
            self.partition_children()
        except ValueError:
            self.children_nodes[0] = TerminalNode(np.unique(self.y)[0], self.node_depth + 1)

    def partition_children(self):
        for cmp, node in self.children_nodes.items():
            node.partition_node()

    def evaluate(self, value):
        if len(self.children_nodes.keys()) == 1:
            return self.children_nodes[0].evaluate(value)
        for comparison in self.children_nodes.keys():
            if comparison == value[self.best_feat_idx]:
                return self.children_nodes[comparison].evaluate(value)

    def print(self, col_names=None):
        for key, node in self.children_nodes.items():
            if not isinstance(node, TerminalNode):
                if isinstance(col_names, list):
                    print('\t' * (self.node_depth + 1) + f'{col_names[self.best_feat_idx]} == {key}')
                else:
                    print('\t' * (self.node_depth + 1) + f'x[{self.best_feat_idx}] == {key}')
                node.print(col_names=col_names)
            else:
                node.print()

    def __repr__(self):
        return str(self.children_nodes)


class TerminalNode:
    val = None
    depth = 0

    def __init__(self, value, depth):
        self.val = value
        self.depth = depth

    def evaluate(self, value):
        return self.val

    def print(self):
        print('\t' * self.depth + f'terminal node, y = {self.val}')

    def __repr__(self):
        return f'<terminal y = {self.val}>'
from nodes import MultiBranchNode
import pandas as pd
import numpy as np


class DecisionTreeModel:
    root_node: MultiBranchNode = None
    debug = True

    def __init__(self, debug=True):
        self.debug = debug

    def __debug(self, output):
        if self.debug:
            print(output)

    @staticmethod
    def __load_data(path='data/train.csv') -> (np.ndarray, np.ndarray):
        df = pd.read_csv(path, header=None).astype('category')
        # df = df.apply(lambda c: c.cat.codes)
        return df.drop(0, axis=1).values, df[0].values

    def fit(self, data_path='data/train.csv'):
        self.__debug(f'[INFO] Loading data from {data_path}...')
        X, y = self.__load_data(data_path)
        self.__debug('[INFO] Creating root node with data...')
        self.root_node = MultiBranchNode(X, y)
        self.__debug('[INFO] Beginning training...')
        self.root_node.partition_node()
        self.__debug('[INFO] Trained.')

    def predict(self, x):
        return self.root_node.evaluate(x)

    def evaluate(self, data_path='data/test.csv'):
        self.__debug(f'[INFO] Loading data from {data_path}...')
        X_t, y_t = self.__load_data(data_path)
        self.__debug(f'[INFO] Evaluating model...')
        num_correct = 0
        total_vals = len(y_t)
        for idx, example in enumerate(X_t):
            num_correct += self.predict(example) == y_t[idx]
        percent_correct = num_correct / total_vals
        self.__debug(f'[DATA] Percent correct: {percent_correct * 100}')
        return percent_correct

    def print_model(self, col_names=None):
        self.root_node.print(col_names=col_names)

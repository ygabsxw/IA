import numpy as np
import pandas as pd
from collections import Counter

class Node:
    """Representa um nó na árvore de decisão."""
    def __init__(self, feature=None, threshold=None, children=None, *, value=None, impurity=None, n_samples=None):
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples

    def is_leaf_node(self):
        return self.value is not None

class BaseDecisionTree:
    """Classe base para os algoritmos de árvore de decisão."""
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        self.feature_types = {col: X[col].dtype for col in self.feature_names}
        X_np = X.values
        y_np = y.values
        self.root = self._grow_tree(X_np, y_np)

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, impurity=self._calculate_impurity(y), n_samples=n_samples)

        best_split = self._find_best_split(X, y)

        if best_split.get('gain', 0) <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, impurity=self._calculate_impurity(y), n_samples=n_samples)

        feature_idx = best_split['feature_idx']
        threshold = best_split['threshold']
        feature_name = self.feature_names[feature_idx]
        
        children_indices_map = self._split(X[:, feature_idx], threshold, feature_name)
        
        children_nodes = {}
        for name, indices in children_indices_map.items():
            if len(indices) == 0:
                children_nodes[name] = Node(value=self._most_common_label(y), impurity=0, n_samples=0)
            else:
                children_nodes[name] = self._grow_tree(X[indices], y[indices], depth + 1)
        
        return Node(feature=feature_name, 
                    threshold=threshold, 
                    children=children_nodes,
                    impurity=self._calculate_impurity(y),
                    n_samples=n_samples)

    def _most_common_label(self, y):
        counter = Counter(y)
        if not counter: return None
        return counter.most_common(1)[0][0]

    def predict(self, X):
        X_test = X.values
        return np.array([self._traverse_tree(x, self.root) for x in X_test])
        
    # Métodos a serem implementados pelas subclasses
    def _find_best_split(self, X, y): raise NotImplementedError
    def _split(self, X_column, threshold, feature_name): raise NotImplementedError
    def _calculate_impurity(self, y): raise NotImplementedError
    def _traverse_tree(self, x, node): raise NotImplementedError

def print_tree(tree, node, depth=0, branch=''):
    """Função para imprimir a árvore de forma legível."""
    if node is None: return
    if node.is_leaf_node():
        print(f"{'|   ' * depth}{branch}Prediz: {node.value} (Impureza: {node.impurity:.2f}, Amostras: {node.n_samples})")
        return
    
    print(f"{'|   ' * depth}{branch}Feature: {node.feature} (Impureza: {node.impurity:.2f}, Amostras: {node.n_samples})")
    
    if isinstance(list(node.children.keys())[0], str): # Categórico Multi-way
        for value, child_node in node.children.items():
            print_tree(tree, child_node, depth + 1, f"├── Se = '{value}': ")
    elif isinstance(list(node.children.keys())[0], bool): # Contínuo ou binário
        if isinstance(node.threshold, (int, float, np.number)):
            print_tree(tree, node.children[True], depth + 1, f"├── <= {node.threshold:.2f}: ")
            print_tree(tree, node.children[False], depth + 1, f"├── > {node.threshold:.2f}: ")
        else: # Categórico binário (CART)
            print_tree(tree, node.children[True], depth + 1, f"├── in {node.threshold[0]}: ")
            print_tree(tree, node.children[False], depth + 1, f"├── not in {node.threshold[1]}: ")
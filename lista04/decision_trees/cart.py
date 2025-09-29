import numpy as np
import pandas as pd
from itertools import chain, combinations
from .base_tree import BaseDecisionTree, Node
from .utils import gini_impurity, gini_gain

class CART(BaseDecisionTree):
    """Implementação do algoritmo CART para classificação."""

    def _calculate_impurity(self, y):
        return gini_impurity(y)

    def _calculate_gain(self, y_parent, y_children):
        return gini_gain(y_parent, y_children)

    def _powerset(self, iterable):
        s = list(iterable)
        # Testa apenas combinações até a metade para evitar duplicatas (ex: {A,B} vs {C} é o mesmo que {C} vs {A,B})
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) // 2 + 1))

    def _split(self, X_column, threshold, feature_name):
        if pd.api.types.is_numeric_dtype(self.feature_types[feature_name]):
            left_indices = np.where(X_column <= threshold)[0]
            right_indices = np.where(X_column > threshold)[0]
        else: # Categórico
            left_categories = set(threshold[0])
            left_indices = np.where(np.isin(X_column, list(left_categories)))[0]
            right_indices = np.where(~np.isin(X_column, list(left_categories)))[0]
        return {True: left_indices, False: right_indices}

    def _find_best_split(self, X, y):
        best_gain = -1
        best_split = {}
        n_features = X.shape[1]
        
        for feat_idx in range(n_features):
            feature_name = self.feature_names[feat_idx]
            X_column = X[:, feat_idx]
            
            if pd.api.types.is_numeric_dtype(self.feature_types[feature_name]):
                unique_values = np.unique(X_column)
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                for threshold in thresholds:
                    children_indices = self._split(X_column, threshold, feature_name).values()
                    y_children = [y[indices] for indices in children_indices]
                    if any(len(child) == 0 for child in y_children): continue
                    gain = self._calculate_gain(y, y_children)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {'gain': gain, 'feature_idx': feat_idx, 'threshold': threshold}
            else: # Categórico
                unique_values = np.unique(X_column)
                if len(unique_values) > 1:
                    for subset in self._powerset(unique_values):
                        right_subset = tuple(set(unique_values) - set(subset))
                        threshold = (subset, right_subset)
                        children_indices = self._split(X_column, threshold, feature_name).values()
                        y_children = [y[indices] for indices in children_indices]
                        if any(len(child) == 0 for child in y_children): continue
                        gain = self._calculate_gain(y, y_children)
                        if gain > best_gain:
                            best_gain = gain
                            best_split = {'gain': gain, 'feature_idx': feat_idx, 'threshold': threshold}
        return best_split

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        feature_idx = self.feature_names.index(node.feature)
        feature_val = x[feature_idx]

        if isinstance(node.threshold, (int, float, np.number)): # Split contínuo
            branch = feature_val <= node.threshold
        else: # Split Categórico binário
            branch = feature_val in node.threshold[0]
            
        return self._traverse_tree(x, node.children[branch])
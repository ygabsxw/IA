import numpy as np
import pandas as pd
from .base_tree import BaseDecisionTree, Node
from .utils import entropy, information_gain

class ID3(BaseDecisionTree):
    """Implementação do algoritmo ID3."""
    
    def _calculate_impurity(self, y):
        return entropy(y)

    def _calculate_gain(self, y_parent, y_children):
        return information_gain(y_parent, y_children)

    def _split(self, X_column, _, feature_name):
        children_indices = {}
        for value in np.unique(X_column):
            children_indices[str(value)] = np.where(X_column == value)[0]
        return children_indices

    def _find_best_split(self, X, y):
        best_gain = -1
        best_split = {}
        n_features = X.shape[1]
        
        for feat_idx in range(n_features):
            feature_name = self.feature_names[feat_idx]
            # ID3 só lida com atributos categóricos/objeto
            if pd.api.types.is_numeric_dtype(self.feature_types[feature_name]):
                continue
            
            X_column = X[:, feat_idx]
            children_indices = self._split(X_column, None, feature_name).values()
            y_children = [y[indices] for indices in children_indices]
            
            if not y_children: continue

            gain = self._calculate_gain(y, y_children)
            if gain > best_gain:
                best_gain = gain
                best_split = {'gain': gain, 'feature_idx': feat_idx, 'threshold': None}
                
        return best_split

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        feature_val = x[self.feature_names.index(node.feature)]
        child_node = node.children.get(str(feature_val))
        
        if child_node is None:
            # Estratégia de fallback: retorna a classe mais comum do nó pai (implícito)
            all_samples_values = [child.value for child in node.children.values() if child.is_leaf_node()]
            return self._most_common_label(all_samples_values) if all_samples_values else list(node.children.values())[0].value
        
        return self._traverse_tree(x, child_node)
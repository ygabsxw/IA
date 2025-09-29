import numpy as np
from collections import Counter
from math import log2

def entropy(y):
    """Calcula a entropia de um conjunto de rótulos."""
    counts = np.array(list(Counter(y).values()))
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9)) # Adiciona epsilon para evitar log(0)

def information_gain(y_parent, y_children):
    """Calcula o ganho de informação."""
    parent_entropy = entropy(y_parent)
    n_parent = len(y_parent)
    children_entropy_weighted_avg = sum(
        (len(child) / n_parent) * entropy(child) for child in y_children
    )
    return parent_entropy - children_entropy_weighted_avg

def split_info(y_children):
    """Calcula a informação de divisão (para a razão de ganho)."""
    n_parent = sum(len(child) for child in y_children)
    proportions = np.array([len(child) / n_parent for child in y_children])
    return -np.sum(proportions * np.log2(proportions + 1e-9))

def gain_ratio(y_parent, y_children):
    """Calcula a razão de ganho."""
    ig = information_gain(y_parent, y_children)
    si = split_info(y_children)
    return ig / si if si > 0 else 0

def gini_impurity(y):
    """Calcula o índice de impureza de Gini."""
    counts = np.array(list(Counter(y).values()))
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities**2)

def gini_gain(y_parent, y_children):
    """Calcula o ganho de Gini (redução na impureza)."""
    parent_gini = gini_impurity(y_parent)
    n_parent = len(y_parent)
    children_gini_weighted_avg = sum(
        (len(child) / n_parent) * gini_impurity(child) for child in y_children
    )
    return parent_gini - children_gini_weighted_avg
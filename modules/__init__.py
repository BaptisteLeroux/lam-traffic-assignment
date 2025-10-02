import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import pinv, null_space, solve, inv
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import truncnorm
from sympy import Matrix
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os

# Optionnel : exposer tout sous un alias unique
__all__ = [
    "np", "matrix_rank", "pinv", "null_space", "solve", "inv",
    "mean_squared_error", "r2_score", "mean_absolute_error",
    "truncnorm", "Matrix", "nx", "pd", "plt", "os"
]

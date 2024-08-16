import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# goal is to take in the t13 hit output csv and convert it to binary level output 
class Binary_Converter:
    def __init__(self):
        pass

    # positive = 1, negative = 0
    def hit_numeric_conv(self, binary_t13_hit_df):
        binary_t13_hit_df = binary_t13_hit_df.replace(['positive', 'negative'], [1,0])
        
        return binary_t13_hit_df
    

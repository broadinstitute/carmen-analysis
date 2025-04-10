import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

#pd.set_option('future.no_silent_downcasting', True) 

# goal is to take in the t13 hit output csv and convert it to binary level output 
class Binary_Converter:
    def __init__(self):
        pass

    # positive = 1, negative = 0
    def hit_numeric_conv(self, binary_t13_hit_df):
        #deprecated approaches
        #binary_t13_hit_df = binary_t13_hit_df.replace(['POSITIVE', 'NEGATIVE'], [1,0]) 
        #binary_t13_hit_df = binary_t13_hit_df.applymap(lambda x: {'POSITIVE': 1, 'NEGATIVE': 0}.get(x, x))
        
        binary_t13_hit_df = binary_t13_hit_df.assign(
                **{col: binary_t13_hit_df[col].map({'POSITIVE': 1, 'NEGATIVE': 0}) for col in binary_t13_hit_df.columns}
            )
        
        return binary_t13_hit_df
    

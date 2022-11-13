
import ast
import random
import pandas as pd
from typing import Type, List, Union, Dict
from enum import Enum

import itertools

class pandas_source:
    def __init__(self, df: pd.DataFrame):

        self.source_df = df

        self.columns = df.columns
        self.range = self.generate_range()
        self.shape = df.shape
        self.description = df.describe()

    def generate_range(self):
        column_data_range = {}

        for col in self.columns:
            dtype = self.source_df[col].dtypes
            if "int" in str(dtype) or "float" in str(dtype) or "date" in str(dtype):
                cur_col_max = self.source_df[col].max()
                cur_col_min = self.source_df[col].min()
                column_data_range[col] = [cur_col_min, cur_col_max]
            else:
                column_data_range[col] = "None"
        return column_data_range

    def merge(self, other: 'pandas_source', left_on, right_on):

        new_df = self.source_df.merge(other.source_df, left_on, right_on)

        return pandas_source(new_df)
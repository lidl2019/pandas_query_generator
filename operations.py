import ast
import random
import pandas as pd
from typing import Type, List, Union, Dict
from enum import Enum
from pandas_query import *
import itertools


class operation:
    def __init__(self, df_name, leading):
        self.df_name = df_name
        self.leading = leading

    def to_str(self):
        # pass
        return ""

    def set_leading(self, b):
        self.leading = b

    def exec(self):
        return eval(self.to_str())


class OP(Enum):
    ge = ">="
    gt = ">"
    le = "<="
    lt = "<"
    eq = "=="


class OP_cond(Enum):
    AND = "&"
    OR = "|"
    # NOT = "-"


class condition():

    def __init__(self, col_name, op: OP, num):
        self.col = col_name
        self.op = op
        self.val = num

    def replace_val(self, val):
        return condition(self.col, self.op, val)
        # self.val = val

    def replace_op(self, op):
        return condition(self.col, op, self.val)
        # self.op = op

    # def to_str(self):

    def __str__(self):
        return f"condition ({self.col} {self.op.value} {self.val} )"


class selection(operation):
    '''
    selection(df_name, [condition(col1, >=, 1), &, condition(col2, <=, 2)])

    '''

    def __init__(self, df_name, conditions, leading=True):
        super().__init__(df_name, leading)
        # self.df_name = df_name
        self.conditions = conditions
        # self.leading = leading
        # self.desire_columns = columns

    def new_selection(self, new_cond):
        return selection(self.df_name, new_cond, self.leading)

    def to_str(self):

        res_str = f"{self.df_name}" if self.leading else ""
        cur_condition = ""

        if len(self.conditions) == 1:
            cur_condition = cur_condition + self.df_name + "[" + "'" + self.conditions[0].col + "'" + "]" + " " + \
                            self.conditions[0].op.value + " " + str(self.conditions[0].val)
            res_str = res_str + "[" + cur_condition + "]"
            return res_str

        for i, condition in enumerate(self.conditions):
            cond = self.conditions[i]
            if type(condition) == OP_cond:
                cur_condition += " " + condition.value + " "
            else:
                cur_condition += "(" + self.df_name + "[" + "'" + cond.col + "'" + "]" + " " + cond.op.value + " " + str(
                    cond.val) + ")"

        res_str = res_str + "[" + cur_condition + "]"
        return res_str

    def __str__(self):
        conditions_ = []
        for c in self.conditions:
            conditions_.append(str(c))
        return f"selection: df_name = {self.df_name} conditions = {conditions_}"

    def exec(self):
        return eval(self.to_str())


class merge(operation):
    def __init__(self, df_name, queries: 'pandas_query', on, leading=False):
        super().__init__(df_name, leading)
        self.operations = queries.operations
        self.queries = queries
        self.on_col = on

    def to_str(self) -> str:
        # print(f"+++++++++++++++++++++{self.on_col}")
        res_str = f"{self.df_name}" if self.leading else ""

        operations_to_str = self.queries.get_query_str(self.queries.pre_gen_query)
        # for op in self.operations:
        #     operations_to_str += op.to_str()

        on_cols = ""
        for col in self.on_col:
            on_cols = on_cols + "'" + col + "'" + ","

        on_cols = on_cols[:-1]

        res_str = res_str + "." + "merge" + "(" + operations_to_str + "," + "on=" + "[" + on_cols + "]" + ")"
        return res_str

    def new_merge(self, new_queries, new_on_col):
        return merge(self.df_name, new_queries, new_on_col, leading=self.leading)

    def exec(self):
        return eval(self.to_str())

    def __str__(self):
        return f"merge: df_name = {self.df_name}, on_col = {self.on_col}"


class projection(operation):
    def __init__(self, df_name, columns, leading=True):
        super().__init__(df_name, leading)
        self.desire_columns = columns
        self.length = len(columns)
        # self.leading = leading
        # self.df_name = df_name

    def to_str(self):
        res_str = f"{self.df_name}" if self.leading else ""

        cur_str = ""
        for column in self.desire_columns:
            cur_str += "'" + column + "'" + ","

        cur_str = cur_str[:-1]

        res_str = res_str + "[[" + cur_str + "]]"

        return res_str

    def new_projection(self, columns):
        return projection(self.df_name, columns, self.leading)

    def __str__(self):
        return f"projection: df_name = {self.df_name}, col = {self.desire_columns}"

    def exec(self):
        return eval(self.to_str())


class group_by(operation):
    def __init__(self, df_name, columns, other_args=None, leading=False):
        super().__init__(df_name, leading)

        self.columns = columns if isinstance(columns, List) else [columns]
        self.other_args = other_args

    def to_str(self):
        other_args = self.other_args if self.other_args else ""
        res_str = f"{self.df_name}" if self.leading else ""
        res_str = res_str + "." + "groupby" + "(" + "by=" + str(self.columns) + other_args + ")"
        return res_str

    def new_groupby(self, columns):
        return group_by(self.df_name, columns, self.other_args, self.leading)

    def __str__(self):
        return f"groupby: {self.columns}"

    def exec(self):
        return eval(self.to_str())


class agg(operation):
    def __init__(self, df_name, dict_columns: Union[str, Dict[str, str]], leading=True):
        """

        :param df_name:
        :param leading:
        :param dict_columns:
        """
        super().__init__(df_name, leading)
        self.dict_key_vals = dict_columns

    def to_str(self):
        res_str = f"{self.df_name}" if self.leading else ""
        res_str = res_str + "." + "agg" + "(" + "'" + str(self.dict_key_vals) + "'" + "," + "numeric_only=True" + ")"
        return res_str

    def new_agg(self, dict_cols):
        return agg(self.df_name, dict_cols, self.leading)

    def __str__(self):
        return f"agg: " + str(self.dict_key_vals)

    def exec(self):
        return eval(self.to_str())

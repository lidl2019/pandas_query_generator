

import ast
import random
import pandas as pd
from typing import Type, List, Union, Dict
from enum import Enum

import itertools
from operations import *
class pandas_query():
    def __init__(self, q_gen_query: List[operation], source: pd.DataFrame = None, verbose = False):

        self.setup_query(q_gen_query)
        if verbose:
            print(self.get_query_str(q_gen_query))
        self._source_ = source # df

        self._source_pandas_q = ""
        self.pre_gen_query = q_gen_query
        self.df_name = q_gen_query[0].df_name
        self.num_merges = 0
        self.operations = [
            "select",
            "merge",
            "order",
            "concat",
            "rename",
            "groupby"
        ]

        self.target = self.execute_query(self.pre_gen_query) # df after operation


    def get_target(self):
        return self.target

    def get_source(self):
        return self._source_.copy()

    def setup_query(self, list_operation: List[operation]):
        for i, op in enumerate(list_operation):
            if i != 0:
                op.set_leading(False)

    def gen_queries(self):
        generated_queries = []
        for operation in self.pre_gen_query:



            possible_new_operations = []



            if isinstance(operation, selection):

                possible_new_conditions = []
                for i, cond in enumerate(operation.conditions):
                    if isinstance(cond, OP_cond):
                        possible_new_conditions.append(cond)
                        continue
                    possible_new_ith_cond = []

                    '''
                    condition1, condition1, condition1
                    '''
                    if type(cond.val) == int or type(cond.val) == float:
                        des = self.get_possible_values(cond.col)
                        stats = ["min", "max", "count", "mean", "std", "25%", "50%", "75%"]

                        for s in stats:

                            new_val = des[s]
                            for operator in OP:
                                new_condition = condition(cond.col, operator, new_val)
                                possible_new_ith_cond.append(new_condition)

                    ### TODO: add other types
                    else:
                        possible_new_ith_cond = [cond]


                    possible_new_conditions.append(possible_new_ith_cond)

                ### print debug

                # for i, new_cond in enumerate(possible_new_conditions):
                #     res_condition = []
                #     for nc in possible_new_conditions:
                #         cur = []
                #         if isinstance(nc, OP_cond):
                #             continue
                #         for c in nc:
                #             cur.append(str(c))
                #         print(cur)


                        # print(len(cur))
                # print(possible_new_conditions)
                possible_selection_operations = self.generate_possible_selection_operations(possible_new_conditions)
                for conds in possible_selection_operations:
                    possible_new_operations.append(operation.new_selection(conds))


            elif isinstance(operation, projection):
                new_operations = self.generate_possible_column_combinations(operation)

                for ops in new_operations:
                    possible_new_operations.append(operation.new_projection(ops))


            elif isinstance(operation, agg):
                possible_dicts = self.generate_possible_agg_combinations(operation)

                for d in possible_dicts:
                    possible_new_operations.append(operation.new_agg(d))

            elif isinstance(operation, group_by):
                possible_groupby_columns = self.generate_possible_groupby_combinations(operation)
                for col in possible_groupby_columns:
                    possible_new_operations.append(operation.new_groupby(col))

            generated_queries.append(possible_new_operations)
            print("===== possible operations generated =====")
        new_generated_queries = []
        # indexes = [0]*len(generated_queries)
        new_generated_queries = itertools.product(*generated_queries)
        print("======= *** start iterating generated queries *** ======")
        l = [item for item in new_generated_queries]
        print(" *** done ***")
        return l

    def get_new_pandas_queries(self, out=1000):
        res = []
        new_queries = self.gen_queries()
        random.shuffle(new_queries)
        new_queries = new_queries[:1000]
        print(f" ==== testing source with {len(new_queries)} queries ==== ")
        df = self._source_
        c = 0
        for i, new_query in enumerate(new_queries):
            if i % (len(new_queries) // 10) == 0:
                print(f"=== {c}% ===")
                c += 10
            # pandas_operation = self.get_query_str(new_query)
            try:
                result_df = self.execute_query(new_query)
            except Exception:
                continue

            new_q_obj = pandas_query(new_query, df)
            new_q_obj.target = result_df
            res.append(new_q_obj)
        random.shuffle(res)
        return res

    def execute_query(self, query):
        query_string = self.get_query_str(query)
        return eval(query_string)

    def generate_possible_groupby_combinations(self, operation: group_by, generate_num=50):

        print("===== generating groupby combinations =====")
        columns = self._source_.columns
        possible_groupby_columns = []
        for col in columns:
            possible_groupby_columns.append(col)
        random.shuffle(possible_groupby_columns)
        return possible_groupby_columns[:generate_num]

    def generate_possible_agg_combinations(self, operation: agg, generate_num=5):
        stats = ["min", "max", "count", "mean", "std"]

        possible_dicts = []

        cur_dict = operation.dict_key_vals

        print("===== generating agg combinations =====")

        if isinstance(cur_dict, str):
            # possible_dicts.append(i for i in stats)
            # return possible_dicts
            return stats
        # else:


    def generate_possible_column_combinations(self, operation: projection, generate_num=50):
        columns = self._source_.columns
        possible_columns = []

        print("===== generating column combinations =====")

        if len(columns) == 1:
            return [columns]

        else:
            res = [list(i) for i in list(itertools.combinations(columns, operation.length))]
            random.shuffle(res)
            return res[:generate_num]

    def generate_possible_selection_operations(self, possible_new_conditions, generate_num=50) -> List[

        List[Union[condition, OP_cond]]]:

        print("===== generating selection combinations =====")

        # print(possible_new_conditions)
        new_conds = []
        clocks = [0] * len(possible_new_conditions)

        for c in range(generate_num):
            possible_cond = []
            for i, new_cond in enumerate(possible_new_conditions):

                if isinstance(new_cond, OP_cond):
                    possible_cond.append(new_cond)
                    continue
                if clocks[i] < len(new_cond):
                    possible_new_ith_condition = new_cond[clocks[i]]
                    clocks[i] += 1
                else:
                    clocks[i] = 0
                    possible_new_ith_condition = new_cond[clocks[i]]

                possible_cond.append(possible_new_ith_condition)

            new_conds.append(possible_cond)
        random.shuffle(new_conds)
        return new_conds[:generate_num]

    def get_possible_values(self, col):

        des = self.get_source_description(self._source_, col)
        return des

    def get_query_str(self, query):
        strs = ""
        for q in query:
            strs += q.to_str()

        return strs

    def get_query_string(self):
        strs = ""
        for q in self.pre_gen_query:
            strs += q.to_str()

        return strs

    def get_source_description(self, dfa: pd.DataFrame, col):
        des = dfa.describe()

        return des[col]

    def to_pandas_template(self):
        cur = ""

    # def parse_query_from_str(self, source_query):

class pandas_queries():
    def __init__(self, queries: List[pandas_query]):
        self.queries = queries

    def check_merge(self, q1:pandas_query, q2:pandas_query):
        cols = q1.get_target().columns.intersection(q2.get_target().columns)
        if len(cols) == 0 or len(cols) == min(len(q1.get_source().columns), len(q2.get_source().columns)):
            return None

        return list(cols)



    def generate_possible_merge_operations(self, max_merge = 3):
        cur_queries = self.queries[:]
        k = 0
        res_hash = {}
        while True:
            if k >= max_merge:
                break
            for i in range(len(cur_queries)-1):
                for j in range(i+1, len(cur_queries)):
                    if str(i)+"+"+str(j) not in res_hash:
                        q1 = cur_queries[i]
                        q2 = cur_queries[j]

                        if q1.get_source().equals(q2.get_source()):
                            print("#### queries with same source detected, skipping to the next queries ####")
                            continue

                        cols = self.check_merge(q1, q2)

                        if cols and max(q1.num_merges, q2.num_merges) < 3:
                            # print(cols)
                            operations = list(q1.pre_gen_query)[:]

                            operations.append(merge(df_name=q1.df_name, queries=q2, on=cols))

                            strs = ""

                            for op in operations:
                                # print("cur op = " + str(op))
                                strs += op.to_str()

                                # print("cur op to str = " + op.to_str())

                            print(f"strs here = {strs}")
                            t = eval(strs)
                            if t.shape[0] == 0:
                                print("no rows exist with the above selection")
                                continue
                            else:
                                print("successfully generated query")
                            try:
                                res_df = q1.get_target().merge(q2.get_target(), on=cols)

                            except Exception:
                                continue

                            new_query = pandas_query(operations, q1.get_source(), verbose=True)
                            new_query.target = res_df
                            new_query.num_merges = max(q1.num_merges, q2.num_merges) + 1

                            cur_queries.append(new_query)
                            res_hash[f"{str(i)}+{str(j)}"] = 0




            k += 1

        return cur_queries
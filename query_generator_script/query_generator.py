import ast
import random
import pandas as pd
from typing import Type, List, Union, Dict
from enum import Enum

import itertools
from configs import *
import helpers as h
from tqdm import tqdm


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
    def __init__(self, df_name, queries: 'pandas_query', on=None, left_on: str = None, right_on: str = None, leading=False):
        super().__init__(df_name, leading)

        # assert (left_on == None and right_on == None) or on == None
        if on is None:
            on = []
        if left_on is None:
            left_on = ""
        if right_on is None:
            right_on = ""
        self.operations = queries.operations
        self.queries = queries
        self.on_col = on
        self.left_on = left_on
        self.right_on = right_on


    def to_str(self) -> str:
        # print(f"+++++++++++++++++++++{self.on_col}")

        if len(self.on_col) > 0:
            res_str = f"{self.df_name}" if self.leading else""

            operations_to_str = self.queries.get_query_str(self.queries.pre_gen_query)
            # for op in self.operations:
            #     operations_to_str += op.to_str()

            on_cols = ""
            for col in self.on_col:
                on_cols = on_cols + "'" + col + "'" + ","

            on_cols = on_cols[:-1]

            res_str = res_str + "." + "merge" + "(" +operations_to_str +"," + "on=" + "[" + on_cols  + "]"+")"
            return res_str
        else:

            res_str = f"{self.df_name}" if self.leading else ""
            operations_to_str = self.queries.get_query_str(self.queries.pre_gen_query)
            res_str = res_str + "." + "merge" + "(" +operations_to_str + "," +"left_on=" + "'" + self.left_on + "'" + ", "\
                       + "right_on=" + "'"  + self.right_on + "'"+ ")"
            return res_str

    def new_merge(self, new_queries, new_on_col = None, new_left_on = None, new_right_on = None):
        return merge(self.df_name, new_queries, new_on_col, new_left_on, new_right_on, leading=self.leading)

    def exec(self):
        return eval(self.to_str())

    def __str__(self):
        return f"merge: df_name = {self.df_name}, on_col = {self.on_col}, left_on = {self.left_on}, right_on = {self.right_on}"

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
        res_str = res_str + "." + "groupby" + "(" + "by=" + str(self.columns) + other_args  + ")"
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
        res_str = res_str + "." + "agg" + "(" + "'" + str(self.dict_key_vals) + "'" + "," + "numeric_only=True"+  ")"
        return res_str

    def new_agg(self, dict_cols):
        return agg(self.df_name, dict_cols, self.leading)

    def __str__(self):

        return f"agg: " + str(self.dict_key_vals)

    def exec(self):
        return eval(self.to_str())



class pandas_query():
    def __init__(self, q_gen_query: List[operation], source: 'TBL_source' = None, verbose = False):

        self.setup_query(q_gen_query)
        if verbose:
            print(self.get_query_str(q_gen_query))
        # self._source_ = source # df
        self._source_ = source
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

    def get_TBL_source(self):
        return self._source_

    def get_target(self):
        return self.target

    def get_source(self):
        return self._source_.source.copy()

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
                        cur = random.choice([OP_cond.OR, OP_cond.AND])

                        possible_new_conditions.append(cur)
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
                                if type(cond.val) == int:
                                    # print(des["min"])
                                    new_val = round(new_val) + random.randrange(0, abs(round(des["std"]+1)), 1)
                                else:
                                    new_val = round(new_val + (des["std"] + 1) * random.random(), 2)
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
        df = self.get_source()
        tbl = self.get_TBL_source()
        c = 0
        g = 0
        for i, new_query in enumerate(new_queries):
            if i % (len(new_queries) // 10) == 0:
                print(f"=== {c}% ===")
                c += 10
            # pandas_operation = self.get_query_str(new_query)
            try:
                result_df = self.execute_query(new_query)
            except Exception:
                continue
            g += 1
            new_q_obj = pandas_query(new_query, tbl)
            new_q_obj.target = result_df
            res.append(new_q_obj)
        random.shuffle(res)

        print(f" ======= {g} new queries generated =======")
        return res

    def execute_query(self, query):
        query_string = self.get_query_str(query)
        return eval(query_string)

    def generate_possible_groupby_combinations(self, operation: group_by, generate_num=50):

        print("===== generating groupby combinations =====")
        columns = self.get_source().columns
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


    def generate_possible_column_combinations(self, operation: projection, generate_num=200):
        columns = self.get_source().columns
        possible_columns = []

        print("===== generating column combinations =====")

        if len(columns) == 1:
            return [columns]

        else:

            res = [list(i) for i in list(itertools.combinations(columns, operation.length))]
            if operation.length > 1 and operation.length < len(list(columns)):
                res = res + [list(i) for i in list(itertools.combinations(columns, operation.length - 1))]
                res = res + [list(i) for i in list(itertools.combinations(columns, operation.length + 1))]

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
                    cur = random.choice([OP_cond.OR, OP_cond.AND])
                    possible_cond.append(cur)
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

        des = self.get_source_description(self.get_source(), col)
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
    def __init__(self, queries: List[pandas_query], self_join = False, verbose = False):
        self.queries = queries
        self.self_join = self_join
        self.result_queries = []
        self.verbose = verbose

        self.un_merged_queries = queries[:]



    def save_unmerged_examples(self, dir, filename):
        count = 0
        f = open(f"{dir}/{filename}.txt", "a")
        for q in self.un_merged_queries:
            # strs = q.
            strs = q.get_query_str(q.pre_gen_query)
            f.write(f"df{count} = {strs} \n")
            count += 1
        print(f" ##### Successfully write the unmerged queries into file {dir}/{filename}.txt #####")
        f.close()
    def shuffle_queries(self):
        random.shuffle(self.queries)


    def check_merge_on(self, q1:pandas_query, q2:pandas_query):
        cols = q1.get_target().columns.intersection(q2.get_target().columns)
        if len(cols) == 0 or len(cols) == min(len(q1.get_source().columns), len(q2.get_source().columns)):
            return None

        return list(cols)

    def check_merge_left_right(self, q1:pandas_query, q2:pandas_query):
        col1 = list(q1.get_target().columns)
        col2 = list(q2.get_target().columns)

        # print(col1)
        # print(col2)



        q1_foreign_keys = q1.get_TBL_source().get_foreign_keys()
        q2_foreign_keys = q2.get_TBL_source().get_foreign_keys()

        foreign_list = {}
        for key in q1_foreign_keys:
            if key in col1:
                for a in q1_foreign_keys[key]:
                    foreign_list[a[0]] = key

        for col in col2:
            if col in foreign_list:
                return [foreign_list[col], col]




        # print(foreign_list)

        '''
        for col in col1:

            if col in q1_foreign_keys:

                for key_and_ref in q1_foreign_keys[col]:

                    if key_and_ref[0] == col:
                        
                        return [col, key_and_ref[0]]
        '''
        return []

    def generate_possible_merge_operations(self, max_merge = 3):
        cur_queries = self.queries[:]
        k = 0
        res_hash = {}
        q_generated = 0
        while True:
            if k >= max_merge:
                break
            for i in tqdm(range(len(cur_queries)-1)):
                for j in range(i+1, len(cur_queries)):
                    if str(i)+"+"+str(j) not in res_hash:



                        q1 = cur_queries[i]
                        q2 = cur_queries[j]


                        # print(f"q1 df name = {q1}")
                        # print(f"q2 df_name = {q2}")


                        if q1.get_source().equals(q2.get_source()) and (self.self_join == False):
                            # print("#### queries with same source detected, skipping to the next queries ####")
                            continue

                        merge_differenet_keys = self.check_merge_left_right(q1, q2)



                        if len(merge_differenet_keys) > 0:
                            if self.verbose:
                                print(f"keys to merge = {merge_differenet_keys}")
                            operations = list(q1.pre_gen_query)[:]

                            operations.append(merge(df_name=q1.df_name, queries=q2, left_on=merge_differenet_keys[0], right_on=merge_differenet_keys[1]))
                            strs = ""

                            for op in operations:
                                # print("cur op = " + str(op))
                                strs += op.to_str()

                                # print("cur op to str = " + op.to_str())
                            if self.verbose:
                                print(f"strs here = {strs}")
                            t = eval(strs)
                            if t.shape[0] == 0:
                                if self.verbose:
                                    print("no rows exist with the above selection")
                                continue
                            else:
                                if self.verbose:
                                    print("successfully generated query")
                            try:
                                res_df = q1.get_target().merge(q2.get_target(), left_on=merge_differenet_keys[0], right_on=merge_differenet_keys[1])

                            except Exception:
                                if self.verbose:
                                    print("Exception occurred")
                                continue
                            if self.verbose:
                                print("++++++++++ add the result query to template +++++++++++++")
                            new_query = pandas_query(operations, q1.get_TBL_source(), verbose=False)
                            new_query.target = res_df
                            new_query.num_merges = max(q1.num_merges, q2.num_merges) + 1

                            cur_queries.append(new_query)

                            q_generated += 1

                            if q_generated % 1000 == 0:
                                print(f"**** {q_generated} queries have generated ****")

                            self.result_queries.append(new_query)
                            res_hash[f"{str(i)}+{str(j)}"] = 0


                        else:
                        ###################################################
                            cols = self.check_merge_on(q1, q2)

                            if cols and max(q1.num_merges, q2.num_merges) < 3 and self.self_join:
                                # print(cols)
                                operations = list(q1.pre_gen_query)[:]

                                operations.append(merge(df_name=q1.df_name, queries=q2, on=cols))

                                strs = ""

                                for op in operations:
                                    # print("cur op = " + str(op))
                                    strs += op.to_str()

                                    # print("cur op to str = " + op.to_str())
                                if self.verbose:
                                    print(f"strs here = {strs}")
                                t = eval(strs)
                                if t.shape[0] == 0:
                                    if self.verbose:
                                        print("no rows exist with the above selection")
                                    continue
                                else:
                                    if self.verbose:
                                        print("successfully generated query")
                                try:
                                    res_df = q1.get_target().merge(q2.get_target(), on=cols)

                                except Exception:
                                    if self.verbose:
                                        print("Exception occurred")
                                    continue
                                if self.verbose:
                                    print("++++++++++ add the result query to template +++++++++++++")

                                new_query = pandas_query(operations, q1.get_TBL_source(), verbose=False)
                                new_query.target = res_df
                                new_query.num_merges = max(q1.num_merges, q2.num_merges) + 1

                                cur_queries.append(new_query)
                                self.result_queries.append(new_query)
                                res_hash[f"{str(i)}+{str(j)}"] = 0

                                q_generated += 1

                                if q_generated % 1000 == 0:
                                    print(f"**** {q_generated} queries have generated ****")




            k += 1

            break

        return cur_queries


class TBL_source():
    # self.source_df

    def __init__(self, df:pd.DataFrame):
        self.source = df
        self.foreign_keys = {}


    def add_edge(self, col_name, other_col_name, other:'TBL_source'):
        self.foreign_keys[col_name] = []
        self.foreign_keys[col_name].append([other_col_name, other])

    def get_foreign_keys(self):
        return self.foreign_keys.copy()

    def equals(self, o:'TBL_source'):
        return self.source.equals(o.source)





def test_patients():

    df = pd.read_csv("./patient_ma_bn.csv")
    q1 = [selection("df", conditions=[condition("Age", OP.gt, 50), OP_cond.OR, condition("Age", OP.le, 70)]),
          projection("df", ["Age", "Sex", "operation", "P1200", "P1600", "Smoking"]), group_by("df", "Sex"), agg("df", "min")
          ]
    q2 = [selection("df", conditions=[condition("Age", OP.gt, 50), OP_cond.AND , condition("Height", OP.le, 160), OP_cond.AND,
                                      condition("TNM_distribution", OP.eq, 1)
                                      ]),
          projection("df", ["Age", "Sex", "P1210", "P100", "Smoking", "Weight"]), group_by("df", "Smoking"), agg("df", "count")
          ]
    pq1 = pandas_query(q1, source=df)
    pq2 = pandas_query(q2, source=df)

    res = pq1.get_new_pandas_queries()[:1000] + pq2.get_new_pandas_queries()[:1000]

    queries = pandas_queries(res)
    queries.generate_possible_merge_operations(3)

def generate_tpch():
    customer = pd.read_csv("./../benchmarks/customer.csv")
    lineitem = pd.read_csv("./../benchmarks/lineitem.csv")
    nation = pd.read_csv("./../benchmarks/nation.csv")
    orders = pd.read_csv("./../benchmarks/orders.csv")
    part = pd.read_csv("./../benchmarks/part.csv")
    partsupp = pd.read_csv("./../benchmarks/partsupp.csv")
    region = pd.read_csv("./../benchmarks/region.csv")
    supplier = pd.read_csv("./../benchmarks/supplier.csv")
    q1 = [selection("customer", conditions=[condition("ACCTBAL", OP.gt, 100), OP_cond.OR, condition("CUSTKEY", OP.le, 70)]),
          projection("customer", ["CUSTKEY", "NATIONKEY", "PHONE", "ACCTBAL", "MKTSEGMENT"])
          ]
    q2 = [selection("customer", conditions=[condition("ACCTBAL", OP.gt, 100), OP_cond.OR, condition("CUSTKEY", OP.le, 70)]),
          projection("customer", ["CUSTKEY", "NATIONKEY", "PHONE", "ACCTBAL", "MKTSEGMENT"]), group_by("customer", "NATIONKEY"),
          agg("customer", "max")
          ]

    q3 = [selection("lineitem", conditions=[condition("SUPPKEY", OP.gt, 100), OP_cond.OR, condition("QUANTITY", OP.gt, 5)]),
          ]
    q4 = [selection("lineitem", conditions=[condition("SUPPKEY", OP.gt, 100), OP_cond.OR, condition("QUANTITY", OP.gt, 5),
                                            OP_cond.AND,
                                            condition("DISCOUNT", OP.gt, 0.05)]),
          projection(
              "lineitem", ["PARTKEY", "SUPPKEY", "LINENUMBER", "QUANTITY", "DISCOUNT", "TAX", "SHIPDATE"]
          )

          ]
    q5 = [
        selection("lineitem", conditions=[condition("SUPPKEY", OP.gt, 100), OP_cond.OR, condition("QUANTITY", OP.gt, 5),
                                          OP_cond.AND,
                                          condition("DISCOUNT", OP.gt, 0.05)]),
        projection(
            "lineitem", ["PARTKEY", "SUPPKEY", "LINENUMBER", "QUANTITY", "RETURNFLAG","DISCOUNT", "TAX", "SHIPDATE", "SHIPMODE"]
        ), group_by("lineitem","RETURNFLAG"), agg("lineitem", "min")

        ]
    q6 = [selection("nation", conditions=[condition("REGIONKEY", OP.gt, 0)]
                    )]

    q7 = [selection("region", conditions=[condition("REGIONKEY", OP.ge, 0)])]

    q8 = [selection("orders", conditions=[condition("TOTALPRICE", OP.gt, 50000.0), OP_cond.OR,condition("SHIPPRIORITY", OP.eq, 0)]),
                    projection(
                        "orders", ["CUSTKEY", "TOTALPRICE", "ORDERPRIORITY", "CLERK"]
                    )

    ]

    q9 = [selection("orders", conditions=[condition("TOTALPRICE", OP.gt, 50000.0), OP_cond.OR,condition("SHIPPRIORITY", OP.eq, 0)]),
                    projection(
                        "orders", ["ORDERSTATUS", "CUSTKEY", "TOTALPRICE", "ORDERPRIORITY", "CLERK"]
                    ), group_by("orders", "ORDERSTATUS"), agg("orders", "max")

    ]
    q10 = [selection("supplier", conditions=[condition("NATIONKEY", OP.gt, 10), OP_cond.OR, condition("ACCTBAL", OP.le, 5000)]),
            projection("supplier", ["S_NAME", "NATIONKEY", "ACCTBAL"])
           ]
    q11 = [selection("supplier", conditions=[condition("NATIONKEY", OP.gt, 10), OP_cond.OR, condition("ACCTBAL", OP.le, 5000)]),
           ]
    q12 = [selection("part", conditions=[condition("RETAILPRICE", OP.gt, 500)]
    )]

    q13 = [selection("partsupp", conditions=[condition("SUPPLYCOSt", OP.le, 1000)]) ]

    # pq1 = pandas_query(q1, source=customer)
    # pq2 = pandas_query(q2, source=customer)
    pq3 = pandas_query(q3, source=lineitem)
    pq4 = pandas_query(q4, source=lineitem)
    pq5 = pandas_query(q5, source=lineitem)
    pq6 = pandas_query(q6, source=nation)
    pq7 = pandas_query(q7, source=region)
    pq8 = pandas_query(q8, source=orders)
    pq9 = pandas_query(q9, source=orders)
    pq10 = pandas_query(q10, source=supplier)
    pq11 = pandas_query(q11, source=supplier)
    pq12 = pandas_query(q12, source=part)
    pq13 = pandas_query(q13, source=partsupp)

    allqueries = [pq1, pq2, pq3, pq4, pq5, pq6, pq7, pq8, pq9, pq10]
    # allqueries = [pq5]
    res = []
    count = 0
    for pq in allqueries:
        print(count)
        res += pq.get_new_pandas_queries()[:100]






def run_TPCH():

    customer = TBL_source(pd.read_csv("./../../../benchmarks/customer.csv"))
    lineitem = TBL_source(pd.read_csv("./../../../benchmarks/lineitem.csv"))
    nation = TBL_source(pd.read_csv("./../../../benchmarks/nation.csv"))
    orders = TBL_source(pd.read_csv("./../../../benchmarks/orders.csv"))
    part = TBL_source(pd.read_csv("./../../../benchmarks/part.csv"))
    partsupp =TBL_source(pd.read_csv("./../../../benchmarks/partsupp.csv"))
    region = TBL_source(pd.read_csv("./../../../benchmarks/region.csv"))
    supplier = TBL_source(pd.read_csv("./../../../benchmarks/supplier.csv"))
    q1 = [selection("customer",
                    conditions=[condition("ACCTBAL", OP.gt, 100), OP_cond.OR, condition("CUSTKEY", OP.le, 70)]),
          projection("customer", ["CUSTKEY", "NATIONKEY", "PHONE", "ACCTBAL", "MKTSEGMENT"])
          ]
    q2 = [selection("customer",
                    conditions=[condition("ACCTBAL", OP.gt, 100), OP_cond.OR, condition("CUSTKEY", OP.le, 70)]),
          projection("customer", ["CUSTKEY", "NATIONKEY", "PHONE", "ACCTBAL", "MKTSEGMENT"]),
          group_by("customer", "NATIONKEY"),
          agg("customer", "max")
          ]

    q3 = [selection("lineitem",
                    conditions=[condition("SUPPKEY", OP.gt, 100), OP_cond.OR, condition("QUANTITY", OP.gt, 5)]),
          ]
    q4 = [
        selection("lineitem", conditions=[condition("SUPPKEY", OP.gt, 100), OP_cond.OR, condition("QUANTITY", OP.gt, 5),
                                          OP_cond.AND,
                                          condition("DISCOUNT", OP.gt, 0.05)]),
        projection(
            "lineitem", ["PARTKEY", "SUPPKEY", "LINENUMBER", "QUANTITY", "DISCOUNT", "TAX", "SHIPDATE"]
        )

        ]

    ### generate links to the foreign keys

    ### generate


    q5 = [
        selection("lineitem", conditions=[condition("SUPPKEY", OP.gt, 100), OP_cond.OR, condition("QUANTITY", OP.gt, 5),
                                          OP_cond.AND,
                                          condition("DISCOUNT", OP.gt, 0.05)]),
        projection(
            "lineitem",
            ["PARTKEY", "SUPPKEY", "LINENUMBER", "QUANTITY", "RETURNFLAG", "DISCOUNT", "TAX", "SHIPDATE", "SHIPMODE"]
        ), group_by("lineitem", "RETURNFLAG"), agg("lineitem", "min")

    ]
    q6 = [selection("nation", conditions=[condition("REGIONKEY", OP.gt, 0)]
                    ), projection("nation", ["REGIONKEY", "N_NAME", "N_COMMENT"])]


    q7 = [selection("region", conditions=[condition("REGIONKEY", OP.ge, 0)])]

    q8 = [selection("orders", conditions=[condition("TOTALPRICE", OP.gt, 50000.0), OP_cond.OR,
                                          condition("SHIPPRIORITY", OP.eq, 0)]),
          projection(
              "orders", ["CUSTKEY", "TOTALPRICE", "ORDERPRIORITY", "CLERK"]
          )

          ]

    q9 = [selection("orders", conditions=[condition("TOTALPRICE", OP.gt, 50000.0), OP_cond.OR,
                                          condition("SHIPPRIORITY", OP.eq, 0)]),
          projection(
              "orders", ["ORDERSTATUS", "CUSTKEY", "TOTALPRICE", "ORDERPRIORITY", "CLERK"]
          ), group_by("orders", "ORDERSTATUS"), agg("orders", "max")

          ]
    q10 = [selection("supplier",
                     conditions=[condition("NATIONKEY", OP.gt, 10), OP_cond.OR, condition("ACCTBAL", OP.le, 5000)]),
           projection("supplier", ["S_NAME", "NATIONKEY", "ACCTBAL"])
           ]
    q11 = [selection("supplier",
                     conditions=[condition("NATIONKEY", OP.gt, 10), OP_cond.OR, condition("ACCTBAL", OP.le, 5000)]),
           ]
    q12 = [selection("part", conditions=[condition("RETAILPRICE", OP.gt, 500)]
                     )]

    q13 = [selection("partsupp", conditions=[condition("SUPPLYCOST", OP.le, 1000)])]

    pq1 = pandas_query(q1, source=customer)
    pq2 = pandas_query(q2, source=customer)
    pq3 = pandas_query(q3, source=lineitem)
    pq4 = pandas_query(q4, source=lineitem)
    pq5 = pandas_query(q5, source=lineitem)
    pq6 = pandas_query(q6, source=nation)
    pq7 = pandas_query(q7, source=region)
    pq8 = pandas_query(q8, source=orders)
    pq9 = pandas_query(q9, source=orders)
    pq10 = pandas_query(q10, source=supplier)
    pq11 = pandas_query(q11, source=supplier)
    pq12 = pandas_query(q12, source=part)
    pq13 = pandas_query(q13, source=partsupp)

    allqueries = [pq1, pq2, pq3, pq4, pq5, pq6, pq7, pq8, pq9, pq10, pq11, pq12, pq13]
    # allqueries = [pq4]
    res = []
    count = 1
    c = pq3.get_new_pandas_queries()
    for pq in allqueries:
        print(f"*** query #{count} is generating ***")
        count += 1
        res += pq.get_new_pandas_queries()[:100]

    print("done")

    pandas_queries_list = pandas_queries(res)
    pandas_queries_list.generate_possible_merge_operations()
    # generate_tpch()
    # print(OP.gt)
    # q1 = [selection("df", conditions=[condition("col1", OP.gt, 1), OP_cond.OR, condition("col2", OP.le, 2)]),
    #       projection("df", ["col1", "col2"]), group_by("df", "col2"), agg("df", "max")]
    # # for q in q1:
    # #     print(q.to_str())
    # d1 = [10, 20, 30, 40, 50]
    # d2 = [1, 2, 3, 4, 5]
    # d3 = ["1", "2", "3", "4", "5"]
    # data = {"col1": d1,
    #         "col2": d2,
    #         "col3": d3}
    # df = pd.DataFrame(data)
    #
    # pq1 = pandas_query(q1, source=df)
    # print(pq1.get_query_string())
    # res = pq1.gen_queries()
    # print(res)

    # for r in res:
    #     cur = []
    #     for g in r:
    #         cur.append(str(g))
        # print(cur)

    # res = pq1.get_new_pandas_queries()[:10]

    # for r in res:
    #     cur = []
    #     for g in r:
    #         cur.append(str(g))
    #     print(cur)



if __name__ == "__main__":
    # run_TPCH()
    customer = pd.read_csv("./../../../benchmarks/customer_1.csv")
    lineitem = pd.read_csv("./../../../benchmarks/lineitem_1.csv")
    nation = pd.read_csv("./../../../benchmarks/nation_1.csv")
    orders = pd.read_csv("./../../../benchmarks/orders_1.csv")
    part = pd.read_csv("./../../../benchmarks/part_1.csv")
    partsupp = pd.read_csv("./../../../benchmarks/partsupp_1.csv")
    region = pd.read_csv("./../../../benchmarks/region_1.csv")
    supplier = pd.read_csv("./../../../benchmarks/supplier_1.csv")
    c = TBL_source(customer)
    l = TBL_source(lineitem)
    n = TBL_source(nation)
    o = TBL_source(orders)
    p = TBL_source(part)
    ps = TBL_source(partsupp)
    r = TBL_source(region)
    s = TBL_source(supplier)


    q1 = [selection("customer",
                    conditions=[condition("C_ACCTBAL", OP.gt, 100), OP_cond.OR, condition("C_CUSTKEY", OP.le, 70)]),
          projection("customer", ["C_CUSTKEY", "C_NATIONKEY", "C_PHONE", "C_ACCTBAL", "C_MKTSEGMENT"])
          ]
    q2 = [selection("customer",
                    conditions=[condition("C_ACCTBAL", OP.gt, 100), OP_cond.OR, condition("C_CUSTKEY", OP.le, 70)]),
          projection("customer", ["C_CUSTKEY", "C_NATIONKEY", "C_PHONE", "C_ACCTBAL", "C_MKTSEGMENT"]),
          group_by("customer", "C_NATIONKEY"),
          agg("customer", "max")
          ]

    q3 = [selection("lineitem",
                    conditions=[condition("L_SUPPKEY", OP.gt, 100), OP_cond.OR, condition("L_QUANTITY", OP.gt, 5)]),
          ]
    q4 = [
        selection("lineitem", conditions=[condition("L_SUPPKEY", OP.gt, 100), OP_cond.OR, condition("L_QUANTITY", OP.gt, 5),
                                          OP_cond.AND,
                                          condition("L_DISCOUNT", OP.gt, 0.05)]),
        projection(
            "lineitem", ["L_PARTKEY", "L_SUPPKEY", "L_LINENUMBER", "L_QUANTITY", "L_DISCOUNT", "L_TAX", "L_SHIPDATE"]
        )

    ]

    ### generate links to the foreign keys

    ### generate

    q5 = [
        selection("lineitem", conditions=[condition("L_SUPPKEY", OP.gt, 100), OP_cond.OR, condition("L_QUANTITY", OP.gt, 5),
                                          OP_cond.AND,
                                          condition("L_DISCOUNT", OP.gt, 0.05)]),
        projection(
            "lineitem",
            ["L_PARTKEY", "L_SUPPKEY", "L_LINENUMBER", "L_QUANTITY", "L_RETURNFLAG", "L_DISCOUNT", "L_TAX", "L_SHIPDATE", "L_SHIPMODE"]
        ), group_by("lineitem", "L_RETURNFLAG"), agg("lineitem", "min")

    ]
    q6 = [selection("nation", conditions=[condition("N_REGIONKEY", OP.gt, 0)]
                    ), projection("nation", ["N_REGIONKEY", "N_NAME", "N_COMMENT"])]

    q7 = [selection("region", conditions=[condition("R_REGIONKEY", OP.ge, 0)])]

    q8 = [selection("orders", conditions=[condition("O_TOTALPRICE", OP.gt, 50000.0), OP_cond.OR,
                                          condition("O_SHIPPRIORITY", OP.eq, 0)]),
          projection(
              "orders", ["O_CUSTKEY", "O_TOTALPRICE", "O_ORDERPRIORITY", "O_CLERK"]
          )

          ]

    q9 = [selection("orders", conditions=[condition("O_TOTALPRICE", OP.gt, 50000.0), OP_cond.OR,
                                          condition("O_SHIPPRIORITY", OP.eq, 0)]),
          projection(
              "orders", ["O_ORDERSTATUS", "O_CUSTKEY", "O_TOTALPRICE", "O_ORDERPRIORITY", "O_CLERK"]
          ), group_by("orders", "O_ORDERSTATUS"), agg("orders", "max")

          ]
    q10 = [selection("supplier",
                     conditions=[condition("S_NATIONKEY", OP.gt, 10), OP_cond.OR, condition("S_ACCTBAL", OP.le, 5000)]),
           projection("supplier", ["S_NAME", "S_NATIONKEY", "S_ACCTBAL"])
           ]
    q11 = [selection("supplier",
                     conditions=[condition("S_NATIONKEY", OP.gt, 10), OP_cond.OR, condition("S_ACCTBAL", OP.le, 5000)]),
           ]
    q12 = [selection("part", conditions=[condition("P_RETAILPRICE", OP.gt, 500)]
                     )]

    q13 = [selection("partsupp", conditions=[condition("PS_SUPPLYCOST", OP.le, 1000)])]

    pq1 = pandas_query(q1, source=c)
    pq2 = pandas_query(q2, source=c)
    pq3 = pandas_query(q3, source=l)
    pq4 = pandas_query(q4, source=l)
    pq5 = pandas_query(q5, source=l)
    pq6 = pandas_query(q6, source=n)
    pq7 = pandas_query(q7, source=r)
    pq8 = pandas_query(q8, source=o)
    pq9 = pandas_query(q9, source=o)
    pq10 = pandas_query(q10, source=s)
    pq11 = pandas_query(q11, source=s)
    pq12 = pandas_query(q12, source=p)
    pq13 = pandas_query(q13, source=ps)

    h.add_foreignkeys(c, "C_NATIONKEY", s, "S_NATIONKEY")
    h.add_foreignkeys(c, "C_NATIONKEY", n, "N_NATIONKEY")
    h.add_foreignkeys(c, "C_CUSTKEY", o, "O_CUSTKEY")
    h.add_foreignkeys(o, "O_ORDERKEY", l, "L_ORDERKEY")
    h.add_foreignkeys(l, "L_PARTKEY", ps, "PS_PARTKEY")
    h.add_foreignkeys(l, "L_PARTKEY", ps, "PS_SUPPKEY")
    h.add_foreignkeys(l, "L_PARTKEY", p, "P_PARTKEY")
    h.add_foreignkeys(l, "L_SUPPKEY", s, "S_SUPPKEY")
    h.add_foreignkeys(r, "R_REGIONKEY", n, "N_REGIONKEY")
    h.add_foreignkeys(s, "S_NATIONKEY", n, "N_NATIONKEY")
    h.add_foreignkeys(p, "P_PARTKEY", ps, "PS_PARTKEY")
    h.add_foreignkeys(s, "S_SUPPKEY", ps, "PS_SUPPKEY")



    allqueries = [pq1, pq2, pq3, pq4, pq5, pq6, pq7, pq8, pq9, pq10, pq11, pq12, pq13]
    # allqueries = [pq4]
    res = []
    count = 1
    c = pq3.get_new_pandas_queries()
    for pq in allqueries:
        print(f"*** query #{count} is generating ***")
        count += 1
        res += pq.get_new_pandas_queries()[:100]

    print("done")

    pandas_queries_list = pandas_queries(res)
    pandas_queries_list.shuffle_queries()
    pandas_queries_list.save_unmerged_examples(dir = Export_Rout, filename="unmerged_queries")
    pandas_queries_list.generate_possible_merge_operations()

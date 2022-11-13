import ast
import random
import pandas as pd
from typing import Type, List, Union, Dict
from enum import Enum

import itertools
from operations import *



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






if __name__ == "__main__":

    customer = pd.read_csv("./../benchmarks/customer.csv")
    lineitem = pd.read_csv("./../benchmarks/lineitem.csv")
    nation = pd.read_csv("./../benchmarks/nation.csv")
    orders = pd.read_csv("./../benchmarks/orders.csv")
    part = pd.read_csv("./../benchmarks/part.csv")
    partsupp = pd.read_csv("./../benchmarks/partsupp.csv")
    region = pd.read_csv("./../benchmarks/region.csv")
    supplier = pd.read_csv("./../benchmarks/supplier.csv")
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


# class query_generator():
#     def __init__(self, query_template):
#         self.query_template = query_template
#         self.sampling_outputs = {}
#     def _generate_new_pandas(self, pred_vals):
#         for key, val in pred_vals.items():
#             if key not in :
#                 print("key not in pandas!")
#             else:
#                 self.query_template[key] = val
#     customer = pd.read_table("./../benchmarks/customer.tbl", comment='#', delim_whitespace=True)
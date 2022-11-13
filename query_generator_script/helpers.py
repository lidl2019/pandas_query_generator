

from query_generator import *


def add_foreignkeys(TBL1:TBL_source, col1, TBL2:TBL_source, col2):
    TBL1.add_edge(col1, col2, TBL2)
    TBL2.add_edge(col2, col1, TBL1)


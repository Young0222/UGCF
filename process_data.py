import pandas as pd
import numpy as np
import io

input_file = './test.txt.raw'
output_file = './test.txt'

query_set = {}

item_idx = 1
with open(input_file,'r',encoding='utf-8') as f_in, io.open(output_file, 'w', encoding='utf-8') as f_out:

    for i,line1 in enumerate(f_in):
        line = line1.strip().split(' qid:')
        relevance = line[0]
        query = line[1]
        if relevance in {'3', '4'}:
            if query not in query_set:
                query_set[query] = [item_idx]
            else:
                tmp = query_set[query]
                tmp.append(item_idx)
                query_set[query] = tmp
        item_idx += 1
    
    for element in query_set:
        f_out.write(element)
        for item_idx in query_set[element]:
            f_out.write(' '+str(item_idx))
        f_out.write('\n')
            
print("Finised!")

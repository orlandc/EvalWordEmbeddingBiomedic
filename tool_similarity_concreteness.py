import os
import sys
import copy
import pandas as pd
import numpy as np
import re

from embedding_evaluation.evaluate import Evaluation
from embedding_evaluation.load_embedding import load_embedding_textfile
from dashtable import data2rst

path = os.getcwd() + '/models'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

files.sort(key=natural_keys)

cabecera = ['Corpus', 'Model', 'method', 'Dim. size', 'Similarity', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Concreteness', '', '']
subcabecera1 = ['', '', '', '', 'usf', '', 'ws353', '', 'men', '', 'vis_sim', '', 'sem_sim', '', 'simlex', '', 'simlex-q1', '', 'simlex-q2', '', 'simlex-q3', '', 'simlex-q4', '', 'mturk771', '', 'rw', '', 'mean', 'std', 'words_with_embedding']
subcabecera2 = ['', '', '', '', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', 'used_pairs', 'all_entities', '', '', '']
cuerpo = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

table = [cabecera, subcabecera1, subcabecera2]

for i, f in enumerate(files):
    k = f
    body = copy.copy(cuerpo)

    datos = k.replace(path+"/", '')
    x = datos.split("/")
    y = x[1].split("_")


    body[0] = x[0]                              #corpus
    
    body[1] = y[0]                              #model
    if(y[0] != 'glove'):
        body[2] = y[1]                          #method
        body[3] = y[4].replace(".txt", '')      #size
    else:
        body[3] = y[3].replace(".txt", '')      #size

    embeddings = load_embedding_textfile(textfile_path=f, sep=" ")

    evaluation = Evaluation() 
    results = evaluation.evaluate(embeddings)
    
    llaves_sim = results["similarity"].keys()
    for q, llav in enumerate(llaves_sim):
        body[(2 * q) + 4] = results["similarity"][llav]['used_pairs']
        body[(2 * q) + 5] = results["similarity"][llav]['all_entities']  

    body[28] = results["concreteness"]['mean']
    body[29] = results["concreteness"]['std']
    body[30] = results["concreteness"]['words_with_embedding']
    
    table.append(body)

    del k
    del x
    del y
    del body

a = np.asarray(table)
my_df = pd.DataFrame(a) 
my_df.to_csv('result.csv', index=False) 

print(data2rst(table, use_headers=True))
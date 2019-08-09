import os
import sys
from wordsim import Wordsim

path = os.getcwd() + '/models'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

for i, f in enumerate(files):
    if (i == 0):
        obj_to_evaluate = Wordsim("en")
        embedding = obj_to_evaluate.load_vector(f)
        result = obj_to_evaluate.evaluate(embedding)
        print(f)
        obj_to_evaluate.pprint(result)

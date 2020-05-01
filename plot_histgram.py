import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

parser=ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--predict_path')
args=parser.parse_args()

valid = pd.read_json(args.test_data_path, lines=True)
predict=pd.read_json(args.predict_path, lines=True)

hist=[]
for i in range(len(valid)):
  for j in predict['predict_sentence_index'][i]:   
    hist.append(j/len(valid['sent_bounds'][i]))

plt.hist(hist,bins=30,edgecolor='black')

plt.xlabel("Relative Location")
plt.ylabel("Count")
plt.title("Histogram")



plt.savefig('histogram.png')

plt.show()
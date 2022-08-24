import pandas as pd
import os
import argparse

# Merge classification results from two different classification models

parser = argparse.ArgumentParser()

# base argument
parser.add_argument('--path_of_csv1', type=str, help='The file path of the csv classification result to be merged.')
parser.add_argument('--path_of_csv2', type=str, help='The file path of the csv classification result to be merged.')
parser.add_argument('--output_path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'merged.csv'),
        help='The output path of the merged csv file.')

cf = parser.parse_args()

if __name__ == '__main__':

    data1 = pd.read_csv(cf.path_of_csv1,header=None)
    data2 = pd.read_csv(cf.path_of_csv2,header=None)

    data=pd.concat([data1,data2],axis=0,ignore_index=True)
    print('merged shape:', data.shape)
    data.to_csv(cf.output_path,index=False,header=False,encoding="utf_8_sig")
    print('saved path:', cf.output_path)
import pandas as pd
from scipy.io import arff
import numpy as np
import glob
import re


def folder(path):

    file_path=glob.glob(path)
    file_path.sort(key=lambda f: int(re.sub('\D', '', f)))

    return file_path

def callmain(Files):
    #code
    Data,Label=[],[]
    for i in range(len(Files)):

        arff_file = arff.loadarff(Files[i])
        df = pd.DataFrame(arff_file[0])
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        data=np.array(df)
        data=data[0:-2]
        cls=data[:,-1]
        for i in range(len(data)):
            data_=data[i].tolist()
            del data_[-1]
            f_data=np.array(data_).astype('float')
            Data.append(f_data.tolist())
            if cls[i]==b'positive':
                Label.append(1)
            else:Label.append(0)
    return Data,Label

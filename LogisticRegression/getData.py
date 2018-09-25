from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def getdata():
    df= pd.DataFrame(pd.read_csv("training.csv"))
    attributes = np.array(df.iloc[:,:46])
    targets = df.iloc[:,46]
    target = targets.reshape(-1,1)
    #target = []
    #for i in targets:
    #    target.append(1 if i ==1 else 0)
    #target = np.array(target).reshape(-1,1)
    print(attributes.shape)
    print(target.shape)

    return attributes,target


import numpy as np
import pandas as pd


def get_split(filename):
    pdata = pd.read_csv(filename)
    tags = pdata['cat'].unique().tolist()
    labels = pdata['label'].unique().tolist()

    for tag in tags:
        envX = pd.DataFrame()
        envY = pd.DataFrame()

        envX = pdata[(pdata['cat'] == tag)]['review']
        envY = pdata[(pdata['cat'] == tag)]['label']
        np.save(tag+"X.npy", envX)
        np.save(tag+"Y.npy", envY)


get_split("../emotion_CI/online_shopping_10_cats.csv")
npd = np.load('书籍X.npy', allow_pickle=True)
print('over')

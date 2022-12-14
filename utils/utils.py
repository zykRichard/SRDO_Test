import numpy as np
import pandas as pd
import logging, argparse, os
from statsmodels.stats.weightstats import DescrStatsW

 

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger



def get_split(filename):
    pdata = pd.read_csv(filename)
    tags = pdata['cat'].unique().tolist()
    labels = pdata['label'].unique().tolist()

    for tag in tags:
        envX = pd.DataFrame()
        envY = pd.DataFrame()

        envX = pdata[(pdata['cat'] == tag)]['review']
        envY = pdata[(pdata['cat'] == tag)]['label']
        np.save()
        np.save("../data/" + tag +"X.npy", envX)
        np.save("../data/" + tag +"Y.npy", envY)


def set_logging(path, filename):
    if not os.path.exists(path):
        os.mkdir(path)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                        datefmt='%d %b %Y %H:%M:%S',
                        filename=path + '/' + filename + '.log',
                        filemode='w+')
    logger = logging.getLogger('Decorr')
    if not logger.handlers or len(logger.handlers) == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='given a model file name')
    args = parser.parse_args()
    return args

def weighted_stat(x, weights):
    n, p = x.shape
    statModel = DescrStatsW(data=x, weights=weights)
    corr_mat = statModel.corrcoef # column * column 
    mean_pairwise_correlation = (np.sum(np.abs(corr_mat))-p)/p/(p-1) # exclude 1; p * (p - 1) correlation remain
    #Compute Condition Number
    eigenvalue = np.linalg.eigvals(corr_mat)
    max_eig = eigenvalue.max()
    min_eig = eigenvalue.min()
    condition_index = max_eig/eigenvalue
    condition_number = np.linalg.cond(corr_mat) # cond(A) = ||A|| * ||A^(-1)||; default by 2-norm
    w_stat = {}
    w_stat['corrcoef'] = corr_mat
    w_stat['mean_corr'] = mean_pairwise_correlation
    w_stat['min_eig'] = min_eig
    w_stat['CI'] = condition_index
    w_stat['CN'] = condition_number
    return w_stat


def data_enhence(data, label, reverse=False):
    enhence_index = np.random.randint(0, len(data), size = len(data) // 3)
    pos = '--' if reverse else '++'
    neg = '++' if reverse else '--'
    for i in enhence_index:
        if label[i] == 1:
            data[i] = pos + data[i]
        elif label[i] == 0:
            data[i] = neg + data[i]
            
        
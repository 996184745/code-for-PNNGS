import time
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

from model.DNNGP import DNNGP
from model.PNNGS import PNNGSImbalanced, PNNGSSStratifiedImbalanced, PNNGSTheLargestCategory


def main(model = "RRBLUP",
         pheno = "Seed volume",
         data_file = "./data/rice/PhenoGenotype_Seed volume.csv",
         CUDA_VISIBLE_DEVICES = '4'):
    #model selection:DNNGP,PNNGS,RF,SVR, RRBLUP
    # model = "RRBLUP"
    # pheno = "Seed volume"
    # data_file = "./data/rice/PhenoGenotype_Seed volume.csv"
    # CUDA_VISIBLE_DEVICES = '4'

    parallel_number = 8
    epoch = 2000
    print("model:", model)
    print("pheno:", pheno)
    print("parallel_number:", parallel_number)
    print("epoch:", epoch)
    print("CUDA_VISIBLE_DEVICES:", CUDA_VISIBLE_DEVICES)
    if model == "DNNGP":
        start_model = time.time()
        save_path = './save_model/DNNGP' + pheno + '.pth'
        average_pearns, test_pearns_list = DNNGP(pheno,
                                              data_file,
                                              n_splits = 10,
                                              epoch = epoch,
                                              CUDA_VISIBLE_DEVICES = '3',
                                              out_channels = 10,
                                              Batch_Size = 16,
                                              save_path = save_path)
        end_model = time.time()
        print('DNNGP Running time: %s Seconds' % (end_model - start_model))

        text_name = model + "_" + pheno + ".txt"
        f = open(text_name, 'w')
        f.write(model)
        f.write("\n")
        f.write(pheno)
        f.write("\n")
        f.write(str(average_pearns))
        f.write("\n")
        f.write(str(list(test_pearns_list)))
        f.write("\n")
        f.write("Running time:" + str(end_model - start_model) + "\n")
        f.close()

    elif model == "PNNGS":
        start_model = time.time()
        save_path = './save_model/PNNGS' + pheno + '.pth'
        average_pearns, test_pearns_list = PNNGSTheLargestCategory(pheno,
                                                 data_file,
                                                 cluster_file,
                                                 n_splits=10,
                                                 epoch= epoch,
                                                 CUDA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES,
                                                 parallel_number= parallel_number,
                                                 Batch_Size=16,
                                                 save_path= save_path)
        end_model = time.time()
        print('PNNGS Running time: %s Seconds' % (end_model - start_model))

        text_name = model + "_" + pheno + ".txt"
        f = open(text_name, 'w')
        f.write(model)
        f.write("\n")
        f.write(pheno)
        f.write("\n")
        f.write(str(average_pearns))
        f.write("\n")
        f.write(str(list(test_pearns_list)))
        f.write("\n")
        f.write("Running time:" + str(end_model - start_model) + "\n")
        f.close()

    elif model == "RF":
        start_model = time.time()
        model = RF(random_state= 0)
        average_pearns, test_pearns_list = RForSVRorRRBLUP(model,
                                                    pheno,
                                                    data_file,
                                                  n_splits=10)
        end_model = time.time()
        print('RF Running time: %s Seconds' % (end_model - start_model))
    elif model == "SVR":
        start_model = time.time()
        model = SVR()
        average_pearns, test_pearns_list = RForSVRorRRBLUP(model,
                                                   pheno,
                                                   data_file,
                                                   n_splits=10)
        end_model = time.time()
        print('SVR Running time: %s Seconds' % (end_model - start_model))
    elif model == "RRBLUP":
        start_model = time.time()
        model = Ridge(random_state= 0)
        average_pearns, test_pearns_list = RForSVRorRRBLUP(model,
                                                   pheno,
                                                   data_file,
                                                   cluster_file,
                                                   n_splits=10)
        end_model = time.time()
        print('RRBLUP Running time: %s Seconds' % (end_model - start_model))


def RForSVRorRRBLUP(model,
            pheno,
            data_file,
            cluster_file,
            n_splits=10):
    '''

    :param model: RF or SVR
    :param pheno:
    :param data_file:
    :param n_splits:
    :return:
    '''
    data_file = pd.read_csv(data_file, header=0, index_col=0)

    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)

    print("X.shape:", X.shape)

    pearns = []
    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    cluster_file = pd.read_csv(cluster_file, header=0, index_col=0)
    cluster_number = cluster_file["cluster"]

    start_model = time.time()
    for fold, (train_index, test_index) in enumerate(skfold.split(X, cluster_number)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        X_y_train = pd.concat([X_train, y_train], axis= 1)
        y_train_cluster = cluster_number.iloc[train_index]
        over = RandomOverSampler()
        X_smote, y_smote = over.fit_resample(X_y_train, y_train_cluster)
        print(y_smote.value_counts())

        y_train_smote = X_smote[pheno]
        X_train_smote = X_smote.drop([pheno], axis=1)

        model.fit(X_train_smote, y_train_smote)

        y_pre = model.predict(X_test)

        pearn, p = stats.pearsonr(y_pre, y_test)
        pearns.append(pearn)
    end_model = time.time()

    print("np.mean(pearns):", np.mean(pearns))
    print("pearns:", pearns)
    print('Model Running time: %s Seconds' % (end_model - start_model))
    return np.mean(pearns), pearns



if __name__ == '__main__':
    # model selection:DNNGP,PNNGS,RF,SVR, RRBLUP
    models = ["PNNGS"]
    phenos = ["Flag leaf length", "Leaf pubescence", "Panicle length",
              "Plant height", "Seed number per panicle", "Seed surface area"]
    crop = "rice"

    # phenos = ["Flower head diameter", "Leaf perimeter","Primary branches",
    #           "Stem colour","Stem diameter at flowering", "Total RGB", ]
    # crop = "sunflower"

    # phenos = ["DaystoSilk_06CL1", "DaystoSilk_065", "DaystoSilk_26M3",
    #           "DaystoSilk_07CL1", "DaystoSilk_07A", "DaystoSilk_06PR", ]
    # crop = "maize"
    #
    for model in models:
        for pheno in phenos:
            data_file = "./data/" + crop + "/PhenoGenotype_" + pheno + ".csv"
            cluster_file = "./result/PCA+cluster_" + pheno + ".csv"
            main(model,
                 pheno,
                 data_file,
                 CUDA_VISIBLE_DEVICES='4')
            print("*" * 30)



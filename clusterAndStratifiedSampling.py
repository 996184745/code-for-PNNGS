import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def getTrainTestDifference(data_file, pheno):
    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)

    print("X.shape:", X.shape)

    differences = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(X):
        y_train, y_test = y[train_index], y[test_index]

        difference = np.abs(np.mean(y_train) - np.mean(y_test))
        differences.append(difference)

    return differences


def kMeansPCA(data_file, pheno, n_clusters = 3):
    X = data_file.drop([pheno], axis=1)

    model = decomposition.PCA(n_components=2)
    model.fit(X)
    X_new = model.fit_transform(X)

    n_clusters = n_clusters
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X_new)
    y_pred = cluster.labels_  #the cluster result
    y_pred_df = pd.DataFrame(data= y_pred, columns=["cluster"], index=X.index)
    y_pred_df["cluster"] = y_pred_df["cluster"]+1
    print("y_pred_df.value_counts():", y_pred_df.value_counts())

    #centroid
    centroid = cluster.cluster_centers_
    print("质心: ", centroid)
    print("centroid.shape:", centroid.shape)
    clusterPlot(X_new, y_pred)

    #save data
    X_new_df = pd.DataFrame(data=X_new, columns=["PCA1", "PCA2"], index=X.index)

    data_file_PCA = pd.merge(X_new_df, y_pred_df, left_index= True, right_index= True)

    data_file_PCA_cluster = data_file_PCA.sort_values(by= ["cluster"])

    data_file_PCA.to_csv("./result/PCA+cluster" + "_" + pheno + ".csv", header=True, sep=",",
                         index=True)  #index = 0 Do not save row index
    data_file_PCA_cluster.to_csv("./result/PCA+cluster+sort" + "_" + pheno + ".csv", header=True, sep=",",
                         index=True)  # index = 0 Do not save row index
    return data_file_PCA

def clusterPlot(X_new, y_pred):
    color = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    fig, ax1 = plt.subplots(1)

    for i in range(n_clusters):
        ax1.scatter(X_new[y_pred == i, 0], X_new[y_pred == i, 1],
                    marker="o",
                    s=8,
                    c=color[i]
                    )
        # ax1.scatter(centroid[:, 0], centroid[:, 1]
        #             , marker="x"
        #             , s=15
        #             , c="black")
    plt.show()

def stratifiedStatistics(data_file, data_file_PCA):
    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)

    print("X.shape:", X.shape)

    Number0 = data_file_PCA["cluster"].value_counts().iloc[0]
    Number1 = data_file_PCA["cluster"].value_counts().iloc[1]
    Number2 = data_file_PCA["cluster"].value_counts().iloc[2]
    print("Number0: ", Number0)
    print("Number1: ", Number1)
    print("Number2: ", Number2)

    differences = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(X):
        train0 = 0
        train1 = 0
        train2 = 0
        for index in train_index:
            if data_file_PCA.iloc[index].loc["cluster"] == 0:
                train0 += 1
            elif data_file_PCA.iloc[index].loc["cluster"] == 1:
                train1 += 1
            elif data_file_PCA.iloc[index].loc["cluster"] == 2:
                train2 += 1
        # difference = np.abs(train0 - Number0) / Number0 \
        #              + np.abs(train1 - Number1) / Number1 \
        #              + np.abs(train2 - Number2) / Number2
        # difference = np.abs(train0 - Number0) / np.sqrt(Number0) \
        #              + np.abs(train1 - Number1) / np.sqrt(Number1) \
        #              + np.abs(train2 - Number2) / np.sqrt(Number2)
        test0 = Number0 - train0
        test1 = Number1 - train1
        test2 = Number2 - train2
        print("test0:", test0)
        print("test1:", test1)
        print("test2:", test2)

        difference = np.abs(round(train0/test0, 2) - 9)\
                     + np.abs(round(train1/test1, 2) - 9)\
                     + np.abs(round(train2/test2, 2) - 9)
        differences.append(difference)

    return differences

def elbowMethod(data_file):
    '''
    :param data_file: phenotype + genomic data
    :return:
    '''
    from scipy.spatial.distance import cdist

    X = data_file.drop([pheno], axis=1)

    model = decomposition.PCA(n_components=2)
    model.fit(X)
    X_new = model.fit_transform(X)

    K = range(1, 10)
    sse_result = []
    for k in K:
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X_new)
        sse_result.append(sum(np.min(cdist(X_new, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_new.shape[0])
    plt.plot(K, sse_result, 'gx-')
    plt.xlabel('k')
    plt.ylabel(u'average degree of distortion')
    plt.show()

def silhouette(data_file):
    '''

    :param data_file: phenotype + genomic data
    :return:
    '''
    from sklearn.metrics import silhouette_score

    X = data_file.drop([pheno], axis=1)

    model = decomposition.PCA(n_components=2)
    model.fit(X)
    X_new = model.fit_transform(X)

    K = range(2, 10)
    score = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_new)
        score.append(silhouette_score(X_new, kmeans.labels_, metric='euclidean'))
    print("score:", score)
    plt.plot(K, score, 'r*-')
    plt.xlabel('k')
    plt.ylabel(u'Silhouette coefficient')
    plt.title(u'The optimal K value determined by the silhouette coefficient')
    plt.show()


# pheno = "Seed number per panicle"
# phenos = ["Flag leaf length", "Leaf pubescence", "Panicle length",
#               "Plant height", "Seed number per panicle", "Seed surface area"]
# crop = "rice"

# pheno = "Flower head diameter"
# phenos = phenos = ["Flower head diameter", "Leaf perimeter","Primary branches",
#               "Stem colour","Stem diameter at flowering", "Total RGB", ]
# crop = "sunflower"


phenos = ["DaystoSilk_06CL1", "DaystoSilk_065", "DaystoSilk_26M3",
           "DaystoSilk_07CL1", "DaystoSilk_07A","DaystoSilk_06PR",]
crop = "maize"


for pheno in phenos:
    data_file = "./data/" + crop + "/PhenoGenotype_" + pheno + ".csv"
    n_splits=10

    data_file = pd.read_csv(data_file, header=0, index_col=0)

    # elbowMethod(data_file)
    # silhouette(data_file)


    n_clusters = 2
    kMeansPCA(data_file, pheno, n_clusters = n_clusters)
#
#
# data_file_PCA = "./result/PCA+cluster" + "_" + pheno + ".csv"
# data_file_PCA = pd.read_csv(data_file_PCA, header=0, index_col=0)
# print(data_file_PCA["cluster"].value_counts())
"""
0    234
1     83
2     59
"""
# differences = stratifiedStatistics(data_file, data_file_PCA)


# differences = getTrainTestDifference(data_file, pheno)
# print("differences:", differences)
# predictionAccuracy = [0.617, 0.625, 0.593, 0.633, 0.606, 0.767, 0.639, 0.709, 0.581, 0.697]
#
# z1 = np.polyfit(differences, predictionAccuracy, 1)
# p1 = np.poly1d(z1)
# print("z1:", z1)
# print("z1:", p1)
#
# print("np.min(differences):", np.min(differences))
# print("np.max(differences):", np.max(differences))
# xpoly = np.arange(np.min(differences), np.max(differences)*1.001, (np.max(differences) - np.min(differences))/10)
# ypoly = p1(xpoly)
# print("xpoly:", xpoly)
# print("ypoly:", ypoly)
#
# fig, ax1 = plt.subplots(1)
# ax1.scatter(differences, predictionAccuracy,
#                     marker="x",
#                     s=15,
#                     c= "green"
#                     )
# plt.show()

# 可以使用上面提到的各种推荐系统算法
import surprise
import os
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold

# 指定文件所在路径
root_dir = os.getcwd()
dfs_path = os.path.join(root_dir, 'Data/datasets/')
ratings_df = pd.read_csv(os.path.join(dfs_path,'ratings_expl.csv'), sep=';',encoding='latin-1',low_memory=False)
# 加载数据
reader = surprise.Reader(rating_scale=(1, 10))
data = surprise.Dataset.load_from_df(ratings_df[['user_id', 'isbn', 'rating']], reader)
kf = KFold(n_splits=5)
# data = Dataset.load_builtin('jester')
# Kfold
algo1 = SVD()
algo2 = surprise.BaselineOnly()
algo3 = surprise.KNNBasic()
algo4 = surprise.CoClustering()
for trainset, testset in kf.split(data):
    # SVD
    algo1.fit(trainset)
    pSVD = algo1.test(testset)
    # 计算并打印RMSE
    print("SVD-")
    accuracy.rmse(pSVD, verbose=True)
    #Baseline
    algo2.fit(trainset)
    pBase = algo2.test(testset)
    print("BaseLine-")
    accuracy.rmse(pBase)
    #Baseline
    algo3.fit(trainset)
    pknn = algo3.test(testset)
    print("KNN(cf)-")
    accuracy.rmse(pknn)
    # CoCluster
    algo4.fit(trainset)
    pCoClust = algo4.test(testset)
    print("CoClustering-")
    accuracy.rmse(pCoClust)


# KNN => Memory Error
# algo3 = surprise.KNNBasic()
# algo3.fit(trainset)
# pKNN = algo3.test(testset)
# accuracy.rmse(pKNN)

# # CoCluster
# algo4 = surprise.CoClustering()
# algo4.fit(trainset)
# pCoClust = algo4.test(testset)
# accuracy.rmse(pCoClust)


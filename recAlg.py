# -*- coding: utf-8 -*-
"""
Created on Wed Sept 9 19:13:36


@author: WLH
"""

import pandas as pd
import os
import sys
import logging
import surprise
import random

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:26:36 2018
Last modified on Aug 27

@author: WLH
"""

import pandas as pd
import os
import sys
import logging
import surprise
import random


# from BookRec_Functions import *


class recommender:

    def __init__(self):
        # Set the running path
        self.root_dir = os.getcwd()
        self.dfs_path = os.path.join(self.root_dir, 'Data/datasets/')
        self.model_path = os.path.join(self.root_dir, 'Data/model.pickle')
        self.users_df, self.items_df, self.ratings_df = self.load_dfs()
        self.pos, self.neg = ['y', 'yes', 'Y', 'Yes'], ['n', 'no', 'N', 'No']
        self.nof_user_ratings = self.ratings_df.user_id.value_counts()

        # print(self.nof_user_ratings.index)

        self.min_nof_ratings = 1
        self.ratings_changed = False
        self._algo_choise = 'CoClustering'  # 这里选择训练的算法SVD Baseline SlopeOne或者KNNBasic
        # Check if there is a save of the model and if yes, load it. If not, train it now
        try:
            _, self.algorithm = surprise.dump.load(self.model_path)   # 加载预训练模型
        except:
            logging.error(('File "model.pickle" was not found in %s.\n If you have already '
                           'trained the Recommender, make sure the file is in the correct directory'), self.model_path)


    def transToList(self, items):
        Res = []
        name = items.columns.values.tolist()
        for i, item in enumerate(items.iterrows()):
            pass
            row = dict.fromkeys(name)
            for n in name:
                # print(item[n])
                row[n] = item[1][n]
                pass
            Res.append(row)
        return Res

    def build_trainset(self):
        '''
        Build the trainset from ratings_df to be used by the <surprise.prediction_algorithms.algo_base.AlgoBase>.fit()
        '''
        reader = surprise.Reader(rating_scale=(1, 10))
        data = surprise.Dataset.load_from_df(self.ratings_df[['user_id', 'isbn', 'rating']], reader)
        self.trainset = data.build_full_trainset()

    def build_recset(self, trainset, fill=None):
        '''
        Return a list of ratings that can be used as a testset in the
        :meth:`test() <surprise.prediction_algorithms.algo_base.AlgoBase.test>`
        method. The ratings are all the ratings that are **not** in the trainset, i.e.
        all the ratings :math:`r_{ui}` where the user :math:`u` is known, the
        item :math:`i` is known, but the rating :math:`r_{ui}`  is not in the
        trainset. As :math:`r_{ui}` is unknown, it is either replaced by the
        :code:`fill` value or assumed to be equal to the mean of all ratings
        :meth:`global_mean <surprise.Trainset.global_mean>`.

        Args:
            trainset (surprise.Trainset.obj) -- The trainset used to fit/train the model.
            fill(float) -- The value to fill unknown ratings. If :code:`None` the
                global mean of all ratings :meth:`global_mean
                <surprise.Trainset.global_mean>` will be used.

        Returns:
            A list of tuples ``(uid, iid, fill)`` where ids are raw ids.
        '''
        trainset = self.trainset
        fill = trainset.global_mean if fill is None else float(fill)
        recset = []

        u = trainset.to_inner_uid(self.user_Id)
        user_items = set([j for (j, _) in trainset.ur[u]])
        recset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                   i in trainset.all_items() if
                   i not in user_items]
        return recset

    def recommend(self, user_Id, nof_rec=4, verbose=True):
        '''
        Recommends Items from database based on user's ratings.

        Args:
            nof_rec (int) -- Number of recommendations to return. Default: 5
            verbose (bool) -- Whether to print the results or not. Default: True

        Returns:
            items(pandas.DataFrame) -- Df of items with top predicted rating for the logged in user.
            top_n(list[][]) -- The predicted points of book.
        '''

        # check if logged in user has rated any items, and if not ask them to rate some in order to be able to get recommendtions
        self.user_Id = user_Id
        if (self.user_Id not in self.nof_user_ratings.index):
            list=self.nof_user_ratings.index.tolist()
            list.sort()
            print(self.user_Id,list)
            print('It seems that you haven\'t rate any books. Here are some books recommended randomly for you!')
            # Here to insert random recommendations
            items=self.rand_recommend()
            return items
        try:
            recset = self.build_recset(self.trainset)
        except:
            self.build_trainset()
            recset = self.build_recset(self.trainset)
        try:
            predictions = self.algorithm.test(recset)
        except:
            logging.error(('File "model.pickle" was not found in %s.\n If you have already '
                           'trained the Recommender, make sure the file is in the correct directory'), self.model_path)

        # get the books with the top predicted rating and construct a pd.DataFrame of them and the ratings
        top_n = []
        for _, iid, _, est, _ in predictions:
            top_n.append((iid, int(est)))
        top_n.sort(key=lambda x: x[1], reverse=True)
        isbn, rating = [], []
        for i, r in top_n[:nof_rec]:
            isbn.append(i)
            rating.append(int(r))
        items = self.items_df.loc[self.items_df.isbn.isin(isbn)]
        items = pd.concat([items.reset_index(drop=True), pd.DataFrame({'rating': rating})], axis=1)

        return self.transToList(items)


    def get_dfs(self):
        '''
        Returns the DataFrames
        '''
        return self.users_df, self.items_df, self.ratings_df



    def save_dfs(self, to_save='all'):
        '''
        Save the selected DataFrames

        Args:
            to_save (str) -- Items to save. One of ['all','users','items','ratings'].
        '''
        if to_save == 'all' or to_save == 'users':
            self.users_df.to_csv(os.path.join(self.dfs_path, 'users_w_ex_ratings.csv'), sep=';', index=False)
        if to_save == 'all' or to_save == 'items':
            self.items_df.to_csv(os.path.join(self.dfs_path, 'items_wo_duplicates.csv'), sep=';', index=False)
        if to_save == 'all' or to_save == 'ratings':
            self.ratings_df.to_csv(os.path.join(self.dfs_path, 'ratings_expl.csv'), sep=';', index=False)

    def load_dfs(self):
        '''
        Load the DataFrames

        Returns:
            users_df -- pandas.DataFrame of users
            items_df -- pandas.DataFrame of items
            ratings_df -- pandas.DataFrame of ratings
        '''
        try:
            users_df = pd.read_csv(os.path.join(self.dfs_path, 'users_w_ex_ratings.csv'), sep=';', encoding='latin-1',
                                   low_memory=False)
            items_df = pd.read_csv(os.path.join(self.dfs_path, 'items_wo_duplicates.csv'), sep=';', encoding='latin-1',
                                   low_memory=False)
            ratings_df = pd.read_csv(os.path.join(self.dfs_path, 'ratings_expl.csv'), sep=';', encoding='latin-1',
                                     low_memory=False)
        except:
            logging.error(('One or more of the files was not found in %s.\n Please make sure you have run '
                           '"BookCrossing data cleansing.ipynb" first.'), self.dfs_path)
            sys.exit(1)
        return users_df, items_df, ratings_df

    def rand_recommend(self, nof_rec=4):  # return top-n full-rating books randomly
        # get the books with the top predicted rating and construct a pd.DataFrame of them and the ratings
        top_n = self.ratings_df[['isbn', 'rating']][self.ratings_df['rating'] == 10]
        # print(top_n)
        row_num = top_n.shape[0]
        rand = random.randint(0, max(row_num - nof_rec, 1))
        isbn, rating = [], []
        for i in zip(top_n['isbn'][rand:rand + nof_rec], top_n['rating'][rand:rand + nof_rec]):
            isbn.append(i[0])
            rating.append(int(i[1]))
        items = self.items_df.loc[self.items_df.isbn.isin(isbn)]
        items = pd.concat([items.reset_index(drop=True), pd.DataFrame({'rating': rating})], axis=1)
        top_n=[];
        for i in range(nof_rec):
            top_n.append((i,'*'))
        return self.transToList(items)

    def model_fit(self):
        '''
        Train model using surprise.SVD algorithm.
        '''
        self.build_trainset()
        algo = self._algo_choise
        if algo == 'SVD':
            self.algorithm = surprise.SVD()
        elif algo == 'Baseline':
            self.algorithm = surprise.BaselineOnly()
        elif algo == 'SlopeOne':
            self.algorithm = surprise.SlopeOne()
        elif algo == 'CoClustering':
            self.algorithm = surprise.CoClustering()
        else:
            self.algorithm = surprise.KNNBasic()

        print('Training Recommender System using %s...' % algo)

        self.algorithm.fit(self.trainset)
        self.ratings_changed = False
        self.save_model()
        print('Done')

    def save_model(self, verbose=True):
        '''
        Save model in ../Data.

        Args:
            verbose (bool): Level of verbosity. If 1, then a message indicates that the dumping went successfully. Default is 0
        '''
        if verbose:
            print('Saving Model...')
        verbose = 1 * verbose
        surprise.dump.dump(self.model_path, predictions=None, algo=self.algorithm, verbose=verbose)

if __name__ == '__main__':
    rec = recommender()
    items = rec.recommend(user_Id=345525)
    i=0
    for item in items:
        i=i+1
        # print('Book "',item[1], '" from "', item[2], '", (%f)'%item[5])
        print('{0}) "{1}({2})" from {3} of {4} in {5}.\nThe url is: {6}'.format(i, item['book_title'], item['isbn'],
                                    item['publisher'],item['book_author'],
                                                                     item['year_of_publication'] ,item['img_m']))
    # When the csv forms changed, you can call model_fit function to update the pre-training model like this:
    # rec.model_fit()
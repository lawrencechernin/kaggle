# FROM https://www.kaggle.com/the1owl/redefining-treatment-0-57456  Sept 12 2017
# This script gives a LB score of 0.57804, 5 fold average score: 0.922406
#  updated:  n_components1=30(20) avg score: 0.92417, LB: 0.57633  best on 9/13 at position #179
#  9/19 standard_cv = 0.970832 * 0.9 CV: 0.920382 LB: 0.56718 new best #110 
#  9/20 max_depth=6(4)  CV: 0.9043 LB: 0.55771 new best #86 


from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb

train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/stage2_test_variants.csv')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

for i in range(56):
    df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')

gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print("len gen_var_lst1:", len(gen_var_lst))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print("len gen_var_lst2:", len(gen_var_lst))
i_ = 0
for gen_var_lst_itm in gen_var_lst:
    if i_ % 100 == 0: print("i:", i_)
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
    i_ += 1

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

print('Pipeline...')
n_components1=20
n_components1=50
n_components1=40
n_components1=25
n_components1=30
n_components2=50
n_components2=75
n_iter=25
n_iter=50
n_iter=25

# find the CV's of each component
standard_cv = 0.970832 * 0.9
pi1_cv = 1.18485
pi2_cv = 1.62459
pi3_cv = 0.989647
sum_cvs = 1.0/standard_cv + 1.0/pi1_cv + 1.0/pi2_cv + 1.0/pi3_cv

standard_wt = 1.0/standard_cv / sum_cvs
pi1_wt = 1.0/pi1_cv / sum_cvs
pi2_wt = 1.0/pi2_cv / sum_cvs
pi3_wt = 1.0/pi3_cv / sum_cvs

print("CV'S, standard_cv:", standard_cv, "pi1_cv:", pi1_cv, "pi2_cv:", pi2_cv, "pi3_cv:", pi3_cv)
print("Weights, standard_wt:", standard_wt, "pi1_wt:", pi1_wt, "pi2_wt:", pi2_wt, "pi3_wt:", pi3_wt)



fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,

        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=n_components1, n_iter=n_iter, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd2', decomposition.TruncatedSVD(n_components=n_components1, n_iter=n_iter, random_state=12))])),
            ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=n_components2, n_iter=n_iter, random_state=12))]))
        ],
        transformer_weights = {
            'standard': standard_wt,
            'pi1': pi1_wt,
            'pi2': pi1_wt,
            'pi3': pi1_wt
        },
    )
    )
])
print("TRAIN HEAD:", train.head())
print("TRAIN columns:", train.columns)

train = fp.fit_transform(train); print("Train shape:", train.shape)
test = fp.transform(test); print("Test shape:", test.shape)
print("TRAIN after ", train)

print("y:", y)
y = y - 1 #fix for zero bound array

denom = 0
eta = 0.03333
max_depth = 7
nrounds=1000
fold = 5 #Change to 5, 1 for Kaggle Limits
for i in range(fold):
    params = {
        'eta': eta,
        'max_depth': max_depth,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), nrounds,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print("Fold:", i, ",Score:", score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)
preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('sub.csv', index=False)

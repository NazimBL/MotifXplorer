#Nazim Belabbaci
#Summer 2022

import xgboost as xgb
import pandas as pd
from matplotlib import pyplot
from xgboost import plot_importance

kmer_size=6
top_features=10

def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#read preprocessed table, see github repository
gata3_data=pd.read_csv("data_final.csv")

##replace seuence column with kmers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence'],kmer_size), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)
#print(gata3_data)

kmer_texts = list(gata3_data['kmers'])
for item in range(len(kmer_texts)):
    kmer_texts[item] = ' '.join(kmer_texts[item])

y_data = gata3_data.iloc[:, 0].values
#print(human_texts[0])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(kmer_texts)

gata3_data['label'].value_counts().sort_index().plot.bar()

seed = 7
test_size = 0.3
# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_data,
                                                    test_size=test_size,
                                                    random_state=seed)

#addd parameters after optimisation
classifier = xgb.XGBClassifier()

classifier.fit(X_train,y_train,verbose=True ,
               early_stopping_rounds=10,eval_metric='aucpr',
               eval_set=[(X_test,y_test)])

# feature importance
#print(classifier.feature_importances_)
# plot
plot_importance(classifier,max_num_features=top_features)
pyplot.show()

#need to output symbols instead of numbers
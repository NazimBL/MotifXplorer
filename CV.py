import pickle
import xgboost as xgb
import pandas as pd

kmer_size=6

def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#read preprocessed table, see github repository
gata3_data=pd.read_csv("data_final.csv")
#print(gata3_data)


##replace seuence column with kmers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence'],kmer_size), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)
#print(gata3_data)

kmer_texts = list(gata3_data['kmers'])
for item in range(len(kmer_texts)):
    kmer_texts[item] = ' '.join(kmer_texts[item])

y_data = gata3_data.iloc[:, 0].values
#print(human_texts[0])

# Creating the Bag of Words model using CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(kmer_texts)

# Save the vectorizer
vec_file = 'vectorizer.pickle'
pickle.dump(cv, open(vec_file, 'wb'))

gata3_data['label'].value_counts().sort_index().plot.bar()

seed = 7
test_size = 0.3
# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_data,
                                                    test_size=test_size,
                                                    random_state=seed)

model=xgb.XGBClassifier()

model.fit(X_train,y_train,verbose=True,
               early_stopping_rounds=20,eval_metric='aucpr',
               eval_set=[(X_test,y_test)])

from sklearn.model_selection import cross_val_score
score=cross_val_score(model,X,y_data,cv=10)
print("Total scores over 10 folds: ")
print(score)
print("\nMean score: ")
print(score.mean())
model.save_model("xgb_model.json")
print("\nModel saved under xgb_model.json")
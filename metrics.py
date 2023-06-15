#Nazim Belabbaci aka NazimBL
#Summer 2022

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import pandas as pd

kmer_size=6
# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#read preprocessed table, see repository
gata3_data=pd.read_csv("data_final.csv")

##replace seuence column with kmers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence'],kmer_size), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)
print(gata3_data)

kmers_text = list(gata3_data['kmers'])
for item in range(len(kmers_text)):
    kmers_text[item] = ' '.join(kmers_text[item])

y_data = gata3_data.iloc[:, 0].values
#print(kmers_text[0])
# Creating the Bag of Words model using CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(kmers_text)

print(X)
gata3_data['label'].value_counts().sort_index().plot.bar()

seed = 7
test_size = 0.3
# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_data,
                                                    test_size=test_size,
                                                    random_state=seed)

# instantiate the classifier
xgb_clf = XGBClassifier()

# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
#print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
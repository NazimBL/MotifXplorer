#Nazim Belabbaci aka NazimBL
#Summer 2022

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Specify the backend framework, e.g., TkAgg


kmer_size = 6

# Function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Read preprocessed table, see repository
gata3_data = pd.read_csv("data_final.csv")

# Replace sequence column with k-mers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence'], kmer_size), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)
#print(gata3_data)

kmers_text = list(gata3_data['kmers'])
for item in range(len(kmers_text)):
    kmers_text[item] = ' '.join(kmers_text[item])

y_data = gata3_data.iloc[:, 0].values

# Creating the Bag of Words model using CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(kmers_text)
#print(X)

#use this to check data balance
#gata3_data['label'].value_counts().sort_index().plot.bar()

seed = 7
test_size = 0.3

# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=test_size, random_state=seed)
# Instantiate the classifier
xgb_clf = XGBClassifier()
# Fit the classifier to the training data
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)

# Confusion matrix
confusion_matrix = pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'))
print("Confusion matrix\n")
print(confusion_matrix)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("Accuracy = %.3f\nPrecision = %.3f\nRecall = %.3f\nF1-Score = %.3f" % (accuracy, precision, recall, f1))

# ROC Curve
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


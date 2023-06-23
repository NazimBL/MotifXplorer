import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

kmer_size = 6
top_features = 10

def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Read preprocessed table, see GitHub repository
gata3_data = pd.read_csv("data_final.csv")

# Replace sequence column with kmers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence'], kmer_size), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)

kmer_texts = list(gata3_data['kmers'])
for item in range(len(kmer_texts)):
    kmer_texts[item] = ' '.join(kmer_texts[item])

y_data = gata3_data.iloc[:, 0].values

# Creating the Bag of Words model using CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(kmer_texts)

feature_names = cv.get_feature_names_out()

gata3_data['label'].value_counts().sort_index().plot.bar()

seed = 7
test_size = 0.3

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=test_size, random_state=seed)

classifier = xgb.XGBClassifier()

classifier.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_test, y_test)])

# Get feature importance scores
importance_types = ['weight', 'cover', 'gain', 'total_gain', 'total_cover']

for importance_type in importance_types:
    importance = classifier.get_booster().get_score(importance_type=importance_type)
    feature_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_k_features = [feature_names[int(x[0][1:])] for x in feature_importance[:top_features]]
    top_k_importance = [x[1] for x in feature_importance[:top_features]]

    # Plotting the feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_features), top_k_importance, align='center')
    ax.set_yticks(range(top_features))
    ax.set_yticklabels(top_k_features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_features} {importance_type.capitalize()} Importance')

    plt.tight_layout()
    plt.savefig(f'top_{top_features}_feature_{importance_type}_importance.png')
    plt.show()

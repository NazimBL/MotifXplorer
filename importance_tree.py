# Nazim Belabbaci
# Summer 2022

import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

kmer_size = 6

def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Read preprocessed table, see github repository
gata3_data = pd.read_csv("data_final.csv")

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

gata3_data['label'].value_counts().sort_index().plot.bar()

seed = 7
test_size = 0.3

# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=test_size, random_state=seed)

classifier = xgb.XGBClassifier()
classifier.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_test, y_test)])

bst = classifier.get_booster()

for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s:' % importance_type, bst.get_score(importance_type=importance_type))

# Get the feature names from the CountVectorizer
feature_names = cv.get_feature_names_out()

# Generate the DOT data for the decision tree
dot_data = xgb.to_graphviz(classifier, num_trees=0, rankdir='UT', yes_color='#0000FF',
                           no_color='#FF0000').source

# Modify the DOT data to replace feature indices with feature names
for i, feature_name in enumerate(feature_names):
    dot_data = dot_data.replace('f%d' % i, feature_name)

# Create the graph from the modified DOT data
graph = graphviz.Source(dot_data)

# Save the graph as an image file (e.g., PNG format)
graph.render(filename='importance_tree', format='png')

# Display the graph
graph.view()

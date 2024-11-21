
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import random
from columnar import columnar

data = pd.read_csv('cleaned_tcc_ceds_music.csv')
data['post_70'] = (data['release_date'] >= 1970)

# Assume df is your DataFrame with 'lyrics' and 'genre' columns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['lyrics'])
y = data['post_70']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Train Naive Bayes
# nb = MultinomialNB()
# nb.fit(X_train, y_train)

# Train Logistic Regression
lr = LogisticRegression(C=1, solver='lbfgs', max_iter=100, verbose=0)
lr.fit(X_train, y_train)

# Train the SVM model:
# Hpyerparameters to consider:
# C, Gamma, Kernel

# You can change the kernel and parameters
# model = SVC(kernel='rbf', C=1.0, gamma='scale')
# model.fit(X_train, y_train)


# Make predictions:
y_pred = lr.predict(X_test)


# Print some predictions:
# Get 10 random songs
random_indices = random.sample(range(len(data)), 10)

songsToPredict = [
    data.iloc[i] for i in random_indices
]

results = []
headers = ['track_name', 'release_date_post_70', 'prediction']
for song in songsToPredict:
    results.append([
        song['track_name'],
        song['release_date'] >= 1970,
        lr.predict(
            vectorizer.transform([song['lyrics']])
        )[0]
    ])

table = columnar(results, headers=headers, no_borders=True, justify='l')

print("** Results **", end="\n\n")
print("Predictions with actual data for 10 random songs:")
print(table)
# Evaluate the model (for SVM):

print("Model Results:", end="\n\n")
print(classification_report(y_test, y_pred))

# Predict and evaluate (logistic regression and NB)
# y_pred = lr.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

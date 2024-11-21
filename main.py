
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import random
from columnar import columnar
from math import floor

choices = ['decade', 'genre', 'post_decade', 'isGenre']

decades = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
genres = ['pop', 'jazz', 'blues', 'hip hop', 'country', 'rock', 'reggae']

chosen_genre = genres[4]
chosen_decade = decades[3]

def getPredictionLabel(column, decade, genre) -> str:
    match(column):
        case 'post_decade':
            return f"Post {decade}'s"
        case 'isGenre':
            return f"Is it {genre}?".title()
        case _:
            return column.title()
        
data = pd.read_csv('cleaned_tcc_ceds_music.csv')
data['post_decade'] = (data['release_date'] >= chosen_decade)
data['isGenre'] = (data['genre'] == chosen_genre)
data['decade'] = data['release_date'].apply(lambda x: int(floor(int(x) / 10) * 10))


chosen_column = choices[2]

prediction_label = getPredictionLabel(chosen_column, chosen_decade, chosen_genre)

sample_size = 10



# Set up data columns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['lyrics'])
y = data[chosen_column]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

# Train Naive Bayes
# nb = MultinomialNB()
# nb.fit(X_train, y_train)

# Predict and evaluate Naive Bayes
# y_pred = nb.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# Train Logistic Regression
lr = LogisticRegression(C=1, solver='lbfgs', max_iter=150, verbose=0)
lr.fit(X_train, y_train)

# Make predictions:
y_pred = lr.predict(X_test)

# Print some predictions:
# Get 10 random songs
random_indices = random.sample(range(len(data)), sample_size)

songsToPredict = [
    data.iloc[i] for i in random_indices
]

correct_guesses = 0

results = []
headers = ['track_name', 'actual', 'prediction']
for song in songsToPredict:
    results.append([
        song['track_name'],
        song[chosen_column],
        lr.predict(
            vectorizer.transform([song['lyrics']])
        )[0]
    ])
    if results[-1][1] == results[-1][2]:
        correct_guesses += 1


table = columnar(results, headers=headers, no_borders=True, justify='l')

print("** Results **", end="\n\n")
print(f"Predicting {prediction_label} for {sample_size} random songs:")
print(table)
print(f"Accuracy of sample: {(correct_guesses / sample_size) * 100}%")



# Predict and evaluate Logistic regression
# y_pred = lr.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


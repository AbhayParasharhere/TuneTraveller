
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import random
from columnar import columnar
from math import floor
from constants import *


def getPredictionLabel(column, decade, genre) -> str:
    match(column):
        case 'post_decade':
            return f"Post {decade}'s"
        case 'isGenre':
            return f"Is it {genre}?".title()
        case _:
            return column.title()


def NBPrediction(X_train, X_test, y_train, y_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = nb.predict(X_test)
    print("Overall Accuracy:", round(accuracy_score(y_test, y_pred), 5))
    print(classification_report(y_test, y_pred))
    
    
def setupColumns(data, chosen_decade, chosen_genre):
    data['post_decade'] = (data['release_date'] >= chosen_decade)
    data['isGenre'] = (data['genre'] == chosen_genre)
    data['decade'] = data['release_date'].apply(lambda x: int(floor(int(x) / 10) * 10))
    

def setupSets(data, chosen_column):
    # Set up data columns
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['lyrics'])
    y = data[chosen_column]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test, vectorizer


def LRPrediction(data, X_train, X_test, y_train, y_test, vectorizer, prediction_label, chosen_column):
    # Train Logistic Regression
    lr = LogisticRegression(C=1, solver='lbfgs', max_iter=150, verbose=0)
    lr.fit(X_train, y_train)

    # Print some predictions:
    # Get 10 random songs
    random_indices = random.sample(range(len(data)), SAMPLE_SIZE)

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
    print(f"Predicting {prediction_label} for {SAMPLE_SIZE} random songs:")
    print(table)
    print(f"Accuracy of sample: {(correct_guesses / SAMPLE_SIZE) * 100}%\n")


    # Full prediction output
    y_pred = lr.predict(X_test)
    print("Overall Accuracy:", round(accuracy_score(y_test, y_pred), 5))
    print(classification_report(y_test, y_pred))


def main():
    data = pd.read_csv(DATABASE_NAME)


    chosen_genre = GENRES[4]
    chosen_decade = DECADES[3]
    chosen_column = CHOICES[2]

    prediction_label = getPredictionLabel(chosen_column, chosen_decade, chosen_genre)
    
    setupColumns(data, chosen_decade, chosen_genre)
    
    X_train, X_test, y_train, y_test, vectorizer = setupSets(data, chosen_column)

    while True:
        try:
            user_input = int(input("Enter 1 for Logistic Regression \nEnter 2 for Naive Bayes\nEnter 3 to Quit\n\n"))
            match user_input:
                case 1:
                    LRPrediction(data, X_train, X_test, y_train, y_test, vectorizer, prediction_label, chosen_column)
                case 2:
                    NBPrediction(X_train, X_test, y_train, y_test)
                case 3:
                    break
                case _:
                    print("Inavlid Number. Please enter a valid number")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
  
main()
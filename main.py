
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
    #y_pred = lr.predict(X_test)
    #print("Overall Accuracy:", round(accuracy_score(y_test, y_pred), 5))
    #print(classification_report(y_test, y_pred))


def main():
    data = pd.read_csv(DATABASE_NAME)
    
    print("Welcome to TuneTraveller!")
    print("Get ready to dive into the world of music! 🎶")
    print("In this interactive experience, we'll guess the genre or decade of 10 random songs based on their lyrics.")
    while True:
        try:
            print("\nMake a selection: ")
            print("1. Genre")
            print("2. Decade")
            print("3. Quit")
            
            user_input = int(input("\nEnter your choice: "))
            match user_input:
                case 1:
                    print("Choose a genre:")
                    print("1. All Genres")
                    print("2. Pop")
                    print("3. Hip Hop")
                    print("4. Blues")
                    print("5. Jazz")
                    print("6. Country")
                    print("7. Rock")
                    print("8. Reggae")
                    
                    genre_choice = int(input("\nEnter your choice: "))
                    chosen_decade = DECADES[0]
                    
                    match genre_choice:
                        case 1:
                            print("You selected: All Genres")
                            chosen_column = CHOICES[1]
                            chosen_genre = GENRES[0]
                        case 2:
                            print("You selected: Pop")
                            chosen_column = CHOICES[3]
                            chosen_genre = GENRES[0]
                            
                        case 3:
                            print("You selected: Hip Hop")
                            chosen_column = CHOICES[3]
                            chosen_genre = GENRES[3]
                        case 4:
                            print("You selected: Blues")
                            chosen_column = CHOICES[3]
                            chosen_genre = GENRES[2]
                        case 5:
                            print("You selected: Jazz")
                            chosen_column = CHOICES[3]
                            chosen_genre = GENRES[1]
                        case 6:
                            print("You selected: Country")
                            chosen_column = CHOICES[3]
                            chosen_genre = GENRES[4]
                        case 7:
                            print("You selected: Rock")
                            chosen_column = CHOICES[3]
                            chosen_genre = GENRES[5]
                        case 8:
                            print("You selected: Rock")
                            chosen_column = CHOICES[5]
                            chosen_genre = GENRES[6]
                        case _:
                            print("Invalid genre selection. Please enter a valid number.")
                            continue
                    
                    if genre_choice in range(1, 9):
                        setupColumns(data, chosen_decade, chosen_genre)
                        label = getPredictionLabel(chosen_column, chosen_decade, chosen_genre)
                        X_train, X_test, y_train, y_test, vectorizer = setupSets(data, chosen_column)
                        LRPrediction(data, X_train, X_test, y_train, y_test, vectorizer, label, chosen_column)
                
                case 2:
                    print("Choose a decade:")
                    print("1. All Decades")
                    print("2. Post 50's")
                    print("3. Post 60's")
                    print("4. Post 70's")
                    print("5. Post 80's")
                    print("6. Post 90's")
                    print("7. Post 2000")
                    
                    decade_choice = int(input("\nEnter your choice: "))
                    chosen_genre = GENRES[0]
                    match decade_choice:
                        case 1:
                            print("You selected: All Decades")
                            chosen_column = CHOICES[0]
                            chosen_decade = DECADES[0]
                        case 2:
                            print("You selected: Post 50's")
                            chosen_column = CHOICES[2]
                            chosen_decade = DECADES[0]
                        case 3:
                            print("You selected: Post 60's")
                            chosen_column = CHOICES[2]
                            chosen_decade = DECADES[1]
                        case 4:
                            print("You selected: Post 70's")
                            chosen_column = CHOICES[2]
                            chosen_decade = DECADES[2]
                        case 5:
                            print("You selected: Post 80's")
                            chosen_column = CHOICES[2]
                            chosen_decade = DECADES[3]
                        case 6:
                            print("You selected: Post 90's")
                            chosen_column = CHOICES[2]
                            chosen_decade = DECADES[4]
                        case 7:
                            print("You selected: Post 2000")
                            chosen_column = CHOICES[2]
                            chosen_decade = DECADES[5]
                        case _:
                            print("Invalid decade selection. Please enter a valid number.")
                            continue
                    
                    if decade_choice in range(1, 8):
                        setupColumns(data, chosen_decade, chosen_genre)
                        label = getPredictionLabel(chosen_column, chosen_decade, chosen_genre)
                        X_train, X_test, y_train, y_test, vectorizer = setupSets(data, chosen_column)
                        LRPrediction(data, X_train, X_test, y_train, y_test, vectorizer, label, chosen_column)
                case 3:
                    print("Goodbye!")
                    break
                case _:
                    print("Inavlid Number. Please enter a valid number")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
  
main()
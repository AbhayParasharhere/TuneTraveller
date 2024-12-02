
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import random
from columnar import columnar
from math import floor
from constants import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


def getPredictionLabel(column, decade, genre) -> str:
    if column == 'post_decade':
        return f"Post {decade}'s"
    elif column == 'isGenre':
        return f"Is it {genre}?".title()
    else:
        return column.title()


def NBPrediction(X_train, X_test, y_train, y_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = nb.predict(X_test)
    print("Overall Accuracy:", round(accuracy_score(y_test, y_pred), 5))
    print(classification_report(y_test, y_pred))
    
def press_enter():
    input("Press Enter to continue...")
    
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


def LRPredictionSample(data, X_train, y_train, vectorizer, prediction_label, chosen_column):
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

def LRPredictionAll(X_train, X_test, y_train, y_test):
    # Train Logistic Regression
    lr = LogisticRegression(C=1, solver='lbfgs', max_iter=150, verbose=0)
    lr.fit(X_train, y_train)
    
    lr = LogisticRegression(C=1, solver='lbfgs', max_iter=150, verbose=0)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    print("Overall Accuracy:", round(accuracy_score(y_test, y_pred), 5))
    print(classification_report(y_test, y_pred))

    
def topic_modeling(data):
    while True:
        user_input = input("\nEnter the number of categories you'd like to generate: ")
        try:
            num_categories = int(user_input)

            count = CountVectorizer(stop_words='english', max_features=5000)
            X = count.fit_transform(data['lyrics'].values)
            
            lda = LatentDirichletAllocation(n_components=num_categories, random_state=123, learning_method='batch')
            X_topics = lda.fit_transform(X)

            print(lda.components_.shape)
            print('*' * 13)

            n_top_words = 7
            feature_names = count.get_feature_names_out()

            for topic_idx, topic in enumerate(lda.components_):
                    print(f'Topic {(topic_idx)}:')
                    print(' '.join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))

            print('*' * 13)
                
            data['topic_category'] = X_topics.argmax(axis=1)
            genre_topic_counts = data.groupby(['genre', 'topic_category']).size().reset_index(name='count')
            pivot_table = genre_topic_counts.pivot(index='genre', columns='topic_category', values='count')

            headers = ["Genre"] + [f"Topic {col}" for col in pivot_table.columns]
            rows = [[genre] + list(pivot_table.loc[genre]) for genre in pivot_table.index]

            table = columnar(rows, headers, no_borders=True, justify='l')
            print(table)
            press_enter()
            break
            
        except ValueError as e:
            print("Sorry, that wasn't a valid number, please try again")
    
def cleanLyrics(file_path):
    stopWords = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    
    try:
        #read in file
        with open(file_path, 'r', encoding='utf-8') as file:
            lyrics = file.read()
         
        #Tokenize the lyrics
        tokens = word_tokenize(lyrics)

        #Remove punctuation and convert to lowercase
        tokens = [re.sub(r'[^\w\s]', '', token).lower() for token in tokens if re.sub(r'[^\w\s]', '', token)]

        #Remove stop words
        filtered_tokens = [token for token in tokens if token not in stopWords]

        #Stem the words
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

        return ' '.join(stemmed_tokens)
    except FileNotFoundError:
        return "File not found. Please check the file path."
    except Exception as e:
        return f"An error occurred: {e}"
    
    
def PredictionUpload(data, X_train, X_test, y_train, y_test, vectorizer, prediction_label, chosen_column, clean_lyrics):
    # Vectorize the cleaned lyrics
    cleaned_lyrics_vector = vectorizer.transform([clean_lyrics])
    
    # Train Logistic Regression
    lr = LogisticRegression(C=1, solver='lbfgs', max_iter=150, verbose=0)
    lr.fit(X_train, y_train)

    # Print some predictions:
    print("\n** Results **", end="\n")
    print(f"Predicting {prediction_label}:")
    prediction = lr.predict(cleaned_lyrics_vector)
    print(prediction[0])
    print("\n")

def uploadRegression(data):
    file_path = (input("\nEnter the file path: ")) #doesnt check if valid file yet
    cleaned_lyrics = cleanLyrics(file_path)
    print("Song uploaded!")

    while True:
        print("\nMake a selection: ")
        print("1. Genre")
        print("2. Decade")
        print("3. Back to main menu")

        user_choice = int(input("\nEnter your choice: "))

        if user_choice == 3:
            break

        elif user_choice == 1:
            while True:
                print("Choose a genre:")
                print("1. All Genres")
                print("2. Pop")
                print("3. Hip Hop")
                print("4. Blues")
                print("5. Jazz")
                print("6. Country")
                print("7. Rock")
                print("8. Reggae")
                print("9. Back to menu")
                
                genre_choice = int(input("\nEnter your choice: "))
                chosen_decade = DECADES[0]

                if genre_choice == 9:
                    break
                
                elif genre_choice == 1:
                    print("You selected: All Genres")
                    chosen_column = CHOICES[1]
                    chosen_genre = GENRES[0]
                elif genre_choice == 2:
                    print("You selected: Is it Pop")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[0]
                    
                elif genre_choice == 3:
                    print("You selected: Hip Hop")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[3]
                elif genre_choice == 4:
                    print("You selected: Blues")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[2]
                elif genre_choice == 5:
                    print("You selected: Jazz")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[1]
                elif genre_choice == 6:
                    print("You selected: Country")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[4]
                elif genre_choice == 7:
                    print("You selected: Rock")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[5]
                elif genre_choice == 8:
                    print("You selected: Raggae")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[6]
                else:
                    print("Invalid genre selection. Please enter a valid number.")
                    continue
                
                if genre_choice in range(1, 9):
                    setupColumns(data, chosen_decade, chosen_genre)
                    label = getPredictionLabel(chosen_column, chosen_decade, chosen_genre)
                    X_train, X_test, y_train, y_test, vectorizer = setupSets(data, chosen_column)
                    PredictionUpload(data, X_train, X_test, y_train, y_test, vectorizer, label, chosen_column, cleaned_lyrics)
                    press_enter()
                    break
                    
        elif user_choice == 2:
            while True:
                print("Choose a decade:")
                print("1. All Decades")
                print("2. Post 60's")
                print("3. Post 70's")
                print("4. Post 80's")
                print("5. Post 90's")
                print("6. Post 2000")
                print("7. Back to menu")
                
                decade_choice = int(input("\nEnter your choice: "))
                chosen_genre = GENRES[0]

                if decade_choice == 7:
                    break

                elif decade_choice == 1:
                    print("You selected: All Decades")
                    chosen_column = CHOICES[0]
                    chosen_decade = DECADES[0]
                elif decade_choice == 2:
                    print("You selected: Post 60's")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[1]
                elif decade_choice == 3:
                    print("You selected: Post 70's")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[2]
                elif decade_choice == 4:
                    print("You selected: Post 80's")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[3]
                elif decade_choice == 5:
                    print("You selected: Post 90's")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[4]
                elif decade_choice == 6:
                    print("You selected: Post 2000")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[5]
                else:
                    print("Invalid decade selection. Please enter a valid number.")
                    continue
                
                if decade_choice in range(1, 7):
                    setupColumns(data, chosen_decade, chosen_genre)
                    label = getPredictionLabel(chosen_column, chosen_decade, chosen_genre)
                    X_train, X_test, y_train, y_test, vectorizer = setupSets(data, chosen_column)
                    PredictionUpload(data, X_train, X_test, y_train, y_test, vectorizer, label, chosen_column, cleaned_lyrics)
                    press_enter()
                    break
        else:
                print("Invalid Number. Please enter a valid number")

def guessRegression(data, useSmallSample):
    while True:
        print("\nMake a selection: ")
        print("1. Genre")
        print("2. Decade")
        print("3. Quit")
        
        user_input = int(input("\nEnter your choice: "))

        if user_input == 3:
            break

        elif user_input == 1:
            while True:
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
                
                if genre_choice == 1:
                    print("You selected: All Genres")
                    chosen_column = CHOICES[1]
                    chosen_genre = GENRES[0]
                elif genre_choice == 2:
                    print("You selected: Pop")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[0]
                elif genre_choice == 3:
                    print("You selected: Hip Hop")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[3]
                elif genre_choice == 4:
                    print("You selected: Blues")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[2]
                elif genre_choice == 5:
                    print("You selected: Jazz")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[1]
                elif genre_choice == 6:
                    print("You selected: Country")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[4]
                elif genre_choice == 7:
                    print("You selected: Rock")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[5]
                elif genre_choice == 8:
                    print("You selected: Reggae")
                    chosen_column = CHOICES[3]
                    chosen_genre = GENRES[6]
                else:
                    print("Invalid genre selection. Please enter a valid number.")
                    continue
                
                if genre_choice in range(1, 9):
                    setupColumns(data, chosen_decade, chosen_genre)
                    label = getPredictionLabel(chosen_column, chosen_decade, chosen_genre)
                    X_train, X_test, y_train, y_test, vectorizer = setupSets(data, chosen_column)
                    if useSmallSample:
                        LRPredictionSample(data, X_train, y_train, vectorizer, label, chosen_column)
                    else:
                        LRPredictionAll(X_train, X_test, y_train, y_test)
                    press_enter()
                    break
            
        elif user_input == 2:
            while True:
                print("Choose a decade:")
                print("1. All Decades")
                print("2. Post 60's")
                print("3. Post 70's")
                print("4. Post 80's")
                print("5. Post 90's")
                print("6. Post 2000")
                
                decade_choice = int(input("\nEnter your choice: "))
                chosen_genre = GENRES[0]
                if decade_choice == 1:
                    print("You selected: All Decades")
                    chosen_column = CHOICES[0]
                    chosen_decade = DECADES[0]
                elif decade_choice == 2:
                    print("You selected: Post 60's")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[1]
                elif decade_choice == 3:
                    print("You selected: Post 70's")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[2]
                elif decade_choice == 4:
                    print("You selected: Post 80's")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[3]
                elif decade_choice == 5:
                    print("You selected: Post 90's")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[4]
                elif decade_choice == 6:
                    print("You selected: Post 2000")
                    chosen_column = CHOICES[2]
                    chosen_decade = DECADES[5]
                else:
                    print("Invalid decade selection. Please enter a valid number.")
                    continue
                
                if decade_choice in range(1, 7):
                    setupColumns(data, chosen_decade, chosen_genre)
                    label = getPredictionLabel(chosen_column, chosen_decade, chosen_genre)
                    X_train, X_test, y_train, y_test, vectorizer = setupSets(data, chosen_column)
                    if useSmallSample:
                        LRPredictionSample(data, X_train, y_train, vectorizer, label, chosen_column)
                    else:
                        LRPredictionAll(X_train, X_test, y_train, y_test)
                    press_enter()
                    break
        else:
            print("Invalid Number. Please enter a valid number")           


def main():
    data = pd.read_csv(DATABASE_NAME)
    
    print("Welcome to TuneTraveller!")
    print("Get ready to dive into the world of music! 🎶")
    print("In this interactive experience, we'll guess the genre or decade of 10 random songs based on their lyrics.")
    while True:
        try:
            print("\nMake a selection: ")
            print("1. Upload your own lyrics")
            print("2. Predict 10 random songs")
            print("3. Predict the whole database")
            print("4. Use topic Modeling to categorize the database")
            print("5. Quit")
            
            user_input = int(input("\nEnter your choice: "))
            if user_input == 1:
                uploadRegression(data)
            elif user_input == 2:
                guessRegression(data, useSmallSample=True) 
            elif user_input == 3:
                guessRegression(data, useSmallSample=False)
            elif user_input == 4:
                topic_modeling(data)
            elif user_input == 5:
                print("Goodbye!")
                break
            else:
                print("Invalid Number. Please enter a valid number")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
  
main()
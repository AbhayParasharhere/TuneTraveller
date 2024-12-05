# TuneTraveller 🎵

TuneTraveller is a Python-based music analysis tool that uses machine learning to predict song genres and decades based on lyrics. It employs various classification algorithms and topic modeling to analyze musical patterns across different eras and styles.

## Features

- **Lyric Analysis**: Upload and analyze your own song lyrics
- **Genre Prediction**: Classify songs into various genres including Pop, Hip Hop, Blues, Jazz, Country, Rock, and Reggae
- **Decade Prediction**: Determine if a song was released before or after specific decades (60s through 2000s)
- **Topic Modeling**: Generate customizable categories to identify common themes across genres
- **Batch Analysis**: Analyze either sample sets of 10 songs or the entire database

## Dependencies

- pandas
- scikit-learn
- nltk
- columnar
- numpy

## Installation/Setup

- Clone the repository
- Create the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

- Install the required dependencies:

```bash
pip install -r requirements.txt
```

- Download required NLTK data:

```bash
python3 extra.py
```

## Usage

Run the main script to start the interactive interface:

```bash
python main.py
```

### Main Menu Options

1. **Upload your own lyrics**: Analyze a text file containing song lyrics
2. **Predict 10 random songs**: Test the model on a small sample set
3. **Predict the whole database**: Run analysis on the complete dataset
4. **Use topic Modeling**: Generate thematic categories across the database
5. **Quit**: Exit the program

### Prediction Options

- **Genre Classification**:

  - All Genres
  - Pop
  - Hip Hop
  - Blues
  - Jazz
  - Country
  - Rock
  - Reggae

- **Decade Classification**:
  - All Decades
  - Post 60's
  - Post 70's
  - Post 80's
  - Post 90's
  - Post 2000

## Technical Details

- Uses TF-IDF vectorization for lyric processing
- Implements Logistic Regression and Naive Bayes classifiers
- Employs Latent Dirichlet Allocation (LDA) for topic modeling
- Includes text preprocessing with stopword removal and stemming

## Data Format

The program expects a CSV file (`DATABASE_NAME` in constants.py) with the following columns:

- track_name
- release_date
- genre
- lyrics

## Performance

The system provides detailed performance metrics including:

- Overall accuracy scores
- Classification reports
- Topic distribution across genres
- Sample predictions with accuracy percentages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Development Team

Hayden Lister
Molly Campbell
Abhay Parashar
Jannis Walker

Developed by JHAM!

## License

Standard MIT license

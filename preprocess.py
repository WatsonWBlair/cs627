import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data():
    """
    Loads the Banking77 dataset and preprocesses it for scikit-learn.

    Returns:
        X_train_tfidf: TF-IDF sparse matrix for training text
        X_test_tfidf: TF-IDF sparse matrix for testing text
        y_train: (pd.Series) Training labels
        y_test: (pd.Series) Testing labels
        vectorizer: The fitted TfidfVectorizer object
    """
    
    # Loading the dataset from Hugging Face
    # This command downloads it and loads it from the local cache
    print("Loading Banking77 dataset...")
    train_ds = load_dataset("PolyAI/banking77", split="train")
    test_ds = load_dataset("PolyAI/banking77", split="test")

    # Now extract text (X) and labels (y)
    # Here, X is the 'text' and y is the 'label'
    
    # Using pandas.Series for easier handling later, though lists work too
    X_train = pd.Series(train_ds['text'])
    y_train = pd.Series(train_ds['label'])
    X_test = pd.Series(test_ds['text'])
    y_test = pd.Series(test_ds['label'])

    # Next pre-processing: TF-IDF Vectorization
    # This is the key step to convert text into numerical features
    # for models like SVM, Naive Bayes, or Decision Trees 
    print("Starting TF-IDF vectorization...")
    
    # Initialize the TF-IDF Vectorizer
    # stop_words='english' removes common words (e.g., 'the', 'is', 'a')
    # max_features=5000 limits the vocabulary to the 5000 most common words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    # Fit the vectorizer ON THE TRAINING DATA ONLY (to learn the vocab)
    # and then transform the training data.
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Use the SAME fitted vectorizer to transform the test data.
    # We only call .transform() here, not .fit_transform()
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Pre-processing complete.")
    print(f"Training data shape (samples, features): {X_train_tfidf.shape}")
    print(f"Test data shape (samples, features): {X_test_tfidf.shape}")

    # Getting the label names
    label_names = train_ds.features['label'].names
    print(f"\nTotal of {len(label_names)} intents (labels).")
    print(f"Example label: {y_train[0]} = {label_names[y_train[0]]}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# --- Main execution ---
if __name__ == "__main__":
    # it will load and process the data.
    X_train, X_test, y_train, y_test, vec = load_and_preprocess_data()
    # Now the data is ready to perform.
    # Right here code which is down is for demonstration and giving an initial start to the team purpose.
    # from sklearn.naive_bayes import MultinomialNB
    # model = MultinomialNB()
    # model.fit(X_train, y_train)
    # print("Model trained!")

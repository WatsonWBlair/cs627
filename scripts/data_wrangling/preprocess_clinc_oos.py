import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data():
    """
    Loads an intent classification dataset (clinc_oos) and preprocesses it for scikit-learn.

    Returns:
        X_train_tfidf: TF-IDF sparse matrix for training text
        X_val_tfidf: TF-IDF sparse matrix for validation text
        X_test_tfidf: TF-IDF sparse matrix for testing text
        y_train: (pd.Series) Training labels
        y_val: (pd.Series) Validation labels
        y_test: (pd.Series) Testing labels
        vectorizer: The fitted TfidfVectorizer object
    """

    # Loading the clinc_oos dataset from Hugging Face as an alternative
    # This command downloads (if needed) and loads from the local cache
    print("Loading clinc_oos dataset...")

    # We load all three available splits: train, validation, and test
    train_ds = load_dataset("clinc_oos", "plus", split="train")
    val_ds = load_dataset("clinc_oos", "plus", split="validation")
    test_ds = load_dataset("clinc_oos", "plus", split="test")

    # Now extract text (X) and labels (y)
    # For clinc_oos:
    #   X is the 'text' column
    #   y is the 'intent' column

    X_train = pd.Series(train_ds['text'])
    y_train = pd.Series(train_ds['intent'])

    X_val = pd.Series(val_ds['text'])
    y_val = pd.Series(val_ds['intent'])

    X_test = pd.Series(test_ds['text'])
    y_test = pd.Series(test_ds['intent'])

    # Next pre-processing: TF-IDF Vectorization
    # This converts the raw text into a numerical feature matrix
    # required by scikit-learn models (SVM, Naive Bayes, etc.)
    print("Starting TF-IDF vectorization...")

    # Initialize the vectorizer
    # max_features=5000 is a good starting point
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    # Fit the vectorizer ON THE TRAINING DATA ONLY
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Use the SAME fitted vectorizer to transform the validation and test data
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Pre-processing complete.")
    print(f"Training data shape: {X_train_tfidf.shape}")
    print(f"Validation data shape: {X_val_tfidf.shape}")
    print(f"Test data shape: {X_test_tfidf.shape}")

    # You can also get the label names
    label_names = train_ds.features['intent'].names
    print(f"\nTotal of {len(label_names)} intents (labels).")
    print(f"Example label: {y_train[0]} = {label_names[y_train[0]]}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, vectorizer

# --- Main execution ---
if __name__ == "__main__":
    # it will load and process the data.

    # Load all the pre-processed data
    X_train, X_val, X_test, y_train, y_val, y_test, vec = load_and_preprocess_data()

    print("\nData is loaded and pre-processed.")
    print("Now our team will import this function into our main training script.")

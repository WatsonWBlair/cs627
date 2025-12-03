import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    """
    Loads the MELD dataset, flattens it for sentence-level
    emotion classification, encodes labels, and preprocesses
    it for scikit-learn.

    Returns:
        X_train_tfidf: TF-IDF sparse matrix for training text
        X_val_tfidf: TF-IDF sparse matrix for validation text
        X_test_tfidf: TF-IDF sparse matrix for testing text
        y_train: (pd.Series) Encoded training emotion labels (int)
        y_val: (pd.Series) Encoded validation emotion labels (int)
        y_test: (pd.Series) Encoded testing emotion labels (int)
        vectorizer: The fitted TfidfVectorizer object
        label_encoder: The fitted LabelEncoder object (to see names)
    """

    # Loading the dataset from Hugging Face
    print("Loading MELD dataset...")
    # This is a standard, safe-to-load dataset
    # It will be downloaded and cached locally
    train_ds = load_dataset("declare-lab/MELD", split="train")
    val_ds = load_dataset("declare-lab/MELD", split="validation")
    test_ds = load_dataset("declare-lab/MELD", split="test")

    # Now extract X (Utterance) and y (Emotion)
    # This dataset is already 'flat', so no need to loop
    print("Extracting X (text) and y (labels)...")
    X_train = pd.Series(train_ds['Utterance'])
    y_train_str = pd.Series(train_ds['Emotion'])

    X_val = pd.Series(val_ds['Utterance'])
    y_val_str = pd.Series(val_ds['Emotion'])

    X_test = pd.Series(test_ds['Utterance'])
    y_test_str = pd.Series(test_ds['Emotion'])

    # Next pre-processing (Part A): Label Encoding
    # We must convert text labels ("joy", "anger") to numbers (0, 1)
    # for scikit-learn models.
    print("Encoding text labels to integers...")
    label_encoder = LabelEncoder()

    # Fit the encoder on the TRAINING labels only
    y_train = label_encoder.fit_transform(y_train_str)

    # Transform the validation and test labels
    y_val = label_encoder.transform(y_val_str)
    y_test = label_encoder.transform(y_test_str)

    # Pre-processing (Part B): TF-IDF Vectorization
    print("Starting TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    # Fit the vectorizer ON THE TRAINING TEXT ONLY
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Use the SAME fitted vectorizer to transform validation and test text
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Pre-processing complete.")
    print(f"Training data shape: {X_train_tfidf.shape}")
    print(f"Validation data shape: {X_val_tfidf.shape}")
    print(f"Test data shape: {X_test_tfidf.shape}")

    # You can see the label mapping
    print(f"\nTotal of {len(label_encoder.classes_)} emotion labels.")
    example_label_int = y_train[0]
    example_label_str = label_encoder.inverse_transform([example_label_int])[0]
    print(f"Example label: {example_label_int} = {example_label_str}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, vectorizer, label_encoder

# --- Main execution ---
if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test, vec, le = load_and_preprocess_data()

    print("\nData is loaded and pre-processed.")
    print("Our team can now import this function into our main training script.")

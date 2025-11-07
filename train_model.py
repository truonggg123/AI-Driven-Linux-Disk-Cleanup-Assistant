import os
import pandas as pd
import numpy as np
import random
import sys
import joblib # Library for saving and loading models
from pathlib import Path

# Machine Learning Libraries
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import spacy
import torch

# ==================== SYNTHETIC DATA CONFIGURATION AND LABELING RULES ====================
# Directory to store trained models
# Initialize spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
    print("[INFO] spaCy model loaded: en_core_web_lg")
except OSError:
    print("[ERROR] spaCy model 'en_core_web_lg' not found.")
    print("Please run: python -m spacy download en_core_web_lg")
    sys.exit(1)

# Initialize Transformers model (using a lightweight model for text embeddings)
try:
    # Use a lightweight and fast model for embeddings
    TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[INFO] Loading Transformers model: {TRANSFORMER_MODEL_NAME}...")
    transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
    transformer_model.eval()  # Evaluation mode
    print("[INFO] Transformers model loaded successfully")
except Exception as e:
    print(f"[WARNING] Could not load Transformers model: {e}")
    print("Program will continue using spaCy only")
    transformer_tokenizer = None
    transformer_model = None

ASSETS_DIR = Path("ml_assets")
MODEL_FILE = ASSETS_DIR / "disk_model.joblib"
ENCODER_FILE = ASSETS_DIR / "label_encoder.joblib"
PCA_FILE = ASSETS_DIR / "pca_transformer.joblib"  # Save PCA transformer for embedding vectors

# Configuration for generating synthetic data (USED FOR TRAINING ONLY)
SYNTHETIC_NUM_FILES = 10000 # Increase number of synthetic samples for better training
SYNTHETIC_MAX_SIZE_MB = 200
SYNTHETIC_MAX_AGE_DAYS = 730
SYNTHETIC_EXTENSIONS = [
    ".pdf", ".docx", ".xlsx", ".zip", ".rar", ".7z",
    ".jpg", ".png", ".mp4", ".mov", ".gif", ".iso",
    ".log", ".tmp", ".bak", ".cache", ".~",
    ".py", ".sh", ".c", ".cpp", ".h", ".html",
    ".txt", ".js", ".json", ".xml", ".db", ""
]

# Keywords for generating more meaningful file names
SYNTHETIC_KEYWORDS = {
    'temp': ['temp', 'temporary', 'tmp', 'cache', 'old', 'backup'],
    'important': ['document', 'report', 'project', 'important', 'final'],
    'media': ['photo', 'image', 'video', 'movie', 'picture'],
    'archive': ['archive', 'old', 'backup', 'old_version']
}

# These thresholds define the Ground Truth for the model
DELETE_SIZE_THRESHOLD = 5 * 1024 * 1024     # 5MB
DELETE_TIME_THRESHOLD = 180                 # 180 days

COMPRESS_SIZE_THRESHOLD = 50 * 1024 * 1024 # 50MB
COMPRESS_TIME_THRESHOLD = 90               # 90 days
COMPRESSED_EXTS = [".zip", ".rar", ".7z", ".tar.gz", ".gz"]
# ==============================================================================


# ----------------- TRANSFORMER EMBEDDING EXTRACTION FUNCTION -----------------
def get_transformer_embedding(text, tokenizer, model, max_length=128):
    """
    Gets embedding from transformer model.
    """
    if tokenizer is None or model is None:
        return None

    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                          truncation=True, padding=True)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Gets mean pooling of token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embeddings.numpy()
    except Exception as e:
        return None

# ----------------- FEATURE EXTRACTION FUNCTION FROM FILE NAME USING SPACY + TRANSFORMERS -----------------
def extract_spacy_features(file_path):
    """
    Extracts features from file name using a combination of spaCy and Transformers.
    Returns: dict containing NLP features
    """
    # Get file name without extension
    file_name = Path(file_path).stem.lower()

    # Process with spaCy
    doc = nlp(file_name)

    # Feature 1: Number of words in file name
    num_words = len([token for token in doc if token.is_alpha])

    # Feature 2: File name length
    name_length = len(file_name)

    # Feature 3: Contains temp/old/backup keywords
    temp_keywords = ['temp', 'tmp', 'cache', 'old', 'backup', 'bak', '~']
    has_temp_keyword = any(keyword in file_name for keyword in temp_keywords)

    # Feature 4: Contains important keywords
    important_keywords = ['important', 'final', 'document', 'report']
    has_important_keyword = any(keyword in file_name for keyword in important_keywords)

    # Feature 5: Embedding vector from spaCy (300D from en_core_web_lg)
    if len(doc) > 0 and doc.vector is not None:
        spacy_embedding = doc.vector  # 300D vector from en_core_web_lg
    else:
        spacy_embedding = np.zeros(300)  # Zero vector if no token

    # Feature 6: Embedding vector from Transformers (384D from all-MiniLM-L6-v2)
    transformer_embedding = get_transformer_embedding(file_name, transformer_tokenizer, transformer_model)
    if transformer_embedding is None or len(transformer_embedding) != 384:
        transformer_embedding = np.zeros(384)  # Zero vector if no transformer

    # Combine both embeddings (will be reduced dimensionally by PCA later)
    # Always 684D: 300D (spaCy) + 384D (Transformers)
    combined_embedding = np.concatenate([spacy_embedding, transformer_embedding])  # 300 + 384 = 684D

    return {
        'num_words': num_words,
        'name_length': name_length,
        'has_temp_keyword': int(has_temp_keyword),
        'has_important_keyword': int(has_important_keyword),
        'embedding_vector': combined_embedding
    }

# ----------------- SYNTHETIC METADATA GENERATION FUNCTION IN MEMORY -----------------
def generate_synthetic_metadata():
    """Generates synthetic metadata in memory for model training."""
    file_data_list = []

    print(f"\n[STEP 1] Start generating {SYNTHETIC_NUM_FILES} synthetic metadata records in memory...")

    # Generate more meaningful file names
    file_prefixes = ['document', 'report', 'photo', 'video', 'temp_file', 'cache',
                     'backup', 'old_file', 'project', 'data', 'log', 'archive']

    for i in range(SYNTHETIC_NUM_FILES):
        ext = random.choice(SYNTHETIC_EXTENSIONS)
        # Random size using log scale
        size_bytes = int(10 ** (random.uniform(2, np.log10(SYNTHETIC_MAX_SIZE_MB * 1024 * 1024))))
        days_ago = random.randint(1, SYNTHETIC_MAX_AGE_DAYS)

        # Create more meaningful file names
        prefix = random.choice(file_prefixes)
        file_name = f"{prefix}_{i:04d}{ext}"

        file_data_list.append({
            'file_path': file_name,
            'size_bytes': size_bytes,
            'extension': ext,
            'days_since_access': days_ago,
        })

    synthetic_df = pd.DataFrame(file_data_list)
    return label_data(synthetic_df)

# -------------------- PROCESSING AND LABELING FUNCTION --------------------
def label_data(df):
    """Calculates Features and assigns Ground Truth labels."""
    if df.empty:
        return df

    temp_extensions = [".log", ".tmp", ".bak", ".cache", ".~", ""]
    df['is_temp_file'] = df['extension'].isin(temp_extensions).astype(int)

    # Use log10 for size
    df['size_log'] = np.log10(df['size_bytes'] + 1)

    # Extract features from file name using spaCy + Transformers
    print("    Extracting features from file name using spaCy + Transformers...")
    spacy_features = []
    embedding_vectors = []

    for idx, file_path in enumerate(df['file_path']):
        features = extract_spacy_features(file_path)
        spacy_features.append({
            'num_words': features['num_words'],
            'name_length': features['name_length'],
            'has_temp_keyword': features['has_temp_keyword'],
            'has_important_keyword': features['has_important_keyword']
        })
        embedding_vectors.append(features['embedding_vector'])

        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(df)} files...")

    # Add spaCy features to DataFrame
    spacy_df = pd.DataFrame(spacy_features)
    df = pd.concat([df, spacy_df], axis=1)

    # Save embedding vectors for later dimensionality reduction
    df['_embedding_vector'] = embedding_vectors

    df['Label'] = 'Keep'

    # --- Rule 1: DELETE ---
    # Update condition to include temp keywords in the file name
    delete_cond = (
        (df['is_temp_file'] == 1) |
        (df['has_temp_keyword'] == 1) |  # Add new condition from spaCy
        ((df['size_bytes'] < DELETE_SIZE_THRESHOLD) & (df['days_since_access'] > DELETE_TIME_THRESHOLD))
    )
    df.loc[delete_cond, 'Label'] = 'Delete'

    # --- Rule 2: COMPRESS (Compress/Archive) ---
    compress_cond = (
        (df['size_bytes'] > COMPRESS_SIZE_THRESHOLD) &
        (df['days_since_access'] > COMPRESS_TIME_THRESHOLD) &
        (~df['extension'].isin(COMPRESSED_EXTS)) &
        (df['has_important_keyword'] == 0)  # Do not compress important files
    )
    df.loc[compress_cond & (df['Label'] == 'Keep'), 'Label'] = 'Compress'

    return df

# ========================= MAIN FUNCTION =========================
def main():

    print("================== ML DISK CLEANUP ASSISTANT - TRAINING ===================")

    # -------------------------------------------------------------
    # PHASE 1: SYNTHETIC DATA GENERATION AND TRAINING
    # -------------------------------------------------------------

    # STEP 1: GENERATE SYNTHETIC DATASET AND ASSIGN LABELS (GROUND TRUTH)
    synthetic_df_labeled = generate_synthetic_metadata()

    print("\n============== SYNTHETIC LABEL DISTRIBUTION (Training Ground Truth) ==============")
    print(synthetic_df_labeled['Label'].value_counts())

    # STEP 2: PREPARE DATA FOR TRAINING
    print("\n[STEP 2] Preparing features for training...")

    # Get embedding vectors and reduce dimensionality using PCA
    embedding_matrix = np.array(list(synthetic_df_labeled['_embedding_vector']))
    print(f"    Embedding vector size: {embedding_matrix.shape}")

    # Use PCA for dimensionality reduction (from 684D if both spaCy and Transformers are available, or 300D if only spaCy)
    # Reduce to 15D to retain more information from both sources
    pca_components = min(15, embedding_matrix.shape[1] - 1)
    pca = PCA(n_components=pca_components, random_state=42)
    embedding_reduced = pca.fit_transform(embedding_matrix)

    # Create column names for embedding features
    embedding_cols = [f'embedding_dim_{i}' for i in range(embedding_reduced.shape[1])]
    embedding_df = pd.DataFrame(embedding_reduced, columns=embedding_cols, index=synthetic_df_labeled.index)

    # Combine all features
    basic_features = synthetic_df_labeled[['size_log', 'days_since_access', 'is_temp_file',
                                           'num_words', 'name_length', 'has_temp_keyword', 'has_important_keyword']]
    X_synth = pd.concat([basic_features, embedding_df], axis=1)

    print(f"    Total number of features: {X_synth.shape[1]}")
    print(f"    Features: {list(X_synth.columns)}")

    y_synth = synthetic_df_labeled['Label']

    le = LabelEncoder()
    y_encoded_synth = le.fit_transform(y_synth)

    if len(synthetic_df_labeled) < 20:
        print("Not enough synthetic data to train the model. Exiting program.")
        sys.exit(0)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_synth, y_encoded_synth, test_size=0.2, random_state=42)

    print("\n[STEP 3] Start training Decision Tree model with features from spaCy + Transformers...")
    model = DecisionTreeClassifier(max_depth=10, random_state=42)  # Increase depth to handle more features
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on synthetic test set: {accuracy*100:.2f}%")

    # -------------------------------------------------------------
    # STEP 4: SAVE MODEL AND TRANSFORMERS
    # -------------------------------------------------------------
    ASSETS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    joblib.dump(pca, PCA_FILE)  # Save PCA transformer for use during prediction

    print("\n[STEP 4] Finished saving model and transformers:")
    print(f"    -> Model saved at: {MODEL_FILE}")
    print(f"    -> Encoder saved at: {ENCODER_FILE}")
    print(f"    -> PCA transformer saved at: {PCA_FILE}")
    print("\nYou can now run disk_cleaner.py to use the model.")

if __name__ == '__main__':
    main()

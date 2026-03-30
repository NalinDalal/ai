"""
Word2Vec: Skip-Gram with Sub-sampling and Negative Sampling
============================================================

Implements the Word2Vec algorithm from:
"Distributed Representations of Words and Phrases and their Compositionality" (Mikolov et al., 2013)

This implementation includes:
- Vocabulary building with frequency thresholding
- Sub-sampling of frequent words (removes "the", "of", "a", etc.)
- Skip-Gram model architecture
- Negative sampling for efficient training
- t-SNE visualization of learned embeddings

Usage:
    python word2vec.py --epochs 5 --embedding_dim 300 --window_size 4
"""

import os
import random
import zipfile
from collections import Counter

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.cuda.empty_cache()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


class Hyperparameters:
    """Configuration for Word2Vec training."""
    
    URL = "http://mattmahoney.net/dc/text8.zip"
    FILENAME = "text8.zip"
    DATASET_NAME = "text8"
    
    MIN_WORD_FREQUENCY = 10
    SUBSAMPLING_THRESHOLD = 1e-5
    
    EMBEDDING_DIM = 300
    WINDOW_SIZE = 4
    NUM_NEGATIVE_SAMPLES = 5
    
    BATCH_SIZE = 512
    EPOCHS = 5
    LEARNING_RATE = 0.003
    
    VALIDATION_WORDS = ['six', 'dog', 'state', 'christianity', 'duke', 'gun']


def download_dataset(url: str, filename: str) -> None:
    """
    Download the text8 dataset from the given URL.
    
    Args:
        url: URL to download from
        filename: Local filename to save as
    """
    if not os.path.isfile(filename):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")


def extract_dataset(filename: str) -> None:
    """Extract the zip file if not already extracted."""
    if not os.path.isfile("text8"):
        with zipfile.ZipFile(filename, 'r') as zipped_file:
            zipped_file.extractall(".")


def load_text_data(filepath: str = "text8") -> list:
    """
    Load and tokenize the text dataset.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        List of words (tokens)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text_data = f.read()
    return text_data.strip().split()


def build_vocabulary(words: list, min_frequency: int) -> dict:
    """
    Build vocabulary mapping words to indices.
    
    Only includes words appearing at least `min_frequency` times.
    
    Args:
        words: List of all words in corpus
        min_frequency: Minimum occurrence count
        
    Returns:
        Dictionary mapping word -> index
    """
    word_counter = Counter(words)
    vocabulary = {}
    
    for index, (word, count) in enumerate(word_counter.most_common()):
        if count < min_frequency:
            break
        vocabulary[word] = index
    
    return vocabulary


def subsample_frequent_words(words: list, threshold: float = 1e-5) -> list:
    """
    Sub-sample frequent words to reduce their dominance.
    
    Uses the formula from the paper:
    P(discard) = 1 - sqrt(threshold / frequency)
    
    High-frequency words (like "the", "of") are more likely to be discarded,
    while rare words are kept.
    
    Args:
        words: List of all words
        threshold: Frequency threshold for sub-sampling
        
    Returns:
        Filtered list of words
    """
    word_counter = Counter(words)
    total = len(words)
    
    def should_discard(word: str) -> bool:
        frequency = word_counter[word] / total
        if frequency > threshold:
            probability = 1 - np.sqrt(threshold / frequency)
            return random.random() < probability
        return False
    
    return [word for word in words if not should_discard(word)]


def compute_negative_sampling_distribution(indexed_words: list) -> np.ndarray:
    """
    Compute the noise distribution for negative sampling.
    
    Raises word frequencies to power 3/4 (magic number from the paper).
    This downweights very common words while upweighting rare ones.
    
    Args:
        indexed_words: List of word indices
        
    Returns:
        Probability distribution for negative sampling
    """
    counts = np.bincount(indexed_words)
    probabilities = counts / counts.sum()
    adjusted_probabilities = probabilities ** 0.75
    return adjusted_probabilities / adjusted_probabilities.sum()


def get_context_words(words: list, center_index: int, max_window_size: int = 5) -> list:
    """
    Get context words around a center word with random window size.
    
    Args:
        words: List of all words
        center_index: Index of the center (target) word
        max_window_size: Maximum window size for context
        
    Returns:
        List of context words
    """
    window_size = random.randint(1, max_window_size)
    start = max(0, center_index - window_size)
    end = min(center_index + window_size + 1, len(words))
    
    return words[start:center_index] + words[center_index + 1:end]


class Word2VecDataset(Dataset):
    """
    PyTorch Dataset for Skip-Gram Word2Vec training.
    
    Each sample returns a center word and its context words.
    """
    
    def __init__(self, indexed_words: list, window_size: int = 4):
        """
        Args:
            indexed_words: List of word indices
            window_size: Maximum context window size
        """
        self.indexed_words = indexed_words
        self.window_size = window_size
    
    def __len__(self) -> int:
        return len(self.indexed_words)
    
    def __getitem__(self, index: int):
        center_word = self.indexed_words[index]
        context_words = get_context_words(
            self.indexed_words, index, self.window_size
        )
        return center_word, context_words


def create_collate_function(
    vocabulary_size: int,
    negative_distribution: np.ndarray,
    num_negative_samples: int
):
    """
    Create a collate function for batching Skip-Gram samples.
    
    Handles:
    - Flattening center-context pairs
    - Generating negative samples for the batch
    
    Args:
        vocabulary_size: Size of the vocabulary
        negative_distribution: Distribution for negative sampling
        num_negative_samples: Number of negative samples per positive pair
        
    Returns:
        Collate function
    """
    neg_dist_tensor = torch.tensor(negative_distribution, dtype=torch.float)
    
    def collate_fn(batch):
        center_words = []
        context_words = []
        
        for center, contexts in batch:
            for context in contexts:
                center_words.append(center)
                context_words.append(context)
        
        center_tensor = torch.LongTensor(center_words)
        context_tensor = torch.LongTensor(context_words)
        
        num_pairs = len(center_tensor)
        neg_samples_flat = torch.multinomial(
            neg_dist_tensor,
            num_pairs * num_negative_samples,
            replacement=True
        )
        neg_samples = neg_samples_flat.view(num_pairs, num_negative_samples)
        
        return center_tensor, context_tensor, neg_samples
    
    return collate_fn


class SkipGramModel(nn.Module):
    """
    Skip-Gram model with negative sampling.
    
    Architecture:
    - Input embedding layer (vocabulary_size -> embedding_dim)
    - Output embedding layer (vocabulary_size -> embedding_dim)
    
    The model learns two sets of embeddings: input (center) and output (context).
    At inference, we typically use only the input embeddings.
    """
    
    def __init__(self, vocabulary_size: int, embedding_dim: int):
        """
        Args:
            vocabulary_size: Number of unique words
            embedding_dim: Dimensionality of word vectors
        """
        super().__init__()
        self.input_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        
        self.input_embeddings.weight.data.uniform_(-1, 1)
        self.output_embeddings.weight.data.uniform_(-1, 1)
    
    def forward(
        self, 
        center_words: torch.Tensor, 
        positive_context: torch.Tensor, 
        negative_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass computing the Skip-Gram loss.
        
        Args:
            center_words: Tensor of center word indices [batch_size]
            positive_context: Tensor of positive context indices [batch_size]
            negative_context: Tensor of negative samples [batch_size, num_neg]
            
        Returns:
            Scalar loss value
        """
        center_vectors = self.input_embeddings(center_words)
        pos_context_vectors = self.output_embeddings(positive_context)
        neg_context_vectors = self.output_embeddings(negative_context)
        
        pos_score = torch.einsum("ij,ij->i", center_vectors, pos_context_vectors)
        neg_score = torch.einsum("ijk,ik->ij", neg_context_vectors, center_vectors)
        
        pos_loss = F.logsigmoid(pos_score).mean()
        neg_loss = F.logsigmoid(-neg_score).mean()
        
        return -(pos_loss + neg_loss)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity score
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)


def find_similar_words(
    query_word: str,
    word_to_idx: dict,
    idx_to_word: dict,
    embeddings: np.ndarray,
    top_k: int = 10
) -> list:
    """
    Find the most similar words to the query word.
    
    Args:
        query_word: Word to find similar words for
        word_to_idx: Vocabulary mapping
        idx_to_word: Reverse vocabulary mapping
        embeddings: Word embedding matrix
        top_k: Number of similar words to return
        
    Returns:
        List of (word, similarity_score) tuples
    """
    query_idx = word_to_idx.get(query_word)
    if query_idx is None:
        return []
    
    query_vector = embeddings[query_idx]
    similarities = []
    
    for word, idx in word_to_idx.items():
        if word == query_word:
            continue
        similarity = cosine_similarity(query_vector, embeddings[idx])
        similarities.append((word, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def plot_word_embeddings(
    embeddings: np.ndarray,
    word_to_idx: dict,
    idx_to_word: dict,
    words_to_plot: list = None,
    n_words: int = 100
):
    """
    Visualize word embeddings using t-SNE.
    
    Args:
        embeddings: Word embedding matrix
        word_to_idx: Vocabulary mapping
        idx_to_word: Reverse vocabulary mapping
        words_to_plot: Specific words to highlight
        n_words: Number of random words to include
    """
    import matplotlib.pyplot as plt
    
    if words_to_plot is None:
        words_to_plot = []
    
    all_indices = list(range(len(idx_to_word)))
    random.shuffle(all_indices)
    selected_indices = all_indices[:n_words]
    
    for word in words_to_plot:
        if word in word_to_idx:
            selected_indices.append(word_to_idx[word])
    
    selected_indices = list(set(selected_indices))
    selected_embeddings = embeddings[selected_indices]
    selected_words = [idx_to_word[idx] for idx in selected_indices]
    
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(selected_indices) - 1))
    embeddings_2d = tsne.fit_transform(selected_embeddings)
    
    plt.figure(figsize=(14, 10))
    
    for i, word in enumerate(selected_words):
        if word in words_to_plot:
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c='red', s=100, zorder=5)
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=12, fontweight='bold')
        else:
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c='blue', alpha=0.6)
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.7)
    
    plt.title("Word2Vec Embeddings (t-SNE Visualization)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig("word2vec_tsne.png", dpi=150)
    plt.show()
    print("Visualization saved to word2vec_tsne.png")


def train_word2vec():
    """Main training function."""
    print("=" * 60)
    print("Word2Vec: Skip-Gram with Sub-sampling and Negative Sampling")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    params = Hyperparameters()
    
    print("\n[1/7] Downloading dataset...")
    download_dataset(params.URL, params.FILENAME)
    extract_dataset(params.FILENAME)
    
    print("\n[2/7] Loading text data...")
    words = load_text_data(params.DATASET_NAME)
    print(f"   Total words: {len(words):,}")
    print(f"   Unique words: {len(set(words)):,}")
    
    print("\n[3/7] Building vocabulary...")
    word_to_idx = build_vocabulary(words, params.MIN_WORD_FREQUENCY)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    print(f"   Vocabulary size (min_freq={params.MIN_WORD_FREQUENCY}): {vocab_size:,}")
    
    words = [w for w in words if w in word_to_idx]
    print(f"   Words after filtering: {len(words):,}")
    
    print("\n[4/7] Sub-sampling frequent words...")
    subsampled_words = subsample_frequent_words(words, params.SUBSAMPLING_THRESHOLD)
    print(f"   Words before sub-sampling: {len(words):,}")
    print(f"   Words after sub-sampling: {len(subsampled_words):,}")
    
    print("\n[5/7] Preparing training data...")
    indexed_words = [word_to_idx[token] for token in subsampled_words]
    neg_sampling_dist = compute_negative_sampling_distribution(indexed_words)
    
    collate_fn = create_collate_function(
        vocab_size, neg_sampling_dist, params.NUM_NEGATIVE_SAMPLES
    )
    
    dataset = Word2VecDataset(indexed_words, params.WINDOW_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True
    )
    print(f"   Training batches: {len(dataloader):,}")
    
    print("\n[6/7] Initializing model...")
    model = SkipGramModel(vocab_size, params.EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
    
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Embedding dimension: {params.EMBEDDING_DIM}")
    print(f"   Window size: {params.WINDOW_SIZE}")
    print(f"   Negative samples: {params.NUM_NEGATIVE_SAMPLES}")
    
    print("\n[7/7] Training...")
    print("-" * 60)
    
    loss_history = []
    
    for epoch in range(params.EPOCHS):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch + 1}/{params.EPOCHS}")
        
        epoch_loss = 0
        for step, (centers, contexts, negatives) in progress_bar:
            centers, contexts, negatives = centers.to(device), contexts.to(device), negatives.to(device)
            
            optimizer.zero_grad()
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            loss_history.append(current_loss)
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n   Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
        
        print("   Top similar words:")
        embeddings = model.input_embeddings.weight.data.cpu().numpy()
        for word in params.VALIDATION_WORDS:
            similar = find_similar_words(word, word_to_idx, idx_to_word, embeddings, 5)
            similar_str = ", ".join([f"{w} ({s:.3f})" for w, s in similar])
            print(f"      {word}: {similar_str}")
        print("-" * 60)
    
    print("\nTraining complete!")
    print(f"Final loss: {loss_history[-1]:.4f}")
    
    print("\nSaving embeddings...")
    embeddings = model.input_embeddings.weight.data.cpu().numpy()
    torch.save(model.state_dict(), "word2vec_model.pt")
    np.save("word_embeddings.npy", embeddings)
    print("Model saved to word2vec_model.pt")
    print("Embeddings saved to word_embeddings.npy")
    
    print("\nGenerating visualization...")
    plot_word_embeddings(embeddings, word_to_idx, idx_to_word, params.VALIDATION_WORDS)
    
    return model, word_to_idx, idx_to_word, embeddings


if __name__ == "__main__":
    model, word_to_idx, idx_to_word, embeddings = train_word2vec()

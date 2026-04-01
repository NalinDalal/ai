"""
train.py - Training script for Seq2Seq translation model
"""

import numpy as np
import sys
sys.path.append('..')

from seq2seq import Seq2Seq
from data_utils import Vocabulary, DataIterator, get_small_dataset, prepare_data


def train(model, train_iter, epochs=100, lr=0.01, clip_grad=5.0, print_every=10):
    """Train the Seq2Seq model."""
    
    losses = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_iter:
            batch_size = X_batch.shape[2]
            
            loss = model.train_step(X_batch, y_batch, 
                                   teacher_forcing_ratio=0.5, 
                                   learning_rate=lr)
            
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if epoch % print_every == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            if epoch % (print_every * 5) == 0:
                test_translation(model, train_iter.src_vocab, train_iter.tgt_vocab)
    
    return losses


def test_translation(model, src_vocab, tgt_vocab, num_samples=3):
    """Test translation with sample sentences."""
    print("\n--- Sample Translations ---")
    
    test_pairs = [
        ("hello", None),
        ("goodbye", None),
        ("thank you", None),
        ("how are you", None),
    ]
    
    for src, _ in test_pairs[:num_samples]:
        src_enc = np.array(src_vocab.encode(src, add_eos=True))
        src_onehot = np.zeros((len(src_vocab), len(src_enc), 1))
        src_onehot[src_enc, np.arange(len(src_enc)), 0] = 1
        
        sos_idx = tgt_vocab.token2idx['<SOS>']
        eos_idx = tgt_vocab.token2idx['<EOS>']
        
        pred = model.predict_greedy(src_onehot, sos_idx=sos_idx, eos_idx=eos_idx, max_len=20)
        translation = tgt_vocab.decode(pred)
        
        print(f"  {src} -> {translation}")


def main():
    print("=== Seq2Seq Translation Training ===\n")
    
    pairs = get_small_dataset()
    print(f"Loaded {len(pairs)} translation pairs\n")
    
    src_sentences = [p[0] for p in pairs]
    tgt_sentences = [p[1] for p in pairs]
    
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}\n")
    
    model = Seq2Seq(
        input_vocab_size=len(src_vocab),
        output_vocab_size=len(tgt_vocab),
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2
    )
    
    print("Model initialized:")
    print(f"  Embedding dim: 64")
    print(f"  Hidden dim: 128")
    print(f"  Num layers: 2\n")
    
    train_iter = DataIterator(
        source_seqs=src_sentences,
        target_seqs=tgt_sentences,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        batch_size=4,
        max_len=20
    )
    
    print("Training...\n")
    losses = train(model, train_iter, epochs=200, lr=0.01, print_every=20)
    
    print("\n=== Final Translations ===")
    test_translation(model, src_vocab, tgt_vocab, num_samples=5)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

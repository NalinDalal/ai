"""
eng2Fre.py - Toy English to French Translator (Seq2Seq with Attention)

A seq2seq model for English→French translation using LSTM encoder-decoder with Bahdanau attention.
Based on Sutskever, Vinyals & Le (2014) and Bahdanau, Cho & Bengio (2014).

Usage:
    python eng2Fre.py --train          # Train the model
    python eng2Fre.py --translate "hello"  # Translate a sentence
    python eng2Fre.py --demo           # Run demo with test sentences
    python eng2Fre.py --attention      # Use attention model (default now)
"""

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import re


class Vocabulary:
    """
    Vocabulary class for mapping tokens to indices and vice versa.
    
    Special tokens:
        <PAD>: Padding token (index 0)
        <SOS>: Start of sequence (index 1) - used as decoder input start
        <EOS>: End of sequence (index 2) - marks end of translation
        <UNK>: Unknown token (index 3) - for out-of-vocabulary words
    """
    
    def __init__(self):
        self.token2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2token = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.n_tokens = 4
    
    def build_vocab(self, sentences: List[str], min_freq: int = 1) -> None:
        """Build vocabulary from a list of sentences."""
        token_freq = {}
        for sent in sentences:
            tokens = self._tokenize(sent)
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        for token, freq in token_freq.items():
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                self.n_tokens += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase words and punctuation."""
        return re.findall(r'\w+|[^\s\w]', text.lower())
    
    def encode(self, sentence: str, max_len: Optional[int] = None, 
               add_sos: bool = False, add_eos: bool = True) -> List[int]:
        """Encode a sentence to a list of token indices."""
        tokens = self._tokenize(sentence)
        indices = [self.token2idx.get(t, self.token2idx['<UNK>']) for t in tokens]
        
        if add_sos:
            indices = [self.token2idx['<SOS>']] + indices
        if add_eos:
            indices.append(self.token2idx['<EOS>'])
        
        if max_len is not None:
            if len(indices) < max_len:
                indices += [self.token2idx['<PAD>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
        
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Decode indices back to a human-readable sentence."""
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, '<UNK>')
            if skip_special and token in ['<PAD>', '<SOS>', '<EOS>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        return self.n_tokens


class Encoder(nn.Module):
    """
    LSTM Encoder that reads the source (English) sequence and produces
    a context vector (hidden state and cell state) for the decoder.
    
    Architecture:
        - Embedding layer: maps token indices to dense vectors
        - LSTM: processes the embedded sequence
    
    For attention, we need ALL encoder hidden states, not just the final one.
    This is different from basic Seq2Seq where only the final state is used.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Source token indices (batch_size, seq_len)
        Returns:
            hidden: Final hidden state (num_layers, batch, hidden_dim)
            cell: Final cell state (num_layers, batch, hidden_dim)
            outputs: All encoder hidden states (batch, seq_len, hidden_dim)
                - Used by attention mechanism to compute context
        """
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell, outputs


class Decoder(nn.Module):
    """
    LSTM Decoder that generates the target (French) sequence from
    the context vector produced by the encoder.
    
    Architecture:
        - Embedding layer: maps token indices to dense vectors
        - LSTM: generates next token based on previous token and hidden state
        - Linear: projects LSTM output to vocabulary size for prediction
    
    Note: This is the basic decoder WITHOUT attention.
    For attention, use DecoderAttention instead.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor, 
                cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input token indices (batch_size, 1)
            hidden: Hidden state from previous step
            cell: Cell state from previous step
        Returns:
            output: Vocabulary logits (batch_size, 1, vocab_size)
            hidden: Updated hidden state
            cell: Updated cell state
        """
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(output)
        return predictions, hidden, cell


class DecoderAttention(nn.Module):
    """
    LSTM Decoder with Bahdanau (Additive) Attention.
    
    The attention mechanism allows the decoder to "look at" all encoder hidden states
    at each decoding step, rather than relying solely on a fixed context vector.
    
    Bahdanau Attention (also called "additive attention"):
        1. Compute attention scores: e_t = v^T * tanh(W_a * s_t + U_a * h_i)
        2. Compute attention weights: α_t = softmax(e_t)
        3. Compute context vector: c_t = Σ α_t[i] * h_i
        
    Where:
        - s_t = decoder hidden state at time t (query)
        - h_i = encoder hidden state at time i (key/value)
        - α_t[i] = attention weight for encoder state i
        
    Reference: Bahdanau, Cho & Bengio (2014) - "Neural Machine Translation by Jointly Learning to Align and Translate"
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim, hidden_dim, num_layers,
            batch_first=True
        )
        
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.U_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor, 
                cell: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.
        
        Args:
            x: Input token indices (batch_size, 1)
            hidden: Decoder hidden state (num_layers, batch, hidden_dim)
            cell: Decoder cell state (num_layers, batch, hidden_dim)
            encoder_outputs: All encoder hidden states (batch, src_len, hidden_dim)
        
        Returns:
            output: Vocabulary logits (batch_size, 1, vocab_size)
            hidden: Updated hidden state
            cell: Updated cell state
        """
        batch_size = x.size(0)
        
        embedded = self.embedding(x)
        
        s_t = hidden[-1].unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        
        scores = self.v_a(torch.tanh(self.W_a(s_t) + self.U_a(encoder_outputs)))
        scores = scores.squeeze(2)
        
        attn_weights = F.softmax(scores, dim=1)
        
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        lstm_input = torch.cat([embedded.squeeze(1), context], dim=1).unsqueeze(1)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        predictions = self.fc(output)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    """
    Complete Sequence-to-Sequence model with Encoder-Decoder architecture.
    
    The model works as follows:
    1. Encoder reads English source sentence and produces context vector
    2. Decoder generates French target sentence one token at a time
    3. Teacher forcing is used during training (50% probability)
    4. Greedy decoding is used during inference
    
    Training objective: Maximize log probability of correct translation
    
    Reference: Sutskever, Vinyals & Le (2014) - "Sequence to Sequence Learning with Neural Networks"
    """
    
    def __init__(self, input_vocab_size: int, output_vocab_size: int,
                 embed_dim: int = 256, hidden_dim: int = 512, 
                 num_layers: int = 2, dropout: float = 0.2,
                 use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.encoder = Encoder(input_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        
        if use_attention:
            self.decoder = DecoderAttention(output_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        else:
            self.decoder = Decoder(output_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        
        self.output_vocab_size = output_vocab_size
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Logits for each position (batch_size, tgt_len, vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        outputs = torch.zeros(batch_size, tgt_len, self.output_vocab_size, device=src.device)
        
        hidden, cell, encoder_outputs = self.encoder(src)
        
        decoder_input = tgt[:, 0:1]
        for t in range(1, tgt_len):
            if self.use_attention:
                output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            else:
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            outputs[:, t] = output.squeeze(1)
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = tgt[:, t:t+1] if teacher_force else top1
        
        return outputs
    
    def translate(self, src: torch.Tensor, sos_idx: int, eos_idx: int, 
                  max_len: int = 50) -> List[int]:
        """
        Translate a single source sentence using greedy decoding.
        
        Args:
            src: Source sequence (1, src_len)
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            max_len: Maximum generation length
        
        Returns:
            List of translated token indices (excluding SOS and EOS)
        """
        self.eval()
        with torch.no_grad():
            hidden, cell, encoder_outputs = self.encoder(src)
            decoder_input = torch.tensor([[sos_idx]], device=src.device)
            
            translations = []
            for _ in range(max_len):
                if self.use_attention:
                    output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
                else:
                    output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                
                top1 = output.argmax(2)
                token = top1.item()
                
                if token == eos_idx:
                    break
                translations.append(token)
                decoder_input = top1
            
            return translations


class TranslationDataset(Dataset):
    """PyTorch Dataset for translation pairs."""
    
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str],
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_len: int = 50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.src_sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.src_vocab.encode(self.src_sentences[idx], max_len=self.max_len, add_eos=True)
        tgt = self.tgt_vocab.encode(self.tgt_sentences[idx], max_len=self.max_len,
                                     add_sos=True, add_eos=True)
        return torch.tensor(src), torch.tensor(tgt)


def get_toy_dataset() -> List[Tuple[str, str]]:
    """
    Get a small toy dataset for testing the translator.
    
    Returns:
        List of (English, French) translation pairs
    """
    pairs = [
        ("hello", "bonjour"),
        ("goodbye", "au revoir"),
        ("thank you", "merci"),
        ("please", "s'il vous plait"),
        ("yes", "oui"),
        ("no", "non"),
        ("good morning", "bonjour"),
        ("good night", "bonne nuit"),
        ("how are you", "comment allez vous"),
        ("i love you", "je t'aime"),
        ("what is your name", "comment vous appelez vous"),
        ("my name is", "je m'appelle"),
        ("where is the bathroom", "ou est la salle de bain"),
        ("i don't understand", "je ne comprends pas"),
        ("speak english", "parlez anglais"),
        ("how much", "combien"),
        ("water", "eau"),
        ("food", "nourriture"),
        ("help", "aide"),
        ("stop", "arret"),
        ("good", "bien"),
        ("bad", "mauvais"),
        ("big", "grand"),
        ("small", "petit"),
        ("day", "jour"),
        ("night", "nuit"),
        ("friend", "ami"),
        ("family", "famille"),
        ("house", "maison"),
        ("car", "voiture"),
    ]
    return pairs


def train_model(model: nn.Module, train_loader: DataLoader, 
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device, epochs: int = 100,
                clip_grad: float = 1.0) -> List[float]:
    """
    Train the seq2seq model.
    
    Args:
        model: The Seq2Seq model
        train_loader: DataLoader for training data
        optimizer: Optimizer for gradient descent
        criterion: Loss function (CrossEntropyLoss)
        device: Device to train on (CPU or CUDA)
        epochs: Number of training epochs
        clip_grad: Gradient clipping threshold
    
    Returns:
        List of average losses per epoch
    """
    model.train()
    losses = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt, teacher_forcing_ratio=0.5)
            
            output = output[:, 1:].contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
    
    return losses


def translate_sentence(model: nn.Module, sentence: str, 
                       src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                       device: torch.device) -> str:
    """
    Translate a single English sentence to French.
    
    Args:
        model: Trained Seq2Seq model
        sentence: English sentence to translate
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device (CPU or CUDA)
    
    Returns:
        Translated French sentence
    """
    model.eval()
    src_enc = torch.tensor([src_vocab.encode(sentence, add_eos=True)]).to(device)
    sos_idx = tgt_vocab.token2idx['<SOS>']
    eos_idx = tgt_vocab.token2idx['<EOS>']
    pred = model.translate(src_enc, sos_idx, eos_idx)
    return tgt_vocab.decode(pred)


def main():
    """Main function for training and testing the translator."""
    parser = argparse.ArgumentParser(description='Toy English→French Translator (Seq2Seq with Attention)')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--translate', type=str, help='Translate a sentence')
    parser.add_argument('--demo', action='store_true', help='Run demo with test sentences')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--no-attention', action='store_true', help='Disable attention mechanism')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    print("Loading data...")
    pairs = get_toy_dataset()
    src_sentences = [p[0] for p in pairs]
    tgt_sentences = [p[1] for p in pairs]
    
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    print(f"Source vocabulary: {len(src_vocab)} tokens")
    print(f"Target vocabulary: {len(tgt_vocab)} tokens\n")
    
    train_dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    use_attention = not args.no_attention
    
    model = Seq2Seq(
        input_vocab_size=len(src_vocab),
        output_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        use_attention=use_attention
    ).to(device)
    
    attention_status = "with Attention (Bahdanau)" if use_attention else "without Attention"
    print(f"Model: Seq2Seq {attention_status}\n")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if args.train or args.demo:
        print(f"Training for {args.epochs} epochs...")
        train_model(model, train_loader, optimizer, criterion, device, epochs=args.epochs)
        print()
    
    if args.translate:
        translation = translate_sentence(model, args.translate, src_vocab, tgt_vocab, device)
        print(f"Translation: {args.translate} -> {translation}")
    
    if args.demo:
        print("Demo translations:")
        test_sentences = ["hello", "thank you", "goodbye", "how are you", "i love you"]
        for src in test_sentences:
            translation = translate_sentence(model, src, src_vocab, tgt_vocab, device)
            print(f"  {src} -> {translation}")
    
    if not (args.train or args.translate or args.demo):
        parser.print_help()


if __name__ == "__main__":
    main()
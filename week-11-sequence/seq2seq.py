"""
seq2seq.py - Seq2Seq Model with Encoder-Decoder LSTM
Based on Sutskever, Vinyals & Le (2014)
"""

import numpy as np
import sys
sys.path.append('../week-9-rnn')
from lstm import LSTMLayer


class Seq2Seq:
    """
    Sequence-to-Sequence model with Encoder-Decoder architecture.
    
    Architecture:
    - Encoder: LSTM that reads input sequence and produces context vector
    - Decoder: LSTM that generates output sequence from context vector
    """
    
    def __init__(self, input_vocab_size, output_vocab_size, 
                 embedding_dim=256, hidden_dim=512, num_layers=2):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self._init_weights()
    
    def _init_weights(self):
        scale = 0.1
        
        self.encoder_embeddings = np.random.randn(
            self.input_vocab_size, self.embedding_dim
        ) * scale
        
        self.decoder_embeddings = np.random.randn(
            self.output_vocab_size, self.embedding_dim
        ) * scale
        
        self.encoder_layers = []
        for _ in range(self.num_layers):
            layer = LSTMLayer(self.embedding_dim, self.hidden_dim)
            self.encoder_layers.append(layer)
        
        self.decoder_layers = []
        for _ in range(self.num_layers):
            layer = LSTMLayer(self.embedding_dim, self.hidden_dim)
            self.decoder_layers.append(layer)
        
        self.W_s = np.random.randn(
            self.output_vocab_size, self.hidden_dim
        ) * scale
        self.b_s = np.zeros((self.output_vocab_size, 1))
    
    def _softmax(self, x):
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def _forward_encoder(self, input_seq):
        batch_size = input_seq.shape[2]
        
        x = self.encoder_embeddings[input_seq.flatten()]  # (emb_dim, batch*seq_len)
        x = x.reshape(self.embedding_dim, input_seq.shape[1], batch_size)
        
        h_states = []
        C_states = []
        
        current_input = x
        for layer in self.encoder_layers:
            output = layer.forward_sequence(current_input)
            current_input = np.tanh(output).reshape(self.embedding_dim, input_seq.shape[1], batch_size)
            
            h_states.append(layer.final_h)
            C_states.append(layer.final_C)
        
        return h_states, C_states
    
    def _forward_decoder(self, target_seq, h_states, C_states, teacher_forcing_ratio=0.5):
        batch_size = target_seq.shape[2]
        seq_len = target_seq.shape[1]
        
        self.decoder_outputs = []
        self.predictions = []
        
        current_h = h_states
        current_C = C_states
        
        decoder_input = target_seq[:, 0:1, :]  # Start token (assumed to be index 0)
        
        for t in range(seq_len):
            x = self.decoder_embeddings[decoder_input.flatten()]
            x = x.reshape(self.embedding_dim, 1, batch_size)
            
            for layer_idx, layer in enumerate(self.decoder_layers):
                output = layer.forward_sequence(x, current_h[layer_idx], current_C[layer_idx])
                new_h = np.tanh(layer.W_hy @ layer.final_h + layer.b_y)
                new_C = layer.final_C
                
                current_h[layer_idx] = new_h
                current_C[layer_idx] = new_C
                x = new_h.reshape(self.embedding_dim, 1, batch_size)
            
            logits = self.W_s @ current_h[-1] + self.b_s
            probs = self._softmax(logits)
            
            self.predictions.append(probs)
            self.decoder_outputs.append(logits)
            
            if np.random.random() < teacher_forcing_ratio and t < seq_len - 1:
                decoder_input = target_seq[:, t+1:t+2, :]
            else:
                decoder_input = np.argmax(probs, axis=0).reshape(1, 1, batch_size)
        
        return self.decoder_outputs, self.predictions
    
    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        """
        Forward pass through encoder-decoder.
        
        Args:
            input_seq: (input_vocab_size, input_len, batch)
            target_seq: (output_vocab_size, output_len, batch)
            teacher_forcing_ratio: probability of using teacher forcing
        """
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.batch_size = input_seq.shape[2]
        
        self.encoder_h_states, self.encoder_C_states = self._forward_encoder(input_seq)
        
        decoder_h = [h.copy() for h in self.encoder_h_states]
        decoder_C = [C.copy() for C in self.encoder_C_states]
        
        self.decoder_outputs, self.predictions = self._forward_decoder(
            target_seq, decoder_h, decoder_C, teacher_forcing_ratio
        )
        
        return self.predictions
    
    def compute_loss(self):
        loss = 0.0
        seq_len = len(self.predictions)
        
        for t in range(seq_len):
            target_idx = self.target_seq[t, :, :]  # (1, batch)
            if target_idx.shape[0] == 1:
                target_idx = target_idx.flatten()
            
            pred = np.clip(self.predictions[t], 1e-7, 1 - 1e-7)
            correct_probs = pred[target_idx, np.arange(self.batch_size)]
            loss += -np.log(correct_probs)
        
        return np.mean(loss) / seq_len
    
    def backward(self, learning_rate=0.001):
        seq_len = len(self.decoder_outputs)
        
        for t in reversed(range(seq_len)):
            target_idx = self.target_seq[t, :, :]
            if target_idx.shape[0] == 1:
                target_idx = target_idx.flatten()
            
            d_logits = self.predictions[t].copy()
            d_logits[target_idx, np.arange(self.batch_size)] -= 1
            d_logits /= self.batch_size * seq_len
            
            for layer in reversed(self.decoder_layers):
                layer.backward_sequence(d_logits.copy())
                d_logits = np.dot(layer.W_hy.T, d_logits)
        
        for layer in self.encoder_layers:
            for _ in range(layer.n_hidden):
                pass
        
        for layer in self.decoder_layers:
            layer.update(lr=learning_rate)
    
    def train_step(self, input_seq, target_seq, teacher_forcing_ratio=0.5, learning_rate=0.001):
        self.forward(input_seq, target_seq, teacher_forcing_ratio)
        loss = self.compute_loss()
        self.backward(learning_rate)
        return loss
    
    def predict_greedy(self, input_seq, sos_idx=0, eos_idx=1, max_len=50):
        """Greedy decoding (beam width = 1)"""
        batch_size = input_seq.shape[2]
        
        self.input_seq = input_seq
        self.batch_size = batch_size
        
        self.encoder_h_states, self.encoder_C_states = self._forward_encoder(input_seq)
        
        decoder_h = [h.copy() for h in self.encoder_h_states]
        decoder_C = [C.copy() for C in self.encoder_C_states]
        
        generated = []
        decoder_input = np.array([[sos_idx]]).reshape(1, 1, 1)
        
        for _ in range(max_len):
            x = self.decoder_embeddings[decoder_input.flatten()]
            x = x.reshape(self.embedding_dim, 1, 1)
            
            for layer_idx, layer in enumerate(self.decoder_layers):
                output = layer.forward_sequence(x, decoder_h[layer_idx], decoder_C[layer_idx])
                new_h = np.tanh(layer.W_hy @ layer.final_h + layer.b_y)
                new_C = layer.final_C
                
                decoder_h[layer_idx] = new_h
                decoder_C[layer_idx] = new_C
                x = new_h.reshape(self.embedding_dim, 1, 1)
            
            logits = self.W_s @ decoder_h[-1] + self.b_s
            probs = self._softmax(logits)
            
            next_token = np.argmax(probs, axis=0)[0]
            generated.append(next_token)
            
            if next_token == eos_idx:
                break
            
            decoder_input = np.array([[next_token]]).reshape(1, 1, 1)
        
        return generated
    
    def predict_beam(self, input_seq, sos_idx=0, eos_idx=1, max_len=50, beam_width=3):
        """Beam search decoding"""
        batch_size = input_seq.shape[2]
        
        self.input_seq = input_seq
        self.batch_size = batch_size
        
        self.encoder_h_states, self.encoder_C_states = self._forward_encoder(input_seq)
        
        self.input_seq = np.zeros((1, 1, 1))
        
        beams = [(0.0, [sos_idx], 
                  [h.copy() for h in self.encoder_h_states],
                  [C.copy() for C in self.encoder_C_states])]
        
        for _ in range(max_len):
            all_candidates = []
            
            for score, seq, h_states, C_states in beams:
                if seq[-1] == eos_idx:
                    all_candidates.append((score, seq, h_states, C_states))
                    continue
                
                decoder_input = np.array([[seq[-1]]]).reshape(1, 1, 1)
                x = self.decoder_embeddings[decoder_input.flatten()]
                x = x.reshape(self.embedding_dim, 1, 1)
                
                for layer_idx, layer in enumerate(self.decoder_layers):
                    output = layer.forward_sequence(x, h_states[layer_idx], C_states[layer_idx])
                    new_h = np.tanh(layer.W_hy @ layer.final_h + layer.b_y)
                    new_C = layer.final_C
                    
                    h_states[layer_idx] = new_h
                    C_states[layer_idx] = new_C
                    x = new_h.reshape(self.embedding_dim, 1, 1)
                
                logits = self.W_s @ h_states[-1] + self.b_s
                probs = self._softmax(logits).flatten()
                
                top_k = np.argsort(probs)[-beam_width:]
                for token in top_k:
                    new_score = score + np.log(probs[token] + 1e-10)
                    new_seq = seq + [token]
                    all_candidates.append((new_score, new_seq, h_states, C_states))
            
            beams = sorted(all_candidates, key=lambda x: x[0] / len(x[1]))[-beam_width:]
        
        return beams[0][1]


class Seq2SeqAttention:
    """
    Seq2Seq with Attention mechanism.
    Based on Bahdanau, Cho & Bengio (2014)
    """
    
    def __init__(self, input_vocab_size, output_vocab_size,
                 embedding_dim=256, hidden_dim=512, num_layers=1):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self._init_weights()
    
    def _init_weights(self):
        scale = 0.1
        
        self.encoder_embeddings = np.random.randn(
            self.input_vocab_size, self.embedding_dim
        ) * scale
        
        self.decoder_embeddings = np.random.randn(
            self.output_vocab_size, self.embedding_dim
        ) * scale
        
        self.W_a = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.U_a = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.v_a = np.random.randn(self.hidden_dim, 1) * scale
        
        self.encoder = LSTMLayer(self.embedding_dim, self.hidden_dim)
        self.decoder = LSTMLayer(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        
        self.W_s = np.random.randn(self.output_vocab_size, self.hidden_dim) * scale
        self.b_s = np.zeros((self.output_vocab_size, 1))
    
    def _attention(self, s_t, h_enc, mask=None):
        scores = []
        for h_j in h_enc:
            score = np.dot(self.v_a.T, np.tanh(np.dot(self.W_a, s_t) + np.dot(self.U_a, h_j)))
            scores.append(score.flatten()[0])
        
        scores = np.array(scores)
        if mask is not None:
            scores = scores * mask - 1e9 * (1 - mask)
        
        attn_weights = np.exp(scores - np.max(scores))
        attn_weights = attn_weights / np.sum(attn_weights)
        
        context = np.sum(attn_weights.reshape(-1, 1) * h_enc, axis=0)
        return context, attn_weights
    
    def _softmax(self, x):
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.shape[2]
        input_len = input_seq.shape[1]
        
        self.input_seq = input_seq
        self.target_seq = target_seq
        
        x = self.encoder_embeddings[input_seq.flatten()]
        x = x.reshape(self.embedding_dim, input_len, batch_size)
        
        self.encoder.forward_sequence(x)
        self.encoder_hiddens = self.encoder.hs
        
        self.decoder_hiddens = []
        self.context_vectors = []
        self.attn_weights = []
        predictions = []
        
        decoder_h = self.encoder.final_h
        decoder_C = self.encoder.final_C
        
        for t in range(target_seq.shape[1]):
            context, attn = self._attention(decoder_h, self.encoder_hiddens)
            self.context_vectors.append(context)
            self.attn_weights.append(attn)
            
            y_t = self.decoder_embeddings[target_seq[t, :, :].flatten()]
            
            decoder_input = np.vstack([y_t.reshape(-1, 1), context.reshape(-1, 1)])
            decoder_input = decoder_input.reshape(-1, 1, batch_size)
            
            logits = self.decoder.forward_sequence(decoder_input, decoder_h, decoder_C)
            decoder_h = self.decoder.final_h
            decoder_C = self.decoder.final_C
            
            self.decoder_hiddens.append(decoder_h)
            
            out_logits = self.W_s @ decoder_h + self.b_s
            probs = self._softmax(out_logits)
            predictions.append(probs)
            
            if np.random.random() < teacher_forcing_ratio and t < target_seq.shape[1] - 1:
                pass
        
        self.predictions = predictions
        return predictions
    
    def compute_loss(self):
        loss = 0.0
        batch_size = self.target_seq.shape[2]
        
        for t, pred in enumerate(self.predictions):
            target_idx = self.target_seq[t, :, :]
            if target_idx.shape[0] == 1:
                target_idx = target_idx.flatten()
            
            pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)
            correct_probs = pred_clipped[target_idx, np.arange(batch_size)]
            loss += -np.log(correct_probs)
        
        return loss / len(self.predictions)

# BERT for Sentiment Analysis: Mathematical Foundations

This document provides a detailed mathematical explanation of how transformer-based models like BERT and RoBERTa approach sentiment analysis. We'll focus particularly on RoBERTa (Robustly Optimized BERT Approach), which builds upon BERT with improved training methodology.

## 1. Tokenization & Word Embeddings

### Subword Tokenization

Unlike whole-word tokenization, BERT uses WordPiece tokenization, while RoBERTa uses Byte-Pair Encoding (BPE) to break words into subword units.

Example of BPE tokenization:
```
"unhappiness" → ["un", "happiness"]
"playing" → ["play", "ing"]
```

### Special Tokens

For sentiment analysis, the input is structured with special tokens:
```
[CLS] This movie was fantastic! [SEP]
```

Where:
- `[CLS]` (Classification token): Used to capture sentence-level information for classification tasks
- `[SEP]` (Separator token): Marks the end of the sequence

### Embedding Layer

Each token is converted to an embedding vector of dimension $d_{model}$ (typically 768 for BERT-base and RoBERTa-base).

Three types of embeddings are combined:

1. **Token Embeddings**: Learned representations of each token in the vocabulary
2. **Position Embeddings**: Encode position information
3. **Segment Embeddings**: Used for distinguishing between different sentences (not typically needed for single sentence sentiment analysis)

The final input embedding is:
$$\text{Input Embedding} = \text{Token Embedding} + \text{Position Embedding} + \text{Segment Embedding}$$

### Positional Encoding

Since transformer models process all tokens in parallel (rather than sequentially), positional information must be explicitly added.

For position $pos$ and dimension $i$:

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

These mathematical formulas ensure:
- Each position has a unique encoding
- The model can extrapolate to sequence lengths not seen during training
- The relative positions have a consistent relationship in the embedding space

## 2. Self-Attention Mechanism

The key innovation in transformer models is the self-attention mechanism, which allows each token to attend to all other tokens in the sequence.

### Queries, Keys, and Values

For an input matrix $X$ (containing the embeddings for all tokens), we compute:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Where:
- $W_Q$, $W_K$, $W_V$ are learnable parameter matrices
- $Q$ (Query): Represents what the token is "looking for"
- $K$ (Key): Represents what each token "contains"
- $V$ (Value): Represents what each token "contributes"

### Attention Score Computation

The scaled dot-product attention is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $QK^T$ is the dot product between queries and keys (forming an attention matrix)
- $\sqrt{d_k}$ is a scaling factor to prevent extremely small gradients
- The softmax operation normalizes the attention weights to sum to 1
- Multiplying by $V$ produces the weighted content from each token

### Mathematical Breakdown:

1. **Compute attention scores**: $S = QK^T$ (produces a matrix where $S_{ij}$ is the score between token $i$ and token $j$)
2. **Scale scores**: $S_{scaled} = \frac{S}{\sqrt{d_k}}$
3. **Apply softmax**: $A = \text{softmax}(S_{scaled})$ (each row sums to 1, representing attention weights)
4. **Get weighted values**: $\text{Output} = AV$

### Multi-Head Attention

Rather than performing a single attention function, transformer models use multiple attention heads in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$$

Where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

This allows the model to:
- Attend to different parts of the sequence simultaneously
- Capture different types of relationships (e.g., syntactic vs. semantic patterns)
- Create a more expressive representation space

## 3. Transformer Layers

### Multi-head Self-Attention

As described above, multi-head attention allows parallel attention computation with different learned projections.

### Layer Normalization

After the attention mechanism, layer normalization is applied:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu$ and $\sigma^2$ are the mean and variance computed across the features
- $\gamma$ and $\beta$ are learnable parameters
- $\epsilon$ is a small constant for numerical stability

### Feed-Forward Network

Each transformer layer contains a feed-forward network after the attention mechanism:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

This is equivalent to two linear transformations with a ReLU activation in between.

### Residual Connections

Residual connections (skip connections) allow gradient flow during backpropagation:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Where Sublayer could be either the multi-head attention or the feed-forward network.

## 4. Final Classification

### [CLS] Token Representation

For sentiment analysis, the final hidden state of the [CLS] token is used as the sentence representation.

Let's call this vector $h_{[CLS]} \in \mathbb{R}^{d_{model}}$. 

### Classification Layer

A simple linear classifier with softmax is applied:

$$P(\text{class}) = \text{softmax}(W \cdot h_{[CLS]} + b)$$

Where:
- $W \in \mathbb{R}^{c \times d_{model}}$ is a weight matrix ($c$ is the number of classes)
- $b \in \mathbb{R}^c$ is a bias vector
- $P(\text{class})$ gives the probability distribution over the sentiment classes

For sentiment analysis, typically:
- $c = 2$ for binary classification (positive/negative)
- $c = 3$ for ternary classification (positive/neutral/negative)
- $c = 5$ for fine-grained sentiment (very negative to very positive)

## 5. Hand-Calculated Example

Let's walk through a simplified example of how BERT/RoBERTa processes a text for sentiment analysis:

Input text: "The movie was incredibly boring, but the acting saved it."

### Step 1: Tokenization
After tokenization (simplified):
```
[CLS], "The", "movie", "was", "incredibly", "boring", ",", "but", "the", "acting", "saved", "it", ".", [SEP]
```

Converting to token IDs (hypothetical):
```
[101, 1996, 3185, 2001, 25087, 8756, 1010, 2021, 1996, 4743, 2605, 2009, 1012, 102]
```

### Step 2: Embedding Layer
Each token ID is mapped to an embedding vector. For simplicity, let's represent with small 4-dimensional vectors (actual models use 768 dimensions):

```
[CLS]  → [0.2, -0.1, 0.3, 0.1]
"The"  → [0.5, 0.2, -0.3, 0.1]
...etc
```

Position embeddings are added to encode position information.

### Step 3: Self-Attention (First Layer)

For the token "boring" at position 5:

1. **Query vector** for "boring": $q_5 = [0.3, -0.2, 0.1, 0.4]$
2. **Computing attention scores** with all tokens (dot products):
   - Score with "incredibly": $q_5 \cdot k_4 = 0.25$ (high)
   - Score with "but": $q_5 \cdot k_7 = 0.15$ (moderate)

3. **After softmax**, attention weights might look like:
   - Attention to "incredibly": 0.3 (strong attention)
   - Attention to "but": 0.2
   - Other tokens: varying smaller weights

4. **Weighted sum of values** creates the output for "boring" in this attention head.

The same process happens across all tokens and multiple attention heads.

### Step 4: Feed-Forward and Multiple Layers

The outputs from multi-head attention pass through feed-forward networks and multiple transformer layers.

### Step 5: Final Classification

After all transformer layers, the [CLS] token's final representation captures the overall sentiment:

Final [CLS] vector (hypothetical): $h_{[CLS]} = [0.5, -0.3, 0.7, -0.2]$

Applied to classification layer:
$W \cdot h_{[CLS]} + b = [−1.2, 0.3, 4.1]$ (logits for negative, neutral, positive)

After softmax:
$P(\text{class}) = [0.02, 0.08, 0.90]$

Result: **Positive** with 90% confidence.

This demonstrates the model's ability to understand contextual nuances like the contrast introduced by "but", which shifts the overall sentiment despite the presence of negative words.

## 6. Pre-training and Fine-tuning

### Pre-training Objectives

RoBERTa is pre-trained using Masked Language Modeling (MLM):
1. Randomly mask 15% of the tokens
2. Train the model to predict the original tokens

Example:
```
Original: "The movie was great"
Masked: "The movie was [MASK]"
Task: Predict "great" at the masked position
```

RoBERTa uses dynamic masking where different tokens are masked in each training epoch.

### Fine-tuning

After pre-training, the model is fine-tuned on labeled sentiment data:
1. Keep the pre-trained transformer layers
2. Add a classification head on top of the [CLS] token
3. Train end-to-end with a smaller learning rate

The loss function for sentiment classification is cross-entropy:

$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

Where:
- $C$ is the number of classes
- $y_c$ is the true label (one-hot encoded)
- $\hat{y}_c$ is the predicted probability for class $c$

## 7. Comparison with Lexicon-Based Models (VADER)

| Feature | BERT/RoBERTa (Deep Learning) | VADER (Lexicon-Based) |
|---------|------------------------------|------------------------|
| Context Understanding | Deep contextual modeling via self-attention | Limited to rule-based context handling |
| Mathematical Basis | Neural networks with non-linear transformations | Weighted sum with heuristic adjustments |
| Complexity | ~110-340M parameters | ~7,500 lexicon entries + rules |
| Training | Pre-training + fine-tuning required | No training needed |
| Adaptability | Can learn from data | Fixed rules and lexicon |
| Computational Cost | High (GPUs typically required) | Very low |
| Memory Footprint | 400MB-1.5GB model size | <1MB lexicon |
| Performance on Subtle Sentiment | Strong | Limited |

## 8. Mathematical Intuition

The power of transformer models like BERT and RoBERTa comes from:

1. **Parallel processing**: All tokens are processed simultaneously rather than sequentially
2. **Contextual awareness**: Every token attends to every other token
3. **Transfer learning**: Leveraging knowledge from massive pre-training
4. **Multiple attention perspectives**: Multi-head attention captures different relationship types
5. **Self-attention mathematics**: Creating dynamic, content-dependent representations

The self-attention mechanism effectively builds a fully-connected graph where edges (attention weights) represent the relevance between tokens. This graph is dynamically computed based on the content, allowing the model to focus on the most relevant tokens for sentiment analysis.
# NLP Sentiment Analysis: Model Mechanics & Mathematics

A comprehensive exploration of sentiment analysis model architecture, mathematical foundations, and internal processes. This repository serves as an educational resource for understanding how popular sentiment analysis models work under the hood.

## 🔍 Featured Models

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type**: Lexicon and rule-based model
- **Specialization**: Social media text and short-form content
- **Key features**: Handles slang, emojis, punctuation emphasis, and contextual modifiers

### RoBERTa (Robustly Optimized BERT Approach)
- **Type**: Transformer-based deep learning model
- **Specialization**: Complex contextual understanding and long-form text
- **Key features**: Self-attention mechanisms, transfer learning, contextual embeddings

## 📊 Model Comparison

| Aspect | VADER | RoBERTa |
|--------|-------|---------|
| Approach | Rule-based, lexicon-driven | Transformer-based, data-driven |
| Context Handling | Heuristic rules (e.g., "but") | Self-attention captures long-range context |
| Training | No training required | Pre-training + task-specific fine-tuning |
| Slang/Emojis | Explicitly coded in lexicon | Learns from data (if present in training) |
| Interpretability | High (transparent rules) | Low (black-box model) |
| Computational Cost | Very low | High (requires GPU for efficient inference) |
| Performance on Long Texts | Limited | Excellent (up to max sequence length) |

## 📚 Resources

### VADER
- [VADER GitHub Repository](https://github.com/cjhutto/vaderSentiment)
- [VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/viewFile/8109/8122)

### RoBERTa
- [RoBERTa GitHub Repository](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

## 🧩 Repository Structure

- `index.html` - Home page with overview and model comparisons
- `vader.html` - Detailed analysis of VADER's architecture and mathematics
- `roberta.html` - Comprehensive exploration of RoBERTa's approach to sentiment analysis
- `css/` - Styling for the pages

## 🚀 Getting Started

This is a GitHub Pages site. To view it locally:

1. Clone this repository
2. Open `index.html` in your browser

To deploy to GitHub Pages:

1. Push to your GitHub repository
2. Enable GitHub Pages in repository settings

## 📝 License

MIT License

## 👥 Contributing

Contributions to expand the model coverage or enhance explanations are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.

## 📞 Contact

For questions or suggestions, please open an issue in this repository.
# NLP Sentiment Analysis: Exploring the Mathematics

This project started when a friend asked me to explain how sentiment analysis models work mathematically. Looking online, I couldn't find resources that clearly explained the mathematical foundations behind popular models like VADER and BERT/RoBERTa in an accessible way.

Most tutorials I found focused on implementation or API usage, while academic papers often contained dense mathematics without clear connections to the actual code. There seemed to be a gap between practical guides and theoretical explanations.

This repository is my attempt to bridge that gap by:

- Explaining the mathematical principles behind lexicon-based models like VADER
- Breaking down the core mathematics of transformer models like BERT/RoBERTa
- Providing notebooks that connect theory to implementation
- Including visualizations to help understand key concepts

This is very much a work in progress, and I'm learning along the way. I hope others who are curious about the mathematical underpinnings of these models might find this helpful in their own learning journey.

## Repository Structure

- **Web Interface** (GitHub Pages)
  - `index.html` - Main landing page
  - `vader-overview.html` - Practical VADER overview
  - `roberta-overview.html` - Practical RoBERTa overview
  - `vader.html` - VADER model mathematical details
  - `roberta.html` - RoBERTa model mathematical details
  - `css/styles.css` - Styling

- **Documentation**
  - `docs/vader_math.md` - Mathematical explanation of VADER
  - `docs/bert_math.md` - Mathematical explanation of BERT/RoBERTa

- **Notebooks**
  - `notebooks/bert_sentiment_analysis.ipynb` - Implementation and analysis of BERT/RoBERTa
  - `notebooks/vader_sentiment_analysis.ipynb` - Implementation and analysis of VADER

## 🔍 Featured Models

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type**: Lexicon and rule-based model
- **Specialization**: Social media text and short-form content
- **Key features**: Handles slang, emojis, punctuation emphasis, and contextual modifiers
- **Mathematical foundation**: Weighted lexicon with rule-based adjustments and sigmoid-like normalization

### RoBERTa (Robustly Optimized BERT Approach)
- **Type**: Transformer-based deep learning model
- **Specialization**: Complex contextual understanding and long-form text
- **Key features**: Self-attention mechanisms, transfer learning, contextual embeddings
- **Mathematical foundation**: Multi-head self-attention with softmax normalization

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

## Getting Started

### Viewing the Web Interface

The easiest way to explore this project is through our GitHub Pages site:

```
https://[your-username].github.io/nlp-sentiment-analysis/
```

### Running the Notebooks Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/nlp-sentiment-analysis.git
   cd nlp-sentiment-analysis
   ```

2. Set up a Python environment (Python 3.8+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Navigate to the `notebooks/` directory and open either notebook.

## 📚 Resources

### VADER
- [VADER GitHub Repository](https://github.com/cjhutto/vaderSentiment)
- [VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/viewFile/8109/8122)

### RoBERTa
- [RoBERTa GitHub Repository](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

## Future Work

- Adding interactive visualizations (GIFs/animations)
- Implementing a live demo with user input
- Expanding model coverage to include additional sentiment analysis approaches
- Adding more mathematical details and derivations

## 📝 License

MIT License

## 👥 Contributing

Contributions to expand the model coverage or enhance explanations are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.
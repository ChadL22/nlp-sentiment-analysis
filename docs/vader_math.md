# VADER Sentiment Analysis: Mathematical Foundations

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically designed for social media texts. This document explores the mathematical foundations and processes that power VADER's sentiment analysis approach.

## 1. Lexicon Construction

The foundation of VADER is its sentiment lexicon, which is a curated list of lexical features (words, emojis, slang) each assigned a valence score between **-4 (extremely negative)** and **+4 (extremely positive)**.

### Example Lexicon Entries:
- "love": +3.0
- "terrible": -3.0
- "😂": +3.4
- "meh": -1.8

The lexicon includes **contextual intensifiers** (e.g., "extremely"), **negations** (e.g., "not"), and **modifiers** (e.g., "kind of").

## 2. Text Preprocessing

Before applying sentiment scoring, VADER performs several preprocessing steps:

### Tokenization
VADER splits the input text into tokens including words, punctuation, and emojis:
```
Input: "I LOVE this movie!! 😊"
Tokens: ["I", "LOVE", "this", "movie", "!", "!", "😊"]
```

### Case Sensitivity
Unlike many NLP tools, VADER preserves case information. Words in ALL CAPS (e.g., "AMAZING") are interpreted as emphasized, increasing their intensity by a factor (typically +0.5).

## 3. Heuristic Rules for Adjusting Sentiment Scores

VADER applies sophisticated rules to modify individual word scores based on grammatical and syntactic cues:

### A. Modifiers (Boosters/Dampeners)

**Boosters** (e.g., "very", "extremely") increase the intensity of the next word's sentiment:
```
"very good" → "good" (+3.0) * booster weight (e.g., +0.5) = +3.5
```

**Dampeners** (e.g., "slightly", "barely") reduce intensity:
```
"slightly good" → "good" (+3.0) * dampener weight (e.g., -0.3) = +2.7
```

Weighted adjustments are empirically derived.

### B. Negations

Words like "not", "never", or "no" invert the polarity of the following sentiment word:
```
"not good" → "good" (+3.0) * negation factor (-1) = -3.0
```

VADER checks for negations within **3 words** of a sentiment term.

### C. Punctuation

**Exclamation Marks (!)**: Amplify sentiment (e.g., +0.5 to compound score)
**Question Marks (?)**: May dampen sentiment (context-dependent)

### D. Capitalization

Words in ALL CAPS increase intensity by approximately +0.5:
```
"GOOD" → "good" (+3.0) + capitalization boost (+0.5) = +3.5
```

### E. Shifters (Conjunctions)

Words like "but" shift focus to the clause following them, emphasizing its sentiment:
```
"The food was good, but the service was terrible."
```
In this case, the negative score of "terrible" is weighted more heavily than the positive score of "good".

## 4. Mathematical Aggregation and Normalization

### Sum Adjusted Scores
First, all the adjusted valence scores from the text are summed.

### Normalization Formula
The raw sum is normalized to a **compound score** between **-1 (most negative)** and **+1 (most positive)** using:

$$\text{Compound Score} = \frac{\text{Sum}}{\sqrt{(\text{Sum})^2 + \alpha}}$$

where $\alpha=15$ (a normalization constant determined empirically).

This is a form of sigmoid-like function that ensures:
- Scores approach but never reach the absolute values of -1 or +1
- There's a smooth transition across the sentiment spectrum
- The function is symmetric around zero

### Classification Thresholds
The compound score is categorized using thresholds:
- **Positive**: compound ≥ 0.05
- **Neutral**: -0.05 < compound < 0.05
- **Negative**: compound ≤ -0.05

### Output Metrics
VADER returns:
1. **Compound Score**: Normalized summary metric (-1 to +1)
2. **Proportions**: % of text classified as positive, neutral, or negative

## 5. Hand-Calculated Example

Let's walk through a complete calculation for the text: "The movie was **extremely good**!! 😊"

### Step 1: Tokenization
```
["The", "movie", "was", "extremely", "good", "!", "!", "😊"]
```

### Step 2: Identify Sentiment-Bearing Words
- "extremely": Booster word
- "good": Positive sentiment word (+3.0)
- "!!": Punctuation
- "😊": Positive emoji (+3.0)

### Step 3: Apply Rules
1. "extremely" (booster) applied to "good" (+3.0) → adjusted to +3.5
2. "!!" (punctuation boost) → +0.3
3. "😊" (emoji from lexicon) → +3.0

### Step 4: Sum
```
3.5 + 0.3 + 3.0 = 6.8
```

### Step 5: Normalize
$$\text{Compound Score} = \frac{6.8}{\sqrt{6.8^2 + 15}} \approx \frac{6.8}{\sqrt{46.24 + 15}} \approx \frac{6.8}{\sqrt{61.24}} \approx \frac{6.8}{7.83} \approx 0.87$$

### Step 6: Classify
0.87 > 0.05, therefore this text is classified as **strongly positive**.

### Step 7: Full Output
```
{
    'neg': 0.0,
    'neu': 0.37,
    'pos': 0.63,
    'compound': 0.87
}
```

## 6. Comparison with Deep Learning Models

### VADER Advantages:
- **No Training Required**: Works out-of-the-box without labeled data
- **Computationally Efficient**: Minimal processing power needed
- **Interpretable**: Transparent rules make decisions easy to understand
- **Social Media Optimized**: Handles informal language, emojis, and slang

### VADER Limitations vs. Deep Learning:
- **Limited Contextual Understanding**: Cannot fully grasp complex or nuanced contexts
- **Domain Adaptation**: Requires manual lexicon updates for new domains or slang
- **Fixed Rules**: Cannot learn from examples or improve with more data
- **Long Text Limitations**: Less effective on long documents with complex structure

## 7. Mathematical Intuition

VADER's scoring mechanism can be thought of as a weighted, rule-enhanced bag-of-words model with a non-linear normalization function. While simpler than deep learning approaches, its mathematical design effectively captures sentiment in short texts with careful attention to linguistic nuances like intensifiers, negations, and punctuation.

The compound score normalization function ensures that:
1. Values are bounded between -1 and +1
2. The function scales proportionally for moderate values
3. Very high or low values asymptotically approach the bounds

This gives VADER a robust way to represent sentiment intensity across various text lengths and styles, particularly in social media contexts.
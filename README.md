# DSLR - Data Science and Logistic Regression

A Harry Potter-themed machine learning project that implements logistic regression from scratch to classify Hogwarts students into their respective houses based on their academic performance in various magical subjects.

## 📖 Overview

This project recreates the Hogwarts Sorting Hat using data science! Students are classified into one of four houses (Gryffindor, Hufflepuff, Ravenclaw, or Slytherin) based on their scores in magical subjects like Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, and more.

The project implements logistic regression using only NumPy and Pandas, without relying on high-level machine learning libraries like scikit-learn.

## 🎯 Features

- **Data Analysis & Visualization**
  - Statistical analysis similar to pandas `describe()`
  - Histogram analysis for feature exploration
  - Pair plots for feature correlation analysis
  - Scatter plots for identifying relationships

- **Machine Learning**
  - Logistic regression implementation from scratch
  - One-vs-all multiclass classification for 4 houses
  - Feature standardization (z-score normalization)
  - Batch and mini-batch gradient descent
  - Model training with convergence detection

- **Prediction & Evaluation**
  - Student house prediction
  - Probability estimates for each house
  - Model persistence (save/load weights)

## 📁 Project Structure

```
dslr/
├── datasets/
│   ├── dataset_train.csv      # Training dataset
│   ├── dataset_test.csv       # Test dataset (no labels)
│   └── dataset_train_small.csv # Smaller training set
├── describe.py                # Statistical analysis tool
├── histogram.py               # Histogram visualization
├── pair_plot.py               # Pair plot visualization
├── scatter_plot.py            # Scatter plot analysis
├── logreg_train.py            # Logistic regression training
├── logreg_predict.py          # Prediction script
├── utils.py                   # Utility functions
├── weights.json               # Trained model weights
├── houses.csv                 # Prediction results
└── test.py                    # Testing utilities
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/itsfwenk/Data-Science-and-Logistic-Regression.git dslr
cd dslr
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn
```

## 📊 Usage

### 1. Data Analysis

Explore the dataset with statistical analysis:
```bash
python describe.py datasets/dataset_train.csv
```

Generate visualizations to understand the data:
```bash
# Histogram analysis
python histogram.py datasets/dataset_train.csv

# Pair plot visualization
python pair_plot.py datasets/dataset_train.csv

# Scatter plot analysis
python scatter_plot.py datasets/dataset_train.csv
```

### 2. Model Training

Train the logistic regression model:
```bash
python logreg_train.py datasets/dataset_train.csv
```

This will:
- Load and preprocess the training data
- Train one-vs-all logistic regression classifiers for each house
- Save the trained weights to `weights.json`
- Display training progress and convergence information

### 3. Making Predictions

Predict house assignments for new students:
```bash
python logreg_predict.py datasets/dataset_test.csv weights.json
```

This will:
- Load the test dataset and trained model weights
- Generate predictions for each student
- Save results to `houses.csv`
- Display prediction probabilities

## 🔬 Technical Implementation

### Logistic Regression Details

- **Algorithm**: One-vs-All (One-vs-Rest) multiclass classification
- **Optimization**: Batch Gradient Descent with optional Mini-batch support
- **Activation**: Sigmoid function with numerical stability (clipping)
- **Features**: 13 magical subjects used as input features
- **Preprocessing**: Z-score normalization for feature scaling
- **Convergence**: Early stopping based on cost function tolerance

### Key Components

1. **ColumnStats Class**: Implements statistical functions (mean, std, min, max, percentiles) from scratch
2. **LogisticRegression Class**: Complete implementation with gradient descent optimization
3. **LogRegPredictor Class**: Handles model inference and probability computation
4. **Feature Engineering**: Automatic handling of missing values and normalization

### Model Architecture

The model uses a one-vs-all approach, training 4 binary classifiers:
- Gryffindor vs Others
- Hufflepuff vs Others
- Ravenclaw vs Others
- Slytherin vs Others

For prediction, the classifier with the highest probability determines the final house assignment.

## 📈 Results

The model achieves effective classification of students into Hogwarts houses based on their academic performance patterns. Each house shows distinct characteristics:

- **Gryffindor**: Often strong in Defense Against the Dark Arts
- **Ravenclaw**: Generally excel in theoretical subjects
- **Hufflepuff**: Balanced performance across subjects
- **Slytherin**: Strategic strengths in specific areas

## 🎨 Visualizations

The project includes several visualization tools:

- **Histograms**: Show grade distributions by house for each subject
- **Pair Plots**: Reveal correlations between different magical subjects
- **Scatter Plots**: Identify the most distinguishing features between houses

## 📝 Files Description

| File | Purpose |
|------|---------|
| `describe.py` | Statistical analysis equivalent to pandas describe() |
| `histogram.py` | Generate histograms to find the most distinguishing feature |
| `pair_plot.py` | Create pair plots to visualize feature relationships |
| `scatter_plot.py` | Generate scatter plots for correlation analysis |
| `logreg_train.py` | Train the logistic regression model |
| `logreg_predict.py` | Make predictions using the trained model |
| `utils.py` | Common utility functions for data loading |
| `weights.json` | Serialized model parameters and metadata |

## 🏆 Educational Value

This project demonstrates:
- Implementation of machine learning algorithms from scratch
- Understanding of logistic regression mathematics
- Data preprocessing and feature engineering
- Gradient descent optimization
- Multiclass classification strategies
- Model evaluation and validation
- Data visualization for insights

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Optimize algorithms

## 📄 License

This project is part of the 42 School curriculum - DSLR (Data Science and Logistic Regression) project.

---

*"It is our choices, Harry, that show what we truly are, far more than our abilities."* - Albus Dumbledore

Transform academic data into house predictions with the magic of machine learning! 🧙‍♂️✨

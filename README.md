# Summative Lab: NLP, Time Series & Neural Networks

Analysis of 997 Amazon customer reviews of e-readers and tablets (mainly Barnes & Noble Nook devices), written for the *Analytics Firm Clients* summative lab.

## Files

| File | Description |
|---|---|
| `C09_M08.ipynb` | Full notebook with code, outputs and written interpretations |
| `amazon_reviews_lab.csv` | Dataset (997 reviews, 9 columns, 2000-2014) |

## What the notebook does

### Part 1: Natural Language Processing
- Cleaning pipeline: lowercase, URL/HTML removal, punctuation removal, tokenization, stopword removal (negations such as "not" kept), lemmatization.
- Word, bigram and trigram frequencies, word cloud, PMI collocations, and word usage by rating.
- **Finding:** reviews focus on devices, battery life, screen and apps. Happy reviews use words like *great, easy, love, price*; unhappy reviews mention *customer, service, return, credit*.

### Part 2: Time Series
- Monthly review count and average rating (Jan 2010 - Jun 2014), 3-month rolling average, decomposition.
- Stationarity tests (ADF and KPSS), log transform and differencing (d = 1).
- ARIMA grid search by AIC/BIC, SARIMA comparison, naive and mean baselines, residual checks, 6-month forecasts.
- **Finding:** review volume grew strongly (log + differencing needed). Average rating is stable at about 4.2-4.3 stars and mostly noise, so a naive forecast is as good as ARIMA.

### Part 3: Neural Networks
- TF-IDF features (5,000 unigrams and bigrams) predicting the star rating.
- Keras MLP with six configurations (depth, dropout, optimizer, class weights), a regression MLP, and the same MLP in PyTorch.
- Evaluation with accuracy, macro-F1, within-one-star accuracy, classification report and confusion matrix.
- **Finding:** all networks overfit the ~680 training reviews. Logistic regression (macro-F1 0.38) was as good as or better than the MLPs (about 0.30). Merging ratings into three sentiment classes was the most effective improvement (accuracy 0.80, macro-F1 0.575).

## How to run

1. Install the packages:
   ```
   pip install pandas numpy matplotlib seaborn nltk statsmodels scikit-learn tensorflow torch wordcloud
   ```
2. Put `amazon_reviews_lab.csv` in the same folder as the notebook.
3. Open `C09_M08.ipynb` in Jupyter and choose **Kernel > Restart & Run All**.

PyTorch and wordcloud are optional. The rest of the notebook runs without them.

## Tools

Python, pandas, NLTK, statsmodels, scikit-learn, TensorFlow/Keras, PyTorch.

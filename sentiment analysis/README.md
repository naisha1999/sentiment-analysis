# sentiment-analysis-using-python
--- Large Data Analysis Course Project ---

This folder is a set of simplified python codes which use sklearn package 
to classify movie reviews.

Two classifiers were used: Naive Bayes and SVM.
Accuracy is 86 %

## usage
`imdbReviews.py` generates `*.pkl` files which are the training and testing datasets.
First, set the dataset directory in the `imdbReviews.py`, then run the code.
```bash
python Reviews.py
```

You will get two `*.pkl` files which are needed for `naive.py`
To do prediction, run the following command.
```bash
python naive.py
```


### End
last modified 04/16/2016

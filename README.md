# movie-sentiment-analysis #
Classifies movie reviews as either negative or postive using a from-scratch Naive Bayes model. It implements bag-of-word features and add-one smoothing, all in python.

## Dataset ##
The movie review dataset is formatted such that each mvoie review is in it's own separate folder, and movie reviews that are positive and negative are spilt into their respective directories. 

```bash
├── reviews
│   ├── test
│   │   ├── neg
│   │   ├── pos
│   ├── train
│   │   ├── neg
│   │   ├── pos
└── └── imdb.vocab
```
The imdb.vocab file is a set of words that we use to correspond with our bag-of-words features.

## Text Preprocessing ##

The dataset is then processed by undergoing noise removal in reference to punctuation/other special characters, setting each remaining character in the dataset to lowercase, and finally tokenization of each sentence in the dataset to words. 

All the preprocessing is done within `pre-process.py` (takes one parameter: folders and files of movie reviews as input) and in addition to the noise removal, lowercasing, and tokenization - it also counts the frequencies of each word and then returns the files as a set of vectors in JSON.

Example output movie review:
```json
{"pos": {"dominion": 1, "is": 4, "a": 1, "good": 1, "not": 1, "blends": 1, "some": 1, "elements": 1, "of": 1, "slasher": 1, "movie": 1, "and": 3, "adventure": 1, "setting": 1, "acting": 1, "acceptable": 1, "the": 1, "film": 1, "fast-paced": 1, "recommended": 1, "for": 1, "any": 1, "buff": 1}}
```

## Analysis ##

Implementation of the classifier is done within `naive-bayes.py` - which takes the processed training data, processed test data, name of file for param output from the model, and the name of file for regular output (with accuracy) as params. 

The classifier is trained on the train data with bag-of-word features and add-one smoothing and then tested it on the test data - the output in this repository is in out.txt with the accuracy of the the evaluation at the last line.

### Refresher on Naive Bayes Classification

`P(c|x) = P(x|c)P(c) / P(x)`

- P(c|x) is the posterior probability of class (c, target) given predictor (x, attributes).
- P(c) is the prior probability of class.
- P(x|c) is the likelihood which is the probability of predictor given class.
- P(x) is the prior probability of predictor.





Made for Machine Learning for Natural Language Processing.

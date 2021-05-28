import sys
import json
import math

def bow_to_list(bow):
    output = []
    for word, freq in bow.items():
        for i in range(freq):
            output.append(word)
    return output

def getinput():
    training = sys.argv[1] # Training file
    test = sys.argv[2] # Test file
    param_file = sys.argv[3] # Param file to be saved
    outputfile = sys.argv[4] # Output file

    vocab = set([line.rstrip() for line in open('reviews/imdb.vocab')]) # All of our words
    documents = []
    classes = {}
    test_docs = {}

    training_file = open(training, "r")
    for line in training_file.readlines():
        vector = json.loads(line)
        documents.append(vector)
        label = list(vector.keys())[0]
        if label in classes:
            classes[label].append(vector[label])
        else:
            classes[label] = [vector[label]]
    training_file.close()

    test_file = open(test, "r")
    for line in test_file.readlines():
        vector = json.loads(line)
        label = list(vector.keys())[0]
        if label in test_docs:
            test_docs[label].append(bow_to_list(vector[label]))
        else:
            test_docs[label] = [bow_to_list(vector[label])]
    test_file.close()

    return documents, classes, vocab, test_docs, param_file, outputfile


def trainNB(documents, classes, vocab):
    total_doc_num = len(documents)
    logprev = {} # Represeting probabilities as log probabilities to prevent floating point underflow
    bow_for_each_class = {}
    logprobs = {} 
    num_of_words_in_each_class = {}
    for label, docs_in_the_class in classes.items():
        num_of_documents_in_this_class = len(docs_in_the_class)
        logprev[label] = math.log2(num_of_documents_in_this_class / total_doc_num) #Again we use log probs here
        bow_for_each_class[label] = {}
        num_of_words_in_each_class[label] = 0
        for doc in docs_in_the_class:
            for word, value in doc.items():
                num_of_words_in_each_class[label] += value
                if word in bow_for_each_class[label]:
                    bow_for_each_class[label][word] += value
                else:
                    bow_for_each_class[label][word] = value

        for word in vocab:
            count = 0
            if word in bow_for_each_class[label]:
                count = bow_for_each_class[label][word]
            logprobs[(word, label)] = math.log2((count + 1) / (num_of_words_in_each_class[label] + len(vocab)))
    return logprev, logprobs, bow_for_each_class


def argmax(d): # Referencing formula
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def testNB(test_doc, classes, vocab, logprev, logprobs):
    sum_of_log_probs = {}
    for label, docs_in_the_class in classes.items():
        sum_of_log_probs[label] = logprev[label]
        for word in test_doc:
            if word in vocab:
                sum_of_log_probs[label] += logprobs[(word, label)]
    return argmax(sum_of_log_probs)

def textprobability (x):
    prob_formatting = ""
    for key, val in x.items():
        w = str(key[0])
        c = str(key[1])
        prob_formatting += 'p(' + w + ' | ' + c + ') = ' + str(val) + '\n'
    return prob_formatting

def allcalculations():
    documents, classes, vocab, test_docs, model_output, predictions_output = getinput()
    logprev, logprobs, bow_in_each_class = trainNB(documents, classes, vocab)
    results = {True: 0, False: 0}
    predictions = "# of Doc     Predicted Review     True Review\n"
    num = 1
    for label, documents in test_docs.items():
        for document in documents:
            test_result = testNB(document, classes, vocab, logprev, logprobs)
            results[test_result == label] += 1
            predictions += "   " + str(num) + "       |       " + test_result + "       |       " + label + "\n"
            num += 1
    param_file = open(model_output, "w")
    model = "Log probability of each class:\n" + str(logprev) + \
            '\n\nLog of each word given each class: \n' + textprobability (logprobs)
    param_file.write(model)
    param_file.close()
    outputfile = open(predictions_output, "w")
    accuracy = results[True] / (results[False] + results[True]) * 100
    predictions += "Total Words: " + str(results) + ". Accuracy of results: " + str(accuracy) + '%'
    outputfile.write(predictions)
    outputfile.close()


allcalculations()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score
from sklearn import svm
from collections import defaultdict

import sys, copy, time, os, glob, itertools
import xml.etree.ElementTree as ET


def read_xml():
    #To do: tokenize
    trainpath = 'training/english'
    documents = []
    labels = []
    
    #read all xml files from folder trainpath
    for file in sorted(glob.glob(os.path.join(trainpath, '*.xml'))):
        authorlist = []
        filename = file[17:-4]
        print(filename)
    
        tree = ET.parse(file)
        root = tree.getroot()
        #print(root.tag)
        #[item for sublist in l for item in sublist]

        for tweet in root.iter('document'):
            tokens = tweet.text.strip().split()
            authorlist.append(tokens)
            #print(tweet.text)
        
        #flatten list and add to list documents
        documents.append(list(itertools.chain(*authorlist)))

    #read correct labels from truth file and append them to labels list
    with open(trainpath + '/truth.txt', encoding='utf-8') as truth:
        for line in truth:
            items = line.split(":::")
            #print(items[1])
            labels.append(items[1])
    
    return documents, labels

    
# a dummy function that just returns its input
def identity(x):
    return x

# When called like python3 LFDassignment3.py train test, use first arg as train and second arg as test.
# Otherwise, use file 'trainset.txt' and split in a part for training and testing
if len(sys.argv) > 1:
    #print(sys.argv[1], sys.argv[2])
    Xtrain, Ytrain = read_corpus(sys.argv[1], use_sentiment)
    Xtest, Ytest = read_corpus(sys.argv[2], use_sentiment)
    # create combined dataset for cross-validation
    Xcross, Ycross = copy.deepcopy(Xtrain), copy.deepcopy(Ytrain)
    for list in Xtest:
        Xcross.append(list)
    for label in Ytest:
        Ycross.append(label)
else:
    X, Y = read_xml()
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]
    Xcross, Ycross = X, Y


# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                        tokenizer = identity)
                        #ngram_range=(1, 2))
                        #stop_words = {'english'})
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)

# HashingVectorizer
hv = HashingVectorizer(preprocessor = identity,
                        tokenizer = identity)




# combine the vectorizer with a SVM Classifier
classifier = Pipeline( [
                        ('vec', vec),
                        #('hv', hv),
                        #('cls', svm.SVC(kernel='linear', C=1.0))]) #SVM linear
                        ('cls', svm.SVC(kernel='rbf', gamma=0.8, C=2))]) #SVM non-linear



# Traindata and labels are put in the classifier to create a predictive model.
# Predicted labels for testdata are then assigned to Yguess. Time for training and testing is calculated.
t0 = time.time()
classifier.fit(Xtrain, Ytrain)
train_time = time.time() - t0
t1 = time.time()
Yguess = classifier.predict(Xtest)
test_time = time.time() - t1

# Print classification report with p, r, f1 per class
print("Classification Report:\n", classification_report(Ytest, Yguess))


# Print training/testing times
print("\nTraining time:\t\t{0} s".format(train_time))
print("Testing time:\t\t{0} s".format(test_time))

# To compute accuracy and f1-score, known labels are compared with the predicted labels
print("\nOverall accuracy:\t{0}".format(accuracy_score(Ytest, Yguess)))
print("F1-score:\t\t{0}".format(f1_score(Ytest, Yguess, average='weighted')))


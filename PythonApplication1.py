from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import csv
import codecs
import numpy as np
import time


# start time
start = time.time()

# classifier 
clf = Pipeline([
                ( 'vect', CountVectorizer() ),
                ( 'tfidf', TfidfTransformer() ),
                ( 'clf', LogisticRegression() )
])

# train 
# list parsing train 
parsing_train = list(csv.reader(codecs.open('news_train.txt', 'r', 'utf_8_sig'), delimiter='\t'))

target = []
data = []

for i in range(0, len(parsing_train)):
    target.append(parsing_train[i][0])
    data.append(parsing_train[i][1] + " " + parsing_train[i][2])


clf.fit(data, target)
# end train 

# test
# list parsing test 
parsing_test = list(csv.reader(codecs.open('news_test.txt', 'r', 'utf_8_sig'), delimiter='\t'))

test = []

for i in range(0, len(parsing_test)):
    test.append(parsing_test[i][0] + " " + parsing_test[i][1])


prediction = clf.predict(test)
# end test


np.savetxt('news_output.txt', prediction, fmt="%s")

# end time
print((time.time() - start)/60)

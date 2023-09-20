import re
import string
import sys
import json
import math
import pyspark
from pyspark.sql import SparkSession
from math import log, sqrt
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, Tokenizer
from pyspark.sql.functions import countDistinct, col
from pyspark.ml.linalg import Vectors, SparseVector

# Initialize the Spark
spark= SparkSession.builder.appName('cs5344 final project').getOrCreate()

# step1 
# Read the tweets file
data = spark.read.csv("covid19_tweets3.csv", header=True)
data = data.dropna(subset=["text","id"])

# Read the stopwords file
stopwords = spark.read.text("stopwords.txt")
stopwords = [row.value.lower() for row in stopwords.collect()]

# step2 
# pre-processing
# Split the text into words, remove stopwords, and remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
words = data.rdd.flatMap(lambda row: ((row.id, regex.sub('', word)) for word in row.text.split() if word.lower() not in stopwords))

# output a file 
# Combine all words by username and add up their text
text_by_user = words.map(lambda x: (x[0], x[1])).reduceByKey(lambda x, y: x + ' ' + y).sortBy(lambda x: x[0], ascending = True)

# output file1
df1 = text_by_user.map(lambda x: (x[0], x[1])).toDF(["id", "text"])
df1 = df1.repartition(1)
df1.write.format("csv").option("header", "true").mode("overwrite").save("output1.csv")


# stp3
# Compute the term frequency (TF) for each word in each document abstract
# key: (id + word)
tf = words.map(lambda x: ((x[0],x[1]), 1)).reduceByKey(lambda x, y: x + y)

# step3 -- output
# Sort the words count by count in descending order and take the top 100
word_fre = tf.map(lambda x: (x[0][1],x[1])).reduceByKey(lambda x, y: x + y)
top_words = word_fre.sortBy(lambda x: x[1], ascending=False).take(500)

# output file2
rdd2 = spark.sparkContext.parallelize(top_words)
df2 = spark.createDataFrame(rdd2, ['word', 'count'])
df2 = df2.repartition(1)
df2.write.csv('output2.csv', header=True, mode='overwrite')


# Compute the document frequency (DF) for each word
# key: (word)
df = words.distinct().map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y)

# compute total number of documents
n = data.select(countDistinct("id")).collect()[0][0]

# step 4
# compute tfidf for each word for each id
tfidf_rdd = tf.map(lambda x: (x[0][1], (x[0][0], x[1]))) \
                  .join(df)\
                  .map(lambda x: ((x[0],x[1][0][0]),(x[1][0][1],x[1][1])))\
                  .map(lambda x: ((x[0][1], x[0][0]), (1 + math.log10(x[1][0])) * math.log10(n/x[1][1])))


# normalization 
# Compute the sum of squares of the TF-IDF values for each abstract
sum_of_squares = tfidf_rdd.map(lambda x: (x[0][0], x[1]**2)).reduceByKey(lambda x, y: x+y)
tfidf_rdd_with_squares = tfidf_rdd.map(lambda x: (x[0][0], (x[0][1], x[1])))\
                                        .join(sum_of_squares)\
                                        .map(lambda x: ((x[0],x[1][0][0]), (x[1][0][1], x[1][1])))

# compute normalized tfidf, output : key = (id, word), value = (normalized tfidf)
tfidf_rdd_normalized= tfidf_rdd_with_squares.map(lambda x: (x[0], (x[1][0]/sqrt(x[1][1]))))

# step 5
# output the normalized sum of tfidf of each word
word_tfidf = tfidf_rdd_normalized.map(lambda x:(x[0][1],x[1])).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1], ascending=False)
top_word_tfidf = word_tfidf.take(500)

# output file3

rdd3 = spark.sparkContext.parallelize(top_word_tfidf)
df3 = spark.createDataFrame(rdd3, ['word', 'count'])
df3 = df3.repartition(1)
df3.write.csv('output3.csv', header=True, mode='overwrite')


print("done")
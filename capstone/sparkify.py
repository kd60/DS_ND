# -*- coding: utf-8 -*-
"""Sparkify.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JUm9CgEBSrEO0bRwjKQSAjaTKrU2roCO

# Sparkify Project Workspace
This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.

You can follow the steps below to guide your data analysis and model building portion of this project.
"""

pip install pyspark

# install openjdk-8-jdk-headless -qq

# Commented out IPython magic to ensure Python compatibility.
# import libraries
# import libraries
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import isnan, count, when, col, desc, udf, col, sort_array, asc, avg
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, LinearSVC, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, PCA, RegexTokenizer, VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import datetime
import time

import pandas as pd
import numpy as np
import re
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# create a Spark session
spark = SparkSession \
    .builder \
    .appName("Sparkify Project") \
    .getOrCreate()

spark.sparkContext.getConf().getAll()

"""# Load and Clean Dataset
In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids.

#### load in the dataset
"""

# !unzip "/content/mini_sparkify_event_data.zip" -d "/content/"

df = spark.read.json("mini_sparkify_event_data.json")

# print the schema
df.printSchema()

# print 5 row of dataframe
df.show(5)

# get the count of the dataset before we do any cleaning - this is 286500 and show some rows od dataframe
# df.count()
row_count = df.count()
print("The number of rows in the dataframe is {}".format(df.count()))
print("The number of columns in the dataframe is {}".format(len(df.columns)))
df.describe(df.columns[:6]).show()
df.describe(df.columns[6:13]).show()
df.describe(df.columns[13:]).show()

"""####  Cleaning the Data"""

# drop rows with missing values in userid and/or sessionid
df = df.dropna(how = 'any', subset = ["userId", "sessionId"])

#Remove records with empty userId
df = df.filter(df['userId'] != '')

print("The number of rows in the dataframe is {}".format(df.count())) #before datacount 286500 now 278154
print("The number of columns in the dataframe is {}".format(len(df.columns)))

"""# Exploratory Data Analysis
When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.

### Define Churn

Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.

### Explore Data
Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.
"""

df_pandas = df.toPandas()
df_pandas

# clearly represent registration and timestamp by creating new columns
new_ts = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime("%m-%d-%Y %H:%M:%S"))
df = df.withColumn('updated_registration', new_ts('registration'))
df = df.withColumn('updated_ts', new_ts('ts'))

# add a new column "downgrade_value" to mark Submit Downgrade
downgrade_value  = udf(lambda x: 1 if x == "Submit Downgrade" else 0, IntegerType())
df = df.withColumn("downgrade_value", downgrade_value("page"))

# label users who have downgraded
dg_window = Window.partitionBy('UserId')
df = df.withColumn("user_downgrade", max('downgrade_value').over(dg_window))

# add a new column "churn_value" to mark cancellation event
churn_value = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0, IntegerType())
df = df.withColumn("churn_value", churn_value("page"))

# label users who have churned
churn_window = Window.partitionBy("userId").rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
df = df.withColumn("user_churn", sum('churn_value').over(churn_window))

plt.grid(True)
df_ch_pd = df.drop_duplicates(['userId']).groupby(['user_churn']).count().sort("user_churn").toPandas()
sns.barplot(data =df_ch_pd, x = 'user_churn',y = 'count')
churn_d = ['Active', 'Cancelled']
x_pos = np.arange(len(churn_d))
plt.xticks(x_pos,churn_d)
plt.title('Number of unique users by Subscription status')
plt.ylabel('Number of users')
plt.xlabel('Subscription status')

# Plotting a bar plot to show gender distribution by Subscription status
plt.grid(True)
df_ch_pd = df.drop_duplicates(['userId', 'gender']).groupby(['user_churn', 'gender']).count().sort("user_churn").toPandas()
sns.barplot(x = 'user_churn', y = 'count', data = df_ch_pd, hue = 'gender')
churn_d = ['Active', 'Cancelled']
y_pos = np.arange(len(churn_d))
plt.xticks(y_pos,churn_d)
plt.title("Gender distribution by Subscription status")
plt.ylabel('Number of users')
plt.xlabel('Subscription status')

# Plotting a bar plot to show level distribution by Subscription status
plt.grid(True)
level_df = df.drop_duplicates(['userId','churn_value', 'level']).groupby(['churn_value', 'level']).count().sort("churn_value").toPandas()
sns.barplot(x = 'churn_value', y = 'count', data = level_df, hue = 'level')
churn_d = ['Active', 'Cancelled']
x_pos = np.arange(len(churn_d))
plt.xticks(x_pos,churn_d)
plt.title("Free\Paid levels distribution by Subscription status")
plt.ylabel('Number of users')
plt.xlabel('Subscription status')

page_df = df.groupby(['page','user_churn']).count().toPandas()
page_df = page_df[page_df['page'] != 'NextSong']
page_df = ((page_df.groupby(['page','user_churn']).sum()/page_df.groupby(['user_churn']).sum())*100).reset_index()
page_df['user_churn'].replace({0:'Active', 1: 'Cancelled'},inplace = True)
plt.figure(figsize=(8,10))
plt.grid(True)
sns.barplot(y = 'page', x = 'count', data = page_df, hue = 'user_churn')
plt.title(" % of events by Subscription status")
sns.set_hls_values
plt.ylabel('Eventes')
plt.xlabel('% of envents')



"""# Feature Engineering
Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
- Write a script to extract the necessary features from the smaller subset of data
- Ensure that your script is scalable, using the best practices discussed in Lesson 3
- Try your script on the full data set, debugging your script if necessary

If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.
"""

# Feature 1 : Total number of  songs listened
feat_1 = df.select('userId', 'song').groupBy('userId').count().withColumnRenamed('count', 'tot_songs')

# Feature 2 : Total time spent 
feat_2 = df.select('userID','length').groupBy('userID').agg({'length':'sum'}).withColumnRenamed('sum(length)', 'listen_time')

# Feature 3 : Number of thumbs-up, Feature 4 : thumbs-down
feat_3 = df.select('userID','page').where(df.page == 'Thumbs Up').groupBy('userID').agg({'page':'count'}).withColumnRenamed('count(page)', 'num_thumb_up')
feat_4 = df .select('userID','page').where(df.page == 'Thumbs Down').groupBy('userID').agg({'page':'count'}).withColumnRenamed('count(page)', 'num_thumb_down')

# Feature 5 : Number of adds to playlist
feat_5 = df.select('userID','page').where(df.page == 'Add to Playlist').groupBy('userID').agg({'page':'count'}).withColumnRenamed('count(page)', 'add_to_playlist')

# Feature 6 : Number of lifetime
feat_6 = df.select('userID','registration','ts').withColumn('lifetime',(df.ts-df.registration)).groupBy('userID').agg({'lifetime':'max'}).withColumnRenamed('max(lifetime)','lt')

# Feature 7 : Total number of friends
feat_7 = df.select('userId', 'page').where(df.page == 'Add Friend').groupBy('userId').count().withColumnRenamed('count', 'tot_friends')

# Feature 8 : Gender of the user
feat_8 = df.select('userId', 'gender').dropDuplicates().replace(['F', 'M'], ['0', '1'], 'gender').select('userId', col('gender').cast('int'))

# Feature 9: Number of help AND  Feature 10 : Number of rolladvert
feat_9 = df.select('userID','page').where(df.page == 'Help').groupBy('userID').agg({'page':'count'}).withColumnRenamed('count(page)', 'help') 
feat_10 = df.select('userID','page').where(df.page == 'Roll Advert').groupBy('userID').agg({'page':'count'}).withColumnRenamed('count(page)', 'rolladvert')

# Feature 11 : Total number of songs listened per session
feat_11  = df.where('page == "NextSong"').groupby(['userId', 'sessionId']).count().groupby('userId')\
          .agg({'count' : 'avg'}).withColumnRenamed('avg(count)', 'avg_played_songs')

# Feature 12 : Total number of artists the user has listened to
feat_12 = df.filter(df.page == "NextSong").select("userId", "artist").dropDuplicates().groupby("userId").count()\
         .withColumnRenamed("count", "tot_artist_played")

# setting the churn label for our model
target = df.select('userId', col('user_churn').alias('label')).dropDuplicates()

# Combining all features with the target churn label
final_data  = feat_1.join(feat_2,'userID','outer') \
    .join(feat_3,'userID','outer') \
    .join(feat_4,'userID','outer') \
    .join(feat_5,'userID','outer') \
    .join(feat_6,'userID','outer') \
    .join(feat_7,'userID','outer') \
    .join(feat_8,'userID','outer') \
    .join(feat_9,'userID','outer') \
    .join(feat_10,'userID','outer') \
    .join(feat_11,'userID','outer') \
    .join(feat_12,'userID','outer') \
    .join(target,'userID','outer') \
    .drop('userID') \
    .fillna(0)

final_data.show(5)

incol = ['tot_songs',
 'listen_time',
 'num_thumb_up',
 'num_thumb_down',
 'add_to_playlist',
 'lt',
 'tot_friends',
 'gender',
 'help',
 'rolladvert',
 'avg_played_songs',
 'tot_artist_played']
assembler = VectorAssembler(inputCols=incol, outputCol="NumFeatures")
final_data = assembler.transform(final_data)

scaler2 = StandardScaler(inputCol="NumFeatures", outputCol="features", withStd=True)
scalerModel = scaler2.fit(final_data)
final_data = scalerModel.transform(final_data)

final_data.take(2)



"""# Modeling
Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.
"""

final_data2 = final_data.select('label','features')

train,validation = final_data2.randomSplit([0.8, 0.2], seed=50)

# Models to train: logistic regression, svm, gradient boosting tree
logistic_reg = LogisticRegression(maxIter=10, regParam =0.0)
gbt = GBTClassifier(maxDepth = 5, maxIter = 10, seed = 42)
svm = LinearSVC (maxIter = 10,  regParam = 0.01)

#Logistic Regression
e1 = MulticlassClassificationEvaluator(metricName='f1')
paramGrid = ParamGridBuilder() \
    .addGrid(logistic_reg.regParam,[0.0, 0.05, 0.1]) \
    .build()
crossval = CrossValidator(estimator=logistic_reg,
                          estimatorParamMaps=paramGrid,
                          evaluator=e1,
                          numFolds=3)
cvModel_q1 = crossval.fit(train)
cvModel_q1.avgMetrics

# Support Vector Machine 
e1 = MulticlassClassificationEvaluator(metricName='f1')
paramGrid = ParamGridBuilder() \
    .addGrid(svm.regParam,[0.01, 0.05, 0.5]) \
    .build()
crossval = CrossValidator(estimator=svm,
                          estimatorParamMaps=paramGrid,
                          evaluator=e1,
                          numFolds=3)
cvModel_q2 = crossval.fit(train)
cvModel_q2.avgMetrics

# Gradient boosted tree
e1 = MulticlassClassificationEvaluator(metricName='f1')
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth,[5, 10]) \
    .build()
crossval = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=e1,
                          numFolds=3)
cvModel_q3 = crossval.fit(train)
cvModel_q3.avgMetrics

gbt_tuned = GBTClassifier(maxDepth=5,maxIter=10,seed=42)
gbt_model = gbt_tuned.fit(train)
results = gbt_model.transform(validation)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(results, {evaluator.metricName: "accuracy"}))

print(evaluator.evaluate(results, {evaluator.metricName: "f1"}))

gbt_model.featureImportances

importances  = [ 0.134,0.0408,0.1519,0.0639,0.0946,0.2436,0.0988,0.0301,0.1421]
feature = ["listen_time", "num_song", "num_thumb_down", \
          'num_thumb_up','add_to_playlist','lt','add_friend','help','rolladvert']
y_pos = np.arange(len(feature))
 
plt.barh(y_pos, importances, align='center')
plt.yticks(y_pos, feature)
plt.xlabel('Importance Score')
plt.title('GBT Feature Importances')
plt.savefig('GBT feature Importance.png', dpi=300)

"""# Final Steps
Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.
"""

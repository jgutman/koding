#!/bin/bash

# Download Spark and Scala
# Set path variables for unzipped download directory
curl -o --progress-bar $HOME/Downloads/scala-2.11.7.tgz http://downloads.typesafe.com/scala/2.11.7/scala-2.11.7.tgz
curl -o --progress-bar $HOME/Downloads/spark-1.5.2.tgz http://mirror.cc.columbia.edu/pub/software/apache/spark/spark-1.5.2/spark-1.5.2.tgz
tar -xzf $HOME/Downloads/scala-2.11.7.tgz -C $HOME/Downloads/
tar -xzf $HOME/Downloads/spark-1.5.2.tgz -C $HOME/Downloads/

SPARK=$HOME'/Downloads/spark-1.5.2'
SCALA=$HOME'/Downloads/scala-2.11.7'

# Install python packages
easy_install deepdist
easy_install gensim
easy_install nltk

# Set up PySpark
cd $SPARK
$SPARK/dev/change-scala-version.sh 2.10
build/mvn -Pyarn -Phadoop-2.4 -Dscala-2.10 -Dhadoop.version=2.4.0 -DskipTests clean package
export MAVEN_OPTS="-Xmx2g -XX:MaxPermSize=512M -XX:ReservedCodeCacheSize=512m"

# Submit test script
REDDIT=$HOME'/Desktop/reddit_classification'
cd $REDDIT
$SPARK/bin/spark-submit $REDDIT'/spark_word2vec.py'
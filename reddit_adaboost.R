require(ggplot2)
setwd('~/Desktop/reddit_classification/')

depth <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 35, 50)
scores <- c(.48826, .53631, .5639, .58172, .59443, 
            .60686, .61582, .6209, .62823, .63216, 
            .63524, 0.63606, .6412, .64337, .64141, 
            .64087, .63767, .61309, .55237)
qplot(depth, scores, ylim = c(.4, 0.8), xlab = 'Tree Depth', ylab = 
        'Model Accuracy', geom = 'smooth', main = 'Adaboost Performance At Varying Depths')

read_logs <- function (filename, ngram) {
  logs <- read.table(filename, header = FALSE, sep = "", col.names = 
                       c("level", "zoom", "lamb", "lambda", "sc", "score"), 
                     skip=9, fill = TRUE)
  keep <- c("lambda", "score")
  logs <- na.omit(logs)
  logs <- logs[keep]
  logs["ngram"] <- ngram
  logs
}

files <- c("logs_1", "logs_2", "logs_3", "logs_4")
data <- lapply(files, function(x) read_logs(x, which(files == x)))
svm <- rbind(data[[1]],data[[2]],data[[3]],data[[4]])
svm$ngram = factor(svm$ngram)
dim(svm)

qplot(log10(lambda), score, data = svm, color = ngram, xlab = 
        "Log Lambda Regularization Parameter", ylab = "Model Accuracy", 
      main = "SVM bag-of-ngrams", ylim = c(0, 1), geom = 'smooth')


require(ggplot2)
setwd("~/Desktop/reddit_classification")
file = "d2v_w2v_results.csv"
file2 = "w2v_results_copy.csv"
data = read.csv(file)
data = read.csv(file2)

qplot(Dimensionality, Accuracy, data = data, geom = c('smooth', 'point'), 
      main = "Doc2Vec Performance as a Function of Dimensionality")

stacked <- with(data=data,
                data.frame(Score = c(Accuracy, Recall),
                           Metric = factor(rep(c("Accuracy","Recall"),
                                               each = NROW(data))),
                           Dimensions = rep(Dimensionality, 2),
                           Context.Window = Context.Size))

p = ggplot(subset(stacked, Context.Window %in% c(5, 6, 8, 10, 12, 15)), 
           aes(Dimensions, Score, color = Metric))
p + geom_smooth() + geom_point() + ggtitle("Dimensionality versus Performance")

q = ggplot(subset(stacked, Dimensions %in% c(100, 150, 200, 250, 300,400, 500, 600)), 
           aes(Context.Window, Score, color = Metric))
q + geom_smooth() + geom_point() +ggtitle("Context Window Size versus Performance")

w = ggplot(data, aes(Dimensionality, Accuracy, color = Weighted))
w + geom_smooth() + geom_point() + ggtitle("Dimensionality versus Performance")

x = ggplot(data, aes(Context.Size, Accuracy, color = Weighted))
x + geom_smooth() + geom_point() + ggtitle("Context Size versus Performance")



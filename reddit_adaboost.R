require(ggplot2)
depth = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
scores = c(.48826, .53631, .5639, .58172, .59443, .60686, .61582, .6209, .62823, .63216, .63524, 0.63606)
qplot(depth, scores, ylim = c(.4, 0.8), xlab = 'Tree Depth', ylab = 'Model Accuracy', geom = 'smooth', main = 'Adaboost Performance At Varying Depths')

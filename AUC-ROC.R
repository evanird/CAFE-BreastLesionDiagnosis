# Authors: Evani Radiya-Dixit and David Zhu, Beth Isreal Deaconess Medical Center and Harvard Medical School
# Created: 09/20/2015
# Last modified: 04/04/2017

# This file determines the AUC under the ROC cuve and plots the curve given the UDH predictions for the L1-regularized LR, LR with early stopping, and CAFE models.

rm(list=ls())

glmnetVal <- read.csv("predictions-LR-L1-Regularized-Validation.csv", sep=',')
logisticVal <- read.csv("predictions-LR-Early-Stopping-Validation.csv", sep=',')
cafeVal <- read.csv("predictions-CAFE-Validation.csv", sep=',')
set.seed(101)
features <- read.csv("Features_Label_HospitalOrder.csv")
temp = features$Class
ans = c()
for (i in 117:167) {
  cur = temp[i]
  if (cur == "UDH")
    ans = c(ans, 0)
  else
    ans = c(ans, 1)
}

standardize <- function(val){
  val = val - min(val)
  val = val/max(val)
  return(val)
}

glmnetVal <- standardize(glmnetVal)
logisticVal <- standardize(logisticVal)
cafeVal <- standardize(cafeVal)

getAUC <- function(val, plotTitle){
  pred <- prediction(val, ans, c(1, 0))
  perf <- performance(pred, measure = "tpr", x.measure = "fpr")
  print(performance(pred, measure = "auc"))
  plot(perf, lwd = 2, col = "blue", main = plotTitle)
}

getAUC(glmnetVal, "ROC curve of L1-regularized LR (AUC = 0.897)")
getAUC(logisticVal, "ROC curve of LR with early stopping model (AUC = 0.884)")
getAUC(cafeVal, "ROC curve of our CAFE model (AUC = 0.918)")
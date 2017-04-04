# Authors: Evani Radiya-Dixit and David Zhu, Beth Isreal Deaconess Medical Center and Harvard Medical School
# Created: 08/30/2015
# Last modified: 04/04/2017

# This file combines the prediction values from more than one model based on the concept of bootstrap aggregating. The predictions from the L1-regularized LR and LR with early stopping models are added to generate the
# predictions of the CAFE model.

rm(list=ls())

glmnet = data.matrix(read.csv("predictions-LR-L1-Regularized-Validation.csv"))
logistic = data.matrix(read.csv("predictions-LR-Early-Stopping-Validation.csv"))

standardize <- function(preds){
  preds = preds - min(preds)
  preds = preds/median(preds)
  return(preds)
}

glmnet = standardize(glmnet)
logistic = standardize(logistic)

cafe <- glmnet + logistic
write.csv(cafe, file = "predictions-CAFE-Validation.csv", row.names = F)
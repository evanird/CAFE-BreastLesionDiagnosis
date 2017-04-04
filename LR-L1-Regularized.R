# Authors: Evani Radiya-Dixit and David Zhu, Beth Isreal Deaconess Medical Center and Harvard Medical School
# Created: 09/03/2015
# Last modified: 04/04/2017

# This file generates predictions of the 51 BIDMC cases using the 116 MGH cases as the training cases. The model used is L1-regularized logistic regression. The algorithm fits the training samples to a logistic curve by 
# minimizing a loss function based on the active feature values. The approach minimizes the absolute difference of each feature from its predicted value on the curve, reducing overfitting. The optimal λ is obtained 
# through cross-validation on the training set.

# Input (active features) = ActiveFeatures_Label_HospitalOrder.csv
# Row 1-116 = 116 MGH Cases (DCIS = 80, UDH = 36)
# Row 117-167 = 51 BIDMC Cases (DCIS = 20, UDH = 31) 

rm(list=ls())
library(glmnet)
library(ROCR)

data = read.csv("ActiveFeatures_Label_HospitalOrder.csv")
classCol = ncol(data)
numCases = nrow(data)
trainingSize = 116
trainingData = data[1:trainingSize,]

lambdavalues = c()
for (k in 1:1000) {
  set.seed(k)
  x = data.matrix(data[,-classCol])
  y = data[,"Class"]
  x.tr = x[c(1:trainingSize),]
  y.tr = y[c(1:trainingSize)]
  cv = cv.glmnet(x.tr, y.tr, family = "bin", type = "auc", nfolds = 10, alpha = 1, keep = T)
  value = cv$lambda.min
  lambdavalues = rbind(lambdavalues, value)
}
median = median(lambdavalues)

fit = glmnet(as.matrix(trainingData[,-classCol]), trainingData[,classCol], alpha = 1, family = "binomial")
testingData = data[(trainingSize + 1):numCases,]
predictions = predict(fit, newx = as.matrix(testingData[,-classCol]), type = "response", s = c(median))
write.csv(predictions, file = "predictions-LR-L1-Regularized-Validation.csv", row.names = F)
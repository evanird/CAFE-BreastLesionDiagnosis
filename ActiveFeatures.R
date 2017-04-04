# Authors: Evani Radiya-Dixit and David Zhu, Beth Isreal Deaconess Medical Center and Harvard Medical School
# Created: 08/09/2015
# Last modified: 04/04/2017

# This file generates active features from the 116 MGH cases. The model used is L1-regularized logistic regression. The algorithm fits the training samples to a logistic curve by minimizing a loss function based on all
# feature values. The approach minimizes the absolute difference of each feature from its predicted value on the curve, reducing overfitting.

# Input (all features) = Features_Label_HospitalOrder.csv
# Row 1-116 = 116 MGH Cases (DCIS = 80, UDH = 36)
# Row 117-167 = 51 BIDMC Cases (DCIS = 20, UDH = 31) 

rm(list=ls())
library(glmnet)
set.seed(101)

data = read.csv("Features_Label_HospitalOrder.csv")
trainingSize = 116
trainingData = data[1:trainingSize,]

x = data.matrix(trainingData[, !is.element(colnames(trainingData), c("Class"))])
y = trainingData[,"Class"]
cv = cv.glmnet(x, y, family = "bin", nfolds = 10, alpha = 1, keep = T)
plot(cv)
f = glmnet(x, y, family = "bin", lambda = cv$lambda.min)
sum(f$beta[,1]!=0)
activeFeatures = labels(f$beta[f$beta[,1]!=0,])
activeFeaturesData = cbind(data.matrix(data[,is.element(colnames(data), activeFeatures)]), data[c("Class")])

write.csv(activeFeaturesData, "ActiveFeatures_Label_HospitalOrder.csv", row.names = F)
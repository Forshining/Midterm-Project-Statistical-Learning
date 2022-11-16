## Package Loading
#install.packages("glmnet")
#install.packages("riskRegression")
#install.packages("rms")
#install.packages("starnet")
#install.packages('readxl')
library(glmnet)
library(riskRegression)
library(rms)
library(starnet)
library(readxl)
library(pROC)
## Data Preprocessing
data = read_excel('./Project Materials/修正数据.xlsx')
# delete the obesity var
library(dplyr)
data <- select(data,-obesity)

data = data[c(which(data$group == 1),which(data$group == 2)),]

data = data[complete.cases(data),]
sam_size = dim(data)[1]

train_index = sort(sample(c(1:sam_size),floor(sam_size*0.75),replace = FALSE))
test_index = c(1:sam_size)[-train_index]
data_train = data[train_index,]
data_test = data[test_index,]

## Variable Selection
select_out = c("OCP","obesity","WELLS","Clinical_Prob1","R_GENEVA","Clinical_Prob2","ECG")
data_train  = data_train[,-which(colnames(data_train) %in% select_out)]
Y_train = data_train[,"group"]
Y_train = as.factor(as.matrix(Y_train))
X_train = data_train[,-which(colnames(data_train) %in% c("group"))]
data_test  = data_test[,-which(colnames(data_test) %in% select_out)]
Y_test = data_test[,"group"]
Y_test = as.factor(as.matrix(Y_test))
X_test = data_test[,-which(colnames(data_test) %in% c("group"))]

train_total = cbind(Y_train,X_train)
X_train = model.matrix(Y_train~.,data = train_total)

test_total = cbind(Y_test,X_test)
X_test = model.matrix(Y_test~.,data = test_total)

## LASSO without splitting
set.seed(99)
model_1 = cv.glmnet(X_train,
                 Y_train,
                 family = "binomial",
                 type.measure = "auc",
                 nfolds = 5,
                 keep = TRUE
                 )

plot(model_1)
## Prediction
### Logistic + LASSO without splitting
lambda = model_1$lambda.1se
pred = predict(model_1,newx = X_test, type = 'response',s = "lambda.1se")
roc = roc.glmnet(model_1$fit.preval, newy = Y_train)
best = model_1$index["min",]
plot(roc[[best]],type = "l")

assess.glmnet(model_1,newx = X_test,newy = Y_test,family = "binomial")
cm = confusion.glmnet(model_1,newx = X_test,newy = Y_test,family = "binomial")

#ACC
sum(Y_test== predict(model_1,newx = X_test, type = 'class',s = "lambda.1se"))/length(Y_test)
#CI
ci.auc(auc(Y_test,pred))

Specificity_without = cm[1,1] / (cm[1,1] + cm[2,1])
Sensitivity_without = cm[2,2] / (cm[2,2] + cm[1,2])
Frequency_without = (cm[1,2] + cm[2,2]) / sum(cm)

## Adaptive Lasso
set.seed(99)
cv.ridge_1 <- cv.glmnet(X_train, Y_train, family='binomial', alpha=0, parallel=TRUE, standardize=TRUE)
best_ridge_coef_1 <- as.numeric(coef(cv.ridge_1, s = cv.ridge_1$lambda.min))[-1]

cv.lasso_1 <- cv.glmnet(X_train, Y_train, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc', penalty.factor=1/abs(best_ridge_coef_1))
plot(cv.lasso_1)
plot(cv.lasso_1$glmnet.fit, xvar="lambda", label=TRUE)
abline(v = log(cv.lasso_1$lambda.min))
abline(v = log(cv.lasso_1$lambda.1se))
coef(cv.lasso_1, s=cv.lasso_1$lambda.1se)
coef <- coef(cv.lasso_1, s='lambda.1se')
assess.glmnet(cv.lasso_1,newx = X_test,newy = Y_test,family = "binomial")
ci.auc(auc(Y_test,predict(cv.lasso_1,newx = X_test, type = 'response',s = "lambda.1se")))



## LASSO with splitting
data_train = data[train_index,]
data_test = data[test_index,]
data_train  = data_train[,-which(colnames(data_train) %in% select_out)]
Y_train = data_train[,"group"]
Y_train = as.factor(as.matrix(Y_train))
X_train = data_train[,-which(colnames(data_train) %in% c("group"))]
data_test  = data_test[,-which(colnames(data_test) %in% select_out)]
Y_test = data_test[,"group"]
Y_test = as.factor(as.matrix(Y_test))
X_test = data_test[,-which(colnames(data_test) %in% c("group"))]
### For training dataset
age_train = X_train$age
HR_train = X_train$HR

age_labels = c(1,2,3,4,5,6,7,8,9)
age_breaks = c(10,20,30,40,50,60,70,80,90,100)

age_train_dummy = cut(x = age_train, breaks = age_breaks, labels = age_labels,right = FALSE)
age_train_dummy = as.matrix(age_train_dummy)

HR_labels = c(1,2,3)
HR_breaks = c(0,60,100,Inf)

HR_train_dummy = cut(x = HR_train, breaks = HR_breaks, labels = HR_labels,right = FALSE)
HR_train_dummy = as.matrix(HR_train_dummy)

X_train$age = age_train_dummy[,1]
X_train$HR = HR_train_dummy[,1]

train_total = cbind(Y_train,X_train)

X_train_split = model.matrix(Y_train~.,data = train_total)

## For test dataset
age_test = X_test$age
HR_test = X_test$HR

age_labels = c(1,2,3,4,5,6,7,8,9)
age_breaks = c(10,20,30,40,50,60,70,80,90,100)

age_test_dummy = cut(x = age_test, breaks = age_breaks, labels = age_labels,right = FALSE)
age_test_dummy = as.matrix(age_test_dummy)

HR_labels = c(1,2,3)
HR_breaks = c(0,60,100,Inf)

HR_test_dummy = cut(x = HR_test, breaks = HR_breaks, labels = HR_labels,right = FALSE)
HR_test_dummy = as.matrix(HR_test_dummy)

X_test$age = age_test_dummy[,1]
X_test$HR = HR_test_dummy[,1]

test_total = cbind(Y_test,X_test)

X_test_split = model.matrix(Y_test~.,data = test_total)

set.seed(99)
model_2 = cv.glmnet(X_train_split,
                    Y_train,
                    family = "binomial",
                    type.measure = "auc",
                    nfolds = 5,
                    keep = TRUE
                    )
plot(model_2)

## Prediction
### Logistic + LASSO with splitting

lambda = model_2$lambda.1se
pred = predict(model_2,newx = X_test_split, type = 'response',s = "lambda.1se")
roc = roc.glmnet(model_2$fit.preval, newy = Y_train)
best = model_2$index["min",]
plot(roc[[best]],type = "l")

assess.glmnet(model_2,newx = X_test_split,newy = Y_test,family = "binomial")
cm = confusion.glmnet(model_2,newx = X_test_split,newy = Y_test,family = "binomial")

Specificity_with = cm[1,1] / (cm[1,1] + cm[2,1])
Sensitivity_with = cm[2,2] / (cm[2,2] + cm[1,2])
Frequency_with = (cm[1,2] + cm[2,2]) / sum(cm)


## xgb 

library(xgboost)




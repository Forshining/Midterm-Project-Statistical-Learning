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
library(dplyr)
## Data Preprocessing
data = read_excel('./Project Materials/修正数据.xlsx')
# delete the obesity var

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
X_train = model.matrix(Y_train~.,data = train_total)[,-1]
#X_train = as.matrix(X_train)
test_total = cbind(Y_test,X_test)
X_test = model.matrix(Y_test~.,data = test_total)[,-1]
#X_test = as.matrix(X_test)

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
cv.ridge_1 <- cv.glmnet(X_train, Y_train, family='binomial', alpha=0, parallel=TRUE, standardize=TRUE,nfolds = 10)
best_ridge_coef_1 <- as.numeric(coef(cv.ridge_1, s = cv.ridge_1$lambda.min))[-1]

cv.lasso_1 <- cv.glmnet(X_train, Y_train, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc', penalty.factor=1/abs(best_ridge_coef_1),nfolds = 10)
plot(cv.lasso_1)
plot(cv.lasso_1$glmnet.fit, xvar="lambda", label=TRUE)
abline(v = log(cv.lasso_1$lambda.min))
abline(v = log(cv.lasso_1$lambda.1se))
coef(cv.lasso_1, s=cv.lasso_1$lambda.1se)
#coef <- coef(cv.lasso_1, s='lambda.1se')
assess.glmnet(cv.lasso_1,newx = X_test,newy = Y_test,family = "binomial")
ci.auc(auc(Y_test,predict(cv.lasso_1,newx = X_test, type = 'response',s = "lambda.1se")))



## LASSO with splitting
data_train_s = data[train_index,]
data_test_s = data[test_index,]
data_train_s  = data_train_s[,-which(colnames(data_train_s) %in% select_out)]
Y_train_s = data_train_s[,"group"]
Y_train_s = as.factor(as.matrix(Y_train_s))
X_train_s = data_train_s[,-which(colnames(data_train_s) %in% c("group"))]
data_test_s  = data_test_s[,-which(colnames(data_test_s) %in% select_out)]
Y_test_s = data_test_s[,"group"]
Y_test_s = as.factor(as.matrix(Y_test_s))
X_test_s = data_test_s[,-which(colnames(data_test_s) %in% c("group"))]
### For training dataset
age_train = X_train_s$age
HR_train = X_train_s$HR

age_labels = c(1,2,3,4,5,6,7,8,9)
age_breaks = c(10,20,30,40,50,60,70,80,90,100)

age_train_dummy = cut(x = age_train, breaks = age_breaks, labels = age_labels,right = FALSE)
age_train_dummy = as.matrix(age_train_dummy)

HR_labels = c(1,2,3)
HR_breaks = c(0,60,100,Inf)

HR_train_dummy = cut(x = HR_train, breaks = HR_breaks, labels = HR_labels,right = FALSE)
HR_train_dummy = as.matrix(HR_train_dummy)

X_train_s$age = age_train_dummy[,1]
X_train_s$HR = HR_train_dummy[,1]

train_total_s = cbind(Y_train_s,X_train_s)


X_train_s = model.matrix(Y_train_s~.,data = train_total_s)[,-1]

## For test dataset
age_test = X_test_s$age
HR_test = X_test_s$HR

age_labels = c(1,2,3,4,5,6,7,8,9)
age_breaks = c(10,20,30,40,50,60,70,80,90,100)

age_test_dummy = cut(x = age_test, breaks = age_breaks, labels = age_labels,right = FALSE)
age_test_dummy = as.matrix(age_test_dummy)

HR_labels = c(1,2,3)
HR_breaks = c(0,60,100,Inf)

HR_test_dummy = cut(x = HR_test, breaks = HR_breaks, labels = HR_labels,right = FALSE)
HR_test_dummy = as.matrix(HR_test_dummy)

X_test_s$age = age_test_dummy[,1]
X_test_s$HR = HR_test_dummy[,1]

test_total_s = cbind(Y_test_s,X_test_s)
#X_test_s = data.matrix(X_test_s)
X_test_s = model.matrix(Y_test_s~.,data = test_total_s)[,-1]

set.seed(99)
model_2 = cv.glmnet(X_train_s,
                    Y_train_s,
                    family = "binomial",
                    type.measure = "auc",
                    nfolds = 5,
                    keep = TRUE
                    )
plot(model_2)

## Prediction
### Logistic + LASSO with splitting

lambda = model_2$lambda.1se
pred = predict(model_2,newx = X_test_s, type = 'response',s = "lambda.1se")
roc = roc.glmnet(model_2$fit.preval, newy = Y_train_s)
best = model_2$index["min",]
plot(roc[[best]],type = "l")

assess.glmnet(model_2,newx = X_test_s,newy = Y_test_s,family = "binomial")
cm = confusion.glmnet(model_2,newx = X_test_s,newy = Y_test_s,family = "binomial")

Specificity_with = cm[1,1] / (cm[1,1] + cm[2,1])
Sensitivity_with = cm[2,2] / (cm[2,2] + cm[1,2])
Frequency_with = (cm[1,2] + cm[2,2]) / sum(cm)

## adaptive lasso with splitting
set.seed(99)
cv.ridge_2 <- cv.glmnet(X_train_s, Y_train_s, family='binomial', alpha=0, parallel=TRUE, standardize=TRUE,nfolds = 10)
#ise and min
best_ridge_coef_2 <- as.numeric(coef(cv.ridge_2, s = cv.ridge_2$lambda.min))[-1]

cv.lasso_2 <- cv.glmnet(X_train_s, Y_train_s, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc', penalty.factor=1/abs(best_ridge_coef_2),nfolds = 10)
plot(cv.lasso_2)
plot(cv.lasso_2$glmnet.fit, xvar="lambda", label=TRUE)
abline(v = log(cv.lasso_2$lambda.min))
abline(v = log(cv.lasso_2$lambda.1se))
coef(cv.lasso_2, s=cv.lasso_2$lambda.1se)
coef <- coef(cv.lasso_2, s='lambda.1se')
assess.glmnet(cv.lasso_2,newx = X_test_s,newy = Y_test_s,family = "binomial")
ci.auc(auc(Y_test,predict(cv.lasso_2,newx = X_test_s, type = 'response',s = "lambda.1se")))



## xgb 

library(xgboost)
library(doParallel)
library(caret)

# without splitting
Y_train_xgb <- as.numeric(Y_train)-1
Y_test_xgb <- as.numeric(Y_test) -1

xgb_grid = expand.grid(
  nrounds = seq(from = 200, to = 1000, by = 100),
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4),
  gamma = c(0,1,2,3),
  colsample_bytree=c(0.5,1),
  min_child_weight=c(1, 2, 3),
  subsample=c(0.8,1)
)

my_control <-trainControl(method="cv", number=5)
xgb_caret <- train(x=X_train, y=Y_train, method='xgbTree', trControl= my_control, tuneGrid=xgb_grid,nthread = 16,verbosity = 0,metric="Accuracy") 
xgb_caret[["bestTune"]]
##
# nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
# 3293     900         3 0.1     2              0.5                3         1

xgb_train <- xgb.DMatrix(data = X_train, label = Y_train_xgb)
xgb_test <- xgb.DMatrix(data = X_test, label = Y_test)

param_list = list(
  eta = 0.1,
  gamma = 2,
  max_depth = 2,
  subsample = 1,
  colsample_bytree = 0.5,
  subsample = 1,
  min_child_weight = 3
)

xgbcv = xgb.cv(params = param_list,
               nrounds = 900,
               data = xgb_train,
               nfold = 5,
               print_every_n = 10,
               maximize = F,
               eval_metric = "auc",
               objective = "binary:logistic")

xgb <- xgb.train(data = xgb_train, params=param_list, nrounds = 900)
xgb.ggplot.importance (xgb.importance (feature_names = colnames(X_train),model = xgb),rel_to_first = TRUE,measure = "Frequency")

pre_xgb = predict(xgb,newdata = xgb_test)
table(Y_test_xgb,round(pre_xgb),dnn=c("true","pre"))
auc(Y_test_xgb,pre_xgb)
roc(Y_test_xgb,pre_xgb)
plot(roc(Y_test_xgb,pre_xgb))


# with splitting
Y_train_xgb_s <- as.numeric(Y_train_s)-1
Y_test_xgb_s <- as.numeric(Y_test_s) -1

xgb_grid = expand.grid(
  nrounds = seq(from = 200, to = 1000, by = 100),
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4),
  gamma = c(0,1,2,3),
  colsample_bytree=c(0.5,1),
  min_child_weight=c(1, 2, 3),
  subsample=c(0.8,1)
)

my_control <-trainControl(method="cv", number=5)
xgb_caret_s <- train(x=X_train_s, y=Y_train_s, method='xgbTree', trControl= my_control, tuneGrid=xgb_grid,nthread = 16,verbosity = 0,metric="Accuracy") 
xgb_caret_s[["bestTune"]]
##
# nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
# 1365     700         2 0.05     0                1                1         1

xgb_train_s <- xgb.DMatrix(data = X_train_s, label = Y_train_xgb_s)
xgb_test_s <- xgb.DMatrix(data = X_test_s, label = Y_test_s)

param_list_s = list(
  eta = 0.05,
  gamma = 0,
  max_depth = 2,
  subsample = 1,
  colsample_bytree = 1,
  subsample = 1,
  min_child_weight = 1
)

xgbcv_s = xgb.cv(params = param_list_s,
               nrounds = 700,
               data = xgb_train_s,
               nfold = 5,
               print_every_n = 10,
               maximize = F,
               eval_metric = "auc",
               objective = "binary:logistic")

xgb_s <- xgb.train(data = xgb_train_s, params=param_list_s, nrounds = 700)
xgb.ggplot.importance (xgb.importance (feature_names = colnames(X_train_s),model = xgb_s),rel_to_first = TRUE,measure = "Frequency")

pre_xgb_s = predict(xgb_s,newdata = xgb_test_s)
table(Y_test_xgb_s,round(pre_xgb_s),dnn=c("true","pre"))
auc(Y_test_xgb_s,pre_xgb_s)
roc(Y_test_xgb_s,pre_xgb_s)
plot(roc(Y_test_xgb_s,pre_xgb_s))

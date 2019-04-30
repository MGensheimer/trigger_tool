rm(list=ls())
graphics.off()
library(ggplot2)
library(dplyr)
library(Matrix)
library(pROC)
library(xgboost)
library(glmnet)
library(data.table)
library(icdcoder)
library(e1071)
#library(icd)

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
data_dir <- '/Users/michael/Documents/research/patient safety/trigger indicators/data/'
output_dir <- '/Users/michael/Documents/research/patient safety/trigger indicators/figures/'

data=read.csv(paste(data_dir,'csi_20180522.csv',sep=''),stringsAsFactors=FALSE)
data <- data %>% mutate(nmri_2_3_4 = (CSI==1 & (CSI_NMRI=='2' | CSI_NMRI=='3' | CSI_NMRI=='4')))
data <- data %>% mutate(nmri_3_4 = (CSI==1 & (CSI_NMRI=='3' | CSI_NMRI=='4')))
data$CSI <- data$CSI==1
data$AgeAtSim[data$AgeAtSim>120] <- 120
data$TotalFx[data$TotalFx<1] <- 1
data$Days_Tx_Missed <- as.numeric(data$Days_Tx_Missed)
data$Days_Tx_Missed[is.na(data$Days_Tx_Missed)] <- 0

data$icd_code <- sub('\\.',"",data$icd_code)
is_icd10 <- grepl('[A-Z]',data$icd_code)
result <- unlist(lapply(data$icd_code[is_icd10], function(x) {convICD(x,'icd10')[1,'icd9']})) #convert ICD10 to ICD9 using CMS GEMS; if multiple matches just use first match
data$icd9 <- data$icd_code
data$icd9[is_icd10] <- result
data$icd9_simplified <- substr(data$icd9,1,3)
data$icd9_simplified[is.na(data$icd9_simplified)] <- 999
data$icd9_simplified <- as.factor(data$icd9_simplified)

#treatment technique processing
data$Technique1 <- as.factor(data$Technique1)
data$Technique2 <- as.factor(data$Technique2)
data$Technique3 <- as.factor(data$Technique3)
data$Technique4 <- as.factor(data$Technique4)
data$Technique5 <- as.factor(data$Technique5)
data$Technique6 <- as.factor(data$Technique6)
data$Technique7 <- as.factor(data$Technique7)
data$Technique8 <- as.factor(data$Technique8)
data$Technique9 <- as.factor(data$Technique9)
data$Technique10 <- as.factor(data$Technique10)
data$Technique11 <- as.factor(data$Technique11)
data$Technique12 <- as.factor(data$Technique12)
data$Technique13 <- as.factor(data$Technique13)
data$HspCode5Coded <- as.factor(data$HspCode5Coded)
data$Technique_MFGcode[is.na(data$Technique_MFGcode) | data$Technique_MFGcode==0] <- 999
data$Technique_MFGcode <- as.factor(data$Technique_MFGcode)

data <- data %>% filter(Technique_MFGcode != 3) #exclude neutron treatments
data_train <- data %>% filter(set==1)
data_test <- data %>% filter(set==2)

#################
#data description
nrow(data_train)
nrow(data_test)
summary(as.factor(data_train$CSI_NMRI))
summary(as.factor(data_test$CSI_NMRI))
summary(data_train$Technique_MFGcode)
summary(data_test$Technique_MFGcode)
summary(data_train$AgeAtSim)
summary(data_test$AgeAtSim)


#######################################################
#Pick model type using cross validation on training set

folds <- 10
set.seed(1)
scrambled_indices <- sample(nrow(data_train),nrow(data_train))
scrambled_indices <- split(scrambled_indices, ceiling(seq_along(scrambled_indices)/(nrow(data_train)/folds)))

myFormula <- formula(~AgeAtSim+NumSites+Technique_MFGcode+
                       TotalFx+DelayedFirstFxInDays+HiddenFields+NewFields+DeletedFields+NewRx+
                       RejectedPorts+PlanDocs+VoidedPlans+PhysicsDocs+UDosePhotonCalc+UPhotonCalcCheck+
                       UPhysicsDocsApprovedAfterStart+Days_Tx_Missed)
cv_results <- data.frame(train_auc=rep(NA,folds),test_auc=rep(NA,folds),test_acc=rep(NA,folds),test_brier=rep(NA,folds))
model_type <- 'logistic_glmnet'
#model_type <- 'xgboost'
if(0) {
  par(mfrow=c(ceiling(folds/3),floor(folds/3)))
  par(mar=c(1,1,1,1))
}
for (fold in seq(folds)) {
  test_indices <- scrambled_indices[[fold]]
  data_train_temp <- data_train[-test_indices,]
  data_test_temp <- data_train[test_indices,]
  x_train <- model.matrix(myFormula,data=data_train_temp)
  row.names(x_train) <- NULL
  x_test <- model.matrix(myFormula,data=data_test_temp)
  row.names(x_test) <- NULL
  if (model_type=='logistic_glmnet') {
    glmnet_fit = cv.glmnet(x_train, data_train_temp$nmri_3_4, family = "binomial", alpha=0,nfolds=10,type.measure='deviance')
    prediction_train <- c(predict(glmnet_fit, newx = x_train, s = "lambda.min", type = "response"))
    prediction_test <- c(predict(glmnet_fit, newx = x_test, s = "lambda.min", type = "response"))
  }
  if (model_type=='xgboost') {
    dtrain <- xgb.DMatrix(data = x_train, label=data_train_temp$nmri_3_4)
    dtest <- xgb.DMatrix(data = x_test, label=data_test_temp$nmri_3_4)
    watchlist=list(train=dtrain, test=dtest)
    model <- xgb.train(data = dtrain, watchlist=watchlist, max.depth = 4, eta = 0.2, subsample=1,
                       nthread = 2, nround = 1000, objective = "binary:logistic", early_stopping_rounds=100, verbose=0)
    prediction_train <- predict(model,x_train)
    prediction_test <- predict(model,x_test)
  }

  cv_results[fold,'test_auc'] <- auc(data_test_temp$nmri_3_4, prediction_test)
  cv_results[fold,'train_auc'] <- auc(data_train_temp$nmri_3_4, prediction_train)
  cv_results[fold,'test_acc'] <- mean(data_test_temp$nmri_3_4==(prediction_test>0.5))
  cv_results[fold,'test_brier'] <- mean( (data_test_temp$nmri_3_4-prediction_test)^2 )
  
  n_bins <- 10
  if(0) { #train folds calibration plot
    breaks <- unique(quantile(prediction_train, probs = seq(0,1,1/n_bins)))
    predBin <- cut(prediction_train, breaks, include = TRUE)
    n_bins <- length(breaks)
    levels(predBin) <- seq(n_bins)
    plot(c(0,1),c(0,1),type='l',lty=2,xaxs="i",yaxs="i",xlab='Predicted probability',ylab='True proportion')
    for (bin in seq(n_bins)) {
      indices <- which(predBin==bin)
      meanPred <- mean(prediction_train[indices])
      trueProp <- mean(data_train_temp$nmri_3_4[indices])
      points(meanPred,trueProp)
    }
  }
  if(0) { #test fold calibration plot
    breaks <- unique(quantile(prediction_test, probs = seq(0,1,1/n_bins)))
    predBin <- cut(prediction_test, breaks, include = TRUE)
    n_bins <- length(breaks)
    levels(predBin) <- seq(n_bins)
    plot(c(0,1),c(0,1),type='l',lty=2,xaxs="i",yaxs="i",xlab='Predicted probability',ylab='True proportion')
    for (bin in seq(n_bins)) {
      indices <- which(predBin==bin)
      meanPred <- mean(prediction_test[indices])
      trueProp <- mean(data_test_temp$nmri_3_4[indices])
      points(meanPred,trueProp)
    }
  }
}
cbind(mean(cv_results$train_auc),mean(cv_results$test_auc),mean(cv_results$test_acc),mean(cv_results$test_brier))

##################################
#fit final model: grade 2-4 events
x_train <- model.matrix(myFormula,data=data_train)
x_test <- model.matrix(myFormula,data=data_test)
glmnet_fit = cv.glmnet(x_train, data_train$nmri_2_3_4, family = "binomial", alpha=0,nfolds=10,type.measure='deviance')
plot(glmnet_fit)
glmnet_fit[['lambda.min']]
prediction_train <- c(predict(glmnet_fit, newx = x_train, s = "lambda.min", type = "response"))
prediction_test <- c(predict(glmnet_fit, newx = x_test, s = "lambda.min", type = "response"))

train_auc <- auc(data_train$nmri_2_3_4, prediction_train)
train_acc <- mean(data_train$nmri_2_3_4==(prediction_train>0.5))
train_brier <- mean( (data_train$nmri_2_3_4-prediction_train)^2 )
test_auc <- auc(data_test$nmri_2_3_4, prediction_test)
test_acc <- mean(data_test$nmri_2_3_4==(prediction_test>0.5))
test_brier <- mean( (data_test$nmri_2_3_4-prediction_test)^2 )

#Performance on training and test sets
cbind(train_auc,train_acc,train_brier,test_auc,test_acc,test_brier)

#ROC curves for training and test sets
pdf(file=paste(output_dir,'roc_grade2to4.pdf',sep=''), width=5.5, height=10)
par(mfrow=c(2,1))
train_roc <- roc(data_train$nmri_2_3_4, prediction_train)
plot(c(0,1),c(0,1),type='l',lty=2,xaxs="i",yaxs="i",xlab='1-Specificity',ylab='Sensitivity',main='Training set')
lines(1-train_roc[['specificities']],train_roc[['sensitivities']])
test_roc <- roc(data_test$nmri_2_3_4, prediction_test)
plot(c(0,1),c(0,1),type='l',lty=2,xaxs="i",yaxs="i",xlab='1-Specificity',ylab='Sensitivity',main='Test set')
lines(1-test_roc[['specificities']],test_roc[['sensitivities']])
dev.off()

#Calibration plots for training and test sets

pdf(file=paste(output_dir,'calib_grade2to4.pdf',sep=''), width=5.5, height=10)
par(mfrow=c(2,1))

n_bins <- 10
breaks <- unique(quantile(prediction_train, probs = seq(0,1,1/n_bins)))
predBin <- cut(prediction_train, breaks, include = TRUE)
n_bins <- length(breaks)
levels(predBin) <- seq(n_bins)
plot(c(0,1),c(0,1),type='l',lty=2,xaxs="i",yaxs="i",xlab='Predicted probability',ylab='True proportion',main='Training set')
for (bin in seq(n_bins)) {
  indices <- which(predBin==bin)
  meanPred <- mean(prediction_train[indices])
  trueProp <- mean(data_train$nmri_2_3_4[indices])
  points(meanPred,trueProp)
}

n_bins <- 10
breaks <- unique(quantile(prediction_test, probs = seq(0,1,1/n_bins)))
predBin <- cut(prediction_test, breaks, include = TRUE)
n_bins <- length(breaks)
levels(predBin) <- seq(n_bins)
plot(c(0,1),c(0,1),type='l',lty=2,xaxs="i",yaxs="i",xlab='Predicted probability',ylab='True proportion',main='Test set')
for (bin in seq(n_bins)) {
  indices <- which(predBin==bin)
  meanPred <- mean(prediction_test[indices])
  trueProp <- mean(data_test$nmri_2_3_4[indices])
  points(meanPred,trueProp)
}

dev.off()

#Model coefficients
print(coef(glmnet_fit, s=glmnet_fit$lambda.min))

#univariate results
predictor_vars <- c('AgeAtSim','NumSites','Technique_MFGcode','TotalFx','DelayedFirstFxInDays','HiddenFields','NewFields','DeletedFields','NewRx','RejectedPorts','PlanDocs','VoidedPlans','PhysicsDocs','UDosePhotonCalc','UPhotonCalcCheck','UPhysicsDocsApprovedAfterStart','Days_Tx_Missed')
for (var_i in seq(length(predictor_vars))) {
  f <- paste('nmri_2_3_4 ~',predictor_vars[var_i])
  m <- glm(f,data=data_train,family='binomial')
  print(summary(m))
  print(exp(cbind("Odds ratio" = coef(m), confint.default(m, level = 0.95))))
}
m <- glm(nmri_2_3_4~Technique_MFGcode,data=data_train,family='binomial')
f <- paste('nmri_2_3_4 ~',predictor_vars[1])
m <- glm(f,data=data_train,family='binomial')


summary(m)
exp(cbind("Odds ratio" = coef(m), confint.default(m, level = 0.95)))

##################################
#fit final model: grade 3-4 events
x_train <- model.matrix(myFormula,data=data_train)
x_test <- model.matrix(myFormula,data=data_test)
glmnet_fit = cv.glmnet(x_train, data_train$nmri_3_4, family = "binomial", alpha=0,nfolds=10,type.measure='deviance')
plot(glmnet_fit)
glmnet_fit[['lambda.min']]
prediction_train <- c(predict(glmnet_fit, newx = x_train, s = "lambda.min", type = "response"))
prediction_test <- c(predict(glmnet_fit, newx = x_test, s = "lambda.min", type = "response"))

predictions_train <- data.frame(StudyID=data_train$StudyID, probability_grade34 = prediction_train)
predictions_test <- data.frame(StudyID=data_test$StudyID, probability_grade34 = prediction_test)
write.table(predictions_train,file=paste(data_dir,'predictions_train.csv',sep=''),row.names=FALSE,sep=',')
write.table(predictions_test,file=paste(data_dir,'predictions_test.csv',sep=''),row.names=FALSE,sep=',')

train_auc <- auc(data_train$nmri_3_4, prediction_train)
train_acc <- mean(data_train$nmri_3_4==(prediction_train>0.5))
train_brier <- mean( (data_train$nmri_3_4-prediction_train)^2 )
test_auc <- auc(data_test$nmri_3_4, prediction_test)
test_acc <- mean(data_test$nmri_3_4==(prediction_test>0.5))
test_brier <- mean( (data_test$nmri_3_4-prediction_test)^2 )

#Performance on training and test sets
cbind(train_auc,train_acc,train_brier,test_auc,test_acc,test_brier)

#ROC curves for training and test sets
pdf(file=paste(output_dir,'roc_grade3to4.pdf',sep=''), width=5.5, height=10)
par(mfrow=c(2,1))
train_roc <- roc(data_train$nmri_3_4, prediction_train)
plot(c(0,1),c(0,1),type='l',lty=2,xaxs="i",yaxs="i",xlab='1-Specificity',ylab='Sensitivity',main='Training set')
lines(1-train_roc[['specificities']],train_roc[['sensitivities']])
test_roc <- roc(data_test$nmri_3_4, prediction_test)
plot(c(0,1),c(0,1),type='l',lty=2,xaxs="i",yaxs="i",xlab='1-Specificity',ylab='Sensitivity',main='Test set')
lines(1-test_roc[['specificities']],test_roc[['sensitivities']])
dev.off()

#Calibration plots for training and test sets

pdf(file=paste(output_dir,'calib_grade3to4.pdf',sep=''), width=5.5, height=10)
par(mfrow=c(2,1))

n_bins <- 10
breaks <- unique(quantile(prediction_train, probs = seq(0,1,1/n_bins)))
predBin <- cut(prediction_train, breaks, include = TRUE)
n_bins <- length(breaks)
levels(predBin) <- seq(n_bins)
plot(c(0,0.5),c(0,0.5),type='l',lty=2,xaxs="i",yaxs="i",xlab='Predicted probability',ylab='True proportion',main='Training set')
for (bin in seq(n_bins)) {
  indices <- which(predBin==bin)
  meanPred <- mean(prediction_train[indices])
  trueProp <- mean(data_train$nmri_3_4[indices])
  points(meanPred,trueProp)
}

n_bins <- 10
breaks <- unique(quantile(prediction_test, probs = seq(0,1,1/n_bins)))
predBin <- cut(prediction_test, breaks, include = TRUE)
n_bins <- length(breaks)
levels(predBin) <- seq(n_bins)
plot(c(0,0.5),c(0,0.5),type='l',lty=2,xaxs="i",yaxs="i",xlab='Predicted probability',ylab='True proportion',main='Test set')
for (bin in seq(n_bins)) {
  indices <- which(predBin==bin)
  meanPred <- mean(prediction_test[indices])
  trueProp <- mean(data_test$nmri_3_4[indices])
  points(meanPred,trueProp)
}

dev.off()

#Model coefficients
print(coef(glmnet_fit, s=glmnet_fit$lambda.min))

#univariate results
predictor_vars <- c('AgeAtSim','NumSites','Technique_MFGcode','TotalFx','DelayedFirstFxInDays','HiddenFields','NewFields','DeletedFields','NewRx','RejectedPorts','PlanDocs','VoidedPlans','PhysicsDocs','UDosePhotonCalc','UPhotonCalcCheck','UPhysicsDocsApprovedAfterStart','Days_Tx_Missed')
for (var_i in seq(length(predictor_vars))) {
  f <- paste('nmri_3_4 ~',predictor_vars[var_i])
  m <- glm(f,data=data_train,family='binomial')
  print(summary(m))
  print(exp(cbind("Odds ratio" = coef(m), confint.default(m, level = 0.95))))
}

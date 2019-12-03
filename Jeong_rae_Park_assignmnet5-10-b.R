library(ggplot2)
library(maps)
library(lattice)
library(caret)
library(gbm)
library(gridExtra)

data(scat)
str(scat)

sum(is.na(scat))

#Remove the Month, Year, Site, Location features
df = subset(scat, select = -c(Month,Year,Site, Location) )

#Set the Species column as the target/outcome and convert it to numeric
df$Species<-as.numeric(factor(df$Species))

#Check if any values are null. If there are, impute missing values using KNN
sum(is.na(scat))
preProcValues <- preProcess(df, method = c("knnImpute","center","scale"))

#Converting every categorical variable to numerical
library('RANN')
train_processed <- predict(preProcValues, df)
sum(is.na(train_processed))

dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))

str(train_transformed)

sum(is.na(train_transformed))

#Graduate Student questions:a. Using feature selection with rfe in caret and the repeatedcv method: Find the top 3
#predictors and build the same models as in 6 and 8 with the same parameters. 
set.seed(100)
index <- createDataPartition(train_transformed$Species, p=0.75, list=FALSE)
trainSet1 <- train_transformed[ index,]
testSet1 <- train_transformed[-index,]

str(trainSet1)
trainSet1$Species<-as.factor(trainSet1$Species)
outcomeName<-'Species'
predictors<-names(trainSet1)[!names(trainSet1) %in% outcomeName]

control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet1)[!names(trainSet1) %in% outcomeName]
Loan_Pred_Profile1 <- rfe(trainSet1[,predictors], trainSet1[,outcomeName],rfeControl = control)
Loan_Pred_Profile1

predictors<-c("CN", "Mass", "d13C", "d15N")

model_gbm<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='gbm')
print(model_gbm)
plot(model_gbm)
predictions<-predict.train(object=model_gbm,testSet1[,predictors],type="raw")
table(predictions)
#confusionMatrix(predictions,testSet1$outcomeName)

model_rf<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='rf')
print(model_rf)
plot(model_rf)
predictions<-predict.train(object=model_rf,testSet1[,predictors],type="raw")
table(predictions)
#confusionMatrix(predictions,testSet1$outcomeName)

model_nnet<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='nnet')
print(model_nnet)
plot(model_nnet)
predictions<-predict.train(object=model_nnet,testSet1[,predictors],type="raw")
table(predictions)
#confusionMatrix(predictions,testSet1$outcomeName)

model_nb<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='nb')
print(model_nb)
plot(model_nb)
predictions<-predict.train(object=model_nb,testSet1[,predictors],type="raw")
table(predictions)
#confusionMatrix(predictions,testSet1$outcomeName)

fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)

#Tune the GBM model using tune length = 20 and: a) print the model summary and b) plot the models
model_gbm<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
print(model_gbm)
plot(model_gbm)

plot_gbm<-plot(varImp(object=model_gbm),main="GBM - Variable Importance")
plot_rf<-plot(varImp(object=model_rf),main="rf - Variable Importance")
#plot_nnet<-plot(varImp(object=model_nnet),main="nnet - Variable Importance")
plot_nb<-plot(varImp(object=model_nb),main="nb - Variable Importance")

grid.arrange(plot_gbm, plot_rf, plot_nb, ncol=2)


#Graduate Student questions:a. Using feature selection with rfe in caret and the repeatedcv method: Find the top 3
#predictors and build the same models as in 6 and 8 with the same parameters. 
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet1)[!names(trainSet1) %in% outcomeName]
Loan_Pred_Profile1 <- rfe(trainSet1[,predictors], trainSet1[,outcomeName],rfeControl = control)
Loan_Pred_Profile1

predictors<-c("CN", "Mass", "d13C", "d15N")

model_gbm<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='gbm')
table(predictions)
table(testSet1[,outcomeName])
confusionMatrix(predictions,as.factor(testSet1[,outcomeName]))
gbm_acc_kappa<-model_gbm$resample
gbm_acc_kappa<-as.data.frame(gbm_acc_kappa)
gbm_acc_kappa <-gbm_acc_kappa[order(gbm_acc_kappa$Accuracy),]
names(gbm_acc_kappa)[3]<-"ExperimentName"
gbm_acc_kappa$ExperimentName <- sub("Resample..", "GBM", gbm_acc_kappa$ExperimentName)
print(gbm_acc_kappa)

model_rf<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='rf')
table(predictions)
table(testSet1[,outcomeName])
confusionMatrix(predictions,as.factor(testSet1[,outcomeName]))
rf_acc_kappa<-model_rf$resample
rf_acc_kappa<-as.data.frame(rf_acc_kappa)
rf_acc_kappa <-rf_acc_kappa[order(rf_acc_kappa$Accuracy),]
names(rf_acc_kappa)[3]<-"ExperimentName"
rf_acc_kappa$ExperimentName <- sub("Resample..", "randomforest", rf_acc_kappa$ExperimentName)
print(rf_acc_kappa)

model_nnet<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='nnet')
table(predictions)
table(testSet1[,outcomeName])
confusionMatrix(predictions,as.factor(testSet1[,outcomeName]))
nnet_acc_kappa<-model_nnet$resample
nnet_acc_kappa<-as.data.frame(nnet_acc_kappa)
nnet_acc_kappa <-nnet_acc_kappa[order(nnet_acc_kappa$Accuracy),]
names(nnet_acc_kappa)[3]<-"ExperimentName"
nnet_acc_kappa$ExperimentName <- sub("Resample..", "neural net", nnet_acc_kappa$ExperimentName)
print(nnet_acc_kappa)

model_nb<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='nb')
table(predictions)
table(testSet1[,outcomeName])
confusionMatrix(predictions,as.factor(testSet1[,outcomeName]))
nb_acc_kappa<-model_nb$resample
nb_acc_kappa<-as.data.frame(nb_acc_kappa)
nb_acc_kappa <-nb_acc_kappa[order(nb_acc_kappa$Accuracy),]
names(nb_acc_kappa)[3]<-"ExperimentName"
nb_acc_kappa$ExperimentName <- sub("Resample..", "naive bayes", nb_acc_kappa$ExperimentName)
print(nb_acc_kappa)

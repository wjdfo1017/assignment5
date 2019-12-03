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

#With a seed of 100, 75% training, 25% testing . Build the following models: randomforest, neuralnet, naive bayes and GBM
set.seed(100)
index <- createDataPartition(train_transformed$Species, p=0.75, list=FALSE)
trainSet1 <- train_transformed[ index,]
testSet1 <- train_transformed[-index,]

str(trainSet1)
trainSet1$Species<-as.factor(trainSet1$Species)
outcomeName<-'Species'
predictors<-names(trainSet1)[!names(trainSet1) %in% outcomeName]

model_gbm<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='gbm')
print(model_gbm)
plot(model_gbm)
predictions<-predict.train(object=model_gbm,testSet1[,predictors],type="raw")
table(predictions)
table(testSet1[,outcomeName])
confusionMatrix(predictions,as.factor(testSet1[,outcomeName]))

model_rf<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='rf')
print(model_rf)
plot(model_rf)
predictions<-predict.train(object=model_rf,testSet1[,predictors],type="raw")
table(predictions)
table(testSet1[,outcomeName])
confusionMatrix(predictions,as.factor(testSet1[,outcomeName]))

model_nnet<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='nnet')
print(model_nnet)
plot(model_nnet)
predictions<-predict.train(object=model_nnet,testSet1[,predictors],type="raw")
table(predictions)
table(testSet1[,outcomeName])
confusionMatrix(predictions,as.factor(testSet1[,outcomeName]))

model_nb<-train(trainSet1[,predictors],trainSet1[,outcomeName],method='nb')
print(model_nb)
plot(model_nb)
predictions<-predict.train(object=model_nb,testSet1[,predictors],type="raw")
table(predictions)
table(testSet1[,outcomeName])
confusionMatrix(predictions,as.factor(testSet1[,outcomeName]))

#For the BEST performing models of each (randomforest, neural net, naive bayes and gbm) create
#and display a data frame that has the following columns:
gbm_acc_kappa<-model_gbm$resample
gbm_acc_kappa<-as.data.frame(gbm_acc_kappa)
gbm_acc_kappa <-gbm_acc_kappa[order(gbm_acc_kappa$Accuracy),]
names(gbm_acc_kappa)[3]<-"ExperimentName"
gbm_acc_kappa$ExperimentName <- sub("Resample..", "GBM", gbm_acc_kappa$ExperimentName)
print(gbm_acc_kappa)

rf_acc_kappa<-model_rf$resample
rf_acc_kappa<-as.data.frame(rf_acc_kappa)
rf_acc_kappa <-rf_acc_kappa[order(rf_acc_kappa$Accuracy),]
names(rf_acc_kappa)[3]<-"ExperimentName"
rf_acc_kappa$ExperimentName <- sub("Resample..", "randomforest", rf_acc_kappa$ExperimentName)
print(rf_acc_kappa)

nnet_acc_kappa<-model_nnet$resample
nnet_acc_kappa<-as.data.frame(nnet_acc_kappa)
nnet_acc_kappa <-nnet_acc_kappa[order(nnet_acc_kappa$Accuracy),]
names(nnet_acc_kappa)[3]<-"ExperimentName"
nnet_acc_kappa$ExperimentName <- sub("Resample..", "neural net", nnet_acc_kappa$ExperimentName)
print(nnet_acc_kappa)

nb_acc_kappa<-model_nb$resample
nb_acc_kappa<-as.data.frame(nb_acc_kappa)
nb_acc_kappa <-nb_acc_kappa[order(nb_acc_kappa$Accuracy),]
names(nb_acc_kappa)[3]<-"ExperimentName"
nb_acc_kappa$ExperimentName <- sub("Resample..", "naive bayes", nb_acc_kappa$ExperimentName)
print(nb_acc_kappa)
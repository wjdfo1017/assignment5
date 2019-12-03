library(ggplot2)
library(maps)
library(lattice)
library(caret)
library(gbm)
library(gridExtra)

data(scat)
str(scat)

sum(is.na(scat))

df = subset(scat)
#Set the Species column as the target/outcome and convert it to numeric
df$Species<-as.numeric(factor(df$Species))

#Remove the Month, Year, Site, Location features
df = subset(scat, select = -c(Month,Year,Site, Location) )


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
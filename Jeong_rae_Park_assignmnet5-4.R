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
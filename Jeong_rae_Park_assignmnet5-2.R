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
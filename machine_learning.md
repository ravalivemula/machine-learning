# Practical-Machine-Learning-Course-Project
## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## Choosing the prediction algorithm

Steps Taken

1.Tidy data. Remove columns with little/no data.

2.Create Training and test data from traing data for cross validation checking

3.Trial 3 methods Random Forrest, Gradient boosted model and Linear discriminant analysis

Fine tune model through combinations of above methods, reduction of input variables or similar. The fine tuning will take into account accuracy first and speed of analysis second.


```
library(ggplot2)
library(caret)
## Loading required package: lattice
library(randomForest)
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
library(e1071)
library(gbm)
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
library(doParallel)
## Loading required package: foreach
## Loading required package: iterators
library(survival)
library(splines)
library(plyr)
setwd("~/GitHub/PracMacLearn")

```
## Load data
1. Load data
2. Remove “#DIV/0!”, replace with an NA value.

```
# load data
setwd("C:/Users/ravali/Downloads/Practical Machine Learning")
training <- read.csv("pml-training.csv", na.strings=c("#DIV/0!"), row.names = 1)
testing <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!"), row.names = 1)
training <- training[, 6:dim(training)[2]]

treshold <- dim(training)[1] * 0.95
#Remove columns with more than 95% of NA or "" values
goodColumns <- !apply(training, 2, function(x) sum(is.na(x)) > treshold  || sum(x=="") > treshold)

training <- training[, goodColumns]

badColumns <- nearZeroVar(training, saveMetrics = TRUE)

training <- training[, badColumns$nzv==FALSE]

training$classe = factor(training$classe)

#Partition rows into training and crossvalidation
inTrain <- createDataPartition(training$classe, p = 0.6)[[1]]
crossv <- training[-inTrain,]
training <- training[ inTrain,]
inTrain <- createDataPartition(crossv$classe, p = 0.75)[[1]]
crossv_test <- crossv[ -inTrain,]
crossv <- crossv[inTrain,]


testing <- testing[, 6:dim(testing)[2]]
testing <- testing[, goodColumns]
testing$classe <- NA
testing <- testing[, badColumns$nzv==FALSE]
#Train 3 different models
mod1 <- train(classe ~ ., data=training, method="rf")
#mod2 <- train(classe ~ ., data=training, method="gbm")
#mod3 <- train(classe ~ ., data=training, method="lda")

pred1 <- predict(mod1, crossv)
#pred2 <- predict(mod2, crossv)
#pred3 <- predict(mod3, crossv)
#show confusion matrices
confusionMatrix(pred1, crossv$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    3    0    0    0
##          B    1 1135    6    0    0
##          C    0    1 1020    4    0
##          D    0    0    0  960    1
##          E    1    0    0    1 1081
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.995, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.996    0.994    0.995    0.999
## Specificity             0.999    0.999    0.999    1.000    1.000
## Pos Pred Value          0.998    0.994    0.995    0.999    0.998
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.184
## Detection Prevalence    0.285    0.194    0.174    0.163    0.184
## Balanced Accuracy       0.999    0.998    0.997    0.997    0.999
#confusionMatrix(pred2, crossv$classe)
#confusionMatrix(pred3, crossv$classe)

#Create Combination Model

#predDF <- data.frame(pred1, pred2, pred3, classe=crossv$classe)
#predDF <- data.frame(pred1, pred2, classe=crossv$classe)

#combModFit <- train(classe ~ ., method="rf", data=predDF)
#in-sample error
#combPredIn <- predict(combModFit, predDF)
#confusionMatrix(combPredIn, predDF$classe)



#out-of-sample error
pred1 <- predict(mod1, crossv_test)
#pred3 <- predict(mod3, crossv_test)
accuracy <- sum(pred1 == crossv_test$classe) / length(pred1)

```
Based on results, the Random Forest prediction was far better than either the GBM or lsa models. The RF model will be used as the sole prediction model. The confusion matrix created gives an accuracy of 99.6%. 

As a double check the out of sample error was calculated. This model achieved 99.7 % accuracy on the validation set.
## Fine Tuning

Assess Number of relevant variables
```
varImpRF <- train(classe ~ ., data = training, method = "rf")
varImpObj <- varImp(varImpRF)
# Top 40 plot
plot(varImpObj, main = "Importance of Top 40 Variables", top = 40)
# Top 25 plot
plot(varImpObj, main = "Importance of Top 25 Variables", top = 25)

```
## Conclusion

The Random Forest method worked very good.

The Confusion Matrix achieved 99.6% accuracy. The Out of Sample Error achieved 99.7 %.

This model will be used for the final calculations.

A Random forest can handle unscaled variables and categorical variables. This is more forgiving with the cleaning of the data.
It worked
## Final Model Testing
```
pml_write_files = function(x){
n = length(x)
for(i in 1:n){
filename = paste0("problem_id_",i,".txt")
write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
}
x <- testing

answers <- predict(mod1, newdata=x)
answers
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
pml_write_files(answers)

```

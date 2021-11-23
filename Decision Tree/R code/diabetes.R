#loading dataset
diabetes<-read.csv(file.choose(),header = T)
View(diabetes)
colnames(diabetes)<-c('Pregnancies','Glucose','Diastolic BP','Skin Thickness','Insulin','BMI','DPFunction','Age','Outcome')
str(diabetes)
diabetes$Outcome<-as.factor(diabetes$Outcome)
table(diabetes$Outcome)
sum(is.na(diabetes))
summary(diabetes)
# Visualization
boxplot(diabetes,col = rainbow(9))$out
round(prop.table(table(diabetes$Class.variable))*100,1)
library(GGally)
ggpairs(diabetes)
ggcorr(diabetes)
library(ggplot2)
attach(diabetes)
v1<-ggplot(diabetes,aes(x=BMI,fill=Outcome,color=Outcome))+geom_histogram(binwidth = 1)+
  labs(title = 'Distribution of BMI by Diabetic Class')
v1+theme_bw()
v2<-ggplot(diabetes,aes(x=Pregnancies,fill=Outcome,color=Outcome))+
  geom_histogram(binwidth = 1)+labs(title = 'Distribution of Pregnancy cases by Diabetic Class')
v2+theme_bw()
v3<-ggplot(diabetes,aes(x=Age,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Age by Diabetic class')
v3+theme_bw()
v4<-ggplot(diabetes,aes(x=`Skin Thickness`,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Triceps skin fold thickness by Diabetic class')
v4+theme_bw()
v5<-ggplot(diabetes,aes(x=`Diastolic BP`,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Diastolic Blood Pressure by Diabetic class')
v5+theme_bw()
v6<-ggplot(diabetes,aes(x=DPFunction,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Diabetes pedigree function by Diabetic class')
v6+theme_bw()
v7<-ggplot(diabetes,aes(x=Glucose,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Plasma Glucose Concentration by Diabetic class')
v7+theme_bw()
norm<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}

diabetes_n<-as.data.frame(lapply(diabetes[-9],norm))
 # spliting of data
train<-diabetes_n[1:538,]
test<-diabetes_n[539:768,]
 # training and test labels
train_labels<-diabetes[1:538,9]
test_labels<-diabetes[539:768,9]

library(randomForest)
randomforest<-randomForest(train_labels~.,train,importance=T)
randomforest
# randoforest model results
# Number of trees = 500
# number of variable tried at each split = 2
'''
Confusion matrix:
     NO YES class.error
NO  297  52   0.1489971
YES  85 104   0.4497354
'''
randomforest_acc<-((297+104)/(297+52+85+104))
randomforest_acc
plot(randomforest)
varImpPlot(randomforest)
# Accuracy Test
test_acc<-mean(test_labels==predict(randomforest,newdata = test))
test_acc
# Confusion Matrix test
Predicted_test<-predict(randomforest,newdata = test)
table(test_labels,Predicted_test)
Updated_test<-cbind(test_labels,Predicted_test)
# Accuracy train
train_acc<-mean(train_labels==predict(randomforest,newdata = train))
train_acc
# Confusion matrix
Predicted_train<-predict(randomforest,newdata = train)
table(train_labels,Predicted_train)
Updated_train<-cbind(train_labels,Predicted_train)

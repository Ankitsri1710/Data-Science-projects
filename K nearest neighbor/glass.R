#loading dataset
glass<-read.csv(file.choose(),header = T)
str(glass)
glass$Type<-as.factor(glass$Type)
library(Hmisc)
describe(glass)
attach(glass)
# Visualization
library(ggplot2)
v1<-ggplot(glass,aes(x=RI,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
  labs(title = "RI distribution by Type")
v1+theme_bw()
v2<-ggplot(glass,aes(x=Na,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
  labs(title =  " Na Proportion by Type ")
v2+theme_bw()
v3<-ggplot(glass,aes(x=Mg,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
  labs(title =  " Mg Proportion by Type ")
v3+theme_bw()
v4<-ggplot(glass,aes(x=Al,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
             labs(title =  " Al Propotion by Type ")
v4+theme_bw()
v5<-ggplot(glass,aes(x=Si,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
     labs(title =  " Si Proportion by Type ")
v5+theme_bw()
v6<-ggplot(glass,aes(x=K,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
  labs(title =  " K Proportion by Type ")
v6+theme_bw()
v7<-ggplot(glass,aes(x=Ca,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
  labs(title =  " Ca Proportion by Type ")
v7+theme_bw()
v8<-ggplot(glass,aes(x=Ba,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
  labs(title =  " Ba Proportion by Type ")
v8+theme_bw()
v9<-ggplot(glass,aes(x=Fe,fill=Type,color=Type))+geom_histogram(bindwidth=1)+
  labs(title =  " Fe Proportion by Type ")
v9+theme_bw()
round(prop.table(table(glass$Type))*100,digits=2)

normalize<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}
norm_glass<-as.data.frame(lapply(glass[,1:9],normalize))

# Spliting of dataset
train_glass<-norm_glass[1:171,]
test_glass<-norm_glass[171:214,]

train_labels<-glass[1:171,10]
test_labels<-glass[171:214,10]

library(class)
pred<-knn(train = train_glass,test = train_glass,cl=train_labels,k=10,prob = T)
pred
table(pred,train_labels) 
mean(train_labels==pred)
library(gmodels)
CrossTable(pred,train_labels,prop.chisq = F,prop.r = F,prop.c = F,prop.t = F)
# model evaluation on train
pred1<-knn(train = train_glass,test = train_glass,cl=train_labels,k=6,prob = T)
CrossTable(pred1,train_labels,prop.chisq = F,prop.r = F,prop.c = F,prop.t = F)
mean(train_labels==pred1)
pred2<-knn(train = train_glass,test = train_glass,cl=train_labels,k=5,prob = T)
CrossTable(pred2,train_labels,prop.chisq = F,prop.r = F,prop.c = F,prop.t = F)
mean(train_labels==pred2)
pred3<-knn(train = train_glass,test = train_glass,cl=train_labels,k=3)
CrossTable(pred3,train_labels,prop.chisq = F,prop.r = F,prop.c = F,prop.t = F)
mean(train_labels==pred3)

# model evaluation on test
p1<-knn(train = test_glass,test = test_glass,cl=test_labels,k=6)
CrossTable(p1,test_labels,prop.chisq = F,prop.r = F,prop.c = F,prop.t = F)
mean(test_labels==p1)
p2<-knn(train = test_glass,test = test_glass,cl=test_labels,k=5)
CrossTable(p2,test_labels,prop.chisq = F,prop.r = F,prop.c = F,prop.t = F)
mean(test_labels==p2)
p3<-knn(train = test_glass,test = test_glass,cl=test_labels,k=3)
CrossTable(p3,test_labels,prop.chisq = F,prop.r = F,prop.c = F,prop.t = F)
mean(test_labels==p3)

# Evaluation
Predicted_Test_Response<-p2
Eval_test<-cbind(test_labels,Predicted_Test_Response)
Predicted_Train_Response<-pred2
Eval_train<-cbind(train_labels,Predicted_Train_Response)
# We select k=5 as both training and test accuracy are close to each other.

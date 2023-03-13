#install.packages('ggplot2')
#install.packages('corrplot')
#install.packages('neuralnet')
#install.packages('caret')
#install.packages('MASS')
#install.packages('kknn')
#install.packages('class')
#install.packages('GGally')
#install.packages('glmnet')
#install.packages('olsrr')
#install.packages('yardstick')
#install.packages('graphics')
#install.packages('ggpol')
library(ggplot2)
library(corrplot)
library(neuralnet)
library(caret)
library(MASS)
library(kknn)
library(class)
library(nnet)
library(GGally)
library(glmnet)
library(olsrr)
library(yardstick)
library(graphics)
library(ggpol)

#Data read
set.seed(12345)
data <- read.csv('heart_failure.csv',header=TRUE)
dim(data)

#Initialization
 ##
 ##read the trained model and data
#split the data
#data_nor <- data
#data_nor[,-13]<- scale(data[,-13])
#index <- sample(seq_len(nrow(data)), size = 0.7 * nrow(data))
#data_train = data_nor[index,]
#data_test = data_nor[-index,]
# splited data
data_train<-readRDS('data_train.rds') # 读取 rds
data_test<-readRDS('data_test.rds') # 读取 rds
lassoBPNNModel<-readRDS('lassoBPNNModel.rds') # improved model
lassoModel<-readRDS('lassoModel.rds') # lasso model
nn<-readRDS('BPNNModel.rds') # bpnn

# visuliaze the data
summary(data)
colnames(data)
head(data)
hist(data$age,col = rgb(202,  178, 114,maxColorValue = 255))
X11()
hist(data$creatinine_phosphokinase,col = rgb(202,  178, 114,maxColorValue = 255))
X11()
hist(data$ejection_fraction,col = rgb(202,  178, 114,maxColorValue = 255))
X11()
hist(data$platelets,col = rgb(202,  178, 114,maxColorValue = 255))
X11()
hist(data$serum_creatinine,col  = rgb(202,  178, 114,maxColorValue = 255))
X11()
hist(data$serum_sodium,col  = rgb(202,  178, 114,maxColorValue = 255))
X11()
hist(data$time,col  = rgb(202,  178, 114,maxColorValue = 255))
X11()
cor_matrix <- cor(data[, sapply(data, is.numeric)])
cor_matrix
corrplot(cor_matrix)
X11()
# Train model
# bpnn
#formula <- factor(fatal_mi) ~ age+anaemia+creatinine_phosphokinase+diabetes+ejection_fraction+high_blood_pressure+platelets+serum_creatinine+serum_sodium+sex+smoking+time
#nn <- neuralnet(formula, data_train,hidden=c(10, 10,5), linear.output=FALSE,threshold=0.005, stepmax=1e+05, learningrate=0.001, lifesign="minimal", algorithm="backprop",act.fct='logistic')
# lasso
x_lasso <- as.matrix(data_train[,-13])
y_lasso <- data_train[,13]
x_lasso_pre <- as.matrix(data_test[,-13])
y_test <- data_test[, 13]
#lassoModel = cv.glmnet(x_lasso, y_lasso, alpha = 1, nfolds = 10)
# get lambda with 1 standard error rule
#lambda.min = lassoModel$lambda.min
#lambda.1se = lassoModel$lambda.1se
#print(lambda.min)
#print(lambda.1se)
#lasso.1se = glmnet(x_lasso, y_lasso, alpha = 1, lambda = lambda.1se)
#print(coef(lasso.1se))
#plot(as.matrix(coef(lasso.1se)))
#title('The result of Lasso choosed weight')
#X11()
#improved bpnn
#formula_lasso <- factor(fatal_mi) ~ age+anaemia+ejection_fraction+serum_creatinine+serum_sodium+time
#train_control <- trainControl(method="cv", number=10)
# Train the neural network using cross-validation
#lassoBPNNModel <- train(formula_lasso, data=data_train, method="nnet", trControl=train_control, 
#                                hidden=c(10, 10,5), linear.output=FALSE, threshold=0.005, 
#                                stepmax=1e+05, learningrate=0.001, lifesign="minimal", 
#                                algorithm="backprop", act.fct='logistic')  

# Prediction
# Improved bpnn
crossnet.predict_lasso<-predict(lassoBPNNModel,data_test)
crosspredict.table_lasso<-table(data_test$fatal_mi,crossnet.predict_lasso)
confusionMatrix(crosspredict.table_lasso)
cm_net_lasso <- confusionMatrix(crosspredict.table_lasso)
# lasso
lasso.pred <- predict(lassoModel, newx = x_lasso_pre, s = "lambda.1se", type = "response")
lasso.pred[lasso.pred < 0.5] = 0
lasso.pred[lasso.pred >= 0.5] = 1
crosspredict.table<-table(data_test$fatal_mi,lasso.pred)
confusionMatrix(crosspredict.table)
confusionMatrix
# bpnn
net.predict<-compute(nn,data_test)$net.result
net.prediction<-c("0","1")[apply(net.predict,1,which.max)]
predict.table<-table(data_test$fatal_mi,net.prediction)
net.prediction
predict.table
confusionMatrix(predict.table)


# Confusion matrix visualization
ggplot() + geom_confmat(aes(x = data_test$fatal_mi, y = crossnet.predict_lasso),
                        normalize = TRUE, text.perc = TRUE)+
  labs(x = "Reference",y = "Prediction",title = 'Confusion matrix of improved bpnn')+
  scale_fill_gradient2(low="darkblue", high="lightgreen")
X11()
ggplot() + geom_confmat(aes(x = data_test$fatal_mi, y = lasso.pred),
                        normalize = TRUE, text.perc = TRUE)+
  labs(x = "Reference",y = "Prediction",title = 'Confusion matrix of lasso')+
  scale_fill_gradient2(low="darkblue", high="lightgreen")
X11()
ggplot() + geom_confmat(aes(x = data_test$fatal_mi, y = net.prediction),
                        normalize = TRUE, text.perc = TRUE)+
  labs(x = "Reference",y = "Prediction",title = 'Confusion matrix of bpnn')+
  scale_fill_gradient2(low="darkblue", high="lightgreen")











# Antler plots, take 2
# 11/29/2018

# Load libraries
library(tidyverse) # For data manipulation
library(caret) # For machine learning
library(pROC) # For AUC
library(ROCR) # For AUC
library(doParallel) # For parallel processing. 
detectCores()
registerDoParallel(cores=7)
getDoParWorkers()


# Get data
d <- read.csv("Nancy_Pham_NewData.csv")
d$Type <- ifelse(d$Type %in% c(1,2), "benign", "malignant") # transform outcome
d <- d[,-2]
head(d) # Check

# Set up empty data.frame for collecting summary performance metrics at varying p
# Collect 10 different p
(performance <- data.frame(p = seq(.45, .90, .05),
                    ntrain = 0,
                    ntest = 0,
                    train_auc = 0,
                    train_var = 0,
                    test_auc = 0,
                    test_var = 0))

J <- 100 #define J

# Set up empty data.frame to collect the calculated AUC for each J
(test_cv_metrics <- matrix(nrow = J, ncol = nrow(performance)))
(train_cv_metrics <- matrix(nrow = J, ncol = nrow(performance)))

set.seed(1121)

for(i in 1:nrow(performance)){
  
  splits <- createDataPartition(d$Type, p = performance$p[i], times = J) 
  performance$ntrain[i] <- length(unique(splits[[1]]))
  performance$ntest[i] <- nrow(d) - performance$ntrain[i]
  
  train_auc <- NULL
  test_auc <- NULL
  
  for(j in 1:J){
    
    train <- d[splits[[j]],]
    test <- d[-splits[[j]],]
    
    # nrow(train)
    # nrow(test)

    model <- train(Type ~.,
               data = train,
               preProcess = "pca",
               method = "glmnet",
               trControl = trainControl(classProbs = T,
                                        summaryFunction = twoClassSummary,
                                        returnResamp = "final"),
               metric = "ROC",
               tuneGrid = expand.grid(alpha = seq(0, 1, .1),
                                      lambda = seq(0, 2, .02)))
    
     
    pred_train <- prediction(predict(model, newdata = train, type = "prob")[,2], 
                       train$Type)
    train_auc[j] <- performance(pred_train,"auc")@y.values
    
    pred_test <- prediction(predict(model, newdata = test, type = "prob")[,2], 
                             test$Type)
    test_auc[j] <- performance(pred_test,"auc")@y.values
    
    cat(j,"\r")
    
  }
  
  performance$train_var[i] <- var(as.numeric(train_auc))
  performance$test_var[i] <- var(as.numeric(test_auc))
  performance$train_auc[i] <- mean(as.numeric(train_auc))
  performance$test_auc[i] <- mean(as.numeric(test_auc))
  
  test_cv_metrics[,i] <- as.numeric(test_auc)
  train_cv_metrics[,i] <- as.numeric(train_auc)
  
 cat(paste("i =", i),"\r")
}

performance
test_cv_metrics
train_cv_metrics

write.csv(performance, "performance.csv")
write.csv(test_cv_metrics, "test_cv_metrics.csv")
write.csv(train_cv_metrics, "train_cv_metrics.csv")


## Plots 

(antler <- gather(performance, sample, value, - ntrain, -ntest, -p) %>%
  separate(sample, into = c("sample","metric")) %>%
  spread(key=metric, value=value) %>%
    mutate(se = sqrt(var * (1/J + ntest/ntrain)),
           lower = auc - 1.96 * se,
           upper = auc + 1.96 *se))

# No error bars
ggplot(filter(antler, p < .95), aes(1/ntrain, auc, col = sample)) +
  geom_point()+
  ylim(.65,.85) +
  xlim(0,.02)+
  geom_smooth(method = "lm", se = F, fullrange=T, ) +
  theme_minimal()

# error bars
ggplot(filter(antler, p < .95), aes(1/ntrain, auc)) +
  geom_point()+
  geom_errorbar(aes(ymin = lower, ymax = upper)) +
  ylim(.5,1) +
  xlim(0,.02)+
  geom_smooth(method = "lm", se = F, fullrange=T, ) +
  theme_minimal() +
  facet_wrap(~sample)


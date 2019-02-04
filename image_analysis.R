# Image analysis
# 11/16/2018

library(tidyverse) # For data manipulation
library(caret) # For machine learning
library(pROC) # For AUC
library(doParallel) # For parallel processing. 
detectCores()
registerDoParallel(cores=4)
getDoParWorkers()

raw <- read.csv("raw_image_data.csv") # This Nancy_Pham_NewData.csv with Type %in% c(1,2) = "benign"

# Here is an implementation of so-called Monte Carlo cross-validation, w/ J = 100
set.seed(1118)
J = 100
n <- nrow(raw)
splits <- createDataPartition(raw$Type, p = .84, times = J) # Define a single set of J splits
(ntrain <- length(unique(splits[[1]])))
(ntest <- n - ntrain)

# Centered and scaled inputs
no_pca <- train(Type ~.,
               data = raw,
               preProcess = c("center","scale"),
               method = "glmnet",
               trControl = trainControl(classProbs = T,
                                        summaryFunction = twoClassSummary,
                                        method = "LGOCV",
                                        returnResamp = "final",
                                        index = splits),
               metric = "ROC",
               tuneGrid = expand.grid(alpha = seq(0, 1, .2),
                       lambda = seq(0, 1, .05)))

# Performance results
no_pca$resample %>%
  summarize(mean = mean(ROC),
            se = sqrt(var(ROC) * (1/J + (ntest/ntrain))),
            lower = mean - 1.96 * se,
            upper = mean + 1.96 * se) %>%
  round(2)

# Fitted model coefficients
coef(no_pca$finalModel, no_pca$bestTune$lambda) %>% 
  round(3) #Inputs are centered and scaled so we can directly compare coefficient effect sizes

# PCA transformed nputs
pca <- train(Type ~.,
                data = raw,
                preProcess = "pca",
                method = "glmnet",
                trControl = trainControl(classProbs = T,
                                         summaryFunction = twoClassSummary,
                                         method = "LGOCV",
                                         returnResamp = "final",
                                         index = splits),
                metric = "ROC",
                tuneGrid = expand.grid(alpha = seq(0, 1, .2),
                                       lambda = seq(0, 1, .05)))

# Performance results
pca$resample %>%
  summarize(mean = mean(ROC),
            se = sqrt(var(ROC) * (1/J + (ntest/ntrain))),
            lower = mean - 1.96 * se,
            upper = mean + 1.96 * se) %>%
  round(2)

# Fitted model coefficients
coef(pca$finalModel, no_pca$bestTune$lambda)

# Fool around with the dependence of estimated performance on n, using glm()
split <- data.frame(prop = seq(.5, .95, .01),
                    ntrain = 0,
                    test_auc = 0,
                    train_auc = 0)

for(i in 1:nrow(split)){
  rows <- createDataPartition(raw$Type, p = split[,1][i], list = F)
  train <- raw[rows, ]
  test <- raw[-rows, ]
  
  split$ntrain[i] <- length(rows)
  
  m2 <- train(Type ~.,
        data = train,
        method = "glmnet")
  
  split$test_auc[i] <- as.numeric(auc(test$Type,
                predict(m2, newdata = test, type = "prob")[,2]))
  
  split$train_auc[i] <- as.numeric(auc(train$Type,
                                      predict(m2, newdata = train, type = "prob")[,2]))
 cat(i, "\r")
}

ggplot(split, aes(prop, test_auc)) +
  geom_point()+
  geom_smooth(method = "lm", se = F)

ggplot(split, aes(prop, train_auc)) +
  geom_point()+
  geom_smooth(method = "lm", se = F)





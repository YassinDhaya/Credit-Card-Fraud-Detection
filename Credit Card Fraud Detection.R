# importing the required libraries
library(dplyr)      # for data manipulation
library(ranger)     # for faster implementaion of random forests
library(caret)      # for classification and regression training
library(caTools)    # for splitting data into training and test set
library(data.table) # for converting data frame to table for faster execution
library(ggplot2)    # for basic plot
library(corrplot)   # for plotting corelation plot between elementslibrary(Rtsne)      # for plotting tsne model
library(ROSE)       # for rose sampling
library(pROC)       # for plotting ROC curve
library(rpart)      # for regression trees
library(rpart.plot) # for plotting decision tree
library(Rborist)    # for random forest model
library(xgboost)    # for xgboost model
#1 importing the dataset
dataset <- setDT(read.csv("data/creditcard.csv"))
#2 data exploration
# exploring the credit card data
head(dataset)
tail(dataset)
# view the table from class column (0 for legit transactions and 1 for fraud)
table(dataset$Class)
# view names of colums  of dataset
names(dataset)
# view summary of amount and histogram
summary(dataset$Amount)
hist(dataset$Amount)
hist(dataset$Amount[dataset$Amount < 100])
# view variance and standard deviation of amount column
var(dataset$Amount)
sd(dataset$Amount)
# check whether there are any missing values in colums
colSums(is.na(dataset))
#4 data visualization
# visualizing the distribution of transcations across time
dataset %>%
  ggplot(aes(x = Time, fill = factor(Class))) + 
  geom_histogram(bins = 100) + 
  labs(x = "Time elapsed since first transcation (seconds)", y = "no. of transactions", title = "Distribution of transactions across time") +
  facet_grid(Class ~ ., scales = 'free_y') + theme()
# correlation of anonymous variables with amount and class
correlation <- cor(dataset[, -1], method = "pearson")
corrplot(correlation, number.cex = 1, method = "color", type = "full", tl.cex=0.7, tl.col="black")
# only use 10% of data to compute SNE and perplexity to 20
tsne_data <- 1:as.integer(0.1*nrow(dataset))
tsne <- Rtsne(dataset[tsne_data,-c(1, 31)], perplexity = 20, theta = 0.5, pca = F, verbose = F, max_iter = 500, check_duplicates = F)
classes <- as.factor(dataset$Class[tsne_data])
tsne_matrix <- as.data.frame(tsne$Y)
ggplot(tsne_matrix, aes(x = V1, y = V2)) + geom_point(aes(color = classes)) + theme_minimal() + ggtitle("t-SNE visualisation of transactions") + scale_color_manual(values = c("#E69F00", "#56B4E9"))
#6 data processing
# scaling the data using standardization and remove the first column (time) from the data set
dataset$Amount <- scale(dataset$Amount)
new_data <- dataset[, -c(1)]
head(new_data)
# change 'Class' variable to factor
new_data$Class <- as.factor(new_data$Class)
levels(new_data$Class) <- c("Not Fraud", "Fraud")
#6 data modeling
# split the data into training set and test set
set.seed(101)
split <- sample.split(new_data$Class, SplitRatio = 0.8)
train_data <- subset(new_data, split == TRUE)
test_data <- subset(new_data, split == FALSE)
dim(train_data)
dim(test_data)
# visualize the training data
train_data %>% ggplot(aes(x = factor(Class), y = prop.table(stat(count)), fill = factor(Class))) +
  geom_bar(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  labs(x = 'Class', y = 'Percentage', title = 'Training Class distributions') +
  theme_grey()
#7 sampling techniques
#ROSE
set.seed(9560)
rose_train_data <- ROSE(Class ~ ., data  = train_data)$data 

table(rose_train_data$Class)
# up sampling
set.seed(90)
up_train_data <- upSample(x = train_data[, -30],
                          y = train_data$Class)
table(up_train_data$Class)  
# down sampling
set.seed(90)
down_train_data <- downSample(x = train_data[, -30],
                              y = train_data$Class)
table(down_train_data$Class) 
# 8 logistic regression model
# fitting the logistic model
logistic_model <- glm(Class ~ ., down_train_data, family='binomial')
summary(logistic_model)
plot(logistic_model)
#9 Plotting the ROC-AUC Curve
logistic_predictions <- predict(logistic_model, test_data, type='response')
roc.curve(test_data$Class, logistic_predictions, plotit = TRUE, col = "blue")
#10 decision tree model
decisionTree_model <- rpart(Class ~ . , down_train_data, method = 'class')
predicted_val <- predict(decisionTree_model, down_train_data, type = 'class')
probability <- predict(decisionTree_model, down_train_data, type = 'prob')
rpart.plot(decisionTree_model)
#11 Random forest model
x = down_train_data[, -30]
y = down_train_data[,30]

rf_fit <- Rborist(x, y, ntree = 1000, minNode = 20, maxLeaf = 13)


rf_pred <- predict(rf_fit, test_data[,-30], ctgCensus = "prob")
prob <- rf_pred$prob

roc.curve(test_data$Class, prob[,2], plotit = TRUE, col = 'blue')
#12 XGBoost model
set.seed(40)

#Convert class labels from factor to numeric
labels <- down_train_data$Class
y <- recode(labels, 'Not Fraud' = 0, "Fraud" = 1)

# xgb fit
xgb_fit <- xgboost(data = data.matrix(down_train_data[,-30]), 
                   label = y,
                   eta = 0.1,
                   gamma = 0.1,
                   max_depth = 10, 
                   nrounds = 300, 
                   objective = "binary:logistic",
                   colsample_bytree = 0.6,
                   verbose = 0,
                   nthread = 7
)
# XGBoost predictions
xgb_pred <- predict(xgb_fit, data.matrix(test_data[,-30]))
roc.curve(test_data$Class, xgb_pred, plotit = TRUE)
#13 Significant Variables
names <- dimnames(data.matrix(down_train_data[,-30]))[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb_fit)
# Nice graph
xgb.plot.importance(importance_matrix[1:10,])
#14 Conclusion: From the above plots and models, we can clarify that XGBoost performed better than logistic and Random Forest Model, although the margin was not very high. We can also fine tune the XGBoost model to make it perform even better. It is really great how models are able to find the distinguishing features between fraud and non-fraud transactions from such a big data.
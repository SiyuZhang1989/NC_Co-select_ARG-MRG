# RF for classifying samples

otu <- read.table('input.txt', sep = '\t', row.names = 1, header = TRUE, fill = TRUE)
head(otu)
#   MAS_AS_10	MAS_AS_11	MAS_AS_12	MAS_AS_13	MAS_AS_14
# Sphingobium	0.004757189	0.006168146	0.005516422	0.005703478	0.006023958
# Micrococcus	0.000454685	0.000259687	0.000140583	0.000160083	0.000176481
# Massilia	0.003384317	0.002464708	0.002437558	0.002563668	0.002550407
otu <- otu[which(rowSums(otu) >= 0.001), ]

group <- read.table('group.txt', sep = '\t', row.names = 1, header = TRUE, fill = TRUE)
head(group)
# samples	groups
# Sample1	OF
# Sample2	NOF
# Sample3	OF
# Sample4	NOF

otu <- data.frame(t(otu))
otu_group <- cbind(otu, group)

set.seed(123)
select_train <- sample(218, 218*0.7)
otu_train <- otu_group[select_train, ]
otu_test <- otu_group[-select_train, ]

# otu_train$groups = factor(otu_train$groups)
# otu_test$groups = factor(otu_test$groups)

library(randomForest)

set.seed(123)
otu_train.forest <- randomForest(groups ~ ., data = otu_train, importance = TRUE)
otu_train.forest

train_predict <- predict(otu_train.forest, otu_train)
compare_train <- table(train_predict, otu_train$groups)
compare_train
sum(diag(compare_train)/sum(compare_train))

test_predict <- predict(otu_train.forest, otu_test)
compare_test <- table(otu_test$groups, test_predict, dnn = c('Actual', 'Predicted'))
compare_test 

importance_otu <- data.frame(importance(otu_train.forest))
head(importance_otu)

importance_otu <- importance_otu[order(importance_otu$MeanDecreaseAccuracy, decreasing = TRUE), ]
head(importance_otu)

write.table(importance_otu, 'importance_otu.txt', sep = '\t', col.names = NA, quote = FALSE)

varImpPlot(otu_train.forest, n.var = min(30, nrow(otu_train.forest$importance)), main = 'Top 30 - variable importance')

set.seed(123)
otu_train.cv <- replicate(10, rfcv(otu_train[-ncol(otu_train)], otu_train$group, cv.fold = 10,step = 1.5), simplify = FALSE)
otu_train.cv

otu_train.cv <- data.frame(sapply(otu_train.cv, '[[', 'error.cv'))
otu_train.cv$otus <- rownames(otu_train.cv)
otu_train.cv <- reshape2::melt(otu_train.cv, id = 'otus')
otu_train.cv$otus <- as.numeric(as.character(otu_train.cv$otus))

library(ggplot2)
library(splines)

p <- ggplot(otu_train.cv, aes(otus, value)) +
  geom_smooth(se = FALSE,	method = 'glm', formula = y~ns(x, 6)) +
  theme(panel.grid = element_blank(), panel.background = element_rect(color = 'black', fill = 'transparent')) +  
  labs(title = '',x = 'Number of OTUs', y = 'Cross-validation error')
p

q <- p + geom_vline(xintercept = 140)
q

importance_otu <- importance_otu[order(importance_otu$MeanDecreaseAccuracy, decreasing = TRUE), ]
head(importance_otu)

otu_select <- rownames(importance_otu)[1:140]

otu_train_top140 <- otu_train[ ,c(otu_select, 'groups')]
otu_test_top140 <- otu_test[ ,c(otu_select, 'groups')]

set.seed(123)
otu_train.forest_140 <- randomForest(groups ~ ., data = otu_train_top140, ntree = 500, mtry = 2, importance = TRUE)
otu_train.forest_140

train_predict <- predict(otu_train.forest_140, otu_train_top140)
compare_train <- table(train_predict, otu_train_top140$groups)
compare_train

test_predict <- predict(otu_train.forest_140, otu_test_top140)
compare_test <- table(otu_test_top140$groups, test_predict, dnn = c('Actual', 'Predicted'))
compare_test

NFAS <- read.table('NFAS_top140.txt', sep = '\t', row.names = 1, header = TRUE, fill = TRUE)

NFAS_predict <- predict(otu_train.forest_140, NFAS)
table(NFAS_predict)

NFAS_predict_table <- data.frame(NFAS_predict)
NFAS_predict_table

write.table(NFAS_predict_table, 'NFAS_predict_table.txt', sep = '\t', col.names = NA, quote = FALSE)

# tuneRF
test.forest <- randomForest(groups ~ ., data = otu_train_top140, ntree = 500, mtry = 18, importance = TRUE)
test.forest
tuneRF(x = otu_train_top140[,-140], y = otu_train_top140$groups, mtryStart = 2, ntreeTry = 500,  stepFactor = 1.5, plot = T,  trace= T, doBest = T)

library(caret)

set.seed(1234)
trControl <- trainControl(method = "cv",number = 10,search = "grid")
rf_default<-train(groups ~ .,
                  data = otu_train_top140,
                  method="rf",
                  metric="Accuracy",
                  trControl=trControl)
rf_default$bestTune$mtry

tuneGrid <- expand.grid(.mtry = c(1: 20))
rf_mtry <- train(groups ~ .,
                 data = otu_train_top140,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 500)
rf_mtry$bestTune$mtry

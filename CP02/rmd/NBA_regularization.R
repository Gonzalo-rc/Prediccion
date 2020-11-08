##NBA REGULARIZATION
library(here) # Comentar
library(tidyverse)
library(janitor) # Clean names
library(skimr) # Beautiful Summarize
library(magrittr) # Pipe operators
library(corrplot) # Correlations
library(ggcorrplot)  # Correlations
library(PerformanceAnalytics) # Correlations
library(leaps) # Model selection
library(MASS)
library(dplyr)
library(readr)
library(gvlma)
library(MASS)
library(car)
library(glmnet)
library(boot)
library(leaps)
library(rsample)
##Carga de datos
raw_data <-  read.csv("../data/nba.csv")
colnames(raw_data)

raw_data %<>% clean_names()
colnames(raw_data)
# delete duplicate
# Remove duplicate rows of the dataframe
raw_data %<>% distinct(player,.keep_all= TRUE)

# delete NA's
raw_data %<>% drop_na() #buscamos el de menor std respecto a su media

log_data <- raw_data %>% mutate(salary=log(salary))
#Lo ideal es que haya relación con el salario pero poca con ass demás
skim(log_data)
# Excluded vars (factor)
vars <- c("player","nba_country","tm")
#Exclusion of categorical features.
nba <- log_data %>% select_at(vars(-vars))

set.seed(03112020)

# We are going to use the 70% of data for training
nba_split <- initial_split(log_data, prop = .7, strata = "salary")
#Training and test are defined
nba_tr <- training(nba_split)
nba_test  <- testing(nba_split)
# We eliminate the the intercept.
nba_tr_x <- model.matrix(salary ~ . -player -nba_country -tm, data = log_data)[, -1]
nba_tr_y <- log_data$salary
nba_test_x <- model.matrix(salary ~ . -player -nba_country -tm, data = log_data)[, -1]
nba_test_y <- log_data$salary

#Elastic net

# maintain the same folds across all models
fold_id <- sample(1:10, size = length(nba_tr_y), replace=TRUE)

# search across a range of alphas
tuning_grid <- tibble::tibble(
  alpha      = seq(0, 1, by = 0.05),
  mse_min    = NA,
  mse_1se    = NA,
  lambda_min = NA,
  lambda_1se = NA
)
tuning_grid

for(i in seq_along(tuning_grid$alpha)) {
  
  # fit CV model for each alpha value
  fit <- cv.glmnet(nba_tr_x, nba_tr_y, alpha = tuning_grid$alpha[i], foldid = fold_id)
  
  # extract MSE and lambda values
  tuning_grid$mse_min[i]    <- fit$cvm[fit$lambda == fit$lambda.min]
  tuning_grid$mse_1se[i]    <- fit$cvm[fit$lambda == fit$lambda.1se]
  tuning_grid$lambda_min[i] <- fit$lambda.min
  tuning_grid$lambda_1se[i] <- fit$lambda.1se
}
tuning_grid
#El que menor que menor lambda nos da es el alpha correspondiente a 1 que corresponde a lasso.
#Calculo del error asociado al lasso con el training
cv_lasso   <- cv.glmnet(nba_tr_x, nba_tr_y, alpha = 1.0)
min(cv_lasso$cvm)
#Prediccion y calculo del error con el dataset de test

pred <- predict(cv_lasso, s = cv_lasso$lambda.min, nba_test_x)
mean((nba_test_y - pred)^2) #el error de la prediccion es menor que 


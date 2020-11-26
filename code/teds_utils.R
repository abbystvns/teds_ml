library(tidyverse)
library(tidymodels)
library(reshape2)
library(gridExtra)


mean_response_plots <- function(teds, myvars, response) {
  fts = melt(teds, response, myvars)
  colnames(fts) = c(response, "feature", "value")

  fts <- fts %>% 
    group_by(feature, value) %>% 
    mutate(count = n()) %>%
    group_by(feature, value, count) %>%
    summarise_at(response, c('mean'=mean, 'sd'=sd))
  fts[, 'std_err'] = fts$sd / sqrt(fts$count)
  
  for(var in myvars) {
    var_fts = filter(fts, feature==var)
    var_fts[, "value"] = as.factor(var_fts$value)
    p = ggplot(var_fts, aes(x=value, y=mean, fill=value)) + 
      geom_bar(stat="identity", position = "dodge") +
      geom_errorbar(aes(ymin=mean-std_err, ymax=mean+std_err), width=.2,
                    position=position_dodge(.9)) +
      ggtitle(var)
    print(p)
  }
}


sbs_response_plots  <- function(teds, myvars, response) {
  fts = melt(teds, response, myvars)
  colnames(fts) = c(response, "feature", "value")
  fts <- fts %>% 
          group_by_at(.vars=c(response, "feature", "value")) %>% 
          summarise(cnt=n())
  fts[, "value"] = as.factor(fts$value)
  fts[,"outcome"] = as.factor(fts[[response]])
  
  
  for (var in myvars) {
    p = ggplot(filter(fts, feature==var), aes(x=value, y=cnt, fill=outcome)) + 
      geom_bar(stat="identity", position = "dodge")  +
      ggtitle(var)
    print(p)
  }
}

teds_rf <- function(teds, myvars, response){
  # RF function using default parameter values (no tuning)
  teds[,response] = as.factor(teds[[response]])
  set.seed(123) #randomization
  # build train/test sets
  teds_split = initial_split(teds, prop=3/4)
  teds_train <- training(teds_split)
  teds_test <- testing(teds_split)
  teds_cv = vfold_cv(teds_train, v=5) #5-fold cv for parameter tuning
  
  # define the recipe
  fm = as.formula(paste(response, "~ ."))
  teds_recipe <- 
    # which consists of the formula (outcome ~ predictors)
    recipe(fm, 
           data = teds)
  
  rf_model <- 
    # specify that the model is a random forest
    rand_forest() %>%
    # select the engine/package that underlies the model
    set_engine("ranger", importance = "impurity") %>%
    # choose either the continuous regression or binary classification mode
    set_mode("classification") 
  
  
  rf_workflow <- workflow() %>%
    # add the recipe
    add_recipe(teds_recipe) %>%
    # add the model
    add_model(rf_model)
  
  rf_fit <- rf_workflow %>%
    # fit on the training set and evaluate on test set
    last_fit(teds_split)
  
  test_performance <- rf_fit %>% collect_metrics()
  test_predictions <- rf_fit %>% collect_predictions()
  
  return(list("model"=rf_workflow, "test_performance"=test_performance, "test_predictions"=test_predictions))
}


teds_rf_cv <- function(teds, myvars, response, rf_grid) {
  # make sure response is a factor variable
  teds[,response] = as.factor(teds[[response]])
  set.seed(123) #randomization
  # build train/test sets
  teds_split = initial_split(teds, prop=3/4)
  teds_train <- training(teds_split)
  teds_test <- testing(teds_split)
  teds_cv = vfold_cv(teds_train, v=5) #5-fold cv for parameter tuning
  
  # define the recipe
  fm = as.formula(paste(response, "~ ."))
  teds_recipe <- 
    # which consists of the formula (outcome ~ predictors)
    recipe(fm, 
           data = teds)
  
  rf_model <- 
    # specify that the model is a random forest
    rand_forest() %>%
    # specify that the `mtry` parameter needs to be tuned
    set_args(mtry = tune(),
             trees = tune()) %>%
    # select the engine/package that underlies the model
    set_engine("ranger", importance = "impurity") %>%
    # choose either the continuous regression or binary classification mode
    set_mode("classification") 
  
  rf_workflow <- workflow() %>%
    # add the recipe
    add_recipe(teds_recipe) %>%
    # add the model
    add_model(rf_model)
  
  # extract results
  rf_tune_results <- rf_workflow %>%
    tune_grid(resamples = teds_cv, #CV object
              grid = rf_grid, # grid of values to try
              metrics = metric_set(accuracy, roc_auc) # metrics we care about
    )
  
  rf_tune_results %>%
    collect_metrics()
  
  param_final <- rf_tune_results %>%
    select_best(metric = "roc_auc")
  
  rf_workflow <- rf_workflow %>%
    finalize_workflow(param_final)
  
  rf_fit <- rf_workflow %>%
    # fit on the training set and evaluate on test set
    last_fit(teds_split)
  
  test_performance <- rf_fit %>% collect_metrics()
  test_predictions <- rf_fit %>% collect_predictions()
  
  # fit tuned model to entire dataset
  #final_model <- fit(rf_workflow, teds)
  return(list("model"=rf_workflow, "test_performance"=test_performance, "test_predictions"=test_predictions, "cv_results"=rf_tune_results))
} 
library(tidyverse)
library(tidymodels)
library(workflowsets)
library(mlbench) #demo data is here
library(DescTools) # for some combinatorics functions.
library(doParallel) #make the training run much faster

#convenience function to insert missing values
miss_vals <- function(x,p=.05) {x[sample(1:length(x), floor(p*length(x)))] <- NA; x}

#dataset is mlbench::BreastCancer with some missing values in Bare.nuclei and Epith.c.size columns
data(BreastCancer)
dataset <- BreastCancer %>%
  dplyr::select(-Id) %>%
  mutate(across(-Class,as.numeric)) %>%
  mutate(across(c(Bare.nuclei, Epith.c.size),miss_vals)) #%>% #add missing vals
  #na.omit() #want to run the code without this line so that rows are only omitted
# if the model uses the column that has a missing value in that row

#specify logistic regression model
lr_spec <- logistic_reg() %>% set_engine("glm")

#specify rf model
rf_spec <-
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#test train split
set.seed(1)
trn_tst_split <- initial_split(dataset, strata = Class)
folds <- vfold_cv(training(trn_tst_split), v=10, strata = Class)

# function to generate all possible combinations of formulas with n variables added to a base formula
add_n_var_formulas <- function(y_var, x_vars, data, n = 2, include_base = T, include_full = T){

  potential_x_vars <-
    names(data) %>% #variable names
    {.[!. %in% c(y_var, x_vars)]} #remove y and existing x vars

  combos <-
    potential_x_vars %>%
    DescTools::CombSet(n, repl=FALSE, ord=FALSE) %>%
    as_tibble() %>%
    unite(col = combination, sep = " + ") %>%
    pull(combination, name = combination)

  if (include_base) {combos[length(combos)+1]<-"1"}
  if (include_full) {combos[length(combos)+1]<-"."}

  base_pred <- paste(x_vars, collapse = " + ")

  new_formulas <-
    combos %>%
    {paste(y_var, " ~ ",paste(base_pred,., sep = " + "))} %>%  #make formula strings
    purrr::map(as.formula) %>% #make formula
    purrr::map(workflowsets:::rm_formula_env)

  names(new_formulas) <- combos

  if(include_base){ names(new_formulas)[names(new_formulas) == "1"] <- "Base Model" }
  if(include_full){names(new_formulas)[names(new_formulas) == "."] <- "All Predictors"}

  new_formulas
}

#make 10 possible formulas to predict `Class` from `Normal.nucleoli` + (something else)
formulas <- add_n_var_formulas(
  y_var = "Class",
  x_vars = c("Normal.nucleoli"),
  data = dataset,
  n=1)

# Create a recipe from each formula
recipes <-
  map(formulas, function(form) {
    recipe(form, data = training(trn_tst_split)) %>%
      step_impute_median(all_predictors())
    #step_naomit(all_predictors(),skip = TRUE)
  })

# Create workflow set
cancer_workflows <-
  workflow_set(
    preproc = recipes,
    models = list(rf = rf_spec,
                  lr = lr_spec)
  )

# Create hyperparameter tuning grid
grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

#Set up for parallel processing
all_cores <- parallel::detectCores(logical = TRUE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

# Fit models
cancer_workflows <-
  cancer_workflows %>%
  workflow_map("tune_grid",
               seed = 1503,
               grid = 25,
               control = grid_ctrl,
               resamples = folds,
               verbose = TRUE)

stopImplicitCluster()

# Plot results
cancer_workflows %>%
  collect_metrics(summarize = FALSE) %>%
  filter(.metric == "roc_auc") %>%
  group_by(wflow_id, model) %>%
  dplyr::summarize(
    ROC = mean(.estimate),
    lower = quantile(.estimate,probs = 0.2),
    upper = quantile(.estimate,probs = 0.8),
    .groups = "drop"
  ) %>%
  mutate(wflow_id = factor(wflow_id),
         wflow_id = reorder(wflow_id, ROC)) %>%
  ggplot(aes(x = ROC, y = wflow_id)) +
  geom_point() +
  geom_errorbar(aes(xmin = lower, xmax = upper), width = .25) +
  labs(title = "Comparing models of the form `Class ~ Normal.nucleoli + ...`",y = "Additional variable")

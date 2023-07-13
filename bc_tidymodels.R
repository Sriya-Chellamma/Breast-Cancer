setwd("C:/Users/Ramachandran/Desktop/Tableau Docs- BBL/Breastcancer")
bc=read.csv("breast.csv")
str(bc)
library(tidymodels)
library(glmnet)
summary(bc)
library(ggplot2)
bc1=initial_split(bc,prop=.70,strata = "diagnosis")
trainbc=training(bc1)
testbc=testing(bc1)
table(trainbc$diagnosis)
table(testbc$diagnosis)
str(testbc)
148/(249+148)
64/(108+64)
trainbc$diagnosis=as.factor(trainbc$diagnosis)
testbc$diagnosis =as.factor(testbc$diagnosis)
bc$diagnosis=as.factor(bc$diagnosis)
myFolds = vfold_cv(trainbc, v = 5,strata = diagnosis)
traincaret = rsample2caret(myFolds)
bcrecip = trainbc %>%
  recipe(diagnosis ~ .) %>%
  # normalize all numeric predictors
  step_normalize(all_numeric()) %>%
  # create dummy variables 
  step_dummy(all_nominal(), - all_outcomes()) %>%
  # remove zero variance predictors
  step_nzv(all_predictors(), - all_outcomes())

logspecbc=logistic_reg(penalty = tune(),mixture = tune()) %>% set_engine("glmnet")

xgbspecbc=boost_tree(mtry = tune(),sample_size = tune(),tree_depth = tune(),
                     trees = 500,learn_rate = tune(),loss_reduction = tune(),
                     min_n = tune()) %>% set_mode("classification") %>%
  set_engine("xgboost")

glmnet_paramsbc=parameters(list(penalty(),mixture()))

xgbparamsbc=parameters(list(min_n(),tree_depth(),learn_rate(),loss_reduction(),
                    sample_size=sample_prop(),finalize(mtry(),trainbc)))


glmnetgridbc=grid_latin_hypercube(glmnet_paramsbc,size = 2)

xgbtreegridbc=grid_latin_hypercube(xgbparamsbc,size = 10)


mymodelsbc=workflow_set(preproc=list(bcrecip),
                        models = list(glmnet=logspecbc,xgbTree=xgbspecbc),
  cross = TRUE) %>% option_add(grid=xgbtreegridbc,id="recipe_xgbTree") %>% 
  option_add(grid=glmnetgridbc,id="recipe_glmnet")

mymodelsbc

bcmetrics=metric_set(bal_accuracy,roc_auc,sensitivity,specificity,precision,f_meas)


str(trainbc)

bcrace=mymodelsbc %>% workflow_map("tune_grid",resamples=myFolds,verbose=TRUE,
                                   control=control_grid(verbose=TRUE),
                                   metrics=bcmetrics)
bcrace %>% collect_metrics(metrics=bcmetrics) %>% group_by(wflow_id)

bcrace %>% collect_metrics(metrics=bcmetrics)

autoplot(bcrace)

results = bcrace %>% extract_workflow_set_result("recipe_glmnet")

bestresults = results %>% select_best(metric = "f_meas")

glmnetwkfl = bcrace %>% extract_workflow("recipe_glmnet") %>% 
  finalize_workflow(bestresults)

glmnetresults = glmnetwkfl %>%
  fit_resamples(resamples = myFolds, 
                metrics = bcmetrics,
                control = control_resamples(save_pred = TRUE))

collect_metrics(glmnetresults)

##Clarify below queries
glmnetfinal = glmnetwkfl %>% last_fit(split = bc1, metrics = bcmetrics)
glmnetfit = fit(glmnetwkfl, trainbc)
glmnettrainpred = predict(glmnetfit, trainbc) %>%
  bind_cols(trainbc[2]) 
conf_mat(glmnettrainpred,
         truth = diagnosis,
         estimate = .pred_class)
conf_mat(glmnetfinal$.predictions[[1]],
         truth = Attrition,
         estimate = .pred_class)

glmnet_final = glmnet_wkfl %>%
  last_fit(split = bc1, metrics = ibm_metrics) 

data.frame(glmnetfinal$.predictions) %>%
  ggplot() +
  geom_density(aes(x = .pred_Yes, fill = Attrition),
               alpha = 0.5)+
  geom_vline(xintercept = 0.5,linetype = "dashed")+
  ggtitle("Predicted class probabilities coloured by attrition")+
  theme_bw()

data.frame(glmnet_final$.predictions) %>% 
  roc_curve(truth = Attrition, .pred_Yes) %>% 
  autoplot()

model1=glm(bc$diagnosis~.,family = "binomial",bc)
str(bc)
bc=bc[,-c(1)]
str(bc)
model1=glm(bc$diagnosis~.,family = "binomial",bc)
summary(model1)
pr1=predict(model1,type="response")
pr1
pr2=ifelse(pr1>0.7,1,0)
table(pr2)
table(pr2,bc$diagnosis)

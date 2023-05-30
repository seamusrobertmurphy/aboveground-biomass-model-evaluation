
#### MSc Inventory Monitoring and Assessment ####
######  Assignment 3; LIDAR AGB Predict #########
############ S.Murphy 19-04-2020 ################

#using k-fold methods to estimate accuracy of model based on following:
#https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/
#library(caret)
#library(klaR)

library(readxl)
ClocaenogField <- read_excel("ClocaenogField.xlsx")
View(ClocaenogField)

#Explore data and normality of predictor variables:
describe(ClocaenogField$MEANH)
describe(ClocaenogField$COVER)
describe(ClocaenogField$SD)
shapiro.test(ClocaenogField$MEANH)
shapiro.test(ClocaenogField$COVER)
shapiro.test(ClocaenogField$SD)

#MODEL 1 (LM1): Linear regression
model_lm1 <- lm(BIOMASS ~ MEANH + COVER + SD, data = ClocaenogField)
summary (model_lm1)

#Run model diagnostics and check linear assumptions
ols_test_normality(model_lm1)
ols_test_breusch_pagan(model_lm1)
dwtest(model_lm1)
ols_coll_diag(model_lm1)
outlierTest(model_lm1, data=Duncan)
ols_test_outlier(model_lm1)
ols_vif_tol(model_lm1)
autoplot(model_lm1)
ols_plot_diagnostics(model_lm1)
ols_plot_cooksd_bar(model_lm1)
ols_plot_dfbetas(model_lm1)
ols_plot_resid_lev(model_lm1)
ols_plot_comp_plus_resid(model_lm1, print_plot = TRUE)

#Bonferroni results significant and influential outliers found.
#Create new dataframe removing outliers and re-run linear model.
#Check again for violations and compare model 1 & 2 with non-linear models.

#remove outliers from rows 9 and 18
ClocaenogField_cleaned <- ClocaenogField[-c(9,18),]

#MODEL 2 (LM_Cleaned): Linear regression using cleaned dataset
model_lm2 <- lm(BIOMASS ~ MEANH + COVER + SD, data = ClocaenogField_cleaned)
summary (model_lm2)

#run model diagnostics again
ols_test_normality(model_lm2)
ols_test_breusch_pagan(model_lm2)
dwtest(model_lm2)
ols_coll_diag(model_lm2)
outlierTest(model_lm2, data=Duncan)
ols_test_outlier(model_lm2)
ols_vif_tol(model_lm2)
autoplot(model_lm2)
ols_plot_diagnostics(model_lm2)
ols_plot_cooksd_bar(model_lm2)
ols_plot_dfbetas(model_lm2)
ols_plot_resid_lev(model_lm2)
ols_plot_comp_plus_resid(model_lm2, print_plot = TRUE)
#same violations found from bonferroni results and variance inflation factors
#resort to stepwise, non-linear and robust regression and compare...

#MODEL 3 (LM-STEP-Full): Stepwise Linear regression using full dataset
# Step-wise backwards 10-fold cross-validation test of model accuracy based on RMSE
# Set seed for reproducibility
# Set up repeated k-fold cross-validation
train.control_lm1 <- trainControl(method = "cv", number = 10)
# Train the model
step.model_lm1 <- train(BIOMASS ~ MEANH + COVER + SD, data = ClocaenogField, method = "leapBackward", tuneGrid = data.frame(nvmax = 1:3), trControl = train.control_lm1)
step.model_lm1$results
summary(step.model_lm1$finalModel)
# Results suggest best 2-variable model contain MEANH and SD
# Step-wise model using most best 2-variables: MEANH and SD
model_lm_step_full <- lm(BIOMASS ~ MEANH + SD, data = ClocaenogField)
summary(model_lm_step_full)

#run diagnostics
ols_test_normality(model_lm_step_full)
ols_test_breusch_pagan(model_lm_step_full)
dwtest(model_lm_step_full)
ols_coll_diag(model_lm_step_full)
outlierTest(model_lm_step_full, data=Duncan)
ols_test_outlier(model_lm_step_full)
ols_vif_tol(model_lm_step_full)
autoplot(model_lm_step_full)
ols_plot_diagnostics(model_lm_step_full)
ols_plot_cooksd_bar(model_lm_step_full)
ols_plot_dfbetas(model_lm_step_full)
ols_plot_resid_lev(model_lm_step_full)
ols_plot_comp_plus_resid(model_lm_step_full, print_plot = TRUE)

# MODEL 4 (NLS) Non-linear least squares regression model using full dataset
# Non-linear least squares regression
model_nls1 <- lm(ClocaenogField$BIOMASS ~ ClocaenogField$MEANH + ClocaenogField$COVER + ClocaenogField$SD + I((ClocaenogField$MEANH + ClocaenogField$COVER + ClocaenogField$SD)^2))
summary(model_nls1)

# MODEL 5 (NLS) Non-linear least squares regression model using cleaned dataset
# Non-linear least squares regression
model_nls1_cleaned <- lm(ClocaenogField_cleaned$BIOMASS ~ ClocaenogField_cleaned$MEANH + ClocaenogField_cleaned$COVER + ClocaenogField_cleaned$SD + I((ClocaenogField_cleaned$MEANH + ClocaenogField_cleaned$COVER + ClocaenogField_cleaned$SD)^2))
summary(model_nls1_cleaned)

# MODEL 6 (RLM.Huber) Robust regression using the Huber M-estimator to reduce outlier influence. Though this does not reduce outliers in predictor variables, so Tukey needed below
rr.huber <- rlm(BIOMASS ~ MEANH + COVER + SD, data = ClocaenogField)
summary(rr.huber)
f.robftest(rr.huber,var = "MEANH")
f.robftest(rr.huber, var = "SD")
f.robftest(rr.huber, var = "COVER")


# MODEL 7 (RLM.Tukey) Robust regression using the Tukey M-estimator that assigns a weight of zero to influential outliers
rr.bisquare <- rlm(BIOMASS ~ MEANH + COVER + SD, data = ClocaenogField, psi = psi.bisquare)
summary(rr.bisquare)
f.robftest(rr.bisquare,var = "MEANH")
f.robftest(rr.bisquare, var = "SD")
f.robftest(rr.bisquare, var = "COVER")

# MODEL 8 (RLM.Tukey) Robust regression using the Tukey M-estimator that assigns a weight of zero to influential outliers
rr.hampel <- rlm(BIOMASS ~ MEANH + COVER + SD, data = ClocaenogField, psi = psi.hampel)
summary(rr.hampel)
f.robftest(rr.hampel,var = "MEANH")
f.robftest(rr.hampel, var = "SD")
f.robftest(rr.hampel, var = "COVER")


# Explore altnerative accuracy assessments:
# sidak p value adjustment
ols_test_breusch_pagan(model_nls1, rhs = TRUE, multiple = TRUE, p.adj = 'sidak')
# holm's p value adjustment
ols_test_breusch_pagan(model_nls1, rhs = TRUE, multiple = TRUE, p.adj = 'holm')
# Global test of model assumptions
library(gvlma)
gvmodel <- gvlma(model_lm1)
summary(gvmodel)
gvmodel_del <- deletion.gvlma(gvmodel)
summary(gvmodel_del)
plot(gvmodel_del)
display.delstats
summary.gvlmaDel 
summary(gvmodel_del, allstats = FALSE)


gvmodel_lm2 <- gvlma(model_lm2)
summary(gvmodel_lm2)
gvmodel_del_lm2 <- deletion.gvlma(gvmodel_lm2)
summary(gvmodel_del_lm2)
plot(gvmodel_del_lm2)
display.delstats
summary.gvlmaDel 
summary(gvmodel_del_lm2, allstats = FALSE)

gvmodel_lm1 <- gvlma(model_lm1_kfold)
summary(model_lm1_kfold)
gvmodel_del_lm2 <- deletion.gvlma(model_lm1_kfold)
summary(gvmodel_del_lm2)
plot(gvmodel_del_lm2)
display.delstats
summary.gvlmaDel 
summary(gvmodel_del_lm2, allstats = FALSE)

#compare models
ols_mallows_cp(model_lm1, model_lm2)
ols_fpe(model_lm1)
ols_hsp(model_lm1)

ols_mallows_cp(model_lm1, model_lm2)
ols_mallows_cp(model_lm1, model_lm_step_full)
ols_mallows_cp(model_lm1, model_nls1)
ols_mallows_cp(model_lm1, model_nls1_cleaned)
ols_mallows_cp(model_lm1, rr.huber)
ols_mallows_cp(model_lm1, rr.bisquare)
ols_mallows_cp(model_lm1, rr.hampel)

AIC(model_lm1)
AIC(model_lm2)
AIC(model_lm_step_full)
AIC(model_nls1)
AIC(model_nls1_cleaned)
AIC(rr.huber)
AIC(rr.bisquare)
AIC(rr.hampel)

BIC(model_lm1)
BIC(model_lm2)
BIC(model_lm_step_full)
BIC(model_nls1)
BIC(model_nls1_cleaned)
BIC(rr.huber)
BIC(rr.bisquare)
BIC(rr.hampel)

glance(model_lm1)
glance(model_lm2)
glance(model_lm_step_full)
glance(model_nls1)
glance(model_nls1_cleaned)
glance(rr.huber)
glance(rr.bisquare)
glance(rr.hampel)

#Explore residual diagnostics
ols_plot_resid_box
ols_plot_resid_fit
ols_plot_resid_hist
ols_plot_resid_qq


#big outliers so choose to run a robust regression using MASS pkg: 
#from https://stats.idre.ucla.edu/r/dae/robust-regression/
#Robust regression is iterated re-weighted least squares (IRLS). The 
#command is rlm in the MASS package. There are several
#weighting functions that can be used for IRLS. We 
#first use the Huber weights and then bi-square weighting. We 
#will then look at the final weights created by the IRLS process. 

#robust methods due to variance of resids and measures of influence:
#from https://stats.idre.ucla.edu/r/dae/robust-regression/
#We can see that the weight given to Mississippi 
#is dramatically lower using the bisquare weighting 
#function than the Huber weighting function and 
#the parameter estimates from these two different 
#weighting methods differ. When comparing the 
#results of a regular OLS regression and a robust 
#regression, if the results are very different, 
#you will most likely want to use the results 
#from the robust regression. Large differences 
#suggest that the model parameters are being 
#highly influenced by outliers. Different functions 
#have advantages and drawbacks. Huber weights can 
#have difficulties with severe outliers, and 
#bisquare weights can have difficulties converging 
#or may yield multiple solutions.

#(robust) sandwich variance estimator for linear regression:
#from: https://thestatsgeek.com/2014/02/14/the-robust-sandwich-variance-estimator-for-linear-regression-using-r/
#This method allowed us to estimate valid standard errors 
#for our coefficients in linear regression, without requiring 
#the usual assumption that the residual errors have constant variance.


# 10 k-fold cross validation
# define training control
train_control_kfold <- trainControl(method="cv", number=10)
metric <- "Accuracy"
# fix the parameters of the algorithm
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
# train the model
model_lm1_kfold <- train(BIOMASS ~ MEANH + SD + COVER, data=ClocaenogField, trControl=train_control_kfold)
# summarize results
print(model_lm1_kfold)
View(model_lm1_kfold)
summary(model_lm1_kfold)

set.seed(7)
accuracy_model_lm1 <- train(BIOMASS ~ MEANH + SD + COVER, data=ClocaenogField, metric=metric, trControl=train_control_kfold)


# repeated 10 k-fold cross validation
# define training control
train_control_kfold_repeat <- trainControl(method="repeatedcv", number=10, repeats=3)
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
# train the model
model_lm1_kfold_repeat <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl = train_control_kfold_repeat)
model_lm2_kfold_repeat <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField_cleaned, trControl=train_control_kfold_repeat)
model_lm_step_full_kfold_repeat <- train(BIOMASS ~ MEANH + SD, data = ClocaenogField, trControl=train_control_kfold_repeat)
model_nls1_kfold_repeat <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_kfold_repeat)
model_nls1_cleaned_kfold_repeat <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField_cleaned, trControl=train_control_kfold_repeat)
rr.huber_kfold_repeat <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_kfold_repeat)
rr.bisquare_kfold_repeat <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_kfold_repeat)
rr.hampel_kfold_repeat <-train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_kfold_repeat)

print(model_lm1_kfold_repeat)
print(model_lm2_kfold_repeat)
print(model_lm_step_full_kfold_repeat)
print(model_nls1_kfold_repeat)
print(model_nls1_cleaned_kfold_repeat)
print(rr.huber_kfold_repeat)
print(rr.bisquare_kfold_repeat)
print(rr.hampel_kfold_repeat)

# Leave-one-out cross validation
# define training control
train_control_LOOVC <- trainControl(method="LOOCV")
# train the model
model_lm1_LOOVC <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_LOOVC)
model_lm2_LOOVC <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField_cleaned, trControl=train_control_LOOVC)
model_lm_step_full_LOOVC <- train(BIOMASS ~ MEANH + SD, data = ClocaenogField, trControl=train_control_LOOVC)
model_nls1_LOOVC <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_LOOVC)
model_nls1_cleaned_LOOVC <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField_cleaned, trControl=train_control_LOOVC)
rr.huber_LOOVC <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_LOOVC)
rr.bisquare_LOOVC <- train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_LOOVC)
rr.hampel_LOOVC <-train(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField, trControl=train_control_LOOVC)

print(model_lm1_LOOVC)
summary(model_lm1_LOOVC)
View(model_lm1_LOOVC)

print(model_lm2_LOOVC)
print(model_lm_step_full_LOOVC)
print(model_nls1_LOOVC)
print(model_nls1_cleaned_LOOVC)
print(rr.huber_LOOVC)
print(rr.bisquare_LOOVC)
print(rr.hampel_LOOVC)

#run diagnostics
shapiro.test(resid(model_lm1_kfold_repeat))
shapiro.test(resid(model_lm2_kfold_repeat))
shapiro.test(resid(model_lm_step_full_kfold_repeat))
shapiro.test(resid(model_nls1_kfold_repeat))
shapiro.test(resid(model_nls1_cleaned_kfold_repeat))
shapiro.test(resid(rr.huber_kfold_repeat))
shapiro.test(resid(rr.bisquare_kfold_repeat))
shapiro.test(resid(rr.hampel_kfold_repeat))

bptest(model_lm1_kfold_repeat)
bptest(model_lm2_kfold_repeat)
bptest(model_lm_step_full_kfold_repeat)
bptest(model_nls1_kfold_repeat)
bptest(model_nls1_cleaned_kfold_repeat)
bptest(rr.huber_kfold_repeat)
bptest(rr.bisquare_kfold_repeat)
bptest(rr.hampel_kfold_repeat)

dwtest(rr.huber)
dwtest(rr.bisquare)
dwtest(rr.hampel)

outlierTest(lm(rr.huber))
outlierTest(lm(rr.bisquare))
outlierTest(lm(rr.hampel))

vif(rr.huber)
vif(rr.bisquare)
vif(rr.hampel)

AIC(model_lm1)
AIC(model_lm2)
AIC(model_lm_step_full)
AIC(model_nls1)
AIC(model_nls1_cleaned)
AIC(rr.huber)
AIC(rr.bisquare)
AIC(rr.hampel)


BIC(model_lm1)
BIC(model_lm2)
BIC(model_lm_step_full)
BIC(model_nls1)
BIC(model_nls1_cleaned)
BIC(rr.huber)
BIC(rr.bisquare)
BIC(rr.hampel)

TheilU(ClocaenogField$BIOMASS, model_lm1, type=1)
theil.wtd(model_lm1, weights = NULL)


bias_training <- rnorm(40, 2, sd = 0.5)
bias(bias_training, ClocaenogField)
bias(samp, pop)
bias(samp, pop, type = 'relative')
bias(samp, pop, type = 'standardized')
dev <- samp - pop
bias(dev)
nom.uncertainty(model_lm1)
lambda3(model_nls1)
CronbachAlpha(model_lm1)
guttman(model_lm1, missing = "complete", standardize = FALSE)


loq(model_lm1)



ols_plot_response(model_lm1)

u_theil_train <- createDataPartition(ClocaenogField$BIOMASS, p=0.5, list = FALSE)
u_theil_trainingData <- ClocaenogField[u_theil_train,]

u_theil_knn_model = train(model_lm1, data = u_theil_train, method = "knn", trControl = trainControl(method = "cv", number = 5), tuneGrid = expand.grid((k = seq(1, 21, by = 2)))
u_theil_knn_model$modelType
summary(u_theil_knn_model)

u_theil_testData <- ClocaenogField[-u_theil_train,]
TheilU(actual_biomass, lm1_residuals)


dim(u_theil_trainingData)

lm1_residuals <-data.frame(lm=model_lm1$residuals)
training_sample <- rnorm (100, 2, sd = 0.5)
dev <- training_sample - ClocaenogField
bias(dev)
bias(mean(training_sample), ClocaenogField)
percent_bias(model_lm1, ClocaenogField)

lod(model_lm1, data = ClocaenogField)

summary(model_lm1)
computeBoundary(b1=2.8, b0=3.3, p=c(.5, .75))

m.nn_lm1	<- matchit(BIOMASS ~ MEANH + SD + COVER, data = ClocaenogField,	method= "nearest",	ratio	= 1)
summary(m.nn)


ape(actual_biomass, lm1_residuals)


MAE(model_lm1)
MAE(model_lm2)
MAE(model_lm_step_full)
MAE(model_nls1)
MAE(model_nls1_cleaned)
MAE(rr.bisquare)
MAE(rr.bisquare)
MAE(rr.bisquare)

coxtest(model_lm1, model_lm1_kfold_repeat, data = ClocaenogField)

rsq.kl(model_lm1_kfold_repeat)
rsq.n(model_lm1_kfold_repeat)
rsq.partial(model_lm1_kfold_repeat)
rsq.sse(model_lm1_kfold_repeat)
rsq.v(model_lm1_kfold_repeat)

AIC(model_lm2)
AIC(model_lm_step_full)
AIC(model_nls1)
AIC(model_nls1_cleaned)
AIC(rr.huber)
AIC(rr.bisquare)
AIC(rr.hampel)


actual_biomass <- data.frame(ClocaenogField$BIOMASS)
actual_biomass_cleaned <- data.frame(ClocaenogField_cleaned$BIOMASS)


kfoldrepeat_residuals_lm1 <-data.frame(lm=model_lm1_kfold_repeat$residuals)
kfoldrepeat_residuals_lm2 <-data.frame(lm=model_lm2_kfold_repeat$residuals)
kfoldrepeat_residuals_lmstep <-data.frame(lm=model_lm_step_full_kfold_repeat$residuals)
kfoldrepeat_residuals_nls <-data.frame(lm=model_nls1_LOOVC$residuals)
kfoldrepeat_residuals_nlsclean <-data.frame(lm=model_nls1_cleaned_LOOVC$residuals)
kfoldrepeat_residuals_rrhuber <-data.frame(lm=rr.huber_LOOVC$residuals)
kfoldrepeat_residuals_rrbisq <-data.frame(lm=rr.bisquare_LOOVC$residuals)
kfoldrepeat_residuals_rrhampel <-data.frame(lm=rr.hampel_LOOVC$residuals)

lmFull_residuals <-data.frame(lm=model_lm1$residuals)
lmClean_residuals <-data.frame(lm=model_lm2$residuals)
lmStep_residuals <-data.frame(lm=model_lm_step_full$residuals)
nlsFull_residuals <-data.frame(lm=model_nls1$residuals)
nlsClean_residuals <-data.frame(lm=model_nls1_cleaned$residuals)
rrHuber_residuals <-data.frame(lm=rr.huber$residuals)
rrTukey_residuals <-data.frame(lm=rr.bisquare$residuals)
rrHampel_residuals <-data.frame(lm=rr.hampel$residuals)


library(DescTools)
TheilU(actual_biomass, lmFull_residuals)
TheilU(actual_biomass_cleaned, lmClean_residuals)
TheilU(actual_biomass, lmStep_residuals)
TheilU(actual_biomass, nlsFull_residuals)
TheilU(actual_biomass_cleaned, nlsClean_residuals)
TheilU(actual_biomass, rrHuber_residuals)
TheilU(actual_biomass, rrTukey_residuals)
TheilU(actual_biomass, rrHampel_residuals)

TheilU(actual_biomass, kfoldrepeat_residuals_lm1)
TheilU(actual_biomass_cleaned, kfoldrepeat_residuals_lm2)
TheilU(actual_biomass, kfoldrepeat_residuals_lmstep)
TheilU(actual_biomass, kfoldrepeat_residuals_nls)
TheilU(actual_biomass_cleaned, kfoldrepeat_residuals_nlsclean)
TheilU(actual_biomass, kfoldrepeat_residuals_rrhuber)
TheilU(actual_biomass, kfoldrepeat_residuals_rrbisq)
TheilU(actual_biomass, kfoldrepeat_residuals_rrhampel)

shapiro.test(resid(model_lm1))
shapiro.test(resid(model_lm2_kfold_repeat))
shapiro.test(resid(model_lm_step_full_kfold_repeat))
shapiro.test(resid(model_nls1_LOOVC))
shapiro.test(resid(model_nls1_cleaned_LOOVC))
shapiro.test(resid(rr.huber_LOOVC))
shapiro.test(resid(rr.bisquare_LOOVC))
shapiro.test(resid(rr.hampel_LOOVC))

bptest(model_lm1)
bptest(model_lm2_kfold_repeat)
bptest(resid(model_lm_step_full_kfold_repeat))
bptest(resid(model_nls1_LOOVC))
bptest(resid(model_nls1_cleaned_LOOVC))
bptest(resid(rr.huber_LOOVC))
bptest(resid(rr.bisquare_LOOVC))
bptest(resid(rr.hampel_LOOVC))


#model diagnostics:
#check for trends 
#check for equal distribution of predictors
plot (ClocaenogField$MEANH, resid (model_lm1))
abline (0,0)
plot (ClocaenogField$MEANH, resid (model_lm1))
abline (0,0)


plot (ClocaenogField$COVER, resid (model2))
abline (0,0)
plot (ClocaenogField$SD, resid (model2))
abline (0,0)



PlotCandlestick(model_lm1)

#homoskedasticity
#predicted values aginst fitted
PlotQQ(model_lm1)
autoplot(model_lm1_kfold_repeat, label.size = 3)

plot(fitted(model_lm1_kfold_repeat), resid(model_lm1_kfold_repeat))
abline(a = coef(model_lm1_kfold_repeat), b = 0, col="green")
plot(fitted(model_nls1_LOOVC), resid(model_nls1_LOOVC))
lines(abline(a = coef(model_nls1_LOOVC), b = 0, col="red"))
plot(fitted(model_nls1_cleaned_LOOVC), resid(model_nls1_cleaned_LOOVC))
lines(abline(a = coef(model_nls1_cleaned_LOOVC), b = 0, col="orange"))
plot(fitted(rr.bisquare_LOOVC), resid(rr.bisquare_LOOVC))
lines(abline(a = coef(rr.bisquare_LOOVC), b = 0, col="yellow"))


plot(fitted(model_lm2_kfold_repeat), resid(model_lm2_kfold_repeat))
lines(abline(a = coef(model_lm2_kfold_repeat), b = 0, col="blue"))

legend(46, 15, legend = c("model1: linear", "model2: poly x^2", "model3: poly x^2 + x^3"),
       col=c("green", "blue", "red", "orange", "yellow"), lwd=3, bty="n", cex=0.9)
lines(smooth.spline(SBI_species$dbh_cm, predict(predict3.z05.55)), col="green", lwd=3, lty=3)


legend(46, 15, legend = c("model1: linear", "model2: poly x^2", "model3: poly x^2 + x^3"),
       col=c("red", "blue", "green"), lwd=3, bty="n", cex=0.9)



lines(smooth.spline(SS_species$dbh_cm, predict(predict2_C14.16)), lwd=3, col="blue")
predict3_C14.16 <- lm(SS_species$tree_agb_kg_C14.16 ~ SS_species$dbh_cm + I(SS_species$dbh_cm^2) + I(SS_species$dbh_cm^3))
lines(smooth.spline(SS_species$dbh_cm, predict(predict3_C14.16)), col="green", lwd=3, lty=3)
legend(46, 15, legend = c("model1: linear", "model2: poly x^2", "model3: poly x^2 + x^3"),
#breusch-pagan test of homoskedasticity
bptest(model2)
bptest(model1)



#normality of residuals
qqnorm  (rstandard(model_lm1))
abline(0,1)
hist(resid(model2))
ols_test_normality(model2)


plot(data$BIOMASS, fitted(modmel2)
     abline(0,1)
     
     sqrt(mean(sqres))
     sqrt(mean(sqres))/mean(data$BIOMASS)*100
     
     

plot(SS_species$dbh_cm, SS_species$tree_agb_kg_C14.16)
abline(predict1_C14.16, col="red", lwd=3)
predict2_C14.16 <- lm(SS_species$tree_agb_kg_C14.16 ~ SS_species$dbh_cm + I(SS_species$dbh_cm^2))
lines(smooth.spline(SS_species$dbh_cm, predict(predict2_C14.16)), lwd=3, col="blue")
predict3_C14.16 <- lm(SS_species$tree_agb_kg_C14.16 ~ SS_species$dbh_cm + I(SS_species$dbh_cm^2) + I(SS_species$dbh_cm^3))
lines(smooth.spline(SS_species$dbh_cm, predict(predict3_C14.16)), col="green", lwd=3, lty=3)
 <-data.frame(lm=model_lm1_kfold_repeat$residuals)


kfoldrepeat_residuals_lm2 <-data.frame(lm=model_lm2_kfold_repeat$residuals)
kfoldrepeat_residuals_lmstep <-data.frame(lm=model_lm_step_full_kfold_repeat$residuals)
kfoldrepeat_residuals_nls <-data.frame(lm=model_nls1_LOOVC$residuals)
kfoldrepeat_residuals_nlsclean <-data.frame(lm=model_nls1_cleaned_LOOVC$residuals)
kfoldrepeat_residuals_rrhuber <-data.frame(lm=rr.huber_LOOVC$residuals)
kfoldrepeat_residuals_rrbisq <-data.frame(lm=rr.bisquare_LOOVC$residuals)
kfoldrepeat_residuals_rrhampel <-data.frame(lm=rr.hampel_LOOVC$residuals)

shapiro.test(resid(model_lm1))
bptest(model_lm2_kfold_repeat)
bptest(model_lm_step_full_kfold_repeat)
bptest(model_nls1_kfold_repeat)
bptest(model_nls1_cleaned_LOOVC)
bptest(rr.huber_LOOVC)
bptest(rr.bisquare_LOOVC)
bptest(rr.hampel_LOOVC)

dwtest(rr.huber)
dwtest(rr.bisquare)
dwtest(rr.hampel)

outlierTest(lm(rr.huber))
outlierTest(lm(rr.bisquare))
outlierTest(lm(rr.hampel))









plot(fitted(model_lm2_kfold_repeat), resid(model_lm2_kfold_repeat))
abline(0,0, col = "green")
hist(resid(model_lm2_kfold_repeat), col = "green")

plot(fitted(model_nls1_LOOVC), resid(model_nls1_LOOVC))
abline(0,0, col = "blue")
hist(resid(model_nls1_LOOVC), col = "blue")

plot(fitted(model_nls1_cleaned_LOOVC), resid(model_nls1_cleaned_LOOVC))
abline(0,0, col = "red")
hist(resid(model_nls1_cleaned_LOOVC), col = "red")

plot(fitted(rr.bisquare_LOOVC), resid(rr.bisquare_LOOVC))
abline(0,0, col = "orange")
hist(resid(rr.bisquare_LOOVC), col = "orange")


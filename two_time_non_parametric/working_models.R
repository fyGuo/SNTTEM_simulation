library(SuperLearner)
library(xgboost)
library(ranger)



# when all the models are correctly specified
# working models
working_model <- function(data, id_1, id_2,
                          ps0 = c(TRUE, FALSE),
                          ps1 = c(TRUE, FALSE),
                          mu01 = c(TRUE, FALSE),
                          mu02 = c(TRUE, FALSE),
                          mu12 = c(TRUE, FALSE)) {
  data1 <- data[id_1,]
  data2 <- data[id_2,]
  rf <- create.Learner("SL.ranger", params = list(num.trees = 500))
  xgboot <- create.Learner("SL.xgboost", params = list(ntrees = 500, max_depth = 4,
                                                       shrinkage=0.1))
  gam <- create.Learner("SL.gam", tune = list(df = 3:8))
  # propensity score model at time 0
  if (ps0 == TRUE) {
    data$ps0 <- NA
    data[id_2,]$ps0 <- SuperLearner(data1$A0, data.frame(L0 = data1$L0),
                                newX = data.frame(L0 = data2$L0),
                                family = binomial(),
                                SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
    data[id_1,]$ps0 <- SuperLearner(data2$A0, data.frame(L0 = data2$L0),
                                newX = data.frame(L0 = data1$L0),
                                family = binomial(),
                                SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
}


  if (ps1 == TRUE) {
    data$ps1 <- NA
    data[id_2,]$ps1 <- SuperLearner(data1$A1, data.frame(L1 = data1$L1,
                                                         A0 = data1$A0),
                                    newX = data.frame(L1 = data2$L1,
                                                      A0 = data2$A0),
                                    family = binomial(),
                                    SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
    data[id_1,]$ps1 <- SuperLearner(data2$A1, data.frame(L1 = data2$L1,
                                                         A0 = data2$A0),
                                    newX = data.frame(L1 = data1$L1,
                                                      A0 = data1$A0),
                                    family = binomial(),
                                    SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
   }

  # ###############
  # outcome regressions at time 1 for variable Y2, with the same cross-fitting strategy
  if (mu12 == TRUE){
    data$mu12 <- NA
    data[id_2,]$mu12 <- SuperLearner(data1[data1$A1==0,]$Y2,
                                     data.frame(L1 = data1[data1$A1==0,]$L1,
                                                A0 = data1[data1$A1==0,]$A0,
                                                L0 = data1[data1$A1==0,]$L0,
                                                Y1 = data1[data1$A1==0,]$Y1),
                                    newX = data.frame(L1 = data2$L1,
                                                      A0 = data2$A0,
                                                      L0 = data2$L0,
                                                      Y1 = data2$Y1),
                                    family = gaussian(),
                                    SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
    data[id_1,]$mu12 <- SuperLearner(data2[data2$A1==0,]$Y2,
                                     data.frame(L1 = data2[data2$A1==0,]$L1,
                                                A0 = data2[data2$A1==0,]$A0,
                                                L0 = data2[data2$A1==0,]$L0,
                                                Y1 = data2[data2$A1==0,]$Y1),
                                     newX = data.frame(L1 = data1$L1,
                                                       A0 = data1$A0,
                                                       L0 = data1$L0,
                                                       Y1 = data1$Y1),
                                     family = gaussian(),
                                     SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
  }


  # ###############
  # outcome regressions at time 0 for variable Y1, with the same cross-fitting strategy
  if (mu01 == TRUE){

    data$mu01 <- NA
    data[id_2,]$mu01 <- SuperLearner(data1[data1$A0 == 0,]$Y1,
                                     data.frame(L0 = data1[data1$A0 == 0,]$L0,
                                                A0 = data1[data1$A0 == 0,]$A0),
                                     newX = data.frame(L0 = data2$L0,
                                                       A0 = 0),
                                     family = gaussian(),
                                     SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
    data[id_1,]$mu01 <- SuperLearner(data2[data2$A0 == 0,]$Y1,
                                     data.frame(L0 = data2[data2$A0 == 0,]$L0,
                                                A0 = data2[data2$A0 == 0,]$A0),
                                     newX = data.frame(L0 = data1$L0,
                                                       A0 = 0),
                                     family = gaussian(),
                                     SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
  }


  #############################
  # out come regression at time 0, this model is based on the dr for mu1
  # here dr_mu1 is a bit triky, because say ID_1 uses its own model to predict dr_mu1,
  # and thus, in the final estimating equation, it never touches on ID_2
  data$dr_mu12 <- NA
data1$dr_mu12 <-  data$dr_mu12[id_1] <-  (1-data$A1[id_1])/(1-data[id_1,]$ps1)*(data$Y2[id_1] -data$mu12[id_1]) + data$mu12[id_1]
data2$dr_mu12 <-  data$dr_mu12[id_2]  <- (1-data$A1[id_2])/(1-data[id_2,]$ps1)*(data$Y2[id_2] -data$mu12[id_2]) + data$mu12[id_2]

  # now we can estimate mu0
  if (mu02 == TRUE) {

    data$mu02<- NA
    data[id_2,]$mu02 <- SuperLearner(data1[data1$A0 ==0,]$dr_mu12,
                                     data.frame(L0 = data1[data1$A0 == 0,]$L0),
                                     newX = data.frame(L0 = data2$L0),
                                     family = gaussian(),
                                     SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
    data[id_1,]$mu02 <- SuperLearner(data2[data2$A0 ==0,]$dr_mu12,
                                     data.frame(L0 = data2[data2$A0 == 0,]$L0),
                                     newX = data.frame(L0 = data1$L0),
                                     family = gaussian(),
                                     SL.library = c(rf$names, xgboot$names, gam$names))$SL.predict
  }
  return(data)
}

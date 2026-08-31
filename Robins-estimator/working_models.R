# when all the models are correctly specified
# working models
working_model <- function(data, id_1, id_2,
                          ps0 = c(TRUE, FALSE),
                          ps1 = c(TRUE, FALSE),
                          mu0 = c(TRUE, FALSE),
                          mu1 = c(TRUE, FALSE)) {
  # propensity score model at time 0
  if (ps0 == TRUE) {
    ps0_model_1 <- glm(A0~ L0, data = data[id_1,], family = binomial(link = "logit"))
    ps0_model_2 <- glm(A0~ L0, data = data[id_2,], family = binomial(link = "logit"))
  } else {
    ps0_model_1 <- glm(A0~ 1, data = data[id_1,], family = binomial(link = "logit"))
    ps0_model_2 <- glm(A0~ 1, data = data[id_2,], family = binomial(link = "logit"))
  }
  # then we use the predicted values for sample id_1 from the models for sample id_2, and vice versa
  data$ps0 <- NA
  data[id_1,]$ps0 <- predict(ps0_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$ps0 <- predict(ps0_model_1, newdata = data[id_2,], type = "response")

  # do the same thing for ps1
  # propensity score model at time 1
  if (ps1 == TRUE) {
    ps1_model_1 <- glm(A1 ~ L1 + A0, data = data[id_1,], family = binomial(link = "logit"))
    ps1_model_2 <- glm(A1 ~ L1 + A0, data = data[id_2,], family = binomial(link = "logit"))
  } else {
    ps1_model_1 <- glm(A1 ~ 1, data = data[id_1,], family = binomial(link = "logit"))
    ps1_model_2 <- glm(A1 ~ 1, data = data[id_2,], family = binomial(link = "logit"))
  }

  # then we use the predicted values for sample id_1 from the models for sample id_2, and vice versa
  data$ps1 <- NA
  data[id_1,]$ps1 <- predict(ps1_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$ps1 <- predict(ps1_model_1, newdata = data[id_2,], type = "response")

  # ###############
  # outcome regressions at time 1, with the same cross-fitting strategy
  if (mu1 == TRUE){
    mu1_model_1 <- glm(Y ~ A0 + L1 + L0 + A0:L0, data = data[id_1,], family = gaussian(link = "identity"))
    mu1_model_2 <- glm(Y ~ A0 + L1 + L0 + A0:L0, data = data[id_2,], family = gaussian(link = "identity"))
  } else {
    mu1_model_1 <- glm(Y ~ 1, data = data[id_1,], family = gaussian(link = "identity"))
    mu1_model_2 <- glm(Y ~ 1, data = data[id_2,], family = gaussian(link = "identity"))
  }
  # do the predictions
  data$mu1 <- NA
  data[id_1,]$mu1 <- predict(mu1_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$mu1 <- predict(mu1_model_1, newdata = data[id_2,], type = "response")

  #############################
  # out come regression at time 0, this model is based on the dr for mu1
  # here dr_mu1 is a bit triky, because say ID_1 uses its own model to predict dr_mu1,
  # and thus, in the final estimating equation, it never touches on ID_2
  data$dr_mu1 <- NA
  data$dr_mu1[id_1] <-  (1-data$A1[id_1])/(1-predict(ps1_model_1, data[id_1,], type = "response"))*
    (data$Y[id_1] - predict(mu1_model_1, data[id_1,])) + predict(mu1_model_1, data[id_1,])
  data$dr_mu1[id_2]  <- (1-data$A1[id_2])/(1-predict(ps1_model_2, data[id_2,], type = "response"))*
    (data$Y[id_2] - predict(mu1_model_2, data[id_2,])) + predict(mu1_model_2, data[id_2,])

  # now we can estimate mu0
  if (mu0 == TRUE) {
    mu0_model_1 <- glm(dr_mu1 ~ L0, data = data[id_1,], family = gaussian(link = "identity"))
    mu0_model_2 <- glm(dr_mu1 ~ L0, data = data[id_2,], family = gaussian(link = "identity"))
  } else {
    mu0_model_1 <- glm(dr_mu1 ~ 1, data = data[id_1,], family = gaussian(link = "identity"))
    mu0_model_2 <- glm(dr_mu1 ~ 1, data = data[id_2,], family = gaussian(link = "identity"))
  }
  # do the predictions
  data$mu0 <- NA
  data[id_1,]$mu0 <- predict(mu0_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$mu0 <- predict(mu0_model_1, newdata = data[id_2,], type = "response")

  return(data)
}

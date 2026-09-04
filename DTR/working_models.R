# when all the models are correctly specified
# working models
working_model <- function(data, id_1, id_2,
                          ps0 = c(TRUE, FALSE),
                          ps1 = c(TRUE, FALSE),
                          mu01 = c(TRUE, FALSE),
                          mu02 = c(TRUE, FALSE),
                          mu12 = c(TRUE, FALSE)) {
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
  # outcome regressions at time 1 for variable Y2, with the same cross-fitting strategy
  if (mu12 == TRUE){
    mu12_model_1 <- glm(Y2 ~ A0 + L1 + L0 + A0:L0, data = filter(data[id_1,], A1==0), family = gaussian(link = "identity"))
    mu12_model_2 <- glm(Y2 ~ A0 + L1 + L0 + A0:L0, data = filter(data[id_2,], A1==0), family = gaussian(link = "identity"))
  } else {
    mu12_model_1 <- glm(Y2 ~ 1, data = filter(data[id_1,], A1==0), family = gaussian(link = "identity"))
    mu12_model_2 <- glm(Y2 ~ 1, data = filter(data[id_2,], A1==0), family = gaussian(link = "identity"))
  }
  # do the predictions
  data$mu12 <- NA
  data[id_1,]$mu12 <- predict(mu12_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$mu12 <- predict(mu12_model_1, newdata = data[id_2,], type = "response")

  # ###############



  #############################
  # out come regression at time 0, this model is based on the dr for mu1
  # here dr_mu1 is a bit triky, because say ID_1 uses its own model to predict dr_mu1,
  # and thus, in the final estimating equation, it never touches on ID_2
  data$dr_mu12 <- NA
  data$dr_mu12[id_1] <-  (1-data$A1[id_1])/(1-predict(ps1_model_2, data[id_1,], type = "response"))*
    (data$Y2[id_1] - predict(mu12_model_2, data[id_1,])) + predict(mu12_model_2, data[id_1,])
  data$dr_mu12[id_2]  <- (1-data$A1[id_2])/(1-predict(ps1_model_1, data[id_2,], type = "response"))*
    (data$Y2[id_2] - predict(mu12_model_1, data[id_2,])) + predict(mu12_model_1, data[id_2,])

  # now we can estimate mu0
  if (mu02 == TRUE) {
    mu02_model_1 <- glm(dr_mu12 ~ L0, data = filter(data[id_1,], A0==0), family = gaussian(link = "identity"))
    mu02_model_2 <- glm(dr_mu12 ~ L0, data = filter(data[id_2,], A0==0), family = gaussian(link = "identity"))
  } else {
    mu02_model_1 <- glm(dr_mu12 ~ 1, data = filter(data[id_1,], A0==0), family = gaussian(link = "identity"))
    mu02_model_2 <- glm(dr_mu12 ~ 1, data = filter(data[id_2,], A0==0), family = gaussian(link = "identity"))
  }
  # do the predictions
  data$mu02 <- NA
  data[id_1,]$mu02 <- predict(mu02_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$mu02 <- predict(mu02_model_1, newdata = data[id_2,], type = "response")

  return(data)
}


ps0 <- function(L0, alpha, correct_model){
  if (correct_model){
    ps0 <- expit(alpha[1] + alpha[2]*L0)
  } else {ps0 <- expit(alpha[1])}
  return(ps0)
}

ps1 <- function(L1, A0, alpha, correct_model){
  if (correct_model){
    ps1 <- expit(alpha[1] + alpha[2]*L1 + alpha[3]*A0)
  } else {ps1 <- expit(alpha[1])}
  return(ps1)
}



mu12 <- function(L0, A0, L1, beta, correct_model){
  if (correct_model){
    mu12 <- beta[1] + beta[2]*L0 + beta[3]*A0 + beta[4]*L0*A0 + beta[5]*L1
  } else {mu12 <- beta[1]}
  return(mu12)
}

mu02 <- function(L0, beta, correct_model){
  if (correct_model){
    mu02 <- beta[1] + beta[2]*L0
  } else {mu02 <- beta[1]}
  return(mu02)
}

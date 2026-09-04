# when all the models are correctly specified
# working models
working_model <- function(data, id_1, id_2,
                          ps0 = c(TRUE, FALSE),
                          ps1 = c(TRUE, FALSE),
                          ps2 = c(TRUE, FALSE),
                          mu03 = c(TRUE, FALSE),
                          mu13 = c(TRUE, FALSE),
                          mu23 = c(TRUE, FALSE)) {
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

  # do the same thing of ps2
  # propensity score model at time 2
   if (ps2 == TRUE) {
    ps2_model_1 <- glm(A2 ~ L2 + A1, data = data[id_1,], family = binomial(link = "logit"))
    ps2_model_2 <- glm(A2 ~ L2 + A1, data = data[id_2,], family = binomial(link = "logit"))
   } else {
    ps2_model_1 <- glm(A2 ~ 1, data = data[id_1,], family = binomial(link = "logit"))
    ps2_model_2 <- glm(A2 ~ 1, data = data[id_2,], family = binomial(link = "logit"))
   }
  # then we use the predicted values for sample id_1 from the models for sample id_2, and vice versa
  data$ps2 <- NA
  data[id_1,]$ps2 <- predict(ps2_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$ps2 <- predict(ps2_model_1, newdata = data[id_2,], type = "response")



  # ###############
  # outcome regressions at time 2 for variable Y3, with the same cross-fitting strategy
  if (mu23 == TRUE){
    mu23_model_1 <- glm(Y3 ~ L0 + L1 + L2 + A0 + A1 + A0:L0 +A1:L1, data = filter(data[id_1,], A2==0), family = gaussian(link = "identity"))
    mu23_model_2 <- glm(Y3 ~ L0 + L1 + L2 + A0 + A1+ A0:L0 +A1:L1, data = filter(data[id_2,], A2==0), family = gaussian(link = "identity"))
  } else {
    mu23_model_1 <- glm(Y3 ~ 1, data = filter(data[id_1,], A2==0), family = gaussian(link = "identity"))
    mu23_model_2 <- glm(Y3 ~ 1, data = filter(data[id_2,], A2==0), family = gaussian(link = "identity"))
  }
  # do the predictions
  data$mu23 <- NA
  data[id_1,]$mu23 <- predict(mu23_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$mu23 <- predict(mu23_model_1, newdata = data[id_2,], type = "response")

  # outcome regression at time 1 for variable Y3,
  if (mu13 == TRUE){
    mu13_model_1 <- glm(mu23 ~ L0 + L1 + A0 + A0:L0, data = filter(data[id_1,], A1==0), family = gaussian(link = "identity"))
    mu13_model_2 <- glm(mu23 ~ L0 + L1 + A0 + A0:L0, data = filter(data[id_2,], A1==0), family = gaussian(link = "identity"))
  } else {
    mu13_model_1 <- glm(mu23 ~ 1, data = filter(data[id_1,], A1==0), family = gaussian(link = "identity"))
    mu13_model_2 <- glm(mu23 ~ 1, data = filter(data[id_2,], A1==0), family = gaussian(link = "identity"))
  }
  # do the predictions
  data$mu13 <- NA
  data[id_1,]$mu13 <- predict(mu13_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$mu13 <- predict(mu13_model_1, newdata = data[id_2,], type = "response")

  # outcome regression at time 0 for varibale Y3,
  if (mu03 == TRUE){
    mu03_model_1 <- glm(mu13 ~ L0, data =  filter(data[id_1,],A0==0), family = gaussian(link = "identity"))
    mu03_model_2 <- glm(mu13 ~ L0, data = filter(data[id_2,],A0==0), family = gaussian(link = "identity"))
  } else {
    mu03_model_1 <- glm(mu13 ~ 1, data = data[id_1,], family = gaussian(link = "identity"))
    mu03_model_2 <- glm(mu13 ~ 1, data = data[id_2,], family = gaussian(link = "identity"))
  }
  # do the predictions
  data$mu03 <- NA
  data[id_1,]$mu03 <- predict(mu03_model_2, newdata = data[id_1,], type = "response")
  data[id_2,]$mu03 <- predict(mu03_model_1, newdata = data[id_2,], type = "response")


  return(data)
}




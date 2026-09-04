# when all the models are correctly specified
# working models
working_model <- function(data) {
  # propensity score model at time 0

    ps0_model <- glm(A0~ L0, data = data, family = binomial(link = "logit"))

  data$ps0 <- predict(ps0_model, newdata = data, type = "response")

  # do the same thing for ps1
  # propensity score model at time 1
    ps1_model <- glm(A1 ~ L1 + A0, data = data, family = binomial(link = "logit"))


  data$ps1 <- predict(ps1_model, newdata = data, type = "response")

  # ###############
  # outcome regressions at time 1 for variable Y2, with the same cross-fitting strategy
    mu1_model <- glm(Y ~ A0 + A0:L0 + A0:exp(L0) + L1, data = filter(data, A1==0), family = gaussian(link = "identity"))

  # do the predictions
  data$mu1 <- NA
  data$mu1 <- predict(mu1_model, newdata = data, type = "response")


  data$mu0 <- 2

  return(data)
}

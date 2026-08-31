## this code is to evaluate the double robustness of the SNTTEM g-estimation
# Under the Homoschesasticity assumption
# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)
library(geex)
library(MASS)
library(matrixcalc)
library(gmm)
source("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Double_robustness/Generate_data.R")
source("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Double_robustness/working_models.R")
sample <- 5000
iter <- 200
psi <- c(0, 0)
seed <- 1652

# set the working models
working_ps1 <- TRUE
working_ps0 <- TRUE
working_mu1 <- TRUE
working_mu0 <- TRUE
ps1_control <- 0

# -3 #-6  #3 #6 # 9
# the conventional estimator
ee_old <- function(data){
  function(theta){
    with(data,
         c(L0*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0),
         (A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
  }
}

plan("multisession")
set.seed(seed)
tic()
est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n, psi0 = psi[1], ps1_control = ps1_control, psi1 = psi[2], theta = 0, var = "hetero")

  # generate a sample splitting id
  id_1 <- sample(1:n, size = n/2, replace = FALSE)
  id_2 <- setdiff(1:n, id_1)

  df <- working_model(df, id_1, id_2, ps1 = working_ps1, ps0 = working_ps0, mu1 = working_mu1, mu0 = working_mu0) # apply the working models
  df$denom1 <- mean((1-df$L0)*(df$A0-df$ps0)*df$ps1)
  df$denom2 <-mean((df$L0)*(df$A0-df$ps0)*df$ps1)
  # with the working models, we can do the estimation
  est_psi <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(0,0)))@"estimates"
  Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0
  ee_old_score <-  cbind(df$L0*(df$A0-df$ps0)*((1-df$A1)/(1-df$ps1)*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0,0)*df$A0),
                         ((1-df$L0)/df$denom1+df$L0/df$denom2)*     (df$A0-df$ps0)/(1-df$ps1)*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, 0)*df$A1 - df$mu1))

  est_cov <-  cov(ee_old_score )
  chisq_stat <- n*t(colMeans(ee_old_score)) %*% ginv(est_cov) %*% (colMeans(ee_old_score)) # conduct the score testing
  reject <- chisq_stat > qchisq(0.95, df = 2) # 2 is the number of parameters we are estimating

  # return the result
  est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2],
                    Y00 = mean(Y00, na.rm = TRUE),
                    reject = reject)
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()



##############################
# The SNTTEM estimator
ee_SNTTEM <- function(data) {
  function(theta){
    with(data,
         c(L0*(A0-ps0)*(Y-gamma1(L1, theta[2])*A1-gamma0(L0, theta[1])*A0 -mu0 ),
           (A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
  }
}

plan("multisession")
tic()
set.seed(seed)

est_SNTTEM <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n, psi0 = psi[1], ps1_control, psi1 = psi[2], theta = 0, var = "hetero")
  df <- working_model(df, ps1 = working_ps1, ps0 = working_ps0, mu1 = working_mu1, mu0 = working_mu0) # apply the working models
  df$denom1 <- mean((1-df$L0)*(df$A0-df$ps0)*df$ps1)
  df$denom2 <-mean((df$L0)*(df$A0-df$ps0)*df$ps1)
  # with the working models, we can do the estimation
  est_psi <-  m_estimate(ee_SNTTEM, data = df, root_control = setup_root_control(start = c(0,0)))@"estimates"
  Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0
  ee_SNTTEM_score <-  cbind(df$L0*(df$A0-df$ps0)*(((1-df$A1)/(1-df$ps1))^(1-df$I1)*(df$Y-gamma1(df$L1, 0)*df$A1-df$mu1)+df$mu1-df$mu0-gamma0(df$L0,0)*df$A0),
                            ((1-df$L0)/df$denom1+df$L0/df$denom2)*     (df$A0-df$ps0)/(1-df$ps1)*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, 0)*df$A1 - df$mu1))
  est_cov <-  cov(ee_SNTTEM_score)
  chisq_stat <- n*t(colMeans(ee_SNTTEM_score)) %*% ginv(est_cov) %*% (colMeans(ee_SNTTEM_score)) # conduct the score testing
  reject <- chisq_stat > qchisq(0.95, df = 2) # 2 is the number of parameters we are estimating

  # return the result
  est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2],
                    Y00 = mean(Y00, na.rm = TRUE),
                    reject = reject)
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()

# the outputs
est_old %>%
  summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
            psi1_sd = sd(est_psi1, na.rm = TRUE)*sqrt(sample),
            psi0_mean = mean(est_psi0, na.rm = TRUE),
            psi0_sd = sd(est_psi0, na.rm = TRUE)*sqrt(sample),
            psi0_eff = 1/(var(est_psi0, na.rm = TRUE)*sqrt(sample)),
            Y00_mean = mean(Y00, na.rm = TRUE),
            Y00_sd = sd(Y00, na.rm = TRUE)*sqrt(sample),
            reject_rate = mean(reject, na.rm = TRUE))

est_SNTTEM %>%
  summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
            psi1_sd = sd(est_psi1, na.rm = TRUE)*sqrt(sample),
            psi0_mean = mean(est_psi0, na.rm = TRUE),
            psi0_sd = sd(est_psi0, na.rm = TRUE)*sqrt(sample),
            psi0_eff = 1/(var(est_psi0, na.rm = TRUE)*sqrt(sample)),
            Y00_mean = mean(Y00, na.rm = TRUE),
            Y00_sd = sd(Y00, na.rm = TRUE)*sqrt(sample),
            reject_rate = mean(reject, na.rm = TRUE))









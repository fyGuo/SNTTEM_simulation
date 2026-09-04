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
sample <- 50000
iter <- 1000
psi <- c(1, 1)
seed <- 10086

# set the working models
working_ps1 <- TRUE
working_ps0 <- TRUE
working_mu1 <- TRUE
working_mu0 <- TRUE



# the conventional estimator

plan("multisession")
set.seed(seed)
tic()
est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n, psi0 = psi[1], psi1 = psi[2], theta = 0, var = "homo")
  df <- working_model(df, ps1 = working_ps1, ps0 = working_ps0, mu1 = working_mu1, mu0 = working_mu0) # apply the working models

  x <- as.matrix(df)
  g <- function(theta, x){
    cbind(x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))*(x[,"Y"]-x[,"mu1"])+x[,"mu1"]-x[,"mu0"]-gamma0(x[,"L0"], theta[1])*x[,"A0"]),
          x[,"I1"]*x[,"L1"]*(x[,"A1"]-x[,"ps1"])*(x[,"Y"]-gamma1(x[,"L1"], theta[2])*x[,"A1"] - x[,"mu1"])) %>%
      return()
  }
  mod <- gmm(g, x = x, t0 = c(1,1), type = "iterative", optfct = "nlminb")
  est_psi <- mod$coefficients
  Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0


  chisq_stat <- t(mod$coefficients) %*% ginv(mod$vcov) %*% mod$coefficients
  reject <- chisq_stat > qchisq(0.95, df = 2)
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
         c(L0*(A0-ps0)*(((1-A1)/(1-ps1))^(1-I1)*(Y-gamma1(L1, theta[2])*A1-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0),
           I1*L1*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
  }
}

plan("multisession")
tic()
set.seed(seed)

est_SNTTEM <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n, psi0 = psi[1], psi1 = psi[2], theta = 0, var = "homo")
  df <- working_model(df, ps1 = working_ps1, ps0 = working_ps0, mu1 = working_mu1, mu0 = working_mu0) # apply the working models

  x <- as.matrix(df)
  g <- function(theta, x){
    cbind(x[,"I1"]*x[,"L1"]*(x[,"A1"]-x[,"ps1"])*(x[,"Y"]-gamma1(x[,"L1"], theta[2])*x[,"A1"] - x[,"mu1"]),
          x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))^(1-x[,"I1"])*(x[,"Y"]-gamma1(x[,"L1"], theta[2])*x[,"A1"]-x[,"mu1"])+x[,"mu1"]-x[,"mu0"]-gamma0(x[,"L0"], theta[1])*x[,"A0"])) %>%
      return()
  }
  mod <- gmm(g, x = x, t0 = c(1,1), type = "iterative", optfct = "nlminb")
  est_psi <- mod$coefficients
  Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0


  chisq_stat <- t(mod$coefficients) %*% ginv(mod$vcov) %*% mod$coefficients
  reject <- chisq_stat > qchisq(0.95, df = 2)
  est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2],
                    Y00 = mean(Y00, na.rm = TRUE),
                    reject = reject)
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()
####################
# the combined estimator
plan("multisession")
tic()
set.seed(seed)

est_combined <- furrr::future_map_dfr(.x = 1:iter,
                                      .f = function(i, n= sample) {
                                        df <- generate_data(n, psi0 = psi[1], psi1 = psi[2], theta = 0, var = "homo")
                                        df <- working_model(df, ps1 = working_ps1, ps0 = working_ps0, mu1 = working_mu1, mu0 = working_mu0) # apply the working models
                                        x <- as.matrix(df)
                                        g <- function(theta, x){
                                          cbind(x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))*(x[,"Y"]-x[,"mu1"])+x[,"mu1"]-x[,"mu0"]-gamma0(x[,"L0"], theta[1])*x[,"A0"]),
                                                x[,"I1"]*x[,"L1"]*(x[,"A1"]-x[,"ps1"])*(x[,"Y"]-gamma1(x[,"L1"], theta[2])*x[,"A1"] - x[,"mu1"]),
                                                x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))^(1-x[,"I1"])*(x[,"Y"]-gamma1(x[,"L1"], theta[2])*x[,"A1"]-x[,"mu1"])+x[,"mu1"]-x[,"mu0"]-gamma0(x[,"L0"], theta[1])*x[,"A0"])) %>%
                                            return()
                                        }
                                        mod <- gmm(g, x = x, t0 = c(1,1), type = "iterative", optfct = "nlminb")
                                        est_psi <- mod$coefficients
                                        Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0


                                        chisq_stat <- t(mod$coefficients) %*% ginv(mod$vcov) %*% mod$coefficients
                                        reject <- chisq_stat > qchisq(0.95, df = 2)
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
            Y00_mean = mean(Y00, na.rm = TRUE),
            Y00_sd = sd(Y00, na.rm = TRUE)*sqrt(sample),
            reject_rate = mean(reject, na.rm = TRUE))

est_SNTTEM %>%
  summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
            psi1_sd = sd(est_psi1, na.rm = TRUE)*sqrt(sample),
            psi0_mean = mean(est_psi0, na.rm = TRUE),
            psi0_sd = sd(est_psi0, na.rm = TRUE)*sqrt(sample),
            Y00_mean = mean(Y00, na.rm = TRUE),
            Y00_sd = sd(Y00, na.rm = TRUE)*sqrt(sample),
            reject_rate = mean(reject, na.rm = TRUE))



est_combined %>%
  summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
            psi1_sd = sd(est_psi1, na.rm = TRUE)*sqrt(sample),
            psi0_mean = mean(est_psi0, na.rm = TRUE),
            psi0_sd = sd(est_psi0, na.rm = TRUE)*sqrt(sample),
            Y00_mean = mean(Y00, na.rm = TRUE),
            Y00_sd = sd(Y00, na.rm = TRUE)*sqrt(sample),
            reject_rate = mean(reject, na.rm = TRUE))





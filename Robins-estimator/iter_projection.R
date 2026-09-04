# this code is to use iterative projeciton approach to find the closest point
# for the projection, which we hope to gain efficiency compared to the traditional method.
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
library(rootSolve)
source("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Robins-estimator/Generate_data.R")
source("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Robins-estimator/working_models.R")
sample <- 10000
iter <- 200
psi <- c(0, 0)
seed <- 1652
varY_control <- 0

# set the working models
working_ps1 <- TRUE
working_ps0 <- TRUE
working_mu1 <- TRUE
working_mu0 <- TRUE

ps1_control <- 0

varY_control <- 0

projection_residual <- function(phi1, phi2,basis, sample) {
  base_matrix <- phi2*basis
  resid <-phi1 - 1/sample*base_matrix %*% solve(sample*t(base_matrix) %*% base_matrix) %*% t(base_matrix) %*% phi1
  return(resid)
}



ee_iter_projection <- function(data){
  function(theta){
    with(data,
         c((1+L0)*projection_residual((A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0),(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1), cbind(L0), sample = sample),
           (1+L1)/(1+2*(varY_control + varY_control^2)*(1-ps1))*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
  }
}



plan("multisession")
set.seed(seed)
tic()
working_ps1 <- TRUE
working_ps0 <- TRUE
working_mu1 <- TRUE
working_mu0 <- TRUE
est_iter_projection <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n, psi0 = psi[1], ps1_control, psi1 = psi[2], theta = 0, varY_control = varY_control)
  df <- working_model(df, ps1 = working_ps1, ps0 = working_ps0, mu1 = working_mu1, mu0 = working_mu0) # apply the working models

  est_psi <-  m_estimate(ee_iter_projection , data = df, root_control = setup_root_control(start = c(0,0)))@"estimates"

  est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2])
  return(est)

},.options = furrr_options(seed = TRUE))

toc()
est_iter_projection %>%
  summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
            psi1_sd = sd(est_psi1, na.rm = TRUE)*sqrt(sample),
            psi0_mean = mean(est_psi0, na.rm = TRUE),
            psi0_sd = sd(est_psi0, na.rm = TRUE)*sqrt(sample),
            psi0_eff = 1/(var(est_psi0, na.rm = TRUE)*(sample)))

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
source("Generate_data.R")
source("working_models_ml.R")
sample <- 6000
iter <-400

# set the propensity score at time 1
ps1_int <- 0
p_I1 <- 0.2
delta <- 0

# set the parameters
psi01 <- 0
psi02 <- 0
psi12 <- 0
seed <-3411
# set the working models
working_ps1 <- TRUE
working_ps0 <- TRUE
working_mu02 <- TRUE
working_mu12 <- TRUE

#the conventional estimator
ee_old <- function(data){
  function(theta){
    with(data,
         c((A0-ps0)*((A0==A1)/((1-ps1)*(1-A0) + A0*ps1)*exp(-gamma02(L0, theta[1])*A0)*(Y2-mu12)+mu12*exp(-gamma02(L0, theta[1])*A0)-mu02),
           I1*(A1-ps1)*(Y2*exp(-gamma12(L1, theta[2])*(A1-A0)) - mu12)))
  }
}

# The SNTTEM estimator
ee_rbs_estimator <- function(data){
  function(theta){
    with(data,
         c((A0-ps0)*(((A0==A1)/((1-ps1)*(1-A0) + ps1*A0))^(1-I1)*exp(-gamma02(L0, theta[1])*A0)*(Y2*exp(-gamma12(L1, theta[2])*(A1-A0))-mu12)
                     +mu12*exp(-gamma02(L0, theta[1])*A0)-mu02),
           I1*(A1-ps1)*(Y2*exp(-gamma12(L1, theta[2])*(A1-A0)) - mu12)))
  }
}


plan("multisession")
set.seed(seed)
tic()
est <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n, psi02 = psi02, psi12 = psi12, ps1_int = ps1_int, p_I1=p_I1)
  # generate a sample splitting id
  id_1 <- sample(1:n, size = n/2, replace = FALSE)
  id_2 <- setdiff(1:n, id_1)

  df <- working_model(df, id_1, id_2, ps1 = working_ps1, ps0 = working_ps0,  mu12 = working_mu12, mu02 = working_mu02) # apply the working models

  # with the working models, we can do the estimation
  est_psi_old <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(0,0)))@"estimates"
  est_psi_rbs <-  m_estimate(ee_rbs_estimator, data = df, root_control = setup_root_control(start = c(0,0)))@"estimates"

  # GMM estimator
  x <- as.matrix(df)
  g <- function(theta, x){
    cbind((x[,"A0"]-x[,"ps0"])*(((x[,"A0"]==x[,"A1"])/((1-x[,"ps1"])*(1-x[,"A0"]) +x[,"A0"]*x[,"ps1"]))*exp(-gamma02(x[,"L0"], theta[1])*x[,"A0"])*(x[,"Y2"]-x[,"mu12"])+x[,"mu12"]*exp(-gamma02(x[,"L0"], theta[1])*x[,"A0"])-x[,"mu02"]),
          (x[,"A0"]-x[,"ps0"])*(((x[,"A0"]==x[,"A1"])/((1-x[,"ps1"])*(1-x[,"A0"]) +x[,"A0"]*x[,"ps1"]))^(1-x[,"I1"])*exp(-gamma02(x[,"L0"], theta[1])*x[,"A0"])*(x[,"Y2"]*exp(-gamma12(x[,"L1"], theta[2])*(x[,"A1"] - x[,"A0"])) - x[,"mu12"])
                                +x[,"mu12"]*exp(-gamma02(x[,"L0"], theta[1])*x[,"A0"])-x[,"mu02"]),
          x[,"I1"]*(x[,"A1"]-x[,"ps1"])*(x[,"Y2"]*exp(-gamma12(x[,"L1"], theta[2])*(x[,"A1"] - x[,"A0"])) - x[,"mu12"]) ) %>%
      return()
  }
  mod <- gmm(g, x = x, t0 = c(1,1), type = "iterative")
  est_psi_gmm <- mod$coefficients

  # return the result
  est <- data.frame(id = c(i,i,i),
                    est_psi02 = c(est_psi_old[1], est_psi_rbs[1], est_psi_gmm[1]),
                    est_psi12 = c(est_psi_old[2], est_psi_rbs[2], est_psi_gmm[2]),
                    method = c("Three-step", "Robins' estimator", "GMM"))
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()



##############################

est$p_I1 <- p_I1


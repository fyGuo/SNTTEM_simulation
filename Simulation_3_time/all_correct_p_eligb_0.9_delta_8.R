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
setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Simulation_3_time")
source("Generate_data.R")
source("working_models.R")
sample <- 20000
iter <- 1000

# set the propensity score at time 1
ps1_int <- 0
ps2_int <- 0

seed <- 34111
p_elig <- 0.9
delta <- 8


# set the working models
working_ps0 <- TRUE
working_ps1 <- TRUE
working_ps2 <- TRUE
working_mu03 <- TRUE
working_mu13 <- TRUE
working_mu23 <- TRUE

#the conventional estimator
ee_old <- function(data){
  function(theta){
    with(data,
         c(L0*(A0-ps0)*((1-A2)/(1-ps2)*(1-A1)/(1-ps1)*(Y3-mu23) +(1-A1)/(1-ps1)*(mu23-mu13) + mu13-gamma03(L0, theta[1])*A0 - mu03),
           I1*L1*(A1-ps1)*((1-A2)/(1-ps2)*(Y3-mu23)+mu23-mu13-gamma13(L1, theta[2])*A1),
           I2*L2*(A2-ps2)*(Y3-gamma23(L2, theta[3])*A2 - mu23)))
  }
}

plan("multisession")
set.seed(seed)
tic()
est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n,  ps1_int = ps1_int, ps2_int = ps2_int, p_elig = p_elig, delta = delta)
# generate a sample splitting id
  id_1 <- sample(1:n, size = n/2, replace = FALSE)
  id_2 <- setdiff(1:n, id_1)

  df <- working_model(df, id_1, id_2,  ps0 = working_ps0, ps1 = working_ps1, ps2 = working_ps2,
                      mu03 = working_mu03, mu13 = working_mu13, mu23 = working_mu23) # apply the working models

  # with the working models, we can do the estimation
  est_psi <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(0,0,0)))@"estimates"
  # ee_score <-  cbind(df$L0*(df$A0-df$ps0)*(df$Y1-gamma01(df$L0, psi01)*df$A0 - df$mu01),
  #                    df$L0*(df$A0-df$ps0)*((1-df$A1)/(1-df$ps1)*(df$Y2-df$mu12)+df$mu12-df$mu02-gamma02(df$L0,psi02)*df$A0),
  #                    df$L1*df$I1*(df$A1-df$ps1)*(df$Y2-gamma12(df$L1, psi12)*df$A1 - df$mu12))
  #
  # est_cov <-  cov(ee_score )
  # chisq_stat <- n*t(colMeans(ee_score)) %*% ginv(est_cov) %*% (colMeans(ee_score)) # conduct the score testing
  # reject <- chisq_stat > qchisq(0.95, df = 3) # 2 is the number of parameters we are estimating

  # return the result
  est <- data.frame(id = i, est_psi03 = est_psi[1], est_psi13 = est_psi[2],est_psi23 = est_psi[3])
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()



##############################
# The SNTTEM estimator
ee_g_estimator <- function(data){
  function(theta){
    with(data,
         c(L0*(A0-ps0)*(((1-A2)/(1-ps2))^(1-I2)*(1-A1)/(1-ps1)^(1-I1)*(Y3-gamma23(L2, theta[3])*A2-mu23) +((1-A1)/(1-ps1))^(1-I1)*(mu23-gamma13(L1, theta[2])*A1-mu13) + mu13-gamma03(L0, theta[1])*A0 - mu03),
           I1*L1*(A1-ps1)*(((1-A2)/(1-ps2))^(1-I2)*(Y3-gamma23(L2, theta[3])*A2-mu23)+mu23-mu13-gamma13(L1, theta[2])*A1),
           I2*L2*(A2-ps2)*(Y3-gamma23(L2, theta[3])*A2 - mu23)))
  }
}

plan("multisession")
tic()
set.seed(seed)

est_g_estimator <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n,  ps1_int = ps1_int, ps2_int = ps2_int, p_elig = p_elig, delta = delta)
  # generate a sample splitting id
  id_1 <- sample(1:n, size = n/2, replace = FALSE)
  id_2 <- setdiff(1:n, id_1)

  df <- working_model(df, id_1, id_2,  ps0 = working_ps0, ps1 = working_ps1, ps2 = working_ps2,
                      mu03 = working_mu03, mu13 = working_mu13, mu23 = working_mu23) # apply the working models


  # with the working models, we can do the estimation
  est_psi <-  m_estimate(ee_g_estimator, data = df, root_control = setup_root_control(start = c(0,0,0)))@"estimates"
  # ee_score <-  cbind(df$L0*(df$A0-df$ps0)*(df$Y1-gamma01(df$L0, psi01)*df$A0 - df$mu01),
  #                    df$L0*(df$A0-df$ps0)*(((1-df$A1)/(1-df$ps1))^(1-df$I1)*(df$Y2-gamma12(df$L1, psi12)*df$A1-df$mu12) + df$mu12-gamma02(df$L0,psi02)*df$A0-df$mu02),
  #                    df$L1*df$I1*(df$A1-df$ps1)*(df$Y2-gamma12(df$L1, psi12)*df$A1 - df$mu12))
  #
  # # est_cov <-  cov( ee_score )
  # # chisq_stat <- n*t(colMeans( ee_score )) %*% ginv(est_cov) %*% (colMeans( ee_score )) # conduct the score testing
  # # reject <- chisq_stat > qchisq(0.95, df = 3) # 2 is the number of parameters we are estimating

  # return the result
  est <- data.frame(id = i, est_psi03 = est_psi[1], est_psi13 = est_psi[2],est_psi23 = est_psi[3])
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()
###################
#the combined estimator
plan("multisession")
tic()
set.seed(seed)

est_gmm <- furrr::future_map_dfr(.x = 1:iter,
                                 .f = function(i, n= sample) {
                                   df <- generate_data(n,  ps1_int = ps1_int, ps2_int = ps2_int, p_elig = p_elig, delta = delta)
                                   # generate a sample splitting id
                                   id_1 <- sample(1:n, size = n/2, replace = FALSE)
                                   id_2 <- setdiff(1:n, id_1)

                                   df <- working_model(df, id_1, id_2,  ps0 = working_ps0, ps1 = working_ps1, ps2 = working_ps2,
                                                       mu03 = working_mu03, mu13 = working_mu13, mu23 = working_mu23) # apply the working models
                                   x <- as.matrix(df)
                                   g <- function(theta, x){
                                     cbind(
                                           x[,"L0"]*(x[,"A0"]-x[,"ps0"])*((1-x[,"A2"])/(1-x[,"ps2"])*(1-x[,"A1"])/(1-x[,"ps1"])*(x[,"Y3"] - x[,"mu23"]) + (1-x[,"A1"])/(1-x[,"ps1"])*(x[,"mu23"] - x[,"mu13"]) + x[,"mu13"] - gamma03(x[,"L0"], theta[1])*x[,"A0"]-x[,"mu03"]) ,
                                           x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A2"])/(1-x[,"ps2"]))^(1-x[,"I2"])*((1-x[,"A1"])/(1-x[,"ps1"]))^(1-x[,"I1"])*(x[,"Y3"] -gamma23(x[,"L2"], theta[3])*x[,"A2"]- x[,"mu23"]) + ((1-x[,"A1"])/(1-x[,"ps1"]))^(1-x[,"I1"])*(x[,"mu23"] -gamma13(x[,"L1"], theta[2])*x[,"A1"] - x[,"mu13"]) + x[,"mu13"] - gamma03(x[,"L0"], theta[1])*x[,"A0"]-x[,"mu03"]) ,
                                           x[,"I1"]*x[,"L1"]*(x[,"A1"]-x[,"ps1"])*((1-x[,"A2"])/(1-x[,"ps2"])*(x[,"Y3"] - x[,"mu23"]) + x[,"mu23"] - gamma13(x[,"L1"], theta[2])*x[,"A1"]-x[,"mu13"])  ,
                                           x[,"I1"]*x[,"L1"]*(x[,"A1"]-x[,"ps1"])*(((1-x[,"A2"])/(1-x[,"ps2"]))^(1-x[,"I2"])*(x[,"Y3"]-gamma23(x[,"L2"], theta[3])*x[,"A2"] - x[,"mu23"]) + x[,"mu23"] - gamma13(x[,"L1"], theta[2])*x[,"A1"]-x[,"mu13"])  ,
                                           x[,"I2"]*x[,"L2"]*(x[,"A2"]-x[,"ps2"])*(x[,"Y3"]-gamma23(x[,"L2"], theta[3])*x[,"A2"] - x[,"mu23"])
                                     ) %>%
                                       return()
                                   }
                                   mod <- gmm(g, x = x, t0 = c(0,0,0), type = "iterative")
                                   est_psi <- mod$coefficients
                                   # vcov <- boot_var(df,ps1 = working_ps1, ps0 = working_ps0, mu01 = working_mu01, mu12 = working_mu12, mu02 = working_mu02, iter = 500)
                                   # chisq_stat <- t(mod$coefficients- c(psi01 ,psi02,psi12 )) %*% solve(vcov) %*% (mod$coefficients -  c(psi01 ,psi02,psi12 ))
                                   # reject <- chisq_stat > qchisq(0.95, df = 3)
                                   est <- data.frame(id = i, est_psi03 = est_psi[1], est_psi13 = est_psi[2],est_psi23 = est_psi[3])
                                   return(est)
                                 },
                                 .options = furrr_options(seed = TRUE))
toc()


# the LS-two step estimator

plan("multisession")
tic()
set.seed(seed)

est_ls <- furrr::future_map_dfr(.x = 1:iter,
                                .f = function(i, n= sample) {
                                  df <- generate_data(n,  ps1_int = ps1_int, ps2_int = ps2_int, p_elig = p_elig, delta = delta)
                                  # generate a sample splitting id
                                  id_1 <- sample(1:n, size = n/2, replace = FALSE)
                                  id_2 <- setdiff(1:n, id_1)

                                  df <- working_model(df, id_1, id_2,  ps0 = working_ps0, ps1 = working_ps1, ps2 = working_ps2,
                                                      mu03 = working_mu03, mu13 = working_mu13, mu23 = working_mu23) # apply the working models

                                  est_psi <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(0,0,0)))@"estimates"


                                  # the alpha before phi_2_bar for phi1
                                  df$phi_1 <- df$I1*df$L1*(df$A1-df$ps1)*((1-df$A2)/(1-df$ps2)*(df$Y3-df$mu23)+df$mu23-df$mu13-gamma13(df$L1,est_psi[2])*df$A1)
                                  df$phi_2_bar <- df$I2*exp(df$L2)*(df$A2-df$ps2)*(df$Y3-gamma23(df$L2, est_psi[3])*df$A2 - df$mu23)
                                  df$phi_2 <- df$I2*df$L2*(df$A2-df$ps2)*(df$Y3-gamma23(df$L2, est_psi[3])*df$A2 - df$mu23)

                                   derivative_ratio <- mean( exp(df$L2)*df$I2*(df$A2-df$ps2)*df$A2*(1+df$L2))/mean(df$I2*df$L2*(df$A2-df$ps2)*df$A2*(1+df$L2))
                                  alpha <- (-mean(df$phi_1*df$phi_2_bar)+derivative_ratio*mean(df$phi_1*df$phi_2))/mean((df$phi_2_bar - derivative_ratio*df$phi_2)^2)

                                  # the beta before phi_1_bar for phi0
                                  df$phi_0 <- df$L0*(df$A0-df$ps0)*((1-df$A2)/(1-df$ps2)*(1-df$A1)/(1-df$ps1)*(df$Y3-df$mu23) +(1-df$A1)/(1-df$ps1)*(df$mu23-df$mu13) + df$mu13-gamma03(df$L0, est_psi[1])*df$A0)
                                  df$phi_1_bar <- df$I1*exp(df$L1)*(df$A1-df$ps1)*((1-df$A2)/(1-df$ps2)*(df$Y3-df$mu23)+df$mu23-df$mu13-gamma13(df$L1,est_psi[2])*df$A1)
                                  derivative_ratio <-  mean( exp(df$L1)*df$I1*(df$A1-df$ps1)*df$A1*(2+df$L1))/mean(df$I1*df$L1*(df$A1-df$ps1)*df$A1*(2+df$L1))
                                  beta <- (-mean(df$phi_0*df$phi_1_bar)+derivative_ratio*mean(df$phi_0*df$phi_1))/mean((df$phi_1_bar - derivative_ratio*df$phi_1)^2)





                                  ee_ls_estimator <- function(data){
                                    function(theta){
                                      with(data,
                                           c(L0*(A0-ps0)*((1-A2)/(1-ps2)*(1-A1)/(1-ps1)*(Y3-mu23) +(1-A1)/(1-ps1)*(mu23-mu13) + mu13-gamma03(L0, theta[1])*A0 - mu03)+
                                               beta*I1*exp(L1)*(A1-ps1)*((1-A2)/(1-ps2)*(Y3-mu23)+mu23-mu13-gamma13(L1, theta[2])*A1),
                                             I1*L1*(A1-ps1)*((1-A2)/(1-ps2)*(Y3-mu23)+mu23-mu13-gamma13(L1, theta[2])*A1) + alpha*I2*exp(L2)*(A2-ps2)*(Y3-gamma23(L2, theta[3])*A2 - mu23),
                                             I2*L2*(A2-ps2)*(Y3-gamma23(L2, theta[3])*A2 - mu23)))
                                    }
                                  }

                                  est_psi <-  m_estimate(ee_ls_estimator, data = df, root_control = setup_root_control(start = c(0,0,0)))@"estimates"

                                  # chisq_stat <- 1# conduct the score testing
                                  # reject <- chisq_stat > qchisq(0.95, df = 3) # 2 is the number of parameters we are estimating

                                  # return the result
                                  est <- data.frame(id = i, est_psi03 = est_psi[1], est_psi13 = est_psi[2],est_psi23 = est_psi[3])
                                  return(est)
                                },
                                .options = furrr_options(seed = TRUE))
toc()

# the outputs
est_old %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample,
            psi13_mean = mean(est_psi13, na.rm = TRUE),
            psi13_avar = var(est_psi13, na.rm = TRUE)*(sample),
            psi23_mean = mean(est_psi23, na.rm = TRUE),
            psi23_avar = var(est_psi23, na.rm = TRUE)*(sample))

est_g_estimator %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample,
            psi13_mean = mean(est_psi13, na.rm = TRUE),
            psi13_avar = var(est_psi13, na.rm = TRUE)*(sample),
            psi23_mean = mean(est_psi23, na.rm = TRUE),
            psi23_avar = var(est_psi23, na.rm = TRUE)*(sample))


est_gmm %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample,
            psi13_mean = mean(est_psi13, na.rm = TRUE),
            psi13_avar = var(est_psi13, na.rm = TRUE)*(sample),
            psi23_mean = mean(est_psi23, na.rm = TRUE),
            psi23_avar = var(est_psi23, na.rm = TRUE)*(sample))



est_ls%>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample,
            psi13_mean = mean(est_psi13, na.rm = TRUE),
            psi13_avar = var(est_psi13, na.rm = TRUE)*(sample),
            psi23_mean = mean(est_psi23, na.rm = TRUE),
            psi23_avar = var(est_psi23, na.rm = TRUE)*(sample))



summary(est_old$est_psi03)
summary(est_g_estimator$est_psi03)
summary(est_gmm$est_psi03)
summary(est_ls$est_psi03)


quantile(est_old$est_psi03, probs = c(0.025, 0.975))
quantile(est_g_estimator$est_psi03, probs = c(0.025, 0.975))
quantile(est_gmm$est_psi03, probs = c(0.025, 0.975))
quantile(est_ls$est_psi03, probs = c(0.025, 0.975))



saveRDS(est_old, file = paste0("est_old_correct","_p_elig_",p_elig,"_delta_",delta,".rds"))
saveRDS(est_g_estimator, file = paste0("est_g_estimator_correct","_p_elig_",p_elig,"_delta_",delta,".rds"))
saveRDS(est_gmm, file = paste0("est_gmm_correct","_p_elig_",p_elig,"_delta_",delta,".rds"))
saveRDS(est_ls, file = paste0("est_ls_correct","_p_elig_",p_elig,"_delta_",delta,".rds"))


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
source("working_models.R")
sample <- 5000
iter <-200

# set the propensity score at time 1
ps1_int <- 2

# set the parameters
psi01 <- 2
psi02 <- 2
psi12 <- 2
seed <- 34111

# set the working models
working_ps1 <- TRUE
working_ps0 <- TRUE
working_mu01 <- TRUE
working_mu02 <- TRUE
working_mu12 <- TRUE

delta <- 4
#the conventional estimator
ee_old <- function(data){
  function(theta){
    with(data,
         c(L0*(A0-ps0)*(Y1-gamma01(L0, theta[1])*A0-mu01),
           L0*(A0-ps0)*((1-A1)/(1-ps1)*(Y2-mu12)+mu12-mu02-gamma02(L0, theta[2])*A0),
           I1*L1*(A1-ps1)*(Y2-gamma12(L1, theta[3])*A1 - mu12)))
  }
}

plan("multisession")
set.seed(seed)
tic()
est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n, psi01 = psi01, psi02 = psi02, psi12 = psi12, ps1_int = ps1_int, delta = delta)
  # generate a sample splitting id
  id_1 <- sample(1:n, size = n/2, replace = FALSE)
  id_2 <- setdiff(1:n, id_1)

  df <- working_model(df, id_1, id_2, ps1 = working_ps1, ps0 = working_ps0, mu01 = working_mu01, mu12 = working_mu12, mu02 = working_mu02) # apply the working models

  # with the working models, we can do the estimation
  est_psi <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(0,0,0)))@"estimates"
  ee_score <-  cbind(df$L0*(df$A0-df$ps0)*(df$Y1-gamma01(df$L0, psi01)*df$A0 - df$mu01),
                     df$L0*(df$A0-df$ps0)*((1-df$A1)/(1-df$ps1)*(df$Y2-df$mu12)+df$mu12-df$mu02-gamma02(df$L0,psi02)*df$A0),
                     df$L1*df$I1*(df$A1-df$ps1)*(df$Y2-gamma12(df$L1, psi12)*df$A1 - df$mu12))

  est_cov <-  cov(ee_score )
  chisq_stat <- n*t(colMeans(ee_score)) %*% ginv(est_cov) %*% (colMeans(ee_score)) # conduct the score testing
  reject <- chisq_stat > qchisq(0.95, df = 3) # 2 is the number of parameters we are estimating

  # return the result
  est <- data.frame(id = i, est_psi01 = est_psi[1], est_psi02 = est_psi[2],est_psi12 = est_psi[3],
                    reject = reject)
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()



##############################
# The SNTTEM estimator
ee_g_estimator <- function(data){
  function(theta){
    with(data,
         c(L0*(A0-ps0)*(Y1-gamma01(L0, theta[1])*A0-mu01),
           L0*(A0-ps0)*(((1-A1)/(1-ps1))^(1-I1)*(Y2-gamma12(L1, theta[3])*A1-mu12)+mu12-gamma02(L0, theta[2])*A0-mu02),
           I1*L1*(A1-ps1)*(Y2-gamma12(L1, theta[3])*A1 - mu12)))
  }
}

plan("multisession")
tic()
set.seed(seed)

est_g_estimator <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n, psi01 = psi01, psi02 = psi02, psi12 = psi12, ps1_int = ps1_int, delta = delta)
  # generate a sample splitting id
  id_1 <- sample(1:n, size = n/2, replace = FALSE)
  id_2 <- setdiff(1:n, id_1)

  df <- working_model(df, id_1, id_2, ps1 = working_ps1, ps0 = working_ps0, mu01 = working_mu01, mu12 = working_mu12, mu02 = working_mu02) # apply the working models

  # with the working models, we can do the estimation
  est_psi <-  m_estimate(ee_g_estimator, data = df, root_control = setup_root_control(start = c(0,0,0)))@"estimates"
  ee_score <-  cbind(df$L0*(df$A0-df$ps0)*(df$Y1-gamma01(df$L0, psi01)*df$A0 - df$mu01),
                     df$L0*(df$A0-df$ps0)*(((1-df$A1)/(1-df$ps1))^(1-df$I1)*(df$Y2-gamma12(df$L1, psi12)*df$A1-df$mu12) + df$mu12-gamma02(df$L0,psi02)*df$A0-df$mu02),
                     df$L1*df$I1*(df$A1-df$ps1)*(df$Y2-gamma12(df$L1, psi12)*df$A1 - df$mu12))

  est_cov <-  cov( ee_score )
  chisq_stat <- n*t(colMeans( ee_score )) %*% ginv(est_cov) %*% (colMeans( ee_score )) # conduct the score testing
  reject <- chisq_stat > qchisq(0.95, df = 3) # 2 is the number of parameters we are estimating

  # return the result
  est <- data.frame(id = i, est_psi01 = est_psi[1], est_psi02 = est_psi[2],est_psi12 = est_psi[3],
                    reject = reject)
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
                                   df <- generate_data(n, psi01 = psi01, psi02 = psi02, psi12 = psi12, ps1_int = ps1_int, delta = delta)
                                   # generate a sample splitting id
                                   id_1 <- sample(1:n, size = n/2, replace = FALSE)
                                   id_2 <- setdiff(1:n, id_1)

                                   df <- working_model(df, id_1, id_2, ps1 = working_ps1, ps0 = working_ps0, mu01 = working_mu01, mu12 = working_mu12, mu02 = working_mu02) # apply the working models
                                   x <- as.matrix(df)
                                   g <- function(theta, x){
                                     cbind(x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(x[,"Y1"]-gamma01(x[,"L0"], theta[1])*x[,"A0"]-x[,"mu01"]),
                                           x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))*(x[,"Y2"]-x[,"mu12"])+x[,"mu12"]-x[,"mu02"]-gamma02(x[,"L0"], theta[2])*x[,"A0"]),
                                           x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))^(1-x[,"I1"])*(x[,"Y2"]-gamma12(x[,"L1"], theta[3])*x[,"A1"] - x[,"mu12"]) + x[,"mu12"] - gamma02(x[,"L0"], theta[2])*x[,"A0"] - x[,"mu02"]),
                                           x[,"I1"]*x[,"L1"]*(x[,"A1"]-x[,"ps1"])*(x[,"Y2"]-gamma12(x[,"L1"], theta[3])*x[,"A1"] - x[,"mu12"]) ) %>%
                                       return()
                                   }
                                   mod <- gmm(g, x = x, t0 = c(0,0,0), type = "iterative")
                                   est_psi <- mod$coefficients
                                   chisq_stat <- t(mod$coefficients- c(psi01 ,psi02,psi12 )) %*% solve(mod$vcov) %*% (mod$coefficients -  c(psi01 ,psi02,psi12 ))
                                   reject <- chisq_stat > qchisq(0.95, df = 3)
                                   est <- data.frame(id = i, est_psi01 = est_psi[1], est_psi02 = est_psi[2],est_psi12 = est_psi[3],
                                                     reject = reject)
                                   return(est)
                                 },
                                 .options = furrr_options(seed = TRUE))
toc()



# the outputs

# the LS-two step estimator
ee_ls_estimator <- function(data){
  function(theta){
    with(data,
         c(L0*(A0-ps0)*(Y1-gamma01(L0, theta[1])*A0-mu01),
           L0*(A0-ps0)*(((1-A1)/(1-ps1))*(Y2-mu12)+mu12-gamma02(L0, theta[2])*A0-mu02) + alpha*I1*exp(L0)*(A1-ps1)*(Y2-gamma12(L1, theta[3])*A1 - mu12),
           I1*L1*(A1-ps1)*(Y2-gamma12(L1, theta[3])*A1 - mu12)))
  }
}

plan("multisession")
tic()
set.seed(seed)

est_ls <- furrr::future_map_dfr(.x = 1:iter,
                                .f = function(i, n= sample) {
                                  df <- generate_data(n, psi01 = psi01, psi02 = psi02, psi12 = psi12, ps1_int = ps1_int, delta = delta)
                                  # generate a sample splitting id
                                  id_1 <- sample(1:n, size = n/2, replace = FALSE)
                                  id_2 <- setdiff(1:n, id_1)

                                  df <- working_model(df, id_1, id_2, ps1 = working_ps1, ps0 = working_ps0, mu01 = working_mu01, mu12 = working_mu12, mu02 = working_mu02) # apply the working models

                                  est_psi <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(0,0,0)))@"estimates"

                                  df$phi_0 <- df$L0*(df$A0-df$ps0)*((1-df$A1)/(1-df$ps1)*(df$Y2-df$mu12)+df$mu12-df$mu02-gamma02(df$L0,est_psi[2])*df$A0)
                                  df$phi_1_bar <- df$I1*exp(df$L0)*(df$A1-df$ps1)*(df$Y2-gamma12(df$L1, est_psi[3])*df$A1 - df$mu12)
                                  df$phi_1 <- df$I1*df$L1*(df$A1-df$ps1)*(df$Y2-gamma12(df$L1, est_psi[3])*df$A1 - df$mu12)
                                  derivative_ratio <- mean( exp(df$L0)*df$I1*(df$A1-df$ps1)*df$A1*(1+df$L1))/mean(df$I1*df$L1*(df$A1-df$ps1)*df$A1*(1+df$L1))
                                  df$alpha <- (-mean(df$phi_0*df$phi_1_bar)+derivative_ratio*mean(df$phi_0*df$phi_1))/mean((df$phi_1_bar - derivative_ratio*df$phi_1)^2)


                                  est_psi <-  m_estimate(ee_ls_estimator, data = df, root_control = setup_root_control(start = c(0,0,0)))@"estimates"

                                  chisq_stat <- 1# conduct the score testing
                                  reject <- chisq_stat > qchisq(0.95, df = 3) # 2 is the number of parameters we are estimating

                                  # return the result
                                  est <- data.frame(id = i, est_psi01 = est_psi[1], est_psi02 = est_psi[2], est_psi12 = est_psi[3], reject = reject)
                                  return(est)
                                },
                                .options = furrr_options(seed = TRUE))
toc()


df <- generate_data(sample, psi01 = psi01, psi02 = psi02, psi12 = psi12, ps1_int = ps1_int, delta = delta)
ps1 <- mean(df$A1)


est_old <- est_old %>%
  summarise(psi01_mean = mean(est_psi01, na.rm = TRUE),
            psi01_avar = var(est_psi01, na.rm = TRUE)*sample,
            psi02_mean = mean(est_psi02, na.rm = TRUE),
            psi02_avar = var(est_psi02, na.rm = TRUE)*(sample),
            psi12_mean = mean(est_psi12, na.rm = TRUE),
            psi12_avar = var(est_psi12, na.rm = TRUE)*(sample))

est_old$method <- "Conventional AIPW"

est_g_estimator <- est_g_estimator %>%
  summarise(psi01_mean = mean(est_psi01, na.rm = TRUE),
            psi01_avar = var(est_psi01, na.rm = TRUE)*sample,
            psi02_mean = mean(est_psi02, na.rm = TRUE),
            psi02_avar = var(est_psi02, na.rm = TRUE)*(sample),
            psi12_mean = mean(est_psi12, na.rm = TRUE),
            psi12_avar = var(est_psi12, na.rm = TRUE)*(sample))
est_g_estimator$method <- "Robin's g-estimator"

est_gmm <- est_gmm%>%
  summarise(psi01_mean = mean(est_psi01, na.rm = TRUE),
            psi01_avar = var(est_psi01, na.rm = TRUE)*sample,
            psi02_mean = mean(est_psi02, na.rm = TRUE),
            psi02_avar = var(est_psi02, na.rm = TRUE)*(sample),
            psi12_mean = mean(est_psi12, na.rm = TRUE),
            psi12_avar = var(est_psi12, na.rm = TRUE)*(sample))
est_gmm$method <- "GMM estimator"


est_ls <- est_ls%>%
  summarise(psi01_mean = mean(est_psi01, na.rm = TRUE),
            psi01_avar = var(est_psi01, na.rm = TRUE)*sample,
            psi02_mean = mean(est_psi02, na.rm = TRUE),
            psi02_avar = var(est_psi02, na.rm = TRUE)*(sample),
            psi12_mean = mean(est_psi12, na.rm = TRUE),
            psi12_avar = var(est_psi12, na.rm = TRUE)*(sample))
est_ls$method <- "LS estimator"

temp <- rbind(est_old, est_g_estimator, est_gmm, est_ls)
temp$ps1 <- ps1



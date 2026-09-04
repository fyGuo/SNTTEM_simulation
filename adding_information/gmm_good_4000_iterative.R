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
setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/adding_information")
source("Generate_data.R")
source("working_models.R")
sample <- 4000
iter <-400

# set the propensity score at time 1
ps1_int <- 0
delta <- 0

# set the parameters
theta1 <- 2
theta2 <- 2
theta3 <- 2
eta <- 2
seed <- 502
#the conventional estimator
ee_old <- function(data){
  function(theta){
    with(data,
         c( (A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1:3])*A0),
            (1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1:3])*A0),
            exp(L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1:3])*A0),
            (A1-ps1)*(Y-gamma1(L1, theta[4])*A1 - mu1)))
  }
}

plan("multisession")
set.seed(seed)
tic()
est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n,theta1  = theta1 , theta2 = theta2 , theta3  = theta3, eta = eta , ps1_int = ps1_int, delta = delta)
  # generate a sample splitting id

  df <- working_model(df) # apply the working models

  # with the working models, we can do the estimation
  est <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(0,0,0,0)))@"estimates"
  blip_down <- sum(est[1:3])
  # return the result
  est <- data.frame(id = i,
                    theta1 = est[1],
                    theta2 = est[2],
                    theta3 = est[3],
                    eta = est[4],
                    blip_down = blip_down)
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()




#the gmm estimator
plan("multisession")
tic()
set.seed(seed)

est_gmm <- furrr::future_map_dfr(.x = 1:iter,
                                 .f = function(i, n= sample) {
                                   df <- generate_data(n,theta1  = theta1 , theta2 = theta2 , theta3  = theta3, eta = eta , ps1_int = ps1_int, delta = delta)
                                   # generate a sample splitting id

                                   df <- working_model(df) # apply the working models
                                   x <- as.matrix(df)
                                   g <- function(theta, x){
                                     cbind((x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))*(x[,"Y"]-x[,"mu1"])+x[,"mu1"]-x[,"mu0"]-gamma0(x[,"L0"], theta[1:3])*x[,"A0"]),
                                           (1+x[,"L0"])*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))*(x[,"Y"]-x[,"mu1"])+x[,"mu1"]-x[,"mu0"]-gamma0(x[,"L0"], theta[1:3])*x[,"A0"]),
                                           exp(x[,"L0"])*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))*(x[,"Y"]-x[,"mu1"])+x[,"mu1"]-x[,"mu0"]-gamma0(x[,"L0"], theta[1:3])*x[,"A0"]),
                                           (1+x[,"L0"])*(x[,"A0"]- x[,"ps0"])/(1+x[,"ps0"])*(x[,"A1"]-x[,"ps1"])*(x[,"Y"]-gamma1(x[,"L1"], theta[4])*x[,"A1"] - x[,"mu1"]),
                                           (x[,"A1"]-x[,"ps1"])*(x[,"Y"]-gamma1(x[,"L1"], theta[4])*x[,"A1"] - x[,"mu1"]) ) %>%
                                       return()
                                   }
                                   mod <- gmm(g, x = x, t0 = c(0,0,0,0), type = "iterative")
                                   est<- mod$coefficients
                                   blip_down <- sum(est[1:3])
                                   # return the result
                                   est <- data.frame(id = i,
                                                     theta1 = est[1],
                                                     theta2 = est[2],
                                                     theta3 = est[3],
                                                     eta = est[4],
                                                     blip_down = blip_down)
                                   return(est)
                                 },
                                 .options = furrr_options(seed = TRUE))
toc()



# the outputs

# the LS-two step estimator
ee_ls_estimator <- function(data){
  function(theta){
    with(data,
         c( (A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1:3])*A0) + alpha*(1+L0)*(A0-ps0)/(1+ps0)*(A1-ps1)*(Y-gamma1(L1, theta[4])*A1 - mu1),
            (1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1:3])*A0) + alpha*(1+L0)*(A0-ps0)/(1+ps0)*(A1-ps1)*(Y-gamma1(L1, theta[4])*A1 - mu1),
            exp(L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1:3])*A0) + alpha*(1+L0)*(A0-ps0)/(1+ps0)*(A1-ps1)*(Y-gamma1(L1, theta[4])*A1 - mu1),
            (A1-ps1)*(Y-gamma1(L1, theta[4])*A1 - mu1)))
  }
}

plan("multisession")
tic()
set.seed(seed)

est_ls <- furrr::future_map_dfr(.x = 1:iter,
                                .f = function(i, n= sample) {
                                  df <- generate_data(n,theta1  = theta1 , theta2 = theta2 , theta3  = theta3, eta = eta , ps1_int = ps1_int, delta = delta)
                                  # generate a sample splitting id

                                  df <- working_model(df) # apply the working models

                                  est <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start =c(0,0,0,0)))@"estimates"
                                  phi_0 <- cbind(
                                    (df$A0-df$ps0)*((1-df$A1)/(1-df$ps1)*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0,est[1:3])*df$A0),
                                    (1+df$L0)*(df$A0-df$ps0)*((1-df$A1)/(1-df$ps1)*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0,est[1:3])*df$A0),
                                    exp(df$L0)*(df$A0-df$ps0)*((1-df$A1)/(1-df$ps1)*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0,est[1:3])*df$A0))
                                  phi_1_bar_der <- cbind(mean((1+df$L0)*(df$A0-df$ps0)/(1+df$ps0)*(df$A1-df$ps1)*df$L1*df$A1),
                                                         mean((1+df$L0)*(df$A0-df$ps0)/(1+df$ps0)*(df$A1-df$ps1)*df$L1*df$A1),
                                                         mean((1+df$L0)*(df$A0-df$ps0)/(1+df$ps0)*(df$A1-df$ps1)*df$L1*df$A1))

                                  phi_1_der <- mean((df$A1-df$ps1)*df$L1*df$A1)
                                  ratio <- phi_1_bar_der/phi_1_der
                                  phi_1_bar_resd <- cbind(
                                    ((1+df$L0)*(df$A0-df$ps0)/(1+df$ps0)*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, est[4])*df$A1 - df$mu1)),
                                    ((1+df$L0)*(df$A0-df$ps0)/(1+df$ps0)*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, est[4])*df$A1 - df$mu1)),
                                    ((1+df$L0)*(df$A0-df$ps0)/(1+df$ps0)*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, est[4])*df$A1 - df$mu1))
                                  )
                                  -  as.matrix((df$A1-df$ps1)*(df$Y-gamma1(df$L1, est[4])*df$A1 - df$mu1)) %*% ratio
                                  phi_0_der <- matrix(c(mean((df$A0-df$ps0)*df$A0 ),
                                                        mean((df$A0-df$ps0)*df$A0*df$L0 ),
                                                        mean((df$A0-df$ps0)*df$A0*exp(df$L0) ),
                                                        mean((1+df$L0)*(df$A0-df$ps0)*df$A0 ),
                                                        mean((1+df$L0)*(df$A0-df$ps0)*df$A0*df$L0 ),
                                                        mean((1+df$L0)*(df$A0-df$ps0)*df$A0*exp(df$L0) ),
                                                        mean(exp(df$L0)*(df$A0-df$ps0)*df$A0 ),
                                                        mean(exp(df$L0)*(df$A0-df$ps0)*df$A0*df$L0 ),
                                                        mean(exp(df$L0)*(df$A0-df$ps0)*df$A0*exp(df$L0) )),
                                                      byrow = T, nrow = 3)
                                  phi_0_der_inv <- ginv(phi_0_der)

                                  u <- t(c(1,1,1)) %*% phi_0_der_inv
                                  df$alpha <- -as.numeric(u %*%  cov(phi_1_bar_resd) %*% t(u)^(-1) *(u %*%  cov(phi_0, phi_1_bar_resd) %*% t(u)))

                                  est <-  m_estimate(ee_ls_estimator, data = df, root_control = setup_root_control(start = est))@"estimates"

                                  blip_down <- t(c(1,1,1)) %*% est[1:3]
                                  # return the result
                                  est <- data.frame(id = i,
                                                    theta1 = est[1],
                                                    theta2 = est[2],
                                                    theta3 = est[3],
                                                    eta = est[4],
                                                    blip_down = blip_down)
                                  return(est)
                                },
                                .options = furrr_options(seed = TRUE))
toc()


df <- generate_data(sample,theta1  = theta1 , theta2 = theta2 , theta3  = theta3, eta = eta , ps1_int = ps1_int, delta = delta)
ps1 <- mean(df$A1)



est_old$method <- "Conventional AIPW"

# est_g_estimator <- est_g_estimator %>%
#   summarise(theta1_mean = mean(theta1, na.rm = TRUE),
#             theta1_avar = var(theta1, na.rm = TRUE)*sample,
#             theta2_mean = mean(theta2, na.rm = TRUE),
#             theta2_avar = var(theta2, na.rm = TRUE)*(sample),
#             theta3_mean = mean(theta3, na.rm = TRUE),
#             theta3_avar = var(theta3, na.rm = TRUE)*(sample),
#             eta_mean = mean(eta, na.rm = TRUE),
#             eta_avar = var(eta, na.rm = TRUE)*(sample),
#             blip_down_mean = mean(blip_down, na.rm = TRUE),
#             blip_down_avar = var(blip_down, na.rm = TRUE)*(sample))
# est_g_estimator$method <- "Robin's g-estimator"


est_gmm$method <- "GMM estimator"



est_ls$method <- "LS estimator"

temp <- rbind(est_old,  est_gmm, est_ls)
temp
# temp$ps1 <- ps1
# temp$p_I1 <- p_I1

saveRDS(temp, file = paste0("gmm_good_4000_iterative.rds"))



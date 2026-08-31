# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)
library(geex)
library(MASS)
library(matrixcalc)
sample<-5000
tic()
iter <- 500
seed <- 123
set.seed(seed)
psi1 <- 1.1
#############################################################
# Scenario 1: two-point case #
generate_data <- function(n) {
  U<-rnorm(n,0,1)
  L0<-rnorm(n,1)
  A0<-rbinom(n,1,plogis(1+L0))
  I1 <- 1-A0
  L1<-rnorm(n,1+0.5*A0+U)
  A1<-rbinom(n,1,plogis(1+0.5*A0+0.75*L1))
  Y<-rnorm(n,1+0.5*A0+L0+L0*A0+A1+L1+psi1*L1*A1+U,1)

  ps0<-predict(glm(A0~L0,family = "binomial"),type = "response")
  ps1<-predict(glm(A1~L1+A0,family = "binomial"),type = "response")
  w1<-(1-A1)/(1-ps1)

  mu1<-cbind(1,A0,L1,L0, A0*L0)%*%coef(lm(Y~A0+L1+L0+A0:L0,subset=A1==0))
  mu0<-cbind(1,L0)%*%coef(lm(mu1~L0,subset=A0==0))

  df <- data.frame(L0, A0, L1, I1, A1, Y, ps0, ps1, w1, mu1, mu0)
  return(df)
}
# returns a matrix with dimen(L0) x dimen(L1)
gamma0 <- function(L0, psi0) {
  return(1+L0%*% t(psi0))
}
gamma1 <- function(L1, psi1) {
  return(1+L1 %*% t(psi1))
}
# First we make a estimating function for psi_1. All the following methods use the
# same EF1
EF1 <- function(psi1, df){
  result <- df$I1*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, psi1)*df$A1 - df$mu1)
  return(colMeans(result))
}

EF1_new <- function(psi1, df){
  result <-df$L1*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, psi1)*df$A1 - df$mu1)
  return(colMeans(result))
}

# Method 1: the default way
# for simplicity, we make d0 = 1
# the estimating equation is

ee_old <- function(data){
  function(theta){
    with(data,
         c(L0*(A0-ps0)*(w1*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0),
           I1*L1*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
  }
}


plan("multisession")
tic()
est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n)
  est_psi <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(1,1)))@"estimates"
  Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0
  est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2],
                    Y00 = mean(Y00, na.rm = TRUE))
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()

## Method 2 :new
ee_new <- function(data){
  function(theta){
    with(data,
         c( L0*(A0-ps0)*(w1^(1-I1)*(Y-gamma1(L1, theta[2])*A1 - mu1) + mu1 -gamma0(L0, theta[1])*A0 - mu0),
            I1*L1*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1))
    )
  }
}


plan("multisession")
tic()
est_new <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  df <- generate_data(n)
  est_psi <-  m_estimate(ee_new, data = df, root_control = setup_root_control(start = c(1,1)))@"estimates"
  Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0
  est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2],
                    Y00 = mean(Y00, na.rm = TRUE))
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()


# then with the estimated cov_matrix we can do the combined estimation


ee_identity <- function(data){
  function(theta){
    with(data,
         c((A0-ps0)*(w1*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0),
           I1*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
  }
}

set.seed(seed)
plan("multisession")


tic()
est_combined <- furrr::future_map_dfr(.x = 1:iter,
                                      .f = function(i, n= sample) {
                                        df <- generate_data(n)
                                        I2 <- diag(1,2)
                                        Z <- rbind(I2, I2)
                                        # now do the estimation with the combined estimating equation
                                        est_identity <- m_estimate(ee_identity, data = df, root_control = setup_root_control(start = c(1,1)))@"estimates"
                                        ee_old_component <-  cbind(df$L0*(df$A0-df$ps0)*(df$w1*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0,est_identity[1])*df$A0),
                                                                   df$L1*df$I1*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, est_identity[2])*df$A1 - df$mu1))
                                        ee_new_component <-  cbind(df$L0* (df$A0-df$ps0)*(df$w1^(1-df$I1)*(df$Y-gamma1(df$L1, est_identity[1])*df$A1 - df$mu1)
                                                                                          + df$mu1 -gamma0(df$L0, est_identity[2])*df$A0 - df$mu0),
                                                                   df$I1*df$L1*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, est_identity[2])*df$A1 - df$mu1))
                                        ee_gmm <- cbind(ee_old_component, ee_new_component)

                                        cov_matrix <- cov(ee_gmm)*sample
                                        cov_matrix_inverse <- ginv(cov_matrix)
                                        ee_gmm <- function(theta){
                                          ee_old_component <-  cbind(df$L0*(df$A0-df$ps0)*(df$w1*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0,theta[1])*df$A0),
                                                                     df$L1*df$I1*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, theta[2])*df$A1 - df$mu1))
                                          ee_new_component <-  cbind(df$L0* (df$A0-df$ps0)*(df$w1^(1-df$I1)*(df$Y-gamma1(df$L1, theta[2])*df$A1 - df$mu1)
                                                                                            + df$mu1 -gamma0(df$L0,theta[1])*df$A0 - df$mu0),
                                                                     df$I1*df$L1*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, theta[2])*df$A1 - df$mu1))
                                          ee_gmm_component <- cbind(ee_old_component, ee_new_component)
                                          return( t(colMeans(ee_gmm_component))  %*% cov_matrix_inverse %*%  colMeans(ee_gmm_component) )
                                        }

                                        est_psi <-  optim(par = c(1, 1), fn = ee_gmm, method = "BFGS", hessian = TRUE)$par
                                        Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0

                                        score_statistics <- ee_gmm(c(1,1))*sample
                                        reject <- (score_statistics > qchisq(0.95, df = 2))
                                        est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2],
                                                          Y00 = mean(Y00, na.rm = TRUE),
                                                          reject = reject)

                                        return(est)
                                      },
                                      .options = furrr_options(seed = TRUE))
toc()

#### compare their performance
est_old %>%
  summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
            psi1_sd = sd(est_psi1, na.rm = TRUE)*sqrt(sample),
            psi0_mean = mean(est_psi0, na.rm = TRUE),
            psi0_sd = sd(est_psi0, na.rm = TRUE)*sqrt(sample),
            Y00_mean = mean(Y00, na.rm = TRUE),
            Y00_sd = sd(Y00, na.rm = TRUE)*sqrt(sample))


est_combined %>%
  summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
            psi1_sd = sd(est_psi1, na.rm = TRUE)*sqrt(sample),
            psi0_mean = mean(est_psi0, na.rm = TRUE),
            psi0_sd = sd(est_psi0, na.rm = TRUE)*sqrt(sample),
            Y00_mean = mean(Y00, na.rm = TRUE),
            Y00_sd = sd(Y00, na.rm = TRUE)*sqrt(sample),
            reject_rate = mean(reject, na.rm = TRUE))

Sys.time()

# calculate the asymptotic variance matrix
var_old <- cov(est_old[,c("est_psi0", "est_psi1")])*sample
var_combined <- cov(est_combined[,c("est_psi0", "est_psi1")])*sample
var_old-var_combined
eigen(var_old -var_combined)
is.positive.semi.definite(var_old-var_combined)
toc()

wald_stat <- numeric(iter)
for (i in 1:iter){
  est_psi <- c(est_old$est_psi0[i], est_old$est_psi1[i])
  wald_stat[i] <- t(est_psi - c(1,1)) %*% ginv(var_old/sample) %*% (est_psi - c(1,1))
}
mean(wald_stat > qchisq(0.95, df = 2))


wald_stat <- numeric(iter)
for (i in 1:iter){
  est_psi <- c(est_combined$est_psi0[i], est_combined$est_psi1[i])
  wald_stat[i] <- t(est_psi - c(1,1)) %*% ginv(var_combined/sample) %*% (est_psi - c(1,1))
}
mean(wald_stat > qchisq(0.95, df = 2))

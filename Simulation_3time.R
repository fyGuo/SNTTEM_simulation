##############################
# this script is for heteroscedasicty in a three-point case
# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)

n<-5000
iter <- 1000
set.seed(134)
#############################################################
# Scenario 2: two-point case #
generate_data <- function(n) {
  U<-rnorm(n,0,1)
  L0<-rnorm(n,1)
  A0<-rbinom(n,1,plogis(1+L0))

  I1 <- 1-A0
  L1<-rnorm(n,1+L0+A0+U)
  A1<-rbinom(n,1,plogis(1+0.5*A0+0.75*L1))

  I2 <- 1-A1
  L2 <- rnorm(n, 1+L1+0.5*A1 + U)
  A2 <- rbinom(n, 1, plogis(1+0.5*A1+0.75*L2))

  Y<- rnorm(n,1+U+L0+A0+L1+A1+L2+A2+L2*A2,sqrt(1+A2))

  ps0<-predict(glm(A0~L0,family = "binomial"),type = "response")
  ps1<-predict(glm(A1~L1+A0,family = "binomial"),type = "response")
  ps2<-predict(glm(A2~L2+A1,family = "binomial"),type = "response")
  w1<-(1-A1)/(1-ps1)
  w2<-(1-A2)/(1-ps2)

  mu2 <- cbind(1, L0, A0, L1, A1, L2)%*%coef(lm(Y~L0+A0+L1+A1+L2,subset=A2==0))
  mu1<-cbind(1,A0,L1,L0)%*%coef(lm(mu2~A0+L1+L0,subset=A1==0))
  mu0<-cbind(1,L0)%*%coef(lm(mu1~L0,subset=A0==0))

  df <- data.frame(L0, A0, L1, I1, A1, L2, I2, A2, Y, ps0, ps1, ps2, w1, w2, mu2, mu1, mu0)
  return(df)
}

# returns a matrix with dimen(L0) x dimen(L1)
gamma0 <- function(L0, psi0) {
    L0 <- rep(1, length(L0))
    return(L0 %*% t(psi0))
}
gamma1 <- function(L1, psi1) {
  L1 <- rep(1, length(L1))
  return(L1 %*% t(psi1))
}
gamma2 <- function(L2, psi2) {
  return(1 + L2 %*% t(psi2))
}

# First we make a estimating function for psi_2. All the following methods use the
# same EF2
EF2 <- function(psi2, df){
  result <- (df$A2-df$ps2)*(df$Y-gamma2(df$L2, psi2)*df$A2 - df$mu2)
  return(colMeans(result))
}



# Method 1: the default way
# for simplicity, we make d0 = 1
# the estimating equation is
f_old <- function(psi0, df) {

  result <- (df$A0-df$ps0)*(df$w2*df$w1*(df$Y-df$mu2) + df$w1*(df$mu2- df$mu1)+
                              df$mu1-df$mu0-gamma0(df$L0, psi0)*df$A0)

  return(colMeans(result))
}
plan("multisession")
tic()
est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= 5000) {
  df <- generate_data(n)
  est_psi0 <- uniroot.all(f_old, c(-10, 10), df = df)
  est_psi0 <- ifelse(length(est_psi0) == 0, NA, est_psi0)
  est <- data.frame(id = i, est_psi0 = est_psi0)
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()


# Method 2: the projection way
# the projection approach relies on an estimate of psi1, so let's make a estimating function for psi1
EF1 <- function(psi1, df){
  result <- (df$A1-df$ps1)*(df$w2*(df$Y - df$mu2) + df$mu2-gamma1(df$L1, psi1)*df$A1 - df$mu1)
  return(colMeans(result))
}

# then let's construct a estimationg equation for psi0, by stepwise LS projections,

EF0_LS <- function(psi0, df, psi1, psi2){
  phi_0 <- (df$A0-df$ps0)*(df$w2*df$w1*(df$Y-df$mu2) + df$w1*(df$mu2- df$mu1)+
                    df$mu1-df$mu0-gamma0(df$L0, psi0)*df$A0)
  df$d_1 <- df$L1
  phi_1 <- df$I1*(df$A1-df$ps1)*df$d_1*(df$w2*(df$Y - df$mu2) + df$mu2-gamma1(df$L1, psi1)*df$A1 - df$mu1)

  # do project phi_0 onto the linear span by phi_1 by OLS
  phi_0_tilde <- matrix(ncol = ncol(phi_0), nrow =nrow(df))
  for (i in 1:ncol(phi_0)) {
    LS <- lm(phi_0[,i] ~ -1+phi_1)
    phi_0_tilde[,i] <-t(phi_0[,i]) - coef(LS)%*%t(phi_1)
  }


  # project phi_0_tilde onto the linear space spanned by phi_2
  phi_1_tilde <- matrix(ncol = ncol(phi_0_tilde), nrow =nrow(df))
  df$d_2 <- df$L2
  phi_2 <- df$I2*(df$A2-df$ps2)*df$d_2*(df$Y - df$mu2 - gamma2(df$L2, psi2)*df$A2)

  for (i in 1:ncol(phi_1_tilde)) {
    LS <- lm(phi_0_tilde[,i] ~ -1+phi_2)
    phi_1_tilde[,i] <-t(phi_0_tilde[,i]) - coef(LS)%*%t(phi_2)
  }

  return(colMeans(phi_1_tilde))
}

plan("multisession")
tic()
est_LS <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= 5000) {
  df <- generate_data(n)
  est_psi2 <- uniroot.all(EF2, c(-10, 10), df = df)
  est_psi1 <- uniroot.all(EF1, c(-10, 10), df = df)
  if(length(est_psi1) != 0 & length(est_psi2) != 0){
    est_psi0 <- uniroot.all(EF0_LS, c(-10, 10), df = df, psi1 = est_psi1, psi2 = est_psi2)
    est_psi0 <- ifelse(length(est_psi0) == 0, NA, est_psi0)
    est <- data.frame(id = i, est_psi0 = est_psi0)
  } else {
    est <- data.frame(id = i, est_psi0 = NA)
  }
  return(est)
},
.options = furrr_options(seed = TRUE))
toc()


# compare the results
est_old  %>% summarise(mean = mean(est_psi0, na.rm = TRUE), sd= sd(est_psi0, na.rm = TRUE))
est_LS %>% summarise(mean = mean(est_psi0, na.rm = TRUE), sd = sd(est_psi0, na.rm = TRUE))

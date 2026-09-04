
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)

n<-5000
iter <- 1000
set.seed(144)
#############################################################
# Scenario 1: two-point case #
generate_data <- function(n) {
       U<-rnorm(n,0,1)
      L0<-rnorm(n,1)
       A0<-rbinom(n,1,plogis(1+L0))
    I1 <- 1
       L1<-rnorm(n,1+L0+0.5*A0+U)
       A1<-rbinom(n,1,plogis(1+0.5*A0+0.75*L1))
       Y<-rnorm(n,1+0.5*A0+L0+A0*L0+A1+L1+A1*L1+U,1+10*A1)

         ps0<-predict(glm(A0~L0,family = "binomial"),type = "response")
         ps1<-predict(glm(A1~L1+A0,family = "binomial"),type = "response")
         w1<-(1-A1)/(1-ps1)

           mu1<-cbind(1,A0,L1,L0)%*%coef(lm(Y~A0+L1+L0,subset=A1==0))
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

   EF1 <- function(psi1, df){
       result <- (df$A1-df$ps1)*(df$Y-gamma1(df$L1, psi1)*df$A1 - df$mu1)
       return(colMeans(result))
     }

   f_old <- function(psi0, df) {

         result <- (df$A0-df$ps0)*(df$w1*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0, psi0)*df$A0)

           return(colMeans(result))
       }
plan("multisession")
 tic()
 est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= 5000) {
     df <- generate_data(n)

       est_psi1 <- uniroot.all(EF1, c(-10, 10), df = df)
       est_psi1 <- if_else(is.na(est_psi1), NA, est_psi1)

         est_psi0 <- uniroot.all(f_old, c(-10, 10), df = df)
         est_psi0 <- if_else(is.na(est_psi0), NA, est_psi0)
         est <- data.frame(id = i, est_psi1, est_psi0)
         return(est)
       },
   .options = furrr_options(seed = TRUE))
 toc()


   # Method 2: the optimal estimating equation in this case
   f_new <- function(psi0, df, psi1) {
       psi1 <-rep(psi1,length(psi0))
       result <- (df$A0-df$ps0)*(df$w1^(1-df$I1)*(df$Y-gamma1(df$L1, psi1)*df$A1 - df$mu1) +
                                    +                               df$mu1 -gamma0(df$L0, psi0)*df$A0 - df$mu0)
       return(colMeans(result))
     }

   plan("multisession")

   tic()
 est_new <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= 5000) {
    df <- generate_data(n)

       est_psi1 <- uniroot.all(EF1, c(-10, 10), df = df)
       est_psi1 <- if_else(is.na(est_psi1), NA, est_psi1)

         est_psi0 <- uniroot.all(f_new, c(-10, 10), df = df, psi1 = est_psi1)
         est_psi0 <- if_else(is.na(est_psi0), NA, est_psi0)
         est <- data.frame(id = i, est_psi1, est_psi0)
         return(est)
       },
  .options = furrr_options(seed = TRUE))
 toc()


   # method 3 estimation via LS projection
   f_projection <- function(psi0, df, psi1) {
       phi_0 <- (df$A0-df$ps0)*(df$w1*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0, psi0)*df$A0)
       df$d_1 <- 1-df$ps0
       phi_1 <- df$I1*(df$A1-df$ps1)*df$d_1*(df$Y-gamma1(df$L1, psi1)*df$A1-df$mu1)

       df$d_2 <- exp(df$ps1)
       phi_2 <- df$I1*(df$A1-df$ps1)*df$d_2*(df$Y-gamma1(df$L1, psi1)*df$A1-df$mu1)

       df$d_3<- df$L1/(1-df$ps1)
       phi_3 <- df$I1*(df$A1-df$ps1)*df$d_3*(df$Y-gamma1(df$L1, psi1)*df$A1-df$mu1)

       df$d_4<- df$L0/exp(df$A0-df$ps0)
       phi_4 <- df$I1*(df$A1-df$ps1)*df$d_4*(df$Y-gamma1(df$L1, psi1)*df$A1-df$mu1)


       df$d_5<- df$L0^2-(df$ps1)
       phi_5 <- df$I1*(df$A1-df$ps1)*df$d_5*(df$Y-gamma1(df$L1, psi1)*df$A1-df$mu1)


       df$d_6<- exp(df$L1/exp(df$A0-df$ps0))+df$ps1
       phi_6 <- df$I1*(df$A1-df$ps1)*df$d_6*(df$Y-gamma1(df$L1, psi1)*df$A1-df$mu1)


       df$d_7<-(df$A0)+exp(df$ps1)+(exp(df$L0)-df$A1+1)
       phi_7 <- df$I1*(df$A1-df$ps1)*df$d_7*(df$Y-gamma1(df$L1, psi1)*df$A1-df$mu1)


       df$d_8<-exp(df$A1+df$L0-10/exp(df$A1))
       phi_8 <- df$I1*(df$A1-df$ps1)*df$d_8*(df$Y-gamma1(df$L1, psi1)*df$A1-df$mu1)

         # do project phi_0 onto the linear span by phi_1 by OLS
         phi_0_tilde <- numeric(ncol(phi_0))
         for (i in 1:ncol(phi_0)) {
             LS <- lm(phi_0[,i] ~ -1+phi_1+phi_2+phi_3+phi_4+phi_5 + phi_6 + phi_7 + phi_8)
             phi_0_tilde[i] <-sum(t(phi_0[,i]) - coef(LS)%*%t(cbind(phi_1,phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8)))
           }

           return(phi_0_tilde/nrow(df))
       }

   plan("multisession")
 tic()
 est_projection <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= 5000) {
     df <- generate_data(n)

       est_psi1 <- uniroot.all(EF1, c(-10, 10), df = df)
       est_psi1 <- if_else(is.na(est_psi1), NA, est_psi1)

         est_psi0 <- uniroot.all(f_projection, c(-10, 10), df = df, psi1 = est_psi1)
         est_psi0 <- if_else(is.na(est_psi0), NA, est_psi0)
         est <- data.frame(id = i, est_psi1, est_psi0)
         return(est)
       },
   .options = furrr_options(seed = TRUE))
 toc()

   #### compare their performance
  est_old %>%
     summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
                             psi1_sd = sd(est_psi1, na.rm = TRUE),
                             psi0_mean = mean(est_psi0, na.rm = TRUE),
                             psi0_sd = sd(est_psi0, na.rm = TRUE))


   est_new %>%
     summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
                             psi1_sd = sd(est_psi1, na.rm = TRUE),
                             psi0_mean = mean(est_psi0, na.rm = TRUE),
                             psi0_sd = sd(est_psi0, na.rm = TRUE))



  est_projection %>%
     summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
                             psi1_sd = sd(est_psi1, na.rm = TRUE),
                             psi0_mean = mean(est_psi0, na.rm = TRUE),
                             psi0_sd = sd(est_psi0, na.rm = TRUE))


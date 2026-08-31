# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)
library(geex)
library(MASS)
library(matrixcalc)

# the function is to generate the data
generate_data <- function(n,  psi02, psi12, ps1_int = 1, p_I1 = 0.5, p1 = 0.05) {
  I0 <- 1
  L0<-runif(n, 0, 0.5)
  A0<-rbinom(n,1,plogis(-1+L0))
  I1 <- rbinom(n, 1, p_I1)
  L1<-rnorm(n,0.5*A0)
  L1 <- ifelse(abs(L1) > 0.5, 0.5, L1)
  A1<-ifelse(rbinom(n,1,p1) == 1, A0, 1-A0)
  Y2<-rbinom(n,1,exp(-2+L0+psi02*L0*A0 + I1*psi12*(1+L1)*(A1-A0) +(1-I1)*psi12*L1*(A1-A0)))
  id <- 1:n
  df <- data.frame(id, I0,L0, A0, L1, I1, A1, Y2)
  return(df)
}

## blip function at time 1 for variable Y2
gamma12 <- function(L1, psi12) {
  return( psi12*(1+L1) )
}

## blip function at time 0 for variable Y2
gamma02 <- function(L0, psi02) {
  return(psi02*(L0))
}


# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)
library(geex)
library(MASS)
library(matrixcalc)

# the function is to generate the data
generate_data <- function(n,theta1, theta2, theta3, eta, delta = 0, ps1_int = 0) {
  L0<-rnorm(n, 0, 1)
  A0<-rbinom(n,1,plogis(1+L0))
  L1<-rnorm(n,1, 1)
  A1<-rbinom(n,1,plogis(ps1_int+0.5*A0+0.75*L1))
  Y<-rnorm(n,1+eta*A1*L1 + L1 + theta1*A0 + theta2*A0*L0 + theta3*A0*exp(L0),
            sqrt(1+delta*A1))
  id <- 1:n
  df <- data.frame(id, L0, A0, L1,  A1,  Y)
  return(df)
}

## blip function at time 1 for variable Y2
gamma1 <- function(L1, eta) {
  return( eta*(L1) )
}

## blip function at time 0 for variable Y2
gamma0 <- function(L0, theta) {
  return(theta[1]*1 + theta[2]*L0 + theta[3]*exp(L0))
}

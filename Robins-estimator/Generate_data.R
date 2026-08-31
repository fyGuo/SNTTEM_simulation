# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)
library(geex)
library(MASS)
library(matrixcalc)

# the function is to generate the data
generate_data <- function(n, psi0, psi1, ps1_control, varY_control, theta) {
  I0 <- rbinom(n, 1, 0.5)
  U<-rnorm(n,0,1)
  L0<-rnorm(n, 0, 1)
  A0<-rbinom(n,1,plogis(1+L0))
  I1 <- 1
  L1<-rnorm(n,1+0.5*A0+theta*U)
  A1<-rbinom(n,1,plogis(ps1_control+0.5*A0+0.75*L1))

    Y<-rnorm(n,I1*(1+L0 + psi0*(0.5*A0 + L0*A0 + L1)+psi1*(A1+L1*A1)+theta*U)+
               (1-I1)*(1+L0 + psi0*(0.5*A0 + L0*A0 + L1)+psi1*(100*A1+L1^2*A1)+theta*U),
             sqrt(1+ varY_control*A1))



  df <- data.frame(I0,L0, A0, L1, I1, A1, Y)
  return(df)
}

## blip function at time 1
gamma1 <- function(L1, psi1) {
 return( psi1*(1+L1) )
}

## blip function at time 0
gamma0 <- function(L0, psi0) {
  return(psi0 * (1+L0))
}


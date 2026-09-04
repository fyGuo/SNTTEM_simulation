# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)
library(geex)
library(MASS)
library(matrixcalc)

# the function is to generate the data
generate_data <- function(n, psi01, psi02, psi12, ps1_int = 1, delta = 0, p_I1 = 0.5) {
  I0 <- 1
  L0<-rnorm(n, 0, 1)
  A0<-rbinom(n,1,plogis(1+L0))
  Y1 <- rnorm(n, 1+ L0 + psi01*A0, 1)
  I1 <- rbinom(n, 1, p_I1)
  L1<-rnorm(n,1+0.5*A0)
  A1<-rbinom(n,1,plogis(ps1_int+0.5*A0+0.75*L1))
  Y2<-rnorm(n,I1*(1+L0+psi02*L0*A0 + psi01*(0.5*A0  + L1)+psi12*(A1+L1*A1))+
              (1-I1)*(1+L0 + psi02*L0*A0 + psi01*(0.5*A0 + L1)+psi12*(100*A1+L1^2*A1))-Y1,
            sqrt(1+delta*A1))
  id <- 1:n
  df <- data.frame(id, I0,L0, A0, L1, I1, A1, Y1, Y2)
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

## blip function at time 0 for variable Y1'
gamma01 <- function(L0, psi01) {
  return(psi01)
}

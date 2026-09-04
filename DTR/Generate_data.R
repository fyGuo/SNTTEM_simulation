# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)
library(geex)
library(MASS)
library(matrixcalc)

# the function is to generate the data
generate_data <- function(n,  psi01, psi02, psi11, psi12, ps1_int = 1, p_I1 = 0.5) {
  I0 <- 1
  L0<-runif(n, 0, 0.5)
  A0<-rbinom(n,1,plogis(-1+L0))
  I1 <- rbinom(n, 1, p_I1)
  L1<-runif(n, -psi11, A0)
  A1<-rbinom(n,1,plogis(ps1_int+0.5*A0+0.75*L1))
  Y2<-rnorm(n, -L1*psi12 + (L0 + psi01)*psi02*A0 + L0 + (psi11+L1)*psi12*A1)
  id <- 1:n
  df <- data.frame(id, I0,L0, A0, L1, I1, A1, Y2)
  return(df)
}

## blip function at time 1 for variable Y2
gamma12 <- function(L1, psi11, psi12) {
  return( psi12*(psi11+L1) )
}

## blip function at time 0 for variable Y2
gamma02 <- function(L0, psi01,  psi02) {
  return(psi02*(L0+psi01
                ))
}


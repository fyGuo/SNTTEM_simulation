# load packages
library(tidyverse)
library(rootSolve)
library(furrr)
library(tictoc)
library(geex)
library(MASS)
library(matrixcalc)

# the function is to generate the data with three time case
generate_data <- function(n, ps1_int, ps2_int, p_elig, delta) {
  I0 <- 1
  L0<-rnorm(n, 0, 1)
  A0<-rbinom(n,1,plogis(1+L0))
  I1 <- rbinom(n,1, p_elig)
  L1<-rnorm(n,1+A0+L0, 1)
  A1<-rbinom(n,1,plogis(ps1_int+0.5*A0+0.75*L1))
  I2 <- rbinom(n,1, p_elig)
  L2 <- rnorm(n, 1+A1+L1, 1)
  A2 <- rbinom(n,1,plogis(ps2_int+0.5*A1+0.75*L2))
  Y3<-rnorm(n,1+L0+L1+L2+A0+A1+A2+A0*L0+A1*L1+A2*L2,
            sqrt(1+delta*A1+delta*A2))
  id <- 1:n
  df <- data.frame(id, I0,L0, A0, I1, L1, A1, I2, L2, A2,Y3 )
  return(df)
}

## blip function at time 2 for variable Y3
gamma23 <- function(L2, psi23) {
  return( psi23*(1+L2) )
}

## blip function at time 1 for variable Y3
gamma13 <- function(L1, psi13) {
  return(psi13*(2+L1))
}

## blip function at time 0 for variable Y3
gamma03 <- function(L0, psi03) {
  return(psi03*(3+L0))
}

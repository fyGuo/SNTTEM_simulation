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
setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/PS1_Efficiency_delta_0_small_sample")



tmp <- read_rds("simulation_results.rds")

ggplot(tmp, aes(x=ps1, y=psi02_avar) ) +
  geom_line(aes(y=psi02_avar, colour=method))  +
  theme_minimal()

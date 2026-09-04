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
setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/PS1_Efficiency_delta_05")



#we extract the data from the previous result
ps1_int <- c(-6, -3, -2, -1, 0, 1, 2, 3)
ps1 <- c(0.0134, 0.1908, 0.353, 0.5838, 0.776, 0.8962,  0.9558, 0.9842)
avar_con <- c( 90.88899, 94.20604,  103.7927, 127.1775, 187.314,  365.6641, 853.6652, 3365.559)
avar_g_estimator <- c(  90.96378, 93.50626, 102.424, 115.753, 160.2348,  271.1679, 611.0462, 2440.932  )
avar_gmm <- c( 91.27625,  94.37877, 102.6201, 116.603, 160.5463,  272.8043, 613.0184,  2424.8 )

# combine the results
temp <- data.frame(ps1_int, ps1, avar_con, avar_g_estimator, avar_gmm, delta = 0)

ggplot(tmp, aes(x=ps1, y=psi02_avar) ) +
  geom_line(aes(y=1/psi02_avar, colour=method))  +
  theme_minimal()

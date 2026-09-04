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
setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Eligibility_homo_PS1_3")

# read


# read PS1_-3 results
dta <- read_rds("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Eligibility_homo_PS1_-3/simulation_results.rds")
dta$ps1 <- 0.1908

dta <- dta[,names(temp)]
temp <- dta

# read PS1_0 results
dta <- read_rds("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Eligibility_homo_PS1_0/simulation_results.rds")
dta$ps1 <- 0.776

temp <- rbind(temp, dta)

# read PS1_3 results
dta <- read_rds("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Eligibility_homo_PS1_3/simulation_results.rds")
dta$ps1 <- 0.9842

temp <- rbind(temp, dta)

temp$ps1 <- paste0("Pr(A[1])==", temp$ps1)

### draw the graph
ggplot(temp, aes(x = p_I1, y = 1/psi02_avar, color = method)) +
  geom_point(size = 3) +
  geom_line() +
  facet_wrap(.~ps1, labeller = "label_parsed") +
  labs(x = expression(Pr(I[1])==1), y = "Asymptotic Efficiency", color = "Estimation Method", shape = "Delta Value") +
  theme_minimal()


ggsave("efficiency_comparison_elgibility_homo.png", width = 10, height = 6)

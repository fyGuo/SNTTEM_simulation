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
source("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Robins-estimator/beta_selection.R")
source("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Robins-estimator/Generate_data.R")
source("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Robins-estimator/working_models.R")

tic()
sample <- 32000
iter <- 200
psi <- c(0, 0)
seed <- 1652
varY_control <- 4
q <- sample/1000 - 1
temp <- data.frame(old_eff = numeric(13), SNTTEM_eff = numeric(13), ps1 = numeric(13))
  ps1_control <- 0
  # set the working models
  working_ps1 <- TRUE
  working_ps0 <- TRUE
  working_mu1 <- TRUE
  working_mu0 <- TRUE

  # the conventional estimator
  ee_old <- function(data){
    function(theta){
      with(data,
           c((1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0),
             (1+L1)/(1+2*(varY_control + varY_control^2)*(1-ps1))*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
    }
  }

  # plan("multisession")
  # set.seed(seed)
  # tic()
  # est_old <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
  #   df <- generate_data(n, psi0 = psi[1], ps1_control = ps1_control, psi1 = psi[2], theta = 0, varY_control = varY_control)
  #
  #   # generate a sample splitting id
  #   id_1 <- sample(1:n, size = n/2, replace = FALSE)
  #   id_2 <- setdiff(1:n, id_1)
  #
  #   df <- working_model(df, id_1, id_2, ps1 = working_ps1, ps0 = working_ps0, mu1 = working_mu1, mu0 = working_mu0) # apply the working models
  #   df$denom1 <- mean((1-df$L0)*(df$A0-df$ps0)*df$ps1)
  #   df$denom2 <-mean((df$L0)*(df$A0-df$ps0)*df$ps1)
  #   # with the working models, we can do the estimation
  #   est_psi <-  m_estimate(ee_old, data = df, root_control = setup_root_control(start = c(0,0)))@"estimates"
  #   Y00 <- df$Y - gamma1(df$L1, est_psi[2]) * df$A1 - gamma0(df$L0, est_psi[1]) * df$A0
  #   ee_old_score <-  cbind(df$L0*(df$A0-df$ps0)*((1-df$A1)/(1-df$ps1)*(df$Y-df$mu1)+df$mu1-df$mu0-gamma0(df$L0,0)*df$A0),
  #                          ((1-df$L0)/df$denom1+df$L0/df$denom2)*     (df$A0-df$ps0)/(1-df$ps1)*(df$A1-df$ps1)*(df$Y-gamma1(df$L1, 0)*df$A1 - df$mu1))
  #
  #   est_cov <-  cov(ee_old_score )
  #   chisq_stat <- n*t(colMeans(ee_old_score)) %*% ginv(est_cov) %*% (colMeans(ee_old_score)) # conduct the score testing
  #   reject <- chisq_stat > qchisq(0.95, df = 2) # 2 is the number of parameters we are estimating
  #
  #   # return the result
  #   est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2],
  #                     Y00 = mean(Y00, na.rm = TRUE),
  #                     reject = reject,
  #                     ps1 = mean(df$A1))
  #   return(est)
  # },
  # .options = furrr_options(seed = TRUE))
  # toc()
  #


  ##############################
  # The SNTTEM estimator


  plan("multisession")
  tic()
  set.seed(seed)

  est_selection <- furrr::future_map_dfr(.x = 1:iter, .f = function(i, n= sample) {
    df <- generate_data(n, psi0 = psi[1], ps1_control, psi1 = psi[2], theta = 0, varY_control = varY_control)
    # generate a sample splitting id
    id_1 <- sample(1:n, size = n/2, replace = FALSE)
    id_2 <- setdiff(1:n, id_1)

    df <- working_model(df, id_1, id_2, ps1 = working_ps1, ps0 = working_ps0, mu1 = working_mu1, mu0 = working_mu0) # apply the working models
    df$denom1 <- mean((1-df$L0)*(df$A0-df$ps0)*df$ps1)
    df$denom2 <-mean((df$L0)*(df$A0-df$ps0)*df$ps1)

    beta <- beta_selection(0:q, df[id_1,])
    ee_selection <- function(data) {
      function(theta){
        with(data,
             c((1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0) +
                 (1+L0)*beta*(ps1)*(A0-ps0)*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1),
               (1+L1)/(1+2*(varY_control + varY_control^2)*(1-ps1))*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
      }
    }
    est_psi_1 <-  m_estimate(  ee_selection , data = df[id_2,], root_control = setup_root_control(start = c(0,0)))@"estimates"
    beta1 <- beta

    beta <- beta_selection(0:q, df[id_2,])
    ee_selection <- function(data) {
      function(theta){
        with(data,
             c((1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0) +
                 (1+L0)*beta*(ps1)*(A0-ps0)*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1),
               (1+L1)/(1+2*(varY_control + varY_control^2)*(1-ps1))*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
      }
    }
    beta2 <- beta
    est_psi_2 <-  m_estimate(  ee_selection , data = df[id_1,], root_control = setup_root_control(start = c(0,0)))@"estimates"

    est_psi <- (est_psi_1 + est_psi_2)/2



    # return the result
    est <- data.frame(id = i, est_psi0 = est_psi[1], est_psi1 = est_psi[2], beta1 = beta1, beta2 = beta2)
    return(est)
  },
  .options = furrr_options(seed = TRUE))




  # the outputs
# est_old %>%
#     summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
#               psi1_avar = var(est_psi1, na.rm = TRUE)*(sample),
#               psi0_mean = mean(est_psi0, na.rm = TRUE),
#               psi0_avar = var(est_psi0, na.rm = TRUE)*(sample),
#               psi0_eff = 1/(var(est_psi0, na.rm = TRUE)*(sample)),
#               Y00_mean = mean(Y00, na.rm = TRUE),
#               Y00_sd = sd(Y00, na.rm = TRUE)*sqrt(sample),
#               reject_rate = mean(reject, na.rm = TRUE),
#               ps1 = mean(ps1, na.rm = TRUE))
est_selection  %>%
    summarise(psi1_mean = mean(est_psi1, na.rm = TRUE),
              psi1_avar = var(est_psi1, na.rm = TRUE)*(sample),
              psi0_mean = mean(est_psi0, na.rm = TRUE),
              psi0_avar = var(est_psi0, na.rm = TRUE)*(sample),
              psi0_eff = 1/(var(est_psi0, na.rm = TRUE)*(sample)))

# temp <- data.frame(psi0 = sqrt(sample)*c(est_old$est_psi0, est_selection$est_psi0),
#                    estimator = c(rep(c("old", "selected"), each = dim(est_old)[1])))

# ggplot(temp, aes(x = psi0, fill = estimator)) +
#   geom_density(alpha = 0.5) +
#   facet_grid(estimator~.) +
#   theme_bw()

table(c(est_selection$beta1, est_selection$beta2)) %>% prop.table()
ggplot(est_selection, aes(x = est_psi0)) +
  geom_density(alpha = 0.5) +
  theme_bw()

shapiro.test(est_selection$est_psi0)
toc()

qqnorm(est_selection$est_psi0)

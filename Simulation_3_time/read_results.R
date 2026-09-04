#read the simulation results out
library(tidyverse)
# elgibility 0.9, delta 8

dta_old <- readRDS("est_old_correct_p_elig_0.9_delta_8.rds")

dta_g_estimator <- readRDS("est_g_estimator_correct_p_elig_0.9_delta_8.rds")

dta_gmm <- readRDS("est_gmm_correct_p_elig_0.9_delta_8.rds")

dta_ls <- readRDS("est_ls_correct_p_elig_0.9_delta_8.rds")

# read the results for psi03
dta_old %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_old$est_psi03 %>% summary()
dta_old$est_psi03 %>% quantile(c(0.05, 0.95))



dta_g_estimator %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_g_estimator$est_psi03 %>% summary()
dta_g_estimator$est_psi03 %>% quantile(c(0.05, 0.95))


dta_gmm %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)

dta_gmm$est_psi03 %>% summary()
dta_gmm$est_psi03 %>% quantile(c(0.05, 0.95))


dta_ls %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_ls$est_psi03 %>% summary()
dta_ls$est_psi03 %>% quantile(c(0.05, 0.95))

# read the results for elgibility 0.1, delta 0 and do the same operation as above
dta_old <- readRDS("est_old_correct_p_elig_0.1_delta_0.rds")
dta_g_estimator <- readRDS("est_g_estimator_correct_p_elig_0.1_delta_0.rds")
dta_gmm <- readRDS("est_gmm_correct_p_elig_0.1_delta_0.rds")
dta_ls <- readRDS("est_ls_correct_p_elig_0.1_delta_0.rds")
dta_old %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_old$est_psi03 %>% summary()
dta_old$est_psi03 %>% quantile(c(0.05, 0.95))

dta_g_estimator %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_g_estimator$est_psi03 %>% summary()
dta_g_estimator$est_psi03 %>% quantile(c(0.05, 0.95))

dta_gmm %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_gmm$est_psi03 %>% summary()
dta_gmm$est_psi03 %>% quantile(c(0.05, 0.95))

dta_ls %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_ls$est_psi03 %>% summary()
dta_ls$est_psi03 %>% quantile(c(0.05, 0.95))


# read results for p_elibg = 0.1, delta = 8
dta_old <- readRDS("est_old_correct_p_elig_0.1_delta_8.rds")
dta_g_estimator <- readRDS("est_g_estimator_correct_p_elig_0.1_delta_8.rds")
dta_gmm <- readRDS("est_gmm_correct_p_elig_0.1_delta_8.rds")
dta_ls <- readRDS("est_ls_correct_p_elig_0.1_delta_8.rds")

dta_old %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_old$est_psi03 %>% summary()
dta_old$est_psi03 %>% quantile(c(0.05, 0.95))

dta_g_estimator %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_g_estimator$est_psi03 %>% summary()
dta_g_estimator$est_psi03 %>% quantile(c(0.05, 0.95))

dta_gmm %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_gmm$est_psi03 %>% summary()
dta_gmm$est_psi03 %>% quantile(c(0.05, 0.95))

dta_ls %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_ls$est_psi03 %>% summary()
dta_ls$est_psi03 %>% quantile(c(0.05, 0.95))

# read results for p_elibg = 0.9, delta = 0
dta_old <- readRDS("est_old_correct_p_elig_0.9_delta_0.rds")
dta_g_estimator <- readRDS("est_g_estimator_correct_p_elig_0.9_delta_0.rds")
dta_gmm <- readRDS("est_gmm_correct_p_elig_0.9_delta_0.rds")
dta_ls <- readRDS("est_ls_correct_p_elig_0.9_delta_0.rds")

dta_old %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_old$est_psi03 %>% summary()
dta_old$est_psi03 %>% quantile(c(0.05, 0.95))

dta_g_estimator %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_g_estimator$est_psi03 %>% summary()
dta_g_estimator$est_psi03 %>% quantile(c(0.05, 0.95))
dta_gmm %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_gmm$est_psi03 %>% summary()
dta_gmm$est_psi03 %>% quantile(c(0.05, 0.95))
dta_ls %>%
  summarise(psi03_mean = mean(est_psi03, na.rm = TRUE),
            psi03_avar = mean((est_psi03-1)^2, na.rm = TRUE)*sample)
dta_ls$est_psi03 %>% summary()
dta_ls$est_psi03 %>% quantile(c(0.05, 0.95))

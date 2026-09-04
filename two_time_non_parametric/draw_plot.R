#check simulation results. Especially focus on the convergence rate of the estimators
setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time_non_parametric")
library(tidyverse)

dta_1000 <- read_rds("simulation1000.rds")

dta_5000 <- read_rds("simulation5000.rds")

#combine results from the read data for sample size 1000
dta <- dta_1000$`Conventiaonl AIPW`
dta$method <- "Conventional AIPW"

temp <- dta_1000$SNTTEM_g_estimator
temp$method <- "Robins' g-estimator"

dta <- rbind(dta, temp)

temp <- dta_1000$GMM
temp$method <- "GMM"

dta <- rbind(dta, temp)

temp <- dta_1000$LS
temp$method <- "LS"
dta <- rbind(dta, temp)
dta$sample <- 1000


#########################
# combine results from the read data for sample size 5000
dta_5k <- dta_5000$`Conventiaonl AIPW`
dta_5k$method <- "Conventional AIPW"
temp <- dta_5000$SNTTEM_g_estimator
temp$method <- "Robins' g-estimator"
dta_5k <- rbind(dta_5k, temp)
temp <- dta_5000$GMM
temp$method <- "GMM"
dta_5k <- rbind(dta_5k, temp)
temp <- dta_5000$LS
temp$method <- "LS"
dta_5k <- rbind(dta_5k, temp)
dta_5k$sample <- 5000

# combine the two data frame
dta <- rbind(dta, dta_5k)

information <- dta %>%  group_by (sample, method)%>%
  summarize(psi01_mean = mean(est_psi01, na.rm = TRUE),
            psi01_avar = var(est_psi01, na.rm = TRUE)*sample,
            psi02_mean = mean(est_psi02, na.rm = TRUE),
            psi02_avar = var(est_psi02, na.rm = TRUE)*(sample),
            psi12_mean = mean(est_psi12, na.rm = TRUE),
            psi12_avar = var(est_psi12, na.rm = TRUE)*(sample),
            .groups = "drop")
information[!duplicated(information),]

dta <- dta %>% group_by(method, sample) %>%
  mutate(psi02 = (est_psi02 - mean(est_psi02))/sd(est_psi02)) %>%
  ungroup()

dta$method <- factor(dta$method, levels = c("Conventional AIPW", "Robins' g-estimator", "GMM", "LS"))
dta$sample <- paste0("Sample=", dta$sample)

# read bias and variance results first



# make ggplot2 density figures for standardized psi02 compared the a standard normal
ggplot(dta) +
  geom_density(aes(x = psi02, fill = method)) +
  facet_grid(sample~method) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), color = "black", size = 1, linetype = "dashed") +
  labs(x = expression(Standardized~psi[0][2]~estimates),
       y = "Density") +
  theme_bw() +
  theme(legend.position = "none")

ggsave("Density_plots.png", width = 12, height = 6)

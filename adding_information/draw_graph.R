# read data
library(tidyverse)
library(ggsci)
# sample 1000
dta <- read_rds("gmm_good_1000.rds")
dta$sample <- 1000
temp <- dta
# sample 2000
dta <- read_rds("gmm_good_2000.rds")
dta$sample <- 2000
temp <- rbind(temp, dta)

# sample 4000
dta <- read_rds("gmm_good_4000.rds")
dta$sample <- 4000
temp <- rbind(temp, dta)

# sample 6000
dta <- read_rds("gmm_good_6000.rds")
dta$sample <- 6000
temp <- rbind(temp, dta)

# sample 8000
dta <- read_rds("gmm_good_8000.rds")
dta$sample <- 8000
temp <- rbind(temp, dta)


#####################
boxplot <- ggplot(temp) +
  geom_boxplot(aes(x = method, y=  blip_down, color = method)) +
  facet_wrap(~sample, nrow = 2)+
  scale_color_aaas() +
  labs( x = NULL, y = expression(Distribution~of~u^t~hat(theta)),
        color = "Method",
        linetype = "Method")+
  theme_bw() +
  theme(axis.text.x = element_blank())

sd <- temp %>% group_by(sample, method) %>%
  summarise(mean = mean( blip_down), quantile = quantile( blip_down, 0.975) -quantile( blip_down, 0.025)) %>%
  ungroup() %>%
  mutate(sd = quantile/ (2*1.96),
         var = sd^2,
         avar = var*sample)

sd_plot <- ggplot(sd) +
  geom_line(aes(x = sample, y=  avar, color = method)) +
  scale_color_aaas() +
  labs( x = "Sample size (n)", y = expression(Empirical~variance%*%n),
        color = "Method",
        linetype = "Method")+
  scale_x_continuous()+
  theme_bw()

library(ggpubr)
ggarrange(boxplot, sd_plot, nrow = 2, labels = c("A", "B"))


ggsave("gmm_good_avar.png", width = 6, height = 8)

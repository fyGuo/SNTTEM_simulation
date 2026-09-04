setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/adding_information/show_case_gmm_bad")

for (sample in seq(400, 1800, 200)) {
  source("show_case.R")
  temp$sample <- sample
  if (sample == 400) {
    result <- temp
  } else{
    result <- rbind(result, temp)
  }
}


boxplot <- ggplot(result) +
  geom_boxplot(aes(x = method, y=  est_psi02, color = method)) +
  facet_wrap(~sample, nrow = 2)+
  scale_color_aaas() +
  labs( x = NULL, y = expression(Distribution~of~hat(theta)),
        color = "Method",
        linetype = "Method")+
  theme_bw() +
  theme(axis.text.x = element_blank())


sd <- result %>% group_by(sample, method) %>%
  summarise(mean = mean(est_psi02),
            quantile = quantile(est_psi02, 0.975) - quantile(est_psi02, 0.025)) %>%
  ungroup() %>%
  mutate(sd = quantile/(1.96*2),
         var = sd^2,
         avar = var*sample)


sd_plot <- ggplot(sd) +
  geom_line(aes(x = sample, y=  avar, color = method)) +
  scale_color_aaas() +
  labs( x = "Sample size (n)", y = expression(Empirical~variance%*%n),
        color = "Method",
        linetype = "Method")+
  scale_x_continuous(breaks = seq(600, 2000, 200))+
  theme_bw()

library(ggpubr)
ggarrange(boxplot, sd_plot, nrow = 2, labels = c("A", "B"))

ggsave("gmm_bad_avar.png", width = 6, height = 8)

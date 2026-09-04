#read all simulation results for the two-time points
# the paths are coded as "Delta_X_Eilg_B", extract X for delta and B for Elig
dir_path <- expand.grid("Delta_",c(0,4,8), "_Elig_", c("01","05","09"))
dir_path$path <- paste0(dir_path$Var1, dir_path$Var2, dir_path$Var3, dir_path$Var4)
dir_path


for (i in 1:dim(dir_path)[1]) {
    path <- dir_path[i, "path"]
    if (i == 1){
      temp <- readRDS(paste0("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time/", path, "/simulation_results.rds"))
      temp$delta <- dir_path[i, "Var2"]
      temp$elig <- as.numeric(dir_path[i, "Var4"])/10
    }
    else {
      df <- readRDS(paste0("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time/", path, "/simulation_results.rds"))
      df$delta <- dir_path[i, "Var2"]
      df$elig <- as.numeric(as.character(dir_path[i, "Var4"]))/10
      temp <- rbind(temp, df)
    }
}

# make a ggplot facet by delta and elig
ggplot(data = temp, aes(x = ps1, y = 1/psi02_avar, color = method, shape = method)) +
  geom_line(size = 0.5) +
  geom_point(size = 1) +
  facet_grid(delta ~ elig, labeller = labeller(
    delta = function(x) paste0("Delta = ", x),
    elig = function(x) paste0("Eligibility = ", x)
  )) +
  labs(x = expression(Pr(A[1])==1),
       y = "Asymptotic power",
       color = "Method",
       shape = "Method") +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time/simulation_results_plot.png", width = 8, height = 6)

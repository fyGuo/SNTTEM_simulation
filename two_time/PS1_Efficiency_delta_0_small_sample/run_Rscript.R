setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/PS1_Efficiency_delta_0_small_sample")
files <- list.files(path = "~/PhD_thesis/Thesis_1_SNTTEM/Codes/PS1_Efficiency_delta_0_small_sample", pattern = "PS1_control+")
for (i in 1:6) {
  source(files[i])
  if(i == 1) {
    result <- temp
  } else{
    result <- rbind(result, temp)
  }
}



saveRDS(result, file = "simulation_results.rds")

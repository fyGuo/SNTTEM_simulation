setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time/Delta_0_Elig_09")
files <- list.files(path = "~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time/Delta_0_Elig_09", pattern = "PS1_control+")
for (i in 1:8) {
  source(files[i])
  if(i == 1) {
    result <- temp
  } else{
    result <- rbind(result, temp)
  }
}
Sys.time()


saveRDS(result, file = "simulation_results.rds")

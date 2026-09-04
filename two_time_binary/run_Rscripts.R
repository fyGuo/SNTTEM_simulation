library(tidyverse)
setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time")

temp <- expand.grid("Delta_", c(0,4,8), "_Elig_", c("01", "05", "09"))
folder_paths <- stringr::str_c("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time/","Delta_", temp$Var2, "_Elig_", temp$Var4)

for (i in 1:length(folder_paths)) {
  setwd(folder_paths[i])
  files <- list.files( pattern = "PS1_control+")
  for (i in 1:8) {
    source(files[i])
    if(i == 1) {
      result <- temp
    } else{
      result <- rbind(result, temp)
    }
  }
  saveRDS(result, file = "simulation_results.rds")
}



Sys.time()



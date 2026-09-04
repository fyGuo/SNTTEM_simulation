setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Eligibility_delta_8_PS1_-3")
files <- list.files(path = "~/PhD_thesis/Thesis_1_SNTTEM/Codes/Eligibility_delta_8_PS1_-3", pattern = "p_I1+")
for (i in 1:4) {
  source(files[i])
  if(i == 1) {
    result <- temp
  } else{
    result <- rbind(result, temp)
  }
}



saveRDS(result, file = "simulation_results.rds")

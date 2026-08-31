setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time_binary/changing_I1")
files <- list.files(path ="~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time_binary/changing_I1", pattern = "eligib_I1+")
for (i in 1:3) {
  source(files[i])
  if(i == 1) {
    result <- temp
  } else{
    result <- rbind(result, temp)
  }
}
Sys.time()


saveRDS(result, file = "simulation_results.rds")

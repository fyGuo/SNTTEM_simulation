setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time_binary/changing_I1_null_effect")
files <- list.files(path ="~/PhD_thesis/Thesis_1_SNTTEM/Codes/two_time_binary/changing_I1_null_effect", pattern = "eligib_I1+")
for (i in 1:3) {
  source(files[i])
  if(i == 1) {
    result <- est
  } else{
    result <- rbind(result, est)
  }
}
Sys.time()


saveRDS(result, file = "simulation_results.rds")

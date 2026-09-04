setwd("~/PhD_thesis/Thesis_1_SNTTEM/Codes/Simulation_3_time")
files <- list.files(path = "~/PhD_thesis/Thesis_1_SNTTEM/Codes/Simulation_3_time", pattern = "all_correct_+")
for (i in 1:4) {
  source(files[i])
}




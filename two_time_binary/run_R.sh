#!/bin/bash
nohup caffeinate -i R CMD BATCH --no-restore --no-save run_Rscripts.R run_Rscripts.out &

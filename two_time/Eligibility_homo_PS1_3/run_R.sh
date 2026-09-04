#!/bin/bash
nohup caffeinate -d -i -m -s R CMD BATCH --no-restore --no-save run_Rscript.R run_Rscript.out &

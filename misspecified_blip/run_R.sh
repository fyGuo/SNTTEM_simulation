#!/bin/bash

#nohup caffeinate -d -i -m -s R CMD BATCH --no-restore --no-save model1_value0.R model1_value0.out &
#nohup caffeinate -d -i -m -s R CMD BATCH --no-restore --no-save model1_value2.R model1_value2.out &
nohup caffeinate -d -i -m -s R CMD BATCH --no-restore --no-save model2_value0.R model2_value0.out &
nohup caffeinate -d -i -m -s R CMD BATCH --no-restore --no-save model2_value2.R model2_value2.out &

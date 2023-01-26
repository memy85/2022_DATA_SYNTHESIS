"

Scenario 1. Cox Proportional Hazard Model

사망/생존에 대해서 cox regression 진행
"
library(tidyverse)
library(here)
library(yaml)
library(ggplot2)

config_path <- here('config/config.yaml')
config <- read_yaml(config_path)
project_path <- config$path_config$project_path
input_path <- here(project_path, 'data/processed/4_evaluate/make_whole_data')
figure_path <- here(project_path, 'figures')

## load file

original <- read_csv(here(input_path, 'original.csv'))
eps01 <- read_csv(here(input_path,'comparison_data_0.1.csv'))



#### Process data

original <- original %>% filter(BSPT_IDGN_AGE < 50)

original['group'] = 'original'
eps01['group'] = 'epsilon 0.1'

original <- original %>% select(-c('PT_SBST_NO'))
eps01 <- eps01 %>% select(-c('PT_SBST_NO'))
##

binded <- rbind(original, eps01)

##
binded

##
eps01$DEAD %>% table

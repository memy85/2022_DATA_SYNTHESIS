library(dtw)
library(tidyverse)
library(glue)
library(here)
library(yaml)

here::i_am('src/4_evaluate/evaluate_timeseries.R')

config_path <- here('config/config.yaml')
config <- read_yaml(config_path)

PROJECT_PATH <- here()
INPUT_PATH <- here(PROJECT_PATH, 'data/processed/2_restore/restore_to_db_form')
OUTPUT_PATH <- here(PROJECT_PATH, 'data/processed/4_evaluate/evalutate_timeseries')
FIGURE_PATH <- here(PROJECT_PATH, 'figures/')


original <- read_csv(here("data/raw", 'CLRC_EX_DIAG.csv'))
eps01 <- read_csv(here(INPUT_PATH,'CLRC_EX_DIAG_0.1.csv'))
eps1 <- read_csv(here(INPUT_PATH,'CLRC_EX_DIAG_1.csv'))
eps10 <- read_csv(here(INPUT_PATH,'CLRC_EX_DIAG_10.csv'))
eps100 <- read_csv(here(INPUT_PATH,'CLRC_EX_DIAG_100.csv'))
eps1000 <- read_csv(here(INPUT_PATH,'CLRC_EX_DIAG_1000.csv'))

original <- original %>% select(c('PT_SBST_NO','TIME'='CEXM_YMD','CEA'='CEXM_RSLT_CONT'))

original['count'] = 1
original <- arrange(original, PT_SBST_NO)

original <- original %>% 
    group_by(PT_SBST_NO) %>% 
    mutate(count=cumsum(count)) %>%
    select(c('PT_SBST_NO','STEP'='count','CEA'))

reshape(original, idvar="PT_SBST_NO", timevar="STEP", direction='wide')


change_data_format <- function(epsilon_data){
    epsilon_data
}


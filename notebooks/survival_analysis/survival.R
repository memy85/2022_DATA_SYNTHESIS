library(tidyverse)
library(here)
library(yaml)

# here::i_am('survival.R')

PROJECT_PATH <- here()
config_path <- here(PROJECT_PATH, "config/config.yaml")
config <- read_yaml(config_path)

# Path

PROJECT_PATH <- config[['path_config']]['project_path']
INPUT_PATH <- here(PROJECT_PATH, 'data/processed/0_preprocess')
OUTPUT_PATH <- here(PROJECT_PATH, 'data/processed/notebooks')

d0 <- read_csv(here(INPUT_PATH, 'D0.csv'))


all <- d0 %>% colnames()


d0 <- d0 %>%
  group_by('PT_SBST_NO') %>%
  fill(all)


# 
d0 %>% 
  mutate(END=TIME) %>% head()

# Time Varying Cox regression analysis
library(survival)
tmerge(d0, tstop = TIME)


cgd0[,1:13] %>% head()

d0

coxph(Surv(tstart, tstop, DEAD_NFRM_DEAD))
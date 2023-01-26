library(tidyverse)
library(here)
library(yaml)
library(ggplot2)

here::i_am('src/4_evaluate/evaluate_hellinger.R')

config_path <- here('config/config.yaml')
config <- read_yaml(config_path)
project_path <- config$path_config$project_path
input_path <- here(project_path, 'data/processed/4_evaluate/make_death')
output_path <- here(project_path, 'data/processed/4_evaluate/compare_hellinger')
figure_path <- here(project_path, 'figures')

ifelse(!dir.exists(file.path(figure_path)), dir.create(file.path(figure_path)), FALSE)
ifelse(!dir.exists(file.path(output_path)), dir.create(file.path(output_path)), FALSE)
## load file


original <- read_csv(here(input_path, 'original.csv'))
eps01 <- read_csv(here(input_path,'comparison_data_0.1.csv'))
eps1 <- read_csv(here(input_path,'comparison_data_1.csv'))
eps10 <- read_csv(here(input_path,'comparison_data_10.csv'))
eps100 <- read_csv(here(input_path,'comparison_data_100.csv'))
eps1000 <- read_csv(here(input_path,'comparison_data_1000.csv'))

## Preprocess data

original['group'] <- 'original'
eps01['group'] <- 'epsilon 0.1'
eps1['group'] <- 'epsilon 1'
eps10['group'] <- 'epsilon 10'
eps100['group'] <- 'epsilon 100'
eps1000['group'] <- 'epsilon 1000'

original <- original %>% select(-c('PT_SBST_NO'))
eps01 <- eps01 %>% select(-c('PT_SBST_NO'))
eps1 <- eps1 %>% select(-c('PT_SBST_NO'))
eps10 <- eps10 %>% select(-c('PT_SBST_NO'))
eps100 <- eps100 %>% select(-c('PT_SBST_NO'))
eps1000 <- eps1000 %>% select(-c('PT_SBST_NO'))

original$BSPT_STAG_VL <- ifelse(original$BSPT_STAG_VL == 'x', 0, original$BSPT_STAG_VL) %>% as.numeric()
original$BSPT_T_STAG_VL <- ifelse(original$BSPT_T_STAG_VL == 'x', 0, original$BSPT_T_STAG_VL) %>% as.numeric()
original$BSPT_N_STAG_VL <- ifelse(original$BSPT_N_STAG_VL == 'x', 0, original$BSPT_N_STAG_VL) %>% as.numeric()
original$BSPT_M_STAG_VL <- ifelse(original$BSPT_M_STAG_VL == 'x', 0, original$BSPT_M_STAG_VL) %>% as.numeric()


change_stage_values <- function(data){
  data$BSPT_STAG_VL <- data$BSPT_STAG_VL %>% as.numeric()
  data$BSPT_T_STAG_VL <- data$BSPT_T_STAG_VL %>% as.numeric()
  data$BSPT_N_STAG_VL <- data$BSPT_N_STAG_VL %>% as.numeric()
  data$BSPT_M_STAG_VL <- data$BSPT_M_STAG_VL %>% as.numeric()
  return(data)
}

eps01 <- change_stage_values(eps01)
eps1 <- change_stage_values(eps1)
eps10 <- change_stage_values(eps10)
eps100 <- change_stage_values(eps100)
eps1000 <- change_stage_values(eps1000)

whole_data <- rbind(original, eps01, eps1, eps10, eps100, eps1000)

change_columns <- function(df, column_name){
    if(column_name == 'BSPT_SEX_CD'){
        df[[column_name]] <- ifelse(df[[column_name]] == "M", 1, 0)
        return(df)
    }
    

    else if(column_name == 'BSPT_FRST_DIAG_CD'){
        convert_diag_code <- function(x){
            switch(x, "C18"=1, "C19"=2, "C20"=3)
        }
        df[[column_name]] <- map(df[[column_name]], convert_diag_code) %>% unlist()
        return(df)
    }
}

whole_data <- change_columns(whole_data, 'BSPT_SEX_CD')
whole_data <- change_columns(whole_data, 'BSPT_FRST_DIAG_CD')


## compare Hellinger Distance. 
'
First set the Hellinger distance function
'
library(distrEx)
library(glue)

HD <- function(x, y){
    hd <- HellingerDist(DiscreteDistribution(x), DiscreteDistribution(y))
    return(hd[['Hellinger distance']])
}

original <- whole_data %>% filter(group == 'original')
eps01 <- whole_data %>% filter(group == 'epsilon 0.1')
eps1 <- whole_data %>% filter(group == 'epsilon 1')
eps10 <- whole_data %>% filter(group == 'epsilon 10')
eps100 <- whole_data %>% filter(group == 'epsilon 100')
eps1000 <- whole_data %>% filter(group == 'epsilon 1000')


##
cols <- original %>% colnames
hellinger_chart <- matrix(nrow=length(cols)-1, ncol=6)
epsilon_datas <- list(eps01, eps1, eps10, eps100, eps1000)

data_idx <- 2
for(data in epsilon_datas){
  
  for(idx in c(1:length(cols))){
    col <- cols[[idx]]
    if(col == 'group'){
        next
    }
    print(glue("column is {col}"))
    x <- original[[col]] %>% na.omit()
    y <- data[[col]] %>% na.omit()

    distance <- HD(x, y)
    hellinger_chart[idx, 1] <- col
    hellinger_chart[idx, data_idx] <- distance
    print(glue('distance is {distance}'))
    }
  data_idx <- data_idx+1
}


hellinger_df <- hellinger_chart %>% as.data.frame()
colnames(hellinger_df) <- c('columns', 'epsilon 0.1', 'epsilon 1', 'epsilon 10', 'epsilon 100', 'epsilon 1000')
hellinger_df %>% write_csv(here(output_path, 'hellinger.csv'))


#############################################
######## Plot the distributions #############
#############################################

## Age
ggplot(whole_data, aes(x = BSPT_IDGN_AGE)) +
  geom_histogram(aes(color = group, fill = group), 
                position = "identity", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  theme(aspect.ratio=1, text = element_text(size = 20))

    
## Sex
ggplot(whole_data, aes(x = BSPT_SEX_CD)) +
  geom_bar(aes(color = group, fill = group), 
                position = "dodge", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  theme(aspect.ratio =1, text = element_text(size = 20))


## First diagnosis code
ggplot(whole_data, aes(x = BSPT_FRST_DIAG_CD)) +
  geom_bar(aes(color = group, fill = group), 
                position = "dodge", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  theme(aspect.ratio =1, text = element_text(size = 20))

# stage value
ggplot(whole_data, aes(x = BSPT_STAG_VL)) +
  geom_bar(aes(color = group, fill = group), 
                position = "dodge", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  theme(aspect.ratio =1, text = element_text(size = 20))


if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('effsize', 'ggrepel', 'ggpubr', 'ggfortify', 'gridExtra', 'cowplot', 'weights', 'tidyverse')

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

neural_reg_data <- read.csv('srp_ridge_results_processed.csv')



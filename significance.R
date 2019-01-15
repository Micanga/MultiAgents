library(dplyr)
library(broom)
library(stringr)
library(reshape2)

adhoc_data <- data.frame(input = readLines('/home/tpin3694/Documents/data.txt')) 

# Clean up results
adhoc_split <- data.frame(str_split_fixed(adhoc_data$input, ',', 3)) %>% 
  `colnames<-`(c('agent_count', 'method', 'iterations')) %>% 
  mutate(iterations = gsub('\\[|\\]', '', iterations)) %>% 
  mutate(iterations = ifelse(iterations=='', 0, iterations))

# Break out results
cleaned <-data.frame(agent_count = adhoc_split$agent_count, 
                     method = adhoc_split$method, 
                     str_split_fixed(adhoc_split$iterations, '\\.', 10))

# Melt columns down to rows
adhoc_melt <- melt(cleaned, id.vars =c('agent_count', 'method')) %>% 
  filter(value>0) %>% 
  select(-variable) %>% 
  mutate(agent_count = as.numeric(agent_count),
         value = as.numeric(value))

# Setup tests
run_test <- function(no_agents, dataframe){
  model_df <- dataframe %>% 
    filter(agent_count==no_agents)
  x <- model_df %>% 
    filter(method=='uct') %>% 
    select(value) %>% 
    as.vector(.)
  y <- model_df %>% 
    filter(method=='uct-h') %>% 
    select(value) %>% 
    as.vector(.)
  return(list('x'=x, 'y'=y))
}

# Conduct test
x <- runif(40, 1,5)
y <- runif(40, 3,8)

t <- run_test(3, adhoc_melt)
t.test(t$x, t$y)


# Test for a difference in graidents
uct_data <- adhoc_melt %>% 
  filter(method=='uct') 
uct_h_data <- adhoc_melt %>% 
  filter(method=='uct-h') 

# Construct linear models
all_lm <- lm(value ~ 0+ agent_count + as.factor(method), data = adhoc_melt)
uct_lm <- lm(value ~ 0 + poly(agent_count, 2), data = uct_data)
uct_h_lm <- lm(value ~ 0 + poly(agent_count, 2), data = uct_h_data)

# Test for a difference
slope_test <- function(gradient1, se1, gradient2, se2){
  test_stat <- (gradient1-gradient2)/sqrt(se1**2+se2**2)
  p_value <- 1-pnorm(test_stat)
  return(list('test_statistic'=test_stat, 'p-value'=p_value))
}

# Clean up linear models
uct_lm_tidy <- tidy(uct_lm)
uct_lm_h_tidy <- tidy(uct_h_lm)

# First order terms
slope_test(uct_lm_tidy$estimate[1], uct_lm_tidy$std.error[1], 
           uct_lm_h_tidy$estimate[1], uct_lm_h_tidy$std.error[1])

# Second order terms
slope_test(uct_lm_tidy$estimate[2], uct_lm_tidy$std.error[2], 
           uct_lm_h_tidy$estimate[2], uct_lm_h_tidy$std.error[2])

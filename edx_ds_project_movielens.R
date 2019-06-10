#####################################
# To create .pdf file from .rmd
#####################################

#install.packages("rmarkdown")
library(rmarkdown)
render('~/edx_ds_project_movielens.Rmd')

###################################
# Create edx set and validation set
###################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind = "Rounding") # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################
# Exploring the Movielens Data
###############################

library(tidyverse)
library(caret)

# Columns and Rows of edx dataset
dim(edx)
edx %>% as_tibble()

# Number of unique movies and number of unique users
edx %>% 
  summarize(num_movies = n_distinct(movieId),
            num_users = n_distinct(userId))

# Distribution of ratings
edx %>% 
  ggplot(aes(x = rating)) + 
  geom_bar() 

# Distribution of number of ratings per movie
edx %>% 
  group_by(movieId) %>% 
  summarize(number_of_ratings_per_movie = n()) %>% 
  ggplot(aes(number_of_ratings_per_movie)) + 
  geom_bar()

# Distribution of number of ratings per user
edx %>% 
  group_by(userId) %>% 
  summarize(number_of_ratings_per_user = n()) %>% 
  ggplot(aes(number_of_ratings_per_user)) + 
  geom_bar() 

# Distribution of average rating per movie
edx %>% 
  group_by(movieId) %>% 
  summarize(rating_avg_per_movie = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(rating_avg_per_movie)) + 
  geom_histogram(bins = 30, color = "black")

# Distribution of average rating per user
edx %>% 
  group_by(userId) %>% 
  summarize(rating_avg_per_user = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(rating_avg_per_user)) + 
  geom_histogram(bins = 30, color = "black")

# distribution of rating count per movie
edx %>% group_by(movieId, title) %>%
  summarize(rating_count_each_movie = n()) %>% 
  filter(rating_count_each_movie <= 10) %>%
  ggplot(aes(rating_count_each_movie)) + 
  geom_histogram(bins = 10, color = "black")

# distribution of rating count per user
edx %>% group_by(userId) %>%
  summarize(rating_count_each_user = n()) %>% 
  filter(rating_count_each_user <= 20) %>%
  ggplot(aes(rating_count_each_user)) + 
  geom_histogram(bins = 10, color = "black")

#########################################################
# Fitting for the predicted rating using different models
# Comparing RMSE for those models
#   1. Just the Average Model
#   2. Movie Effect Model
#   3. Movie + User Effect Model
#   4. Regularized Movie + User Effect Model
#########################################################

# The function that computes the RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# 1. Just the Average Model

mu_hat <- mean(edx$rating)
predited_ratings_1 <- mu_hat
rmse_model_1 <- RMSE(predicted_ratings_1, validation$rating)
rmse_model_1

rmse_results <- data_frame(method = "Just the average", RMSE = rmse_model_1)

# 2. The Movie Effect Model

mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# distribution of average rating effect of each movie, b_i
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# predicted ratings considering movie effects
predicted_ratings_2 <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

rmse_model_2 <- RMSE(predicted_ratings_2, validation$rating)
rmse_model_2

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = rmse_model_2))

# 3. Movie + User Effect Model

edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# distribution of average rating effect of each user, b_u
user_avgs %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black"))

# Predicted ratings considering both movie effects and user effects
predicted_ratings_3 <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_model_3 <- RMSE(predicted_ratings_3, validation$rating)
rmse_model_3

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = rmse_model_3))

# 4. Regularization

# Choosing the tuning parameter for regularized movie + user effect model
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

# Regularized Movie + User Effect Model using lambda = 5.25
lambda <- 5.25
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) #, n_i = n())
user_reg_avgs <- edx %>%
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# predicted rating for regularized movie and user effects model
predicted_ratings_4 <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_model_4 <- RMSE(predicted_ratings_4, validation$rating)
rmse_model_4

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = rmse_model_4))

rmse_results %>% knitr::kable()

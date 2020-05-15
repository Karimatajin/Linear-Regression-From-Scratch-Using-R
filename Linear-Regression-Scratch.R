# Author : Karima Tajin
# Date : 25 February 2020
# Machine Learning in R 
# Project 2 : Linear Regression

# Write a linear regression algorithm from scratch, which prints coefficients and accuracy metrics (e.g. R2) 
# and plots the actual versus predicted response variables.
# Compare the R2 of the results using your algorithm to the R2 of the results using lm() and predict().
# Document each step of the code to demonstrate you understand what each line of code does. The code has to describe the steps of the linear regression model. 
# For your algorithm, you may use basic statistical functions and graphing functions but NOT machine learning functions (i.e. no lm(), no predict() and related functions). 
# Note, you must explain what solve() is doing if you use it (i.e. an explanation for a general audience more than “The function solves the equation”).

# using TrainData_Group1.csv to train and test our algorithm

# the equation of linear regression is Y = βo + β1X + ∈ where
# X  : independent variable ( we use to make prediction)
# βo : Intercept ( it is the prediction value you get when X=0)
# β1 : slope (it explains the change in Y when X changes by 1 unit)
# ∈  : residual value

### steps of linear regression:
#1.load the data 
#2.dividing the data into train and test data:
#3.train the model using the training dataset
##3-1 create X Matrix of training data.
##3-2 create Y Matrix.
##3-3 solve for the β vector.
#4.make predictions using the test dataset
##4-1 create the X matrix of test data
##4-2 compute Ŷ = βX 
#5.test the model
##5-1 compute the sum of squared errors (RSS)
##5-2 compute the total sum of squares (TSS)
##5-3 compute R^2 = 1 – RSS / TSS
##5-4 plot predicted Y vs actual Y 

# setting the working directory:
setwd("/Users/karimaidrissi/Desktop/DSSA 5201 ML/week of linear regression")

# load the data:
KarimaData= read.csv('TrainData_Group1.csv', header = TRUE)

# dimension of the data:
dim(KarimaData) # we have 1000 observations and 6 variables ( 5 X's and 1 intercept)

#2. split Karima dataset into training and testing dataset:
## we are going to split ramdomly our data set into 80% train and 20% test.
row.number <- sample(1:nrow(KarimaData), 0.8*nrow(KarimaData)) # random simple using sample()
training = KarimaData[row.number,] # 800 observation to train the data
testing = KarimaData[-row.number,] # 200 observation to test the data
dim(training)    # dimension of training data, 800 rows with 6 variables
dim(testing)     # dimension testing data, 200 rows with 6 variables

rm(row.number)  # to clean our environment

# seperate X and Y variables
trainingX <- subset(training, select = -c(Y)) # select all X's column for training 
trainingY <- subset(training, select = c(Y)) # select only Y column for training
testingX <- subset(testing, select = -c(Y)) # select all X's column for testing
testingY <- subset(testing, select = c(Y)) # select only Y column for testing
#3.train the model using the training dataset:
linearRegression <- function(x,y) { # create a function of x and y data
  # vector of 1's with the same amount af rows.
  intercept <- rep(1, nrow(x))
  # Add intercept column to x
  x <- cbind(intercept, x)
  
  matrix_X <- as.matrix(x) # create x matrix of our feature variables
  vector_Y <- as.matrix(y) # create y vector of the response variable
  # after having our feature variable and response vector, we can use them to solve 
  # the equation β = ((X^T*x)^-1)* (X^T)*y we can calculate this equation in R with 
  # β = solve(t(X) %*% X) %*% t(X) %*% y where:
  # t() :function takes the transpose of a matrix.
  # solve() :calculate the inverse of any matrix.
  # %*% :operator to multiplicate the matrices.
  # implement closed-form solution
  betas  <- solve(t(matrix_X) %*% matrix_X) %*% t(matrix_X) %*% vector_Y
  betas <- round(betas, 2)# round betas to 2 decimal places
  return(betas) # return the value of betas 
} # close our LinearRegression function

#4.make predictions using the test dataset
## is similar to linearRegression function 
# but now we will use beta value that we found to get Ŷ = βX 

PredictY <- function(x, betas) {
  betas_matrix <- t(as.matrix(betas)) # transpose of beta matrix
  intercept <- rep(1, nrow(x))   # vector of 1's with the same amount af rows in x data.
  x <- cbind(intercept, x)   # Add intercept column to x
  matrix_X <- t(as.matrix(x)) # transpose of x data matrix
  Ŷ <- betas_matrix %*% matrix_X # to find Ŷ, we multiply beta matrix by x data matrix
  return(Ŷ) # return  Ŷ from the function
} # ends predictY function

# compute betas using the linearRegression function that we created :
betas <- linearRegression(trainingX,trainingY)
dim(betas) # dimension of betas is 6 values of Y intercept
print(betas) 

# compute Ŷ with PredictY function using 
Ŷ <- PredictY(testingX, betas)
dim(Ŷ) 
print(Ŷ) # there'r 200 values from the test regression

#5.test the model
# test the model by comparing our result with R's built in function by computing R2.
errors <- function(Y, Ŷ){
  Y <- as.matrix(Y)
  Ŷ <- t(as.matrix(Ŷ))
  RSS = sum((Y- Ŷ)^2)  # compute the sum squared errors
  # RSS gives a measure of error of prediction, the lower it is the more our model is accurate 
  TSS = sum((Y - mean(Ŷ))^2) # compute the total sum of squares
  R2 <- 1 - (RSS/TSS) # R2 represents the proportion of variance
  RMSE <- sqrt(mean((Ŷ - Y)^2)) # Root mean square error we will use it to evaluate our model
  return(list(R2 = R2, RMSE = RMSE, RSS = RSS, TSS = TSS)) # return list of R2 and RMSE
} #ends our error function

error <- errors(testingY, Ŷ)
error
#$R2[1] 0.9909298
#$RMSE[1] 0.5832482 
#$RSS[1] 68.03569
#$TSS[1] 7500.974

# we will use lm function which can be used to create a simple linear regression model, 
# in karimaData dataset We have Y column that will depend on 5 columns of X's (X1-x5)

Rversion <- lm(formula = Y ~ X1 + X2 + X3 + X4 + X5, data =KarimaData)
# summary of linear model function:
summary(Rversion)

# Overall, both R2 values from the regressionlinear function from scatch and using also built in function with R 
# are close to 1 which means the two method are very close.
# however, the value of TSS and RSS are extremly differents but the value of R2 is so close to 0.99


# plotting the graph with actual Y values against the predicted value
dev.off()
testingY <- as.matrix(testingY) # Y values actual
dim(testingY) # dimension of Y values
resultY <- PredictY(testingX, betas) # predicted Y values
resultY <- as.matrix(resultY) # convert the predicted Y values to matrix
dim(resultY)
plot(x=testingY, y=resultY, col=3, pch="*", font.lab=4, main="Actual VS. Predicted Values in R",xlab = "Testing Y Values", ylab="Predicted Values")

# the graph is linear with a few outliers, the reason why R2 isn't equal to 1.





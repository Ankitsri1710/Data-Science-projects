# Ordinal Logistic Regression
library(carData)
library(MASS)

data(WVS)
?WVS
View(WVS)

summary(WVS)

table(WVS$poverty)

# Ordinal Logistic Regression
# Proportional Odds Logistic Regression - polr
model <- polr(poverty ~ religion + degree + country + age + gender, data = WVS, Hess = TRUE)
summary(model)
?polr


# Significance check of coefficients
summary_table <- coef(summary(model))
p_val <- pnorm(abs(summary_table[, "t value"]), lower.tail = FALSE)* 2
summary_table <- cbind(summary_table, "p value" = round(p_val,3))
summary_table


# Prediction on new data
new_data <- data.frame("religion"= "yes","degree"="no","country"="Norway","age"=30,"gender"="male")

prob <- predict(model,new_data, type = "p")
prob

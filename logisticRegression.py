
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# read data
featureData = pandas.read_csv('customer_data.csv',usecols=['age','salary'])
purchasedData = pandas.read_csv('customer_data.csv',usecols=['purchased'])
purchased = pandas.DataFrame(purchasedData)
#change to a data frame
df = pandas.DataFrame(featureData)
#use scaler
scaler = MinMaxScaler()
normalized_values= scaler.fit_transform(df)
normalized_df = pandas.DataFrame(normalized_values, columns=df.columns)
#add purchased to df
concatned_df = pandas.concat([normalized_df,purchased],sort=False,axis=1)
# Create two separate data frames for purchased and not purchased items
purchased_df = concatned_df[concatned_df['purchased'] == 1]
not_purchased_df =concatned_df[concatned_df['purchased'] == 0]

# Create a scatter plot with age and salary as x and y axes, respectively
plt.scatter(purchased_df['age'], purchased_df['salary'], color='blue', label='Purchased')
plt.scatter(not_purchased_df['age'], not_purchased_df['salary'], color='red', label='Not Purchased')

# Set the x and y axis labels and the plot title
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Customer Age vs. Salary')

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()
#split dataset to training and testing 80:20
age= concatned_df['age']
age_train = age[0:320]

salary = concatned_df['salary']
salary_train = salary[0:320]

purchased_train = purchased[0:320]

age_test = age[320:]

salary_test = salary[320:]

purchased_test = purchased[320:]

# concatenate the age_train and salary_train into one input variable
x = pandas.concat([age_train,salary_train],sort=False,axis=1)
x_test = pandas.concat([age_test,salary_test],sort=False,axis=1)

# Create a logistic regression object and fit the model to the data
lr = LogisticRegression()
# lr.fit(x,purchased_train.values.ravel())
lr.fit(x,purchased_train)
# Print the coefficients of the logistic regression model
print(lr.coef_)
# Use the optimized hypothesis function to make predictions on the testing set
y_pred = lr.predict(x_test)

# Calculate the accuracy of the model on the testing set
accuracy = accuracy_score(purchased_test, y_pred)

# Print the accuracy of the model on the testing set
print('Accuracy:', accuracy)
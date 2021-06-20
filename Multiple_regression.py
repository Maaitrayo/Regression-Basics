import pandas 
from sklearn import linear_model
import matplotlib.pyplot as plt

model = pandas.read_csv('cars.csv')

X = model[['Weight', 'Volume']]
y = model['CO2']

m = model['Weight']
v = model['Volume']

regr = linear_model.LinearRegression()
regr.fit(X,y)

print('Enter the weight of your car and the volume of your engine: ')
weight = int(input('\tEnter the weight of the car: '))
volume = int(input('\n\tEnter the volume of the engine: '))

#predict the value of the Co2 emission by entering the values of the [[weight, volume]] 
predictedCO2 = regr.predict([[weight, volume]])

print('The predicted CO2 emission is: ',predictedCO2)

#prints the coefficient of the weight and volume
print('The coefficient of the weight and volume respectively are: ', regr.coef_)

plt.subplot(1,2,1)
plt.title('Weight VS CO2')
plt.scatter(m,y)

plt.subplot(1,2,2)
plt.title('Volume VS CO2')
plt.scatter(v,y)
plt.show()
from keras.models import Sequential
from keras.layers import Dense
import numpy
seed = 6
numpy.random.seed(seed)

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# Split into X and Y
X = dataset[:, :8]
Y = dataset[:, 8]

# Model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

# Evaluation
scores = model.evaluate(X, Y)
print(' %s: %.2f%%'%(model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
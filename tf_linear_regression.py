import tensorflow as tf
import pandas as pd

#Define data
X = tf.placeholder(tf.float32,shape=[200,3])
Y = tf.placeholder(tf.float32,shape=[200,1])

#Define Weights to learn
W = tf.Variable(tf.ones([3,1]))
W0 = tf.Variable(tf.ones(1))

#Define model
Y_hat = tf.matmul(X,W) + W0

#Define cost function
cost = tf.reduce_mean((Y - Y_hat)**2)

#Choose optimisation technique and initialise
optimiser = tf.train.GradientDescentOptimizer(0.00003315)
train = optimiser.minimize(cost)
#Read advertising dataset
advertising = pd.read_csv('/home/sambeet/data/Advertising.csv')

#Define training and testing data
train_X = advertising[['TV','Radio','Newspaper']]
train_Y = advertising[['Sales']]

#Initialise variables
init = tf.global_variables_initializer()

#Define session to run operations:
sess = tf.Session()
sess.run(init)

#Run gradient descent step for multiple iterations: 
for iteration in range(0,10000):
    sess.run(train,{X: train_X,Y: train_Y})

#Get latest values of W, W0 and cost
curr_W, curr_W0, curr_cost = sess.run([W, W0, cost], {X: train_X, Y: train_Y})

#Close session
sess.close()

#Print coefficients and R-squared
print 'Coeff:', curr_W, curr_W0
print 'R-Squared:', 1 - (curr_cost/train_Y.var().values)

import tensorflow as tf
from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')


nodes1 = 500
nodes2 = 500
nodes3 = 500


classes = 2
batch_size = 100

len(train_x)

x = tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
  layer1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),nodes1])),
            'biases':tf.Variable(tf.random_normal([nodes1]))}
  layer2 = {'weights':tf.Variable(tf.random_normal([nodes1,nodes2])),
            'biases':tf.Variable(tf.random_normal([nodes2]))}
  layer3 = {'weights':tf.Variable(tf.random_normal([nodes2,nodes3])),
             'biases':tf.Variable(tf.random_normal([nodes3]))}
  layer_out = {'weights':tf.Variable(tf.random_normal([nodes3,classes])),
            'biases':tf.Variable(tf.random_normal([classes]))}
  
  l1 = tf.add(tf.matmul(data,layer1['weights']),layer1['biases'])
  l1 = tf.nn.relu(l1)
  
  l2 = tf.add(tf.matmul(l1,layer2['weights']),layer2['biases'])
  l2 = tf.nn.relu(l2)

  l3 = tf.add(tf.matmul(l2,layer3['weights']),layer3['biases'])
  l3 = tf.nn.relu(l3)
  
  l_out = tf.add(tf.matmul(l3,layer_out['weights']),layer_out['biases'])
  
  return l_out

def train_neural_network():
  prediction = neural_network_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,labels = y))
  optimizer = tf.train.AdamOptimizer().minimize(cost)
  
  n_epochs = 1000
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
      epoch_loss =  0
      i = 0
      while(i < len(train_x)):
        start = i 
        end = i + batch_size
        i = end
        batch_x  = np.array(train_x[start:end])
        batch_y  = np.array(train_y[start:end])
        _, c  =  sess.run([optimizer,cost],  feed_dict = {x:batch_x,y:batch_y})
        epoch_loss += c
      print('Epoch : ',epoch,'/',n_epochs,' loss : ',epoch_loss)
      
      
      
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    print('Accuracy : ',accuracy.eval({x:test_x, y:test_y}))
    
train_neural_network()

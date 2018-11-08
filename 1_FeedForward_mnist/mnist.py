import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
tf.logging.set_verbosity(old_v)

nodes1 = 500
nodes2 = 500
nodes3 = 500


classes = 10
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):
  layer1 = {'weights':tf.Variable(tf.random_normal([784,nodes1])),
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
  
  n_epochs = 10
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
      epoch_loss =  0
      for _ in range(int(mnist.train.num_examples/batch_size)):
        train_data,train_labels = mnist.train.next_batch(batch_size)
        _, c  =  sess.run([optimizer,cost],  feed_dict = {x:train_data,y:train_labels})
        epoch_loss += c
      print('Epoch : ',epoch,'/',n_epochs,' loss : ',epoch_loss)
      
      
      
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    print('Accuracy : ',accuracy.eval({x:mnist.test.images,y: mnist.test.labels}))
    
train_neural_network()

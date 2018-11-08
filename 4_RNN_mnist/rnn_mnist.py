import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn 


old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
tf.logging.set_verbosity(old_v)

n_epochs = 3
classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float32',[None,n_chunks,chunk_size])
y = tf.placeholder('float32')

def recurrent_neural_network_model(data):
  layer = {'weights':tf.Variable(tf.random_normal([rnn_size,classes])),
            'biases':tf.Variable(tf.random_normal([classes]))}
  
  data  = tf.transpose(data,   [1,0,2])
  data  = tf.reshape(data  ,[-1, chunk_size])
  data  = tf.split(data, n_chunks, 0)

  lstm_cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell')
  outputs, states  =  rnn.static_rnn(lstm_cell, data, dtype=tf.float32)


  l_out = tf.add(tf.matmul(outputs[-1],layer['weights']),layer['biases'])
  
  return l_out

def train_neural_network():
  prediction = recurrent_neural_network_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,labels = y))
  optimizer = tf.train.AdamOptimizer().minimize(cost)
    
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
      epoch_loss =  0
      for _ in range(int(mnist.train.num_examples/batch_size)):
        train_data,train_labels = mnist.train.next_batch(batch_size)
        train_data = train_data.reshape((batch_size,n_chunks,chunk_size))

        _, c  =  sess.run([optimizer,cost],  feed_dict = {x:train_data,y:train_labels})
        epoch_loss += c
      print('Epoch : ',epoch,'/',n_epochs,' loss : ',epoch_loss)
      
      
      
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    print('Accuracy : ',accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)),y: mnist.test.labels}))
    
train_neural_network()

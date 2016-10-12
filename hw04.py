import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib import learn 

# processing the data in train files.
with open('train.txt') as f:
    train_content = f.read()
train_words = train_content.replace('\n', ' ').split(' ')
vocab_processor = learn.preprocessing.VocabularyProcessor(1)
train_words = [x[0] for x in vocab_processor.fit_transform(train_words)]
X_train, y_train = train_words[:-1], train_words[1:]

#starting the tensorflow sessions and initializing the place holder.
sess = tf.InteractiveSession()
vocab_size, embeded_size, batch_size, hidden_size = len(vocab_processor.vocabulary_), 30, 20, 100
X = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.int32, [None])
E = tf.Variable(tf.random_uniform([vocab_size, embeded_size], -1, 1))
embd = tf.nn.embedding_lookup(E, X)

w1 = tf.Variable(tf.random_uniform([embeded_size, hidden_size], minval=-1, maxval=1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
h1 = tf.nn.relu(tf.matmul(embd, w1) + b1)

w2 = tf.Variable(tf.random_uniform([hidden_size, vocab_size], minval=-1, maxval=1))
b2 = tf.Variable(tf.constant(0.1, shape=[vocab_size]))
y_pred = tf.matmul(h1, w2) + b2
sm = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, y)

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(sm)
loss = tf.reduce_mean(sm)

sess.run(tf.initialize_all_variables())

for i in range(len(X_train) // batch_size - 1):
    train_step.run(feed_dict={
            X: X_train[batch_size* i: batch_size *(i + 1)],
            y: y_train[batch_size* i: batch_size *(i + 1)]
        })

# processing the data in test files.
with open('test.txt') as f:
    test_content = f.read()
test_words = [x[0] for x in vocab_processor.fit_transform(test_content.replace('\n', ' ').split(' '))]
X_test, y_test = test_words[:-1], test_words[1:]

print math.exp(loss.eval(feed_dict={X: X_test, y: y_test}))

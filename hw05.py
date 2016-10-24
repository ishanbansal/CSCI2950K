import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np

with open('lesmiserables_train.txt') as f:
    train_content = f.read()
train_words = train_content.replace('\n', ' ').split(' ')
with open('lesmiserables_test.txt') as f:
    test_content = f.read()
test_words = test_content.replace('\n', ' ').split(' ')

vocab_processor = learn.preprocessing.VocabularyProcessor(1)
vocab_processor.fit(train_words + test_words)
test_words = [x[0] for x in vocab_processor.transform(test_words)]
X_test, y_test = test_words[:-1], test_words[1:]
train_words = [x[0] for x in vocab_processor.transform(train_words)]
X_train, y_train = train_words[:-1], train_words[1:]

sess = tf.InteractiveSession()
NUM_STEPS = 20
BATCH_SIZE = 25
EMBEDDING_SIZE = 50
VOCAB_SIZE = len(vocab_processor.vocabulary_)
H_SIZE = 256

X = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_STEPS])
y = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_STEPS])
keep_prob = tf.placeholder(tf.float32)

E = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1, 1))
embd = tf.nn.embedding_lookup(E, X)
embd_dropped = tf.nn.dropout(embd, keep_prob)

blstm = tf.nn.rnn_cell.BasicLSTMCell(H_SIZE)
initial_state = blstm.zero_state(BATCH_SIZE, tf.float32)
outputs, state = tf.nn.dynamic_rnn(
    blstm, embd_dropped, initial_state=initial_state)
h = tf.reshape(outputs, [BATCH_SIZE * NUM_STEPS, H_SIZE])

w1 = tf.Variable(tf.random_uniform([H_SIZE, VOCAB_SIZE], minval=-1, maxval=1))
b1 = tf.Variable(tf.constant(0.1, shape=[VOCAB_SIZE]))
logits = tf.matmul(h, w1) + b1

y_reshaped = tf.reshape(y, [BATCH_SIZE * NUM_STEPS])
weights = tf.constant(1.0, shape=[BATCH_SIZE * NUM_STEPS])
loss_by_example = tf.nn.seq2seq.sequence_loss_by_example(
    [logits], [y_reshaped], [weights])
loss = tf.reduce_sum(loss_by_example) / BATCH_SIZE

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())

np_state = [initial_state[0].eval(), initial_state[1].eval()]
num_iter = int((len(X_train) - NUM_STEPS) / BATCH_SIZE)
for i in range(num_iter):
    batch_X, batch_y = [], []
    for j in range(BATCH_SIZE):
        batch_X.append(
            X_train[i * BATCH_SIZE + j: i * BATCH_SIZE + j + NUM_STEPS])
        batch_y.append(
            y_train[i * BATCH_SIZE + j: i * BATCH_SIZE + j + NUM_STEPS])
    batch_X = np.array(batch_X)
    batch_y = np.array(batch_y)
    loss_val, state_tuple, _ = sess.run([loss, state, train_step], feed_dict={
        X: batch_X,
        y: batch_y,
        keep_prob: 0.5,
        initial_state[0]: np_state[0],
        initial_state[1]: np_state[1]
    })
    np_state = [state_tuple[0], state_tuple[0]]

total_loss = 0.0
np_state = [initial_state[0].eval(), initial_state[1].eval()]
num_iter = int((len(X_test) - NUM_STEPS) / BATCH_SIZE)
for i in range(num_iter):
    batch_X, batch_y = [], []
    for j in range(BATCH_SIZE):
        batch_X.append(
            X_test[i * BATCH_SIZE + j: i * BATCH_SIZE + j + NUM_STEPS])
        batch_y.append(
            y_test[i * BATCH_SIZE + j: i * BATCH_SIZE + j + NUM_STEPS])
    batch_X = np.array(batch_X)
    batch_y = np.array(batch_y)
    loss_val, state_tuple = sess.run([loss, state], feed_dict={
        X: batch_X,
        y: batch_y,
        keep_prob: 1,
        initial_state[0]: np_state[0],
        initial_state[1]: np_state[1]
    })
    np_state = [state_tuple[0], state_tuple[0]]
    total_loss += loss_val

ans = total_loss / num_iter
print('perplexity = ' + str(ans))


import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float') / 255
x_test = x_test.reshape(-1, 784).astype('float') / 255

y_train = tf.keras.utils.to_categorical(y_train, 10).astype('float')
y_test = tf.keras.utils.to_categorical(y_test, 10).astype('float')

X = tf.placeholder(dtype = tf.float32, shape = [None, 784], name = 'input')
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
y_true_cls = tf.argmax(y_true, axis=1)


W1 = tf.Variable(tf.random_normal([784,300]), dtype=tf.float32, name='weight1')
b1 = tf.Variable(tf.random_normal([1,300]), name='bias1', dtype=tf.float32)

W2 = tf.Variable(tf.random_normal([300, 10]), dtype=tf.float32, name='weight2')
b2 = tf.Variable(tf.random_normal([1,10]), name='bias2', dtype=tf.float32)

logits = tf.matmul(X, W1, name='op1') + b1
logits = tf.nn.relu(logits)

logits = tf.matmul(logits, W2, name='op2') + b2
#print('l2',logits)

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

acc = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true_cls), dtype=tf.float32))


loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss_mean = tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

batch_size = 100

sess = tf.Session()

sess.run(tf.global_variables_initializer())

batches = int(x_train.shape[0] / batch_size)
curr = 0
for _ in range(10):
    for batch in range(batches):
        st = batch * batch_size
        en = st + batch_size
        if batch == batches - 1:
            feed_dict = {
                X : x_train[st : , : ],
                y_true:y_train[st : , : ]
            }
        else:
            feed_dict = {
                X : x_train[st:en, : ],
                y_true:y_train[st:en, : ]
            }
        #print('sten',st, en)
        #print('curr',curr, curr+batch_size)
        curr += batch_size
        sess.run(optimizer, feed_dict=feed_dict)
    #print('train_acc',sess.run(acc, feed_dict=feed_dict))
    #print('train_los',sess.run(loss_mean, feed_dict=feed_dict))
    feed_dict = {X:x_test, y_true:y_test}
    print('test_acc',sess.run(acc, feed_dict=feed_dict))
    #print('test_los',sess.run(loss_mean, feed_dict=feed_dict))


sess.close()
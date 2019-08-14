import numpy as np
import random
import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.allocator_type = "BFC"


def leaky_relu(x, leak=0.1):
    return tf.maximum(x, x * leak)


class Model:
    def __init__(self, sess, name, testfold):
        self.sess = sess
        self.name = name
        self.test = testfold
        self.val = random.choice([i for i in range(10)].remove(testfold))
        self.learning_rate = tf.Variable(0.001, trainable=False)
        self.batch_size = 80
        self.__build_net()

    def save(self, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(self.sess, save_path="models/spec", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(self.sess, "models/dnn")
        print(' * model restored ')

    def __build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)

            self.X = tf.placeholder("float", [None, 128, 345])
            X_input = tf.reshape(self.X, [-1, 128, 345, 1])
            self.Y = tf.placeholder("float", [None, 10])

            reg = tf.contrib.layers.l2_regularizer(0.001)

            conv1 = tf.layers.conv2d(inputs=X_input, filters=32, kernel_size=[3, 7], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            bn1 = leaky_relu(tf.layers.batch_normalization(conv1, training=self.training))
            conv2 = tf.layers.conv2d(inputs=bn1, filters=32, kernel_size=[3, 5], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            bn2 = leaky_relu(tf.layers.batch_normalization(conv2, training=self.training))
            pool1 = tf.layers.max_pooling2d(inputs=bn2, pool_size=[4, 4], padding="SAME", strides=[4, 4])
            #Output size: 32, 87, 32

            conv3 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 1], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            bn3 = leaky_relu(tf.layers.batch_normalization(conv3, training=self.training))
            conv4 = tf.layers.conv2d(inputs=bn3, filters=64, kernel_size=[3, 1], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            bn4 = leaky_relu(tf.layers.batch_normalization(conv4, training=self.training))
            pool2 = tf.layers.max_pooling2d(inputs=bn4, pool_size=[4, 1], padding="SAME", strides=[4, 1])
            #Output size: 8, 87, 64

            conv5 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[1, 5], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            bn5 = leaky_relu(tf.layers.batch_normalization(conv5, training=self.training))
            conv6 = tf.layers.conv2d(inputs=bn5, filters=128, kernel_size=[1, 5], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            bn6 = leaky_relu(tf.layers.batch_normalization(conv6, training=self.training))
            pool3 = tf.layers.max_pooling2d(inputs=bn6, pool_size=[1, 3], padding="SAME", strides=[1, 3])
            #Output size: 8, 29, 128

            conv7 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            bn7 = leaky_relu(tf.layers.batch_normalization(conv7, training=self.training))
            conv8 = tf.layers.conv2d(inputs=bn7, filters=256, kernel_size=[3, 3], padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            bn8 = leaky_relu(tf.layers.batch_normalization(conv8, training=self.training))
            pool4 = tf.layers.max_pooling2d(inputs=bn8, pool_size=[2, 2], padding="SAME", strides=2)
            #Output size: 4, 15, 256

            flat = tf.reshape(pool4, [-1, 4 * 15 * 256])
            fc1 = tf.layers.dense(inputs=flat, units=512, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=reg)
            fc1 = leaky_relu(fc1)
            dropout4 = tf.layers.dropout(inputs=fc1, rate=0.5, training=self.training)

            self.logits = tf.layers.dense(inputs=dropout4, units=10, kernel_regularizer=reg)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + tf.reduce_sum(reg_loss)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.loss, self.optimizer, self.accuracy], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

    def val_test_acc(self, spec, labels, fold):
        accuracy = 0
        dsize = 0

        for i in range((len(labels[fold]) // self.batch_size) + 1):
            if i == len(labels[fold]) // self.batch_size:
                a = self.get_accuracy(spec[fold][i * self.batch_size:],
                                      labels[fold][i * self.batch_size:])
                accuracy += a * len(labels[fold][i * self.batch_size:])
                dsize += len(labels[fold][i * self.batch_size:])
            else:
                a = self.get_accuracy(spec[fold][i * self.batch_size:(i + 1) * self.batch_size],
                                      labels[fold][i * self.batch_size:(i + 1) * self.batch_size])
                accuracy += a * self.batch_size
                dsize += self.batch_size
        accuracy = accuracy / dsize
        return accuracy

    def train_epoch(self, spec, labels, epoch):
        if epoch == 79 or epoch == 139:
            self.learning_rate /= 10
        avg_loss = 0
        datasize = 0
        accuracy = 0
        for fold in range(10):
            if fold != self.test and fold != self.val:
                for i in range((len(labels[fold]) // self.batch_size) + 1):
                    if i == len(labels[fold]) // self.batch_size:
                        c, _, a = self.train(spec[fold][i * self.batch_size:],
                                          labels[fold][i * self.batch_size:])
                        avg_loss += c * len(labels[fold][i * self.batch_size:])
                        datasize += len(labels[fold][i * self.batch_size:])
                        accuracy += a * len(labels[fold][i * self.batch_size:])
                    else:
                        c, _, a = self.train(spec[fold][i * self.batch_size:(i + 1) * self.batch_size],
                                          labels[fold][i * self.batch_size:(i + 1) * self.batch_size])
                        avg_loss += c * self.batch_size
                        datasize += self.batch_size
                        accuracy += a * self.batch_size
        avg_loss = avg_loss / datasize
        accuracy = accuracy / datasize
        vaccuracy = self.val_test_acc(spec, labels, self.val)
        print("Epoch:", '%04d' % (epoch + 1), "loss=", avg_loss, "training accuracy=", accuracy, "val accuracy=", vaccuracy)

        if epoch % 10 == 9:
            accuracy = self.val_test_acc(spec, labels, self.test)
            print('Accuracy:', accuracy)
        return vaccuracy


with open('Spectrograms/specs.dat', 'rb') as spf:
    spec = np.load(spf)
    spec = np.asarray(spec)
spf.close()

with open ('Labels/labels.dat', 'rb') as lbf:
    labels = np.load(lbf)
    labels = np.asarray(labels)
lbf.close()

print('Read spec and label success')

testfold = 0
epoch_save = 20
vacc = []

sess = tf.Session(config=tf_config)

with tf.device('/device:GPU:2'):
    m = Model(sess, "Model", testfold)

sess.run(tf.global_variables_initializer())

print('Learning started.')


for epoch in range(200):
    vacc.append(m.train_epoch(spec, labels, epoch))
    if epoch % epoch_save == (epoch_save - 1):
        m.save(global_step=epoch)

print(vacc)
'''
with open('model_spec_0.csv', 'w', newline='') as vafile:
    writer = csv.writer(vafile)
    writer.writerows(vacc)
vafile.close()
'''
print("Learning Finished")

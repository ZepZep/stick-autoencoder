import tensorflow as tf
import numpy as np
from stickgenerator import get_stick


learning_rate = 0.001
inputs_ = tf.placeholder(tf.float32, (None, 128, 128, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 128, 128, 1), name='targets')

### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 128x128x16
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(4,4), strides=(4,4), padding='same')
# Now 32x32x16
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 32x32x16
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(4,4), strides=(4,4), padding='same')
# Now 8x8x16
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 8x8x16
encoded = tf.layers.max_pooling2d(conv3, pool_size=(8,8), strides=(8,8), padding='same')
# Now 1x1x16
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 8x8x16
conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 8x8x16
upsample2 = tf.image.resize_images(conv4, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 32x32x16
conv5 = tf.layers.conv2d(inputs=upsample2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 32x32x16
upsample3 = tf.image.resize_images(conv5, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 128x128x16
conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 128x128x16
logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
#Now 128x128x1
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)
# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


print("Creating data")

N_STICKS = 4000
stick_list = np.zeros((N_STICKS, 128, 128))
corp_pos_list = np.random.random((N_STICKS, 5))

for i, corp_pos in enumerate(corp_pos_list):
    stick_list[i] = np.array(get_stick(corp_pos), dtype=np.float32)

print(stick_list.shape)

print("Data created")

sess = tf.Session()
epochs = 100
batch_size = 200
# Set's how much noise we're adding to the MNIST images
noise_factor = 0.1

sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for i in range(N_STICKS // batch_size):
        imgs = stick_list[i*batch_size:(i+1)*batch_size].reshape((-1, 128, 128, 1))
        # Get images from the batch
        # imgs = batch[0].reshape((-1, 28, 28, 1))

        # Add random noise to the input images
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        # Noisy images as inputs, original images as targets
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,
                                                         targets_: imgs})
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Training loss: {:.4f}".format(batch_cost))
# 1.How to use Deep Taylor Decomposition 

## (1). Define model structure from trained model for calculation of relevance score


weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='scopename')
activations = tf.get_collection('activation_collection_name')
X = activations[0]

conv_ksize = "[convolution layerfilter size]"#[1, 4, 4 , 1]
pool_ksize = "[pooling layer filter size]"#[1 ,4, 4, 1]
conv_strides "[convolution layer stride size]"= #[1, 1, 1, 1]
pool_strides ="[pooling layer stride size]" #[1, 4, 4, 1]

weights.reverse()
activations.reverse()

taylor = Taylor(activations, weights, conv_ksize, pool_ksize, conv_strides, pool_strides, 'Taylor',part)

Rs = []
for i in range("number of class"):
    Rs.append(taylor(i))

## (2).run session for getting relvance score
model.sess.run([Rs],feed_dict={X:batch_in, model.keep_prob :p})


# 2. How to use Network Dissection
##(1). please write down!



# 1.How to use Deep Taylor Decomposition 

## (1). Define model structure from trained model for calculation of relevance score

<pre><code>
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
</code></pre>

## (2).run session for getting relvance score
<pre><code>
model.sess.run([Rs],feed_dict={X:batch_in, model.keep_prob :p})
</code></pre>
ref : [Explaining NonLinear Classification Decisions with Deep Taylor Decomposition]: https://arxiv.org/pdf/1512.02479.pd
</hr>

# 2. How to use Network Dissection
## (1). please write down!



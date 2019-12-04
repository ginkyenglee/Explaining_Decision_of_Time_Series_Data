class Taylor:

    def __init__(self, activations, weights, conv_ksize, pool_ksize, conv_strides, pool_strides, name,part):

        self.last_ind = len(activations)
        for op in activations[::-1]:
            self.last_ind -= 1
            if any([word in op.name for word in ['conv', 'pooling', 'dense']]):
                break

        self.activations = activations
        self.weights = weights
        self.conv_ksize = conv_ksize
        self.pool_ksize = pool_ksize
        self.conv_strides = conv_strides
        self.pool_strides = pool_strides
        self.name = name
        self.part = part

    def __call__(self, logit):

        with tf.name_scope(self.name):
            Rs = []
            j = 0

            for i in range(len(self.activations) - 1):

                if i is self.last_ind:

                    if 'conv' in self.activations[i].name.lower():
                        Rs.append(self.backprop_conv_input(self.activations[i + 1], self.weights[j], Rs[-1], self.conv_strides))
                        #print ('backprop_conv_input: {},{}'.format(i,j))
                    else:
                        Rs.append(self.backprop_dense_input(self.activations[i + 1], self.weights[j], Rs[-1]))
                        #print ('backprop_dense_input: {},{}'.format(i,j))
                    continue

                if i is 0:
                    Rs.append(self.activations[i][:,logit,None])
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j][:,logit,None], Rs[-1]))
                    #print ('backprop_dense: {},{}'.format(i,j))
                    j += 1
                    continue

                elif 'dense' in self.activations[i].name.lower():
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j], Rs[-1]))
                    #print ('backprop_dense: {},{}'.format(i,j))
                    j += 1
                elif 'reshape' in self.activations[i].name.lower():
                    shape = self.activations[i + 1].get_shape().as_list()
                    #print ('reshape: {},{}'.format(i,j))
                    shape[0] = -1
                    Rs.append(tf.reshape(Rs[-1], shape))
                elif 'conv' in self.activations[i].name.lower():
                    Rs.append(self.backprop_conv(self.activations[i + 1], self.weights[j], Rs[-1], self.conv_strides))
                    #print ('backprop_conv: {},{}'.format(i,j))
                    j += 1
                elif 'pooling' in self.activations[i].name.lower():

                    # Apply average pooling backprop regardless of type of pooling layer used, following recommendations by Montavon et al.
                    # Uncomment code below if you want to apply the winner-take-all redistribution policy suggested by Bach et al.
                    #
                    if 'avg' in self.activations[i].name.lower():
                        pooling_type = 'avg'
                    else:
                        pooling_type = 'max'
                    Rs.append(self.backprop_pool(self.activations[i + 1], Rs[-1], self.pool_ksize, self.pool_strides, pooling_type))
                    #print ('backprop_pool: {},{}'.format(i,j))
                    #Rs.append(self.backprop_pool(self.activations[i + 1], Rs[-1], self.pool_ksize, self.pool_strides, 'avg'))
                else:
                    raise Error('Unknown operation.')
            if self.part =="whole":
                print ('whole LRP result')
                return Rs
            elif self.part =="part":
                print ('last input LRP result')
                return Rs[-1]

    def backprop_conv(self, activation, kernel, relevance, strides, padding='SAME'):
        W_p = tf.maximum(0., kernel)
        z = nn_ops.conv2d(activation, W_p, strides, padding) + 1e-10
        s = relevance / z
        c = nn_ops.conv2d_backprop_input(tf.shape(activation), W_p, s, strides, padding)
        
        return activation * c

    def backprop_pool(self, activation, relevance, ksize, strides, pooling_type, padding='SAME'):
        if pooling_type.lower() in 'avg':
            z = nn_ops.avg_pool(activation, ksize, strides, padding) + 1e-10
            s = relevance / z
            c = gen_nn_ops._avg_pool_grad(tf.shape(activation), s, ksize, strides, padding)
            #print ('--avg pooling')
            return activation * c
        else:
            z = nn_ops.max_pool(activation, ksize, strides, padding) + 1e-10
            s = relevance / z
            #c = gen_nn_ops._max_pool_grad(activation, z, s, ksize, strides, padding)
            c = gen_nn_ops.max_pool_grad(activation, z, s, ksize, strides, padding)
            #print ('--max pooling')
            return activation * c

    def backprop_dense(self, activation, kernel, relevance):
        W_p = tf.maximum(0., kernel)
        z = tf.matmul(activation, W_p) + 1e-10
        s = relevance / z
        c = tf.matmul(s, tf.transpose(W_p))
        return activation * c

    def backprop_conv_input(self, X, kernel, relevance, strides, padding='SAME', lowest=0., highest=1.):
        W_p = tf.maximum(0., kernel)
        W_n = tf.minimum(0., kernel)

        L = tf.ones_like(X, tf.float32) * lowest
        H = tf.ones_like(X, tf.float32) * highest

        z_o = nn_ops.conv2d(X, kernel, strides, padding)
        z_p = nn_ops.conv2d(L, W_p, strides, padding)
        z_n = nn_ops.conv2d(H, W_n, strides, padding)

        z = z_o - z_p - z_n + 1e-10
        s = relevance / z

        c_o = nn_ops.conv2d_backprop_input(tf.shape(X), kernel, s, strides, padding)
        c_p = nn_ops.conv2d_backprop_input(tf.shape(X), W_p, s, strides, padding)
        c_n = nn_ops.conv2d_backprop_input(tf.shape(X), W_n, s, strides, padding)

        return X * c_o - L * c_p - H * c_n
    
    #FIND THE ROOT POINT USING W^2 RULE
    def backprop_conv_input2(self, X, kernel, relevance, strides, padding='SAME', lowest=-1., highest=1.):
        V = kernel * kernel
        N =  V/ tf.ones_like(V , tf.float32) *1
        
        O = nn_ops.conv2d_backprop_input(tf.shape(X), N, relevance, strides, padding)

        return O

    #pixel intensity
    def backprop_dense_input(self, X, kernel, relevance, lowest=0., highest=1.):
        W_p = tf.maximum(0., kernel)
        W_n = tf.minimum(0., kernel)

        L = tf.ones_like(X, tf.float32) * lowest
        H = tf.ones_like(X, tf.float32) * highest

        z_o = tf.matmul(X, kernel)
        z_p = tf.matmul(L, W_p)
        z_n = tf.matmul(H, W_n)

        z = z_o - z_p - z_n + 1e-10
        s = relevance / z

        c_o = tf.matmul(s, tf.transpose(kernel))
        c_p = tf.matmul(s, tf.transpose(W_p))
        c_n = tf.matmul(s, tf.transpose(W_n))

        return X * c_o - L * c_p - H * c_n
    
    #FIND THE ROOT POINT USING W^2 RULE
    def backprop_dense_input2(self, X, kernel, relevance, lowest=0., highest=1.):
        V = kernel * kernel
        N =  V/ tf.ones_like(V , tf.float32) *1

        return nn_ops.conv2d_backprop_input(tf.shape(X), N, relevance, strides, padding)
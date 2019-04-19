import tensorflow as tf

class DQN:
    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, scope):
        
        self.K= K
        self.scope = scope
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            #considering input as 4 series of images
            self.X = tf.placeholder(tf.float32, shape = (None, IM_SIZE, IM_SIZE, 4), name = 'X') 
            #order: (num_samples, height, width, "color")
            
            #RL variables
            self.G = tf.placeholder(tf.float32, shape = (None, ), name = 'G')
            self.actions = tf.placeholder(tf.int32, shape = (None, ), name = 'actions')
            
            #convolution
            Z =self.X/255.0
            Z= tf.transpose(Z, [0,2,3,1])
            
            i = 0
            for num_output_filters, filtersz, stridesz in conv_layer_sizes:
                #print("debugging: ")
                #print((num_output_filters, filtersz, stridesz))
                Z = tf.contrib.layers.conv2d(Z, num_output_filters, filtersz, stride = stridesz, activation_fn=tf.nn.relu)
                i += 1
                
            #fully connected layers
            Z = tf.contrib.layers.flatten(Z)
            for M in hidden_layer_sizes:
                Z = tf.contrib.layers.fully_connected(Z, M)
            
            #final layer
            self.predict_op = tf.contrib.layers.fully_connected(Z, K)
            
            #also one_hot_encode_all_predictions(actions)
            selected_action_values = tf.reduce_sum(self.predict_op* tf.one_hot(self.actions, self.K), reduction_indices=[1])
            
            cost = tf.reduce_sum(tf.square(self.G- selected_action_values))
                
            self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
     
            self.cost = cost
    
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key= lambda x:x.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda x:x.name)

        ops = []
        for p, q in zip(mine, theirs):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)
    
    def set_session(self, session):
        self.session = session
        
    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict = {self.X: states})
    
    def update(self, states, actions, targets):
        c,_ = self.session.run([self.cost, self.train_op], feed_dict = {self.X:states, self.G:targets, self.actions:actions})
        return c
    
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])
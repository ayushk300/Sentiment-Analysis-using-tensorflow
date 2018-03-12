import tensorflow as tf
import numpy as np
from create_sentimental_feature_set import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
n_nodes_hl1 = 500 
n_nodes_hl2 = 500 #no of nodes in layer 2 
n_nodes_hl3 = 500

n_classes = 2 
batch_size = 100 # batches of 100 images around we will feed. 

#height X width
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def nueral_network_model(data):
	hidden_1_layer = {'weights':tf.variable(tf.random.normal([len(train_x[0]),n_nodes_hl1])),
					'baises': tf.variable(tf.random.normal(n_nodes_hl1))}
					
	hidden_2_layer = {'weights':tf.variable(tf.random.normal([n_nodes_hl1,n_nodes_hl2])),
					'baises': tf.variable(tf.random.normal(n_nodes_hl2))}
					
	hidden_3_layer = {'weights':tf.variable(tf.random.normal([n_nodes_hl2,n_nodes_hl3])),
					'baises': tf.variable(tf.random.normal(n_nodes_hl3))}
					
	output_layer = {'weights':tf.variable(tf.random.normal([n_nodes_hl3,n_classes])),
					'baises': tf.variable(tf.random.normal(n_classes))}
					
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['baises'])
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['baises'])
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['baises'])
	l3 = tf.nn.relu(l1)
	
	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['baises'])
	output = tf.nn.relu(output)
	
	return output
	
def train_neural_network(x):
	prediction = nueral_network_model(x)
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions,y))
	optimizer  = tf.train.AdamOptimizer().minimize(cost)
	
	#epochs is basically feed forward + backpropogation
	hm_epochs = 10 
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		
		for epoch in hm_epochs:
			epoch_loss = 0 

            while (i<len(train_x)):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

				_ , c = session.run([optimizer, cost],feed_dict = {x:batch_x,y:batch_y})
                epoch_loss+=c
                i +=batch_size

            print("Epoch",epoch,"completed out of ",hm_epochs,"loss:",loss)
        print(prediction)
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = 	tf.reduce_mean(tf.cast(correct,'float'))
        print("accuracy:",accuracy.eval({x:test_x,y:test_y}))


train_neural_network(x)

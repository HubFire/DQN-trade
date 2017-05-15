import tensorflow as tf 
import pandas as pd
from collections import deque
import numpy as np
from stock_env import StockEnv
import random


TRADE_PERIOD =240 
GAMMA = 0.9 #discount factor 
EPSILON = 0.9
REPLAY_SIZE = 500
BATCH_SIZE = 32

class DQN_Trade():
	def __init__(self):
		self.replay_buffer = deque()
		self.time_step = 0
		self.epsilon = EPSILON
		self.state_dim = 40
		self.action_dim = 3
		#build NN and train mothed
		self.buildNetwork()
		self.train_method()
		#init TF session
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()
		check_point = tf.train.get_checkpoint_state('saved_network')
		if check_point and check_point.model_checkpoint_path:
			self.saver.restore(self.session,check_point.model_checkpoint_path)
			print 'load model  success'
		else:
			print 'can not find old network weight'
	def buildNetwork(self):
		#first layer ,100 units
		W1 = self.weight_variable([40,100])
		b1 = self.bias_variable([100])
		W2 = self.weight_variable([100,3])
		b2 = self.bias_variable([3])

		#input layer
		self.state_input = tf.placeholder('float',[None,40])
		#hiden layer
		layer_1 = tf.nn.relu(tf.matmul(self.state_input,W1)+b1)
		# Q value layer
		self.Q_value = tf.matmul(layer_1,W2)+b2

	def train_method(self):
		self.action_input = tf.placeholder('float',[None,3])
		self.y_input = tf.placeholder('float',[None])
		Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input),reduction_indices=1)
		self.cost = tf.reduce_mean(tf.square(self.y_input-Q_action))
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

	def train_net(self):
		# get random  sample from replay buffer 
		minibach = random.sample(self.replay_buffer, BATCH_SIZE)
		state_bach = [data[0] for data in minibach]
		action_bach = [data[1] for data in minibach]
		reward_bach = [data[2] for data in minibach]
		next_state_bach = [data[3] for data in minibach]

		#calcuate Q
		Y_bach =[]
		next_Q = self.Q_value.eval(feed_dict = {self.state_input:next_state_bach})
		for i in range(0,BATCH_SIZE):
			done = minibach[i][4]
			if done :
				Y_bach.append(reward_bach[i])
			else:
				Y_bach.append(reward_bach[i]+GAMMA*np.max(next_Q[i]))
		self.optimizer.run(feed_dict ={
			self.y_input:Y_bach,
			self.action_input:action_bach,
			self.state_input:state_bach
			})

	def precive(self,state,action,reward,state_,done):
		self.time_step +=1
		one_hot_action = np.zeros(3)
		one_hot_action[action] =1
		self.replay_buffer.append((state,one_hot_action,reward,state_,done))
		if len(self.replay_buffer) >REPLAY_SIZE:
			self.replay_buffer.popleft()
		if len(self.replay_buffer) >100: # after 100 step ,pre  train
			self.train_net()

	def egreedy_action(self,state):
		Q_value  = self.Q_value.eval(feed_dict={
			self.state_input:[state]
			})
		Q_value = Q_value[0]

		if self.epsilon<=0.1:
			epsilon_rate =1
		else:
			epsilon_rate =0.95
		if self.time_step > 200 :
			self.epsilon=epsilon_rate*self.epsilon

		if random.random()<= self.epsilon:
			return random.randint(0, 2)
		else:
			return np.argmax(Q_value)

	def action(self,state):
		Q_value  = self.Q_value(feed_dict={
			self.state_input:[state]
			})
		Q_value = Q_value[0]
		return np.argmax(Q_value)

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial  = tf.constant(0.01,shape=shape)
		return tf.Variable(initial)

	def save_model(self,step):
		saver = self.saver
		saver.save(self.session, 'saved_network/'+'network' + '-dqn',global_step=step)
	
	def getLoss(self):
		pass


def main(train = False):
	data = np.loadtxt('./data.csv',delimiter = ',',skiprows=1)
	data = data[230:-1]  #delete the first day data
	angent = DQN_Trade()
    
	for i in range(0,10):
		iters =len(data)/240
		for iter_step in range(0,iters):
			#print iter_step
			iter_data =data[iter_step*240:iter_step*240+240]
			env =StockEnv(iter_data)
			s = env.reset()
			while True:
				action = angent.egreedy_action(s)
				s_,reward,done =env.gostep(action)
				print action
				angent.precive(s,action,reward,s_,done)
				s= s_
				if done:
					break
		angent.save_model(step=i)		


main()	





import pandas as pd

class StockEnv():
	def __init__(self,data):
		self.action_space = ['b','s','n']
		self.n_actions = len(self.action_space)
		self.data = data
		self.step = 0
		self.hold = 0
		self.cash = 0

	def reset(self):
		data = self.data[self.step]
		self.hold =0
		self.cash =0
		return data 

	def gostep(self,action): 
		cash = self.cash
		hold = self.hold
		self.step +=1
		s = self.data[self.step-1]
		oldPrice =s[0]
		
		if action == 0: # buy
			self.hold+=1
			self.cash-= oldPrice
		elif action ==1: #sell
			self.hold -=1
			self.cash += oldPrice
		else:
			pass  # nothing

		if self.step ==239:
			done = True
		else:
			done = False
		s_ = self.data[self.step]
		new_pro = s_[0]*self.hold+self.cash
		old_pro = oldPrice*hold+cash
		reward = new_pro-old_pro
		return s_,reward,done

# df = pd.read_csv('./data.csv')
# env = StockEnv(df)
# s0= env.reset()
# s1,reward,done = env.gostep(1)
# s2,reward,done = env.gostep(1)
# print reward
# #print done


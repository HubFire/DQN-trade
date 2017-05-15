import pandas as pd 


def  getDayDf(path):
	df = pd.read_csv(path,usecols=['time','close','volume'])
	return df

def getMinDf(path):
	df = pd.read_csv(path,usecols=['year','mon','day','time','close','volume'])
	return df 

def normalDayDf(df):
	first = df.loc[0,]
	new_df  = df
	new_df['close'] = new_df['close'] /first['close']
	new_df['volume'] = new_df['volume'] /first['volume']
	return new_df

def getFitures(index,min_df,day_df):
	if index>= 9:
		fiture = dict()
		i=9
		while i>=0:
	   		data_min = min_df.loc[index-i,]
	   		col_close = 'close_min{}'.format(9-i)
	   		col_volume ='volume_min{}'.format(9-i)
	   		fiture[col_close] = data_min['close']
	   		fiture[col_volume] = data_min['volume']
	   		i-=1
		year ,mon,day= min_df.loc[index,'year'],min_df.loc[index,'mon'],min_df.loc[index,'day']
		time = int(day+100*mon+10000*year)
		end =list(day_df['time']).index(time)
		start = end -10
		for i in range(0,10):
	   		data_day = day_df.loc[start+i,]
	   		col_close = 'close_day{}'.format(i)
	   		col_volume ='volume_day{}'.format(i)
	   		fiture[col_close] = data_day['close']
	   		fiture[col_volume] = data_day['volume']
		return fiture

def create_data(min_df,day_df):
	cols =[]
	for i in range(0,10):
		cols.append('close_min{}'.format(i))
		cols.append('volume_min{}'.format(i))
	for i in range(0,10):
		cols.append('close_day{}'.format(i))
		cols.append('volume_day{}'.format(i))	
	df  = pd.DataFrame(columns=cols)
	for index in min_df.index:
		if index>8:  
			data = getFitures(index,min_df,day_df)
			df.loc[index]=data
	return df 
    



day_df = getDayDf('./sz000001_day.csv')
min_df = getMinDf('./sz000001_min.csv')

normal_day = normalDayDf(day_df)
normal_min = normalDayDf(min_df)
#print normal_day.head(n=10)
#print normal_min.head(n=10)
#feature = getFitures(9,normal_min,normal_day)
#print normal_day.index
#print normal_min.index
df = create_data(normal_min,normal_day)
df.to_csv('./data.csv',index=False)
print df.head()


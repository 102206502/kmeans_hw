# -*- coding: utf-8 -*-
import pandas as pd

class testmain(object):
	"""docstring for main"""
	def __init__(self):
		self.df = pd.read_csv('iris.data.csv', header = None)
	
	def main(self):
		#load petal length(3rd col), petal width(4th col)
		print(self.df.head())
		print(self.df.tail())
		# y = df.iloc[]
if __name__ == '__main__':
	test = testmain()
	test.main()
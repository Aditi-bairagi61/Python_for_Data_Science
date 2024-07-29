# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:41:58 2024

@author: HP
"""

#-----------------------24 April 2024-------------------#

#This is codes for matplotlib 
'''easily understand the data is symetrically
 distributed
 diagnostics tools for data scientists
 -boxplot, histogram
 All diagonistics tools uses matplotlib
 matplolib and seabon gives the similar kind of function
 matplotlib is a major package and inside that their is pyplot
 #you are giving alias name as plt to that matplotlib
 #why alias name is used :
     instead of using  matplolib.pyplot each time
     we can give the short name and use this in program
     #so here alias name is plt
'''
#select the tools->references->ipython console->graphics
#->backend->automatic
import matplotlib.pyplot as plt
X=range(1,50)
Y=[value*3 for value in X]
print("values of X:")
print(*range(1,50))

'''This is equivalence to-
i in range(1,50)
   print(i,end='')
'''
print("values o y(thrice of X:",Y)
print(Y)
###################################
#draw a line using matplotlib

#plot lines and/or markers to the axes
#from below line select all and then run dont run code line by line
plt.plot(X,Y)
#set the X aix label of the current axis
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#set a title
plt.title("draw a line")
#display the line
plt.show()

##################################################
#positive and negative slop of line
#label in the x axis, y axis and a title
import matplotlib.pyplot as plt
# x axis values
x=[1,2,3]
#y axis values
y=[2,4,1]
#plot lines and/or markers to the Axes
plt.plot(x,y)
#set the x axis label of the current axis
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('sample graph!')
#display a figure
plt.show()

###############################################
#write a python program to plot two or more lines
#on same plot with suitable legends of each lines
import matplotlib.pyplot as plt
#line 1 points
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]

#plotting line 1 points
plt.plot(x1,y1,label="line 1")
#plotting the line 2 points
plt.plot(x2,y2,label="line 2")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('two or more lines on same plot with suitable legend')
plt.legend() #in right corner of plot it display 2 lines(line1, line2)
plt.show()

##################################################
#write python program to plot two or more
#lines on same plot but with different 
#linewidth and color
#synatx: linewidth=3, color='blue'
import matplotlib.pyplot as plt
#line 1 points
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]

#plotting line 1 points
plt.plot(x1,y1,color='blue',linewidth=3 ,label="line 1")
#plotting the line 2 points
plt.plot(x2,y2,color='red',linewidth=5,label="line 2")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('two or more lines on same plot with suitable legend')
plt.legend() #in right corner of plot it display 2 lines(line1, line2)
plt.show()

#########################################################
#write python code to plot two or more lines with
#different styles
#synatx: linestyle='dotted', linestyle='dashed'

import matplotlib.pyplot as plt
#line 1 points
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]

#plotting line 1 points
plt.plot(x1,y1,color='blue',linewidth=3 ,label="line 1",linestyle='dotted')
#plotting the line 2 points
plt.plot(x2,y2,color='red',linewidth=5,label="line 2",linestyle='dashed')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('two or more lines on same plot with suitable legend')
plt.legend() #in right corner of plot it display 2 lines(line1, line2)
plt.show()


#------------------------25 April 2024-----------------#
#######################################################
#write a python program to plot two or more lines
#and set the line markers
import matplotlib.pyplot as plt
x=[1,4,5,6,7]
# y axis values
y=[2,6,3,6,3]
#plotting the points
plt.plot(x,y,color='red',linestyle='dashdot',linewidth=3,marker='o',markerfacecolor='blue',markersize=12)
#set the y-limits of the current axes.
plt.ylim(1,8)
#set the x-limits of the current axes
plt.xlim(1,8)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#giving title to graph
plt.title('display marker')
#show function used to show the plot
plt.show()

##################################################

#-----------bar graph----------#
#write this code in quick reference pages#
#bcz, this code is required several times
'''disadvantage of BAr graph :it cannot be suitable for 
#big data
#advantage :it is suitable for small type of data'''
#write python programming to display a bar
# chart of the
#popularity of programming languages
import matplotlib.pyplot as plt
x=['java','python','php','javascript','C#','C++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
#list comprehension
x_pos=[i for i,_ in enumerate(x)] #list comprehension
plt.bar(x_pos,popularity,color='blue')
plt.xlabel("Languages")
plt.ylabel("popularity")
plt.title("popularity of programming languages\n"+
          "Worldwise, oct 2017 compared to yr ago")
plt.xticks(x_pos,x)
#turn on grid
plt.minorticks_on()
plt.grid(which='major',linestyle='-',linewidth='0.5',color='red')
plt.show()

#################################################
#yticks and hbar(horizontal graph)
import matplotlib.pyplot as plt
x=['java','python','php','javascript','C#','C++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
#list comprehension
x_pos=[i for i,_ in enumerate(x)] #list comprehension
plt.barh(x_pos,popularity,color='green')
plt.xlabel("Languages")
plt.ylabel("popularity")
plt.title("popularity of programming languages\n"+
          "Worldwise, oct 2017 compared to yr ago")
plt.yticks(x_pos,x)
#turn on grid
plt.minorticks_on()
plt.grid(which='major',linestyle='-',linewidth='0.5',color='red')
plt.show()

####################################################

#write a python program to plot bar graph
#with different color to each bar


import matplotlib.pyplot as plt
x=['java','python','php','javascript','C#','C++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
#list comprehension
x_pos=[i for i,_ in enumerate(x)] #list comprehension
plt.bar(x_pos,popularity,color=['blue','red','violet','yellow','black','gray'])
plt.xlabel("Languages")
plt.ylabel("popularity")
plt.title("popularity of programming languages\n"+
          "Worldwise, oct 2017 compared to yr ago")
plt.xticks(x_pos,x)
#turn on grid
plt.minorticks_on()
plt.grid(which='major',linestyle='-',linewidth='0.5',color='red')
plt.show()

##################################################

#-----------------Histogram------------------------#
'''histogram: whether your data is symmetriclly
distributed for this we are using histogram
symmetrically and normal distribution of data
no skewness 
left skewed:from left side is long run 
right skewed:from right size is long run

histogram output is same as bar graph but here in hist
all bars are connected to each other'''
import matplotlib.pyplot as plt
blood_sugar=[113,85,90,150,149,88,93,115,135,80,77,82,129]
plt.hist(blood_sugar,rwidth=0.5,bins=4)

'''histogram showing norml,prediabetci and diabetic
 patients distribution
 
 80-100:normal
 100-125:Pre-diabetic
 125 onwards:diabetic
'''

plt.xlabel("sugar level")
plt.ylabel("no of paients")
plt.title("blood sugar chart")
plt.hist(blood_sugar,bins=[80,100,125,150],rwidth=0.95,color='red')

###############################################
#---------------Box Plot----------------------#
'''above one dot and below one dot are the outliers
in output
'''
#import libraries
import matplotlib.pyplot as plt
import numpy as np

#creating dataset
np.random.seed(10)
data=np.random.normal(100,20,200)
fig=plt.figure(figsize=(10,7))
#creating plot
plt.boxplot(data)
#show data
plt.show()

###############################################
#plot the multiple boxplot
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
data_1=np.random.normal(100,10,200)
data_2=np.random.normal(90,20,200)
data_3=np.random.normal(80,30,200)
data_4=np.random.normal(70,40,200)
data=[data_1,data_2,data_3,data_4]
fig=plt.figure(figsize=(10,7))
#creating axes instance
ax=fig.add_axes([0,0,1,1])
#creating plot
bp=ax.boxplot(data)
#show plot
plt.show()

#############################################

#----------------------29 april 2024-------------#
#how to plot graph on dataset
#seaborn library
import seaborn as sns
import pandas as pd

cars=pd.read_csv("C:/Users/HP/Desktop/DS/Cars.csv")
cars.head()
cars.columns
sns.relplot(x='HP',y='MPG',data=cars)
sns.relplot(x='HP',y='MPG', data=cars,kind='line')
sns.catplot(x='HP',y='MPG', data=cars,kind='box')
sns.catplot(x='VOL',y='SP', data=cars,kind='box')
sns.distplot(cars.HP)

'''
exploratory data analysis
#1. measure the central tendency
2. measure the dispersion
3. third moment business decision 
4.moment business decision
5. probability distribution
6. graphical representation(histofram, Boplot)'''
cars.describe()

#graphical representation
import matplotlib.pyplot as plt
import numpy as np
#if data quantity is small then you can print bar graph
plt.bar(height=cars.HP,x=np.arange(1,82,1))
sns.distplot(cars.HP)
#data is right skewed
plt.boxplot(cars.HP)
#there are several otliers in hp columns
#similar operations are expected from other 3 columns
sns.distplot(cars.MPG)
#data is slight it left distributed
plt.boxplot(cars.MPG)
#there are no outliers
sns.distplot(cars.VOL)
#data is slighit left distributed
plt.boxplot(cars.VOL)
sns.distplot(cars.SP)
plt.boxplot(cars.SP)
sns.distplot(cars.WT)
plt.boxplot(cars.WT)
#there are several outliers

######################################################
#now let us plot joint plot, joint plot is to show scatter
#histogram
#HOW many times each value occured = count plot

import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])
#now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#92 HP value occured 7 times
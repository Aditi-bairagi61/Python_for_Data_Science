# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 08:54:57 2024

@author: HP
"""

#pandas having two types 1) series , 2) columns
#two ways to use the series we can import data from files 
#other way is we can provide the list as a series from user
import pandas as pd
songs2=pd.Series([145,142,38,13],name='counts')
#it is easy to inspect the index of aseries 
songs2.index

#the index can be string  based as well
#in which case pandas indicats that
# datatype for the index is object(not string)
songs3=pd.Series([145,142,38,13],name='counts')
index=['paul','john','george','ringo']
songs3.index
songs3

import pandas as pd
f1=pd.read_csv('age.csv')
f1
df.pd.read_excel('Bahaman.xlsx')
#None, NAn, nan and null are synonyms
#the series object behaves similarly to a numpy array


import numpy as np
numpy_ser=np.array([145,142,38,13])
songs3[1]
#142
print(numpy_ser[1])
#they both have methods in a common
songs3.mean()
numpy_ser.mean()

#############################
#the pandas series data structure provides support
#for the basic CRUD operations
#-create,read, update,and delete


george=pd.Series([10,7,1,22], # creation of series
index=['1990','3324','6785','4325'],
name='george_songs')
george

#in the following code not giving index
#so by default it will take integer index
#which is unique
george=pd.Series([10,7,1,22], # creation of series

name='george_songs')
george
############################################
#reading the index value
george=pd.Series([10,7,1,22], # creation of series
index=['1990','3324','6785','4325'],
name='george_songs')
george
george['3324']

#we can iterate over data in a series as well
#when iterating over a series
for item in george:
    print(item)
    
#########################################
#updating
#updating values in a series can be a little tricky as well.
#to  update a value
#for a given index label, the standard
#index assignments operation works
george['3324']=68
george

#############################################
#3 April
#delete item using index
george=pd.Series([10,7,1,22], # creation of series
index=['1990','3324','6785','4325'],
name='george_songs')
george
s=pd.Series([2,3,4],index=[1,2,3])
del s[1] #here index 1 is 3 so 3 is get deleted
s
george

#convert types 
#string use.astype(str)
#numeric use pd.to_numeric
#integer use.astype(int)
#note that this will fail with NaN
#datetime use pd.to_datetime
import pandas as pd 
songs_66=pd.Series([3.0,None,11.0,9.0],
index=['George','ringo','john','paul'],
name='counts')
songs_66.dtypes  #print float as a data type
pd.to_numeric(songs_66.apply(str))
#ther will be error
pd.to_numeric(songs_66.astype(str),errors='coerce')
#if we pass errors='coerce',
#we can see that it supports many formats
songs_66.dtypes
#dealing with none
#the .fillna method will replace them with a given values
songs_66=songs_66.fillna(-1)
songs_66=songs_66.astype(int)
print(songs_66)
songs_66.dtypes

#NaN values can be dropped from 
#the series using .dropna
#how to drop null value from dataset or series
songs_66=pd.Series([3.0,None,11.0,9.0],
index=['George','ringo','john','paul'],
name='counts')
songs_66=songs_66.dropna() #drop null value i.e index 1 gets droped
songs_66

###############################################
#how to append one series with another series
#append ,combning, and joining two series
songs_69=pd.Series([7,9,34,5],
index=['ram','sham','krishna','ghansham'],
name='counts')
#to concatenate two series together , simply use the .append
songs=songs_66.append(songs_69)
print(songs)
songs=songs.astype(int) #default data type is float now using this statement it goes to in int type
print(songs)
songs.dtypes

##################################################
#plotting Series
#why using dot in matplotlib.plot 
#bcz, in matplotlib their is a package called pyplot
import matplotlib.pyplot as plt

#so here plt is a alias name
fig=plt.figure()
#when we dont use plt name so each time need to write
#matplotlib.pyplot.figure()
songs_69.plot()
plt.legend()

#printing grap for 2 series
fig=plt.figure()
songs_69.plot(kind='bar')
songs_66.plot(kind='bar',color='r')
plt.legend()

####################################################

#histogram is used for distribution 
import numpy as np
data=pd.Series(np.random.randn(500),name='500_random')
fig=plt.figure()
ax=fig.add_subplot(111)
data.hist()

################################################
# 4 April 2024
#what is pandas dataframe
#pandas dataframe is a two dimensional data structure
#an immuatable 
#to check the version of pandas

import pandas as pd
pd.__version__

#create using constructor
#create pandas dataframe from list
import pandas as pd
technologies=[["Spark",20000,"30days"],
              ["pandas",5600,"40days"]]
df=pd.DataFrame(technologies)
print(df)

#since we have not given label to columns and 
#indexes, Dataframe by default assigns
#add column and row labels to the dataframe
column_names=["courses","fee","Duration"]
row_label=["a","b"]
df=pd.DataFrame(technologies,columns=column_names,index=row_label)
print(df)

#################################################
#print the data types
df.dtypes
##################################
#you can also assign custom 
#data types to columns
#set custom types to DataFrame
import pandas as pd
#below is a dictionary having keys and values
#courses,fee,duration,discount are keys 
#tech={'course':[],'fee':[],'duration':[],'discount':[]}
technologies={'courses':["spark","pyspark","hadoop","python","pandas","oracle","java"],
              'Fee':[2000,2500,2600,2200,2400,2100,2900],
              'Duration':['30days','40days','50days','60days','70days','80days','90days'],
              'Discount':[11.8,23.7,67.5,45.6,34.6,12.5,99.4]}
df=pd.DataFrame(technologies)
print(df.dtypes)
#########################################
#convert all types to best possible types
df2=df.convert_dtypes()
print(df.dtypes)
#change all columns to same type
df=df.astype(str)
print(df.dtypes)
#Change type for one or multiple columns
df=df.astype({"Fee":int,"Discount":float})
print(df.dtypes)

#when i want to chnage the columns datatype  of multiple columns
#then following steps follows
#like create one list and mention the column name 
#and then set the data type using astype
df=pd.DataFrame(technologies)
df.dtypes
cols=['Fee','Discount']
df[cols]=df[cols].astype('float')
df.dtypes

#ignore the error so here courses have object as a data type
#when i change the courses to int it will show error
#so it will ignore the error using following sentence
#df=df.astype({"courses":int}) error in this line
df=df.astype({"courses":int},errors='ignore')
df.dtypes

#generate error
df=df.astype({"courses":int},errors='raise')

#converts feed column to numeric type
df=df.astype(str)
print(df.dtypes)
df['Discount']=pd.to_numeric(df['Discount'])  #it will convert data type to float
df.dtypes

#########################
#before runnin code below select the directory
import pandas as pd
#create dataframe from disctionary
technologies={'courses':["spark","pyspark","hadoop"],
               'Fee':[2000,2500,10],
              'Duration':['70days','80days','90days'],
              'Discount':[1000,3323,4874]}
df=pd.DataFrame(technologies)
df
#convert dataframe to csv
df.to_csv('data_file.csv') #csv file is created in a selected folder
df=pd.read_csv('data_file.csv')
print(df)

#########################################################
#pandas dataframe -Basic operations
#create dataframe with None/ull to work with examples
#giving labels to rows
import pandas as pd
import numpy as np
technologies=({'courses':["spark","pyspark","hadoop","python","pandas",None,"oracle","java"],
              'Fee':[2000,2500,2600,2200,np.nan,2400,2100,2900],
              'Duration':['30days','40days','50days','60days','70days','80days','90days','67days'],
              'Discount':[11.8,23.7,67.5,45.6,34.6,12.5,99.4,90.3]})
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df=pd.DataFrame(technologies,index=row_labels)
print(df)
#############################################
#5 April 2024
#DataFrame properties
df.shape #hit will display 8 rows and 4 columns
#(8,4)
df.size
#32
df.columns
df.columns.values
df.index
df.dtypes
df.info

################################################
#accessing one column contents
df['Fee']
#Accessing two columns contents
cols=['courses','Duration']
df[cols]
df[['courses','Duration']]

#select certain rows and assign it to another DataFrame
#df[start:end]
#df[rows,columns]
#df[start_row:end_row,start_column:end_column]
#df[:,:2] here 
#all rows and all columns df[:,:] or df
df2=df[6:] #start from 6 row upto end
df2
df2=df[:6] #prints 0 to 5
df2
###############################################
#Accessing certain cell from column 'Duration'
df['Duration'][3]

#Subtracting specific value from a column
df['Fee']=df['Fee']-500
df['Fee']
#############################################
#df.describe
#mean, median, standard deviation
#df describe is only appicable to the integer or numeric data
df.describe()
#it will show 5 number summary

##############################################
#rename()- renames pandas DataFrame columns
df=pd.DataFrame(technologies,index=row_labels)

#assign new header by setting new column names
df.columns=['A','B','C','D']
df
######################################################
#whenever u want to do chnages in a row then axis=0
#whenever u want to do changes in a column then axis=1
#Rename column names using rename() method
df=pd.DataFrame(technologies,index=row_labels)
df.columns=['A','B','C','D']
df2=df.rename({'A':'C1','B':'C2'},axis=1)
df2
df2=df.rename({'C':'C3','D':'C4'},axis='columns')
df2
df2=df.rename(columns={'A':'C1','B':'C2','C':'C3','D':'C4'})
df2

#######################################################
#drop DataFrame rows and columns
df=pd.DataFrame(technologies,index=row_labels)

#drop rows by labels
df1=df.drop(['r1','r2'])
df1
#delete rows by position/index
df1=df.drop(df.index[1])
df1
df1=df.drop(df.index[[1,3]])
df1
#delete rows by index range
df1=df.drop(df.index[2:])
df1

#when you have default indexes for rows
df=pd.DataFrame(technologies)
df1=df.drop(0)
df1
df=pd.DataFrame(technologies)
df1=df.drop([0,3],axis=0) #it will delete row0 and row3
df1
df1=df.drop(range(0,2)) #it will delete 0 and 1
df1

#------------------------------------------------
#10 April 2024
######################################
#drop column by index
print(df.drop(df.columns[1],axis))

#explicitly using parameter name 'labels'
import pandas as pd
import numpy as np
technologies=({'courses':["spark","pyspark","hadoop","python","pandas",None,"oracle","java"],
              'Fee':[2000,2500,2600,2200,np.nan,2400,2100,2900],
              'Duration':['30days','40days','50days','60days','70days','80days','90days','67days'],
              'Discount':[11.8,23.7,67.5,45.6,34.6,12.5,99.4,90.3]})
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df=pd.DataFrame(technologies)
df2=df.drop(labels=["Fee"],axis=1)
#Alternatively you can also use columns instead of label
df2=df.drop(columns=["Fee"],axis=1)
################################################
#Drop column by index
print(df.drop(df.columns[1],axis=1))
df=pd.DataFrame(technologies)
#using inplace=true
df.drop(df.columns[2],axis=1,inplace=True)
print(df)
#drop one or more columns by label name
df2=df.drop(['courses','Fee'],axis=1)
print(df2)
##########################################
#drop two or more columns by index
df=pd.DataFrame(technologies)
df2=df.drop(df.columns[[0,1]],axis=1)
print(df2)

######################################3#
#drop columns from list of columns
df=pd.DataFrame(technologies)
df.columns # it will gives column names
liscol=["courses","Fee"]
df2=df.drop(liscol,axis=1)
print(df2)

##################################################
#remove columns from DataFrame
#when you want to delete multiple  columns then list inside list
df=pd.DataFrame(technologies)
df
df.drop(df.columns[1,axis=1,inplace=True)
df.drop(df.columns[[1,2]],axis=1,inplace=True) #delete multiple columns using list inside list
df

###################################################
'''when u want to acess the column using index
#then function is used i.e iloc, and loc
#when you access columns using index iloc
#when you access columns using name loc
#difference between iloc and loc must in interview 
#very imp iloc and loc
#whenever ur accessing either rows and columns using index 
#number then iloc method is used 
'''
import pandas as pd
import numpy as np

df=pd.DataFrame(technologies)
df
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df-pd.DataFrame(technologies,index=row_labels)
print(df)
df=pd.DataFrame(technologies)
df=pd.DataFrame(technologies,index=row_labels)
#below is the syntax for iloc
#df.iloc[startrow:endrow,startcolumn:endcolumn]
#below are quick sample
df2=df.iloc[:,0:2] #it display all rows  and columns from 0 and 1
df2
''' above 2 lines explain here this line uses the slicing operator to get DataFrame
item by index.
the first slice[:] indicates to return all rows
the second slice specifies that only columns
between 0 adnd 2(excluding 2) should be returned.'''

df2=df.iloc[0:2,:]
df2

''' above 2 line explain In this case the first slice[0:2] is 
requesting  only rows 0 through 1 of the Dataframe
the second slice[:] indicates that all columns are requires.
'''
#slicing specific rows and columns using iloc  attribute
df3=df.iloc[1:2,1:3]  #it displays 1st row and display 1st and 2nd columns
df3

#another example
df3=df.iloc[:,1:3] 
df3
#the second operator [1:3] yields columns 1 and 3 only
#select rows by integer index
df2=df.iloc[2] #select row by index it will select only 2nd row
df2
#below line display 2nd 3rd and 6th row 
df2=df.iloc[[2,3,6]] #select rows by index list 
print(df2)
df2=df.iloc[1:5] #select  rows by integer index range
print(df2)
df2=df.iloc[:1] #select first row
df2=df.iloc[:3] #select first three rows
df2=df.iloc[-1:] #select last row
df2=df.iloc[-3:] #select last 3 row
df2=d.iloc[::2] #selects alternate rows 

#--------------------------------------------------#
#12 April 2024
import pandas as pd
import numpy as np
technologies=({'courses':["spark","pyspark","hadoop","python","pandas",None,"oracle","java"],
              'Fee':[2000,2500,2600,2200,np.nan,2400,2100,2900],
              'Duration':['30days','40days','50days','60days','70days','80days','90days','67days'],
              'Discount':[11.8,23.7,67.5,45.6,34.6,12.5,99.4,90.3]})
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df=pd.DataFrame(technologies,index=row_labels)
#select rows by index labels 
df2=df.loc[['r2']]
df2
df2=df.loc[['r2','r3','r6']] #select rows by index label
df2=df.loc['r1':'r5'] #select rows by label index range
df2=df.loc['r1':'r5':2] #select alternate rows with index

#by using df[] notation
df2=df['courses'] #it will select only courses column
#select multiple columns
df2=df[["courses","Fee","Duration"]] #here accessing 3 columns in list inside list

#using loc[] to take column slices
#loc[] syntax to slice columns
#df.loc[:,start:stop:step]
#select multiple columns
df2=df.loc[:,["courses","Fee","Duration"]]
#select random columns
df2=df.loc[:,["courses","Fee","Discount"]]
#select column between two columns
df2=df.loc[:,'Fee':'Discount']
#select column by range
df2=df.loc[:,'Duration':]
#select columns by range
#All column upto duration 
df2=df.loc[:,:'Duration']
#select every alternate column
df2=df.loc[:,::2]

################################################
#pandas.DataFrame.query() by examples
#query all rows with courses equals 'spark'
#imp give the row name in single code
df2=df.query("courses=='spark'")
print(df2)

################################################
#not equals condition
df2=df.query("courses !='spark'") #it will exclude spark row and display other rows
df2

##################################################
#pandas add column to DataFrame
import pandas as pd
import numpy as np
technologies={'courses':["spark","pyspark","hadoop","pandas"],
              'Fee':[22000,25000,23000,24000],
              'Discount':[0.1,0.2,0,0.5]}
df=pd.DataFrame(technologies)
print(df)

#################################
#pandas add column to dataframe
#add new column to the dataframe
tutors=['ram','sham','ghansham','ganesh']
df2=df.assign(TutorsAssigned=tutors)
print(df2)

########################################
#add multiple columns to the dataframe
MNCCompanies=['TATA','HCL','Infosys','Google']
df2=df.assign(MNC=MNCCompanies,tutors=tutors)
df2

###############################################
#using lambda function you can derive new column
#derive new column from existing column
df=pd.DataFrame(technologies)
df2=df.assign(Discount_percent=lambda x: x.Fee*x.Discount/100)
print(df2)

###############################################
#append column to exisiting pandas dataframe
#add new column to the exisitng dataframe
df=pd.DataFrame(technologies)
df["MNCCompanies"]=MNCCompanies
print(df)

#############################################
#append the column at specific position
df=pd.DataFrame(technologies)
df.insert(0,'Tutors',tutors) #tutors column place in 0th position
print(df)

################################################
#when ever you want to rename multiple columns then use dictionary
import pandas as pd
import numpy as np
technologies=({'courses':["spark","pyspark","hadoop","python","pandas","oracle","java"],
              'Fee':[2000,2500,2600,2200,2400,2100,2900],
              'Duration':['30days','40days','50days','60days','70days','80days','90days'],
              })
df=pd.DataFrame(technologies)
df.columns
#pandas rename column name
#rename a multiple column
df.rename(columns={'courses':'courses_list','Fee':'Courses_fee',
                   'Duration':'Courses_duration'},inplace=True )
print(df.columns)
df.columns

#-------------------------------------------------------#
#15 April 2024
#Finding number or rows and columns in a dataframe

import pandas as pd
import numpy as np
technologies=({'courses':["spark","pyspark","hadoop","python","pandas",None,"oracle","java"],
              'Fee':[2000,2500,2600,2200,np.nan,2400,2100,2900],
              'Duration':['30days','40days','50days','60days','70days','80days','90days','67days'],
              'Discount':[11.8,23.7,67.5,45.6,34.6,12.5,99.4,90.3]})
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df=pd.DataFrame(technologies,index=row_labels)
rows_count=len(df.index)
rows_count
rows_count=len(df.axes[0])
rows_count
column_count=len(df.axes[1])
column_count

##############################################
df=pd.DataFrame(technologies)
row_count=df.shape[0] #returns number of rows
row_count
col_count=df.shape[1] #return number of columns
print(row_count)
print(col_count)

#using DataFrame.apply() to apply function and column
import pandas as pd
import numpy as np
data={"A":[1,2,3],"B":[4,5,6],"C":[7,8,9]}
df=pd.DataFrame(data)
print(df)
def add_3(x):
      return x+3
df2=df.apply(add_3)
df2

#####################################################3
#print only one column
df2=((df.A).apply(add_3)) #A is a colum name
print(df2)

#####################################
#using apply function to single column
def add_4(x):
    return x+4
df["B"]=df["B"].apply(add_4)
df["B"]
#Apply to multiple columns
df[['A','B']]=df[['A','B']].apply(add_4)
df

#apply a lambda function to each column
df2=df.apply(lambda x:x+10)
df2

#instead using apply function you can use transform 
#it will going to show same result
import pandas as pd
import numpy as np
data={"A":[1,2,3],"B":[4,5,6],"C":[7,8,9]}
df=pd.DataFrame(data)
print(df)
def add_2(x): #when running function select function block then run otherwise it will shows error
    return x+3
df=df.transform(add_2)
print(df)

###########################################
#there are three function 
#1.Apply, 2.Transform , 3.map

#using pandas.DataFrame.map() to single column
df['A']=df['A'].map(lambda A:A/2.)
print(df)

###############################################
#using numpy function on a single column
#using DataFrame.apply() & [] operator
df=pd.DataFrame(data)
import numpy as np
df['A']=df['A'].apply(np.square) #it will take square of columns
print(df)

#take the sqaure of a perticular column
#using numpy.square() method
#using numpy.square() and [] operator
import pandas as pd
import numpy as np
data={"A":[1,2,3],"B":[4,5,6],"C":[7,8,9]}
df=pd.DataFrame(data)
print(df)
df['A']=np.square(df['A']) # it take only 1st column square
print(df)

#######################################################
#pandas groupby() function
technologies=({'courses':["spark","pyspark","hadoop","python","pandas","Hadoop","spark","python","NA"],
              'Fee':[2000,2500,2600,2200,2400,2100,2900,7600,7600],
              'Duration':['30days','40days','50days','60days','70days','80days','90days','67days','90days'],
              'Discount':[1000,2300,4500,3400,5689,None,328,3231,0]})
df=pd.DataFrame(technologies)
print(df)
#use groupby() to compute the sum
df2=df.groupby(['courses']).sum()
print(df2)

####################################################
#apply groupby to multiple columns
df2=df.groupby(['courses','Duration']).sum()
print(df2)

#####################################################
#add index to the grouped data
#using reset index function you can reset the index
#add row index to the group by result
df3=df.groupby(['courses','Duration']).sum().reset_index()
print(df3)

technologies=({'courses':["spark","pyspark","hadoop","python","pandas","Hadoop","spark","python","NA"],
              'Fee':[2000,2500,2600,2200,2400,2100,2900,7600,7600],
              'Duration':['30days','40days','50days','60days','70days','80days','90days','67days','90days'],
              'Discount':[1000,2300,4500,3400,5689,None,328,3231,0]})
df=pd.DataFrame(technologies)
print(df)
df.columns

#get the list of all column names from headers
column_headers=list(df.columns.values)
print("the column header:",column_headers)
###################3#######
#using list(df) to get the column headers as a list
column_headers=list(df.columns)
column_headers
#using list(df) to get the list of all columns names
column_headers=list(df)
column_headers


#------------------------------------------------------------
#16 April 2024
#pandas shuffle DataFrame rows
#shuffliing is import for splitting - traning and testing
import pandas as pd
technologies=({'courses':["spark","pyspark","hadoop","python","pandas","Hadoop","spark","python","NA"],
              'Fee':[2000,2500,2600,2200,2400,2100,2900,7600,7600],
              'Duration':['30days','40days','50days','60days','70days','80days','90days','67days','90days'],
              'Discount':[1000,2300,4500,3400,5689,None,328,3231,0]})
df=pd.DataFrame(technologies)
print(df)
#shuffle the dataframe rows and return all rows
df1=df.sample(frac=1)  #1 means 100%
print(df1)
df2=df.sample(frac=0.5) #0.5 means 50% shuffling of rows
print(df2)
df3=df.sample(frac=0.3)
print(df3)

###############################################
#create new index starting from zero
df1=df.sample(frac=1).reset_index()  #1 means present , 0 means absent
print(df1)

################################
#drop shuffle index
#it will drop the shuffle index
df1=df.sample(frac=1).reset_index(drop=True)
print(df1)

#joins
#inner join joins only the common row that are present in both the tables
#only common rows taken in inner join others are get omitted

#1st Dataframe
import pandas as pd
technologies=({'courses':["spark","pyspark","python","pandas"],
              'Fee':[2000,2300,3322,7890],
              'Duration':['30days','60days','78daays','89days']})
index_labels=['r1','r2','r3','r4']
df1=pd.DataFrame(technologies,index=index_labels)

#2nd dataframe
import pandas as pd
technologies2=({'courses':["spark","java","python","go"],
              'Discount':[2000,4000,7000,9000]})
index_labels2=['r1','r2','r3','r4']
df2=pd.DataFrame(technologies2,index=index_labels2)

#pandas join 
df3=df1.join(df2, lsuffix="_left", rsuffix="_right")
print(df3)
#if we not explicitly mention join then it is left join

#################################################
#pandas inner join DataFrames
df3=df1.join(df2,lsuffix="_left",rsuffix="_right",how='inner')
print(df3)

#pandas left join DataFrames
df3=df1.join(df2,lsuffix="_left",rsuffix="_right",how='left')
print(df3)

#pandas right join DataFrames
df3=df1.join(df2,lsuffix="_left",rsuffix="_right",how='right')
print(df3)

#using pandas.merge()
df3=pd.merge(df1,df2)

#using DataFrame.merge()
df3=df1.merge(df2)

################################################
#use pandas.concat()  to concat two dataframes
import pandas as pd
df=pd.DataFrame({'courses':["spark","pyspark","python","pandas"],
                 'Fee':[2000,25000,4500,43943]})
df1=pd.DataFrame({'courses':["pandas","hadoop","hyperion","java"],
                  'Fee':[3400,7890,9087,7833]})
#using pandas.concat() to concat two DataFrames
data=[df,df1]
df2=pd.concat(data)
df2
######################################################
#concatenate multiple dataframes using pandas.concat()
 import pandas as pd
df=pd.DataFrame({'courses':["spark","pyspark","python","pandas"],
                 'Fee':[2000,25000,4500,43943]})
df1=pd.DataFrame({'courses':["unix","hadoop","hyperion","java"],
                  'Fee':[3400,7890,9087,7833]})
df2=pd.DataFrame({'Duration':['30days','70days','50days','65days','90days'],
                  'Discount':[1000,7044,8944,3943,9000]})
#appending multiple dataframes
df3=pd.concat([df,df1,df2])
print(df3)

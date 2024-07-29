# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:29:14 2024

@author: HP
"""
#this is code for numpy library
#---------------23 april 2024------------------------#
#index out of bound
from numpy import array
a=([10,30,40])
print(a[4]) #error 
#index row of two dimensional array
from numpy import array
#define array
data=array([[11,22],[33,44],[55,66]])
#index data
print(data[0,]) #0 th row and all columns
#[11 22]


###################################3
#slice one -dimensional array
from numpy import array
#define array
data=array([11,22,33,44,55])
print(data[1:4])
#[22 33 44]

#####################################
#negative slicing of a one dimensional array
from numpy import array
#define array
data=array([11,22,33,44,55])
print(data[-2:])

######################################
#split input and output data
from numpy import array
#define array
data=array([11,22,33,44,55])
print(data[-2:])

#########################################
#split input and output data
from numpy import array
#define array
data=array([[11,22,33],[44,55,66],[77,88,99]])
#seperate data
X,y=data[:,:-1],data[:,-1]
X
y

##########################################
#broadcast scalar to one-dimensional array
from numpy import array
#define array
a=array([1,2,3])
print(a)
#define scalar
b=2
print(b)
#broadcast
c=a+b
print(c)

#----------------L1  norm---------------------------#

'''vector L1 norm 
the L1 norm is calculated as the num 
of the abslute vector values,
where absolue value of a scalar uses the
notation |a1|.
in effect the norm is a calculation of the 
manhattan distance from the origin of the vector space.
||v||=|a1|+|a2|+|a3|
'''
#where it is used L1 norm and L2 norm is used to find out the distance
#it is used in euclidean distance and manhattan disatnce
#que, what is meaning of L1 and L2 norm

from numpy import array
from numpy.linalg import norm
#define array
a=array([1,2,3])
print(a)
#calculate norm
l1=norm(a,1)
print(l1)

#############################
#----------------L2 norm---------------------------#
#vector L2 norm
'''the notation for the L2 norm of a vector
To calculate the L2 norm of a vector,
take the square root of the sum of the squared
vector values.
Another name for L2 nrm of a vector
is euclidean distance.
This is often used for calculating
the error in machine learning models'''
#l2=sqaure root x1^2+x2^2+x3^2
from numpy import array
from numpy.linalg import norm
#define vector
a=array([1,2,3])
print(a)
#calculate norm
l2=norm(a)
print(l2)

#########################################

#triangular matrices 
#triangular matrices used in image processing
from numpy import array
from numpy import tril
from numpy import triu
#define sqaure matrix
M=array([[1,2,3],[1,2,3],[1,2,3]])
print(M)
#output
'''print(M)
[[1 2 3]
 [1 2 3]
 [1 2 3]]'''
M # when i only print M the it will shows comma
#output
'''array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])'''
#lower triangular matrix
lower=tril(M)
print(lower)  
#upper triangular matrix
upper=triu(M)
print(upper)

###############################################
#diagonal matrix
from numpy import array
from numpy import diag
#define square matrix
M=array([[1,2,3],[1,2,3],[1,2,3]])
print(M)
#extract diagonal vector
d=diag(M)
print(d)
#create diagonal matrix from vector
D=diag(d)
print(D)

######################################
#identity matrix
 from numpy import identity
 I=identity(3)
 print(I)

#########################################
#orthogonal matrix

'''the matrix is said to be an orthogonal
matrix if the product of matrix
and its transpose gives an identity value'''

from numpy import array
from numpy.linalg import inv
#define orthogonal matrix
Q=array([[1,0],[0,-1]])
print(Q)
#inverse equivalence
V=inv(Q)
print(Q.T)
print(V)
#identity equivalence
I=Q.dot(Q.T)
print(I)

#-----------------------24 April 2024--------------#
#Transpose of matrix
from numpy import array
#define matrix
A=array([[1,2],[3,4],[5,6]])
print(A)
#calculate transpose
C=A.T
print(C)

###################################
#inverse matrix
from numpy import array
from numpy.linalg import inv
#define  matrix
A=array([[1.0,2.0],[3.0,4.0]])
print(A)
#invert matrix
B=inv(A)
print(B)

#multiply A and B
I=A.dot(B)
print(I)

#####################################
#sparse matrix
'''in sparse matrix their are more no of 
zeros in a rows'''
#how many no of 0's and 1's it will display using csr method
#todense gives us the original matrix
from numpy import array
from scipy.sparse import csr_matrix
#create dense matrix
A=array([[1,0,0,1,0,0],[0,0,2,0,0,1],[0,0,0,2,0,0]])
print(A)
#convert to sparse matrix (CSR method)
S=csr_matrix(A)
print(S)
#reconstruct dense matrix
B=S.todense()
print(B)
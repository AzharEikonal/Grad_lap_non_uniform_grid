import numpy as np
import matplotlib.pyplot as plt

def X(x):
    return 2*np.pi*x*h +0.2*np.sin(2*np.pi*x*h)

# function 
def f(x):
    return np.sin(x)
x_min=0
x_max=2*np.pi
M=50
h=(x_max-x_min)/(M-1)
x=np.zeros((M,1))
x=[x_min+i*h for i in range(M)]
## ghost nodes
x_left_1=-h
x_left_2= -2*h
x_left_3= -3*h
x_right_1= M*h
x_right_2= (M+1)*h
x_right_3= (M+2)*h

x_left_1_mid=(x_left_1+x[0])/2
x_left_2_mid= (x_left_1+ x_left_2)/2
x_left_3_mid= (x_left_2+x_left_3)/2

x_right_1_mid= (x_right_1+x[M-1])/2
x_right_2_mid= (x_right_1 +x_right_2)/2
x_right_3_mid= (x_right_2+x_right_3)/2

# ploting the x over interval
plt.figure()
plt.plot(x, np.zeros((M,1)), 'ro')
plt.xlabel('uniform grid')
plt.show()
print(x)

x_mid=np.zeros((M-1,1))
x_mid=[(x[i]+x[i+1])/2 for i in range(M-1)]

x_non= np.zeros((M,1))
x_non=[ X(x[i]) for i in range(M)]

plt.figure()
plt.plot(x_non, np.zeros((M,1)), 'ro')
plt.xlabel('non uniform grid')
plt.show()
print(x_non)



x_non_mid=np.zeros((M-1,1))
x_non_mid=[X(x_mid[i]) for i in range(M-1)]

u=np.zeros((M,1))

## initail condition    
u=[f(x_non[i]) for i in range(M)]

## boundary conditions
u[0]=0 
u[M-1]=0

## grad u
grad_u= np.zeros((M,1))
grad_u[0]=0
grad_u[M-1]= 0


## divergence of grad u is laplacian of u (second derivative)
div_grad_u=np.zeros((M-1,1))

## u(x) at midpoints of the cells in uniform grid
u_mid=np.zeros((M-1,1))
u_mid= [f((x[i]+x[i+1])/2) for i in range(M-1)]

# u(x) at midpoints of the cells in non-uniform grid
u_mid_non=np.zeros((M-1,1))
u_mid_non=[ f(X((x[i]+x[i+1])/ 2 )) for i in range(M-1)] ##very important

u_mid_non_left_1= f(X(x_left_1_mid))
u_mid_non_left_2= f(X(x_left_2_mid))
u_mid_non_left_3= f(X(x_left_3_mid))

u_mid_non_right_1= f(X(x_right_1_mid))
u_mid_non_right_2= f(X(x_right_2_mid))
u_mid_non_right_3= f(X(x_right_3_mid))

grad_u[0]= (-u_mid_non[1] +27*u_mid_non[0] - 27*u_mid_non_left_1 +u_mid_non_left_2) /(-x_non_mid[1] +27*x_non_mid[0] -27* X(x_left_1_mid) + X(x_left_2_mid))
grad_u[1]= (-u_mid_non[2] +27*u_mid_non[1] - 27*u_mid_non[0] +u_mid_non_left_1) /(-x_non_mid[2] +27*x_non_mid[1] -27* x_non_mid[0] + X(x_left_1_mid))

grad_u[M-1]= (-u_mid_non_right_2 +27*u_mid_non_right_1 - 27*u_mid_non[M-2] +u_mid_non[M-3]) /(-X(x_right_2_mid) +27* X(x_right_1_mid) -27* u_mid_non[M-2] + u_mid_non[M-3])
grad_u[M-2]= (-u_mid_non_right_1 +27*u_mid_non[M-2] - 27*u_mid_non[M-3] +u_mid_non[M-4]) /(-X(x_right_1_mid) +27* u_mid_non[M-2] -27* u_mid_non[M-3] + u_mid_non[M-4])


## calculate Grad of u
for i in range(2,M-2):
    grad_u[i]= (-u_mid_non[i+1] +27*u_mid_non[i] -27*u_mid_non[i-1] +u_mid_non[i-2]) / (-x_non_mid[i+1]+27*x_non_mid[i]-27*x_non_mid[i-1]+x_non_mid[i-2])

grad_u_left_1= (-u_mid_non[0] +27*u_mid_non_left_1 -27*u_mid_non_left_2 +u_mid_non_left_3) /(-x_non_mid[0]+27*X(x_left_1_mid)-27*X(x_left_2_mid)+X(x_left_3_mid))

grad_u_right_1= (-u_mid_non_right_2 +27*u_mid_non_right_1 -27*u_mid_non[M-2] +u_mid_non[M-3]) /(-X(x_right_2_mid)+27*X(x_right_1_mid)-27*u_mid_non[M-2]+u_mid_non[M-3])



## calculate divergence of grad u
for i in range(1,M-2):
    div_grad_u[i] = (-grad_u[i+2]+27*grad_u[i+1]-27*grad_u[i]+grad_u[i-1])/(-x_non[i+2]+27*x_non[i+1]-27*x_non[i]+x_non[i-1])

div_grad_u[0]= (-grad_u[2] +27* grad_u[1] -27*grad_u[0] +grad_u_left_1)/(-x_non[2]+27*x_non[1]-27*x_non[0]+ X(x_left_1_mid))
div_grad_u[M-2]= (-grad_u_right_1 +27* grad_u[M-1] -27*grad_u[M-2] +grad_u[M-3])/(-X(x_right_1_mid)+27*x_non[M-1]-27*x_non[M-2]+x_non[M-3])

## ploting the grad u over nonuniform grid
plt.figure()
plt.plot(x_non,grad_u)
plt.xlabel('x')
plt.ylabel('grad u')
plt.title('grad u over non uniform grid')
plt.show()

## ploting the laplacian u over nonuniform grid (at the cells of non uniform grid)
plt.figure()
plt.plot(x_non_mid,div_grad_u)
plt.xlabel('x')
plt.ylabel('laplacian u')
plt.title('laplacian of u over non uniform grid')
plt.show()













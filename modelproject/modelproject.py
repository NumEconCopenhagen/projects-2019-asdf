#Import necessary packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
%matplotlib inline

#Define the function
def cir_euler(r0, a, b, sigma, T, n, m):
    
    np.random.seed(844)
    h = T/n
    r = np.empty([n+1, m])
    I = np.empty([n+1, m])
    r[0,:] = r0
    I[0,:] = 1
    cols_r_euler = []
    cols_I_euler = []
    
    for j in range(m):
        
        #Create columns for dataframe
        cols_r_euler.append('r_path_' + str(j+1))
        cols_I_euler.append('I_path_' + str(j+1))
        
        for i in range(n):
            
            r[i+1,j]=r[i,j]+a*(b-r[i, j])*h+sigma*math.sqrt(max(r[i,j],0))*math.sqrt(h)*np.random.normal()
            I[i+1,j]=I[i,j]*(1-r[i,j]*h)
            
    return pd.DataFrame(r, columns = cols_r_euler), pd.DataFrame(I, columns = cols_I_euler)


#Set parameters
r0 = 0.025 #Interest rate at time t = 0
a = 0.2 
b = 0.025
sigma = 0.1
T = 1 #Maturity
n = 100 #Number of steps
m = 3 #Number of sample paths of r and I

#Run the cir_euler function and store results
r, I = cir_euler(r0, a, b, sigma, T, n, m);

#Merge the output from cir_euler into one dataframe
sample_paths = pd.concat([r,I], axis = 1)

#Show some paths
sample_paths.head(n = 11)

#Create a figure
fig = plt.figure(figsize = (12,5))

#Adjust and plot the interest rate figure
ax_left = fig.add_subplot(1,2,1)
ax_left.set_title('Interest Rate Sample Paths')
ax_left.set_ylabel('Interest Rate')
ax_left.set_xlabel('Number of Steps (n)')
ax_left.plot(r);

#Adjust and plot the discount bond price figure
ax_right = fig.add_subplot(1,2,2)
ax_right.set_title('Discount Bond Price')
ax_right.set_ylabel('Price')
ax_right.set_xlabel('Number of Steps (n)')
ax_right.plot(I);

#Calcualte the numerical estimate as the mean of the discount bond price at time T
num_estimate = np.mean(I.iloc[-1,:])

#Calculate the analytical solution found in (2)
eta = np.sqrt(a**2+2*sigma**2)
A = (2*eta*np.exp((a+eta)*T/2)/((eta+a)*(np.exp(eta*T)-1)+2*eta))**(2*a*b/(sigma**2))
B = 2*(np.exp(eta*T)-1)/((eta+a)*(np.exp(eta*T)-1)+2*eta)
P = A*np.exp(-r0*B)

#Calcualte the mean absolute error
MAE = abs(num_estimate-P)

print('Numerical Estimate: 'f'{num_estimate:.5f}')
print('Theoretical value: 'f'{P:.5f}')
print('Mean Absolute Error: 'f'{MAE:.5f}')

#Set parameters
r0 = 0.005
a = 0.4
b = 0.05
sigma = 0.1
T = 2
n_list = [2**x for x in range(9)] #Create a list of n's
m = 10000


def cir_euler_output():
    
    MAE_euler = np.empty([len(n_list), 1])
    
    eta = np.sqrt(a**2+2*sigma**2)
    A = (2*eta*np.exp((a+eta)*T/2)/((eta+a)*(np.exp(eta*T)-1)+2*eta))**(2*a*b/(sigma**2))
    B =  2*(np.exp(eta*T)-1)/((eta+a)*(np.exp(eta*T)-1)+2*eta)
    P = A*np.exp(-r0*B)
        
    for number in range(0,len(n_list)):
        
        n = n_list[number]
        res_r_euler, res_I_euler = cir_euler(r0, a, b, sigma, T, n, m)
        mean = np.mean(res_I_euler.iloc[-1,:])
        
        #Mean absolute error
        MAE_euler[number] = abs(mean - P)
        
    return pd.DataFrame(MAE_euler, columns = ['Mean Absolute Error'], index = n_list) 


#Create a dataframe of the output
output_euler = pd.DataFrame(cir_euler_output())
output_euler.index.name = 'n'

#Display the output
output_euler.head(n = len(n_list))


#Define the function
def cir_milstein(r0, a, b, sigma, T, n, m):
    
    np.random.seed(844)
    h = T/n
    r = np.empty([n+1, m])
    I = np.empty([n+1, m])
    r[0,:] = r0
    I[0,:] = 1
    cols_r_mil = []
    cols_I_mil = []
    
    for j in range(m):
        
        #Create columns for dataframe
        cols_r_mil.append('r_path_' + str(j+1))
        cols_I_mil.append('I_path_' + str(j+1))
        
        for i in range(n):
            
            r[i+1,j]=r[i,j]+(a*(b-r[i,j])-(sigma**2)*0.25)*h+sigma*np.sqrt(max(r[i,j],0))*np.sqrt(h)*np.random.normal()\
            +sigma*0.25*((1/np.sqrt(abs(r[i,j])))*(a*(b-r[i,j])-(sigma**2)*0.25)-2*np.sqrt(max(r[i,j],0))*a)*np.random.normal()*h*np.sqrt(h)\
            -0.5*(a**2)*(b-r[i,j])*h**2+(sigma**2)*0.25*((np.random.normal())**2)*h
            
            I[i+1,j]=I[i,j]*(1-r[i,j]*h+0.5*(r[i,j]**2-a*(b-r[i,j]))*h**2-0.5*sigma*np.sqrt(max(r[i,j],0))*np.random.normal()*h*np.sqrt(h))
            
    return pd.DataFrame(r, columns = cols_r_mil), pd.DataFrame(I, columns = cols_I_mil)

def cir_milstein_output():
    
    MAE_mil = np.empty([len(n_list), 1])
    
    eta = np.sqrt(a**2+2*sigma**2)
    A = (2*eta*np.exp((a+eta)*T/2)/((eta+a)*(np.exp(eta*T)-1)+2*eta))**(2*a*b/(sigma**2))
    B =  2*(np.exp(eta*T)-1)/((eta+a)*(np.exp(eta*T)-1)+2*eta)
    P = A*np.exp(-r0*B)
        
    for number in range(0,len(n_list)):
        n = n_list[number]
        res_r_mil, res_I_mil = cir_milstein(r0, a, b, sigma, T, n, m)
        mean = np.mean(res_I_mil.iloc[-1,:])
        
        #Mean absolute error
        MAE_mil[number] = abs(mean - P)
        
    return pd.DataFrame(MAE_mil, columns = ['Mean Absolute Error'], index = n_list) 

output_mil = pd.DataFrame(cir_milstein_output())
output_mil.index.name = 'n'
output_mil.head(n = len(n_list))

#Create a figure
fig = plt.figure(figsize = (6,5))

ax = fig.add_subplot(1,1,1)
ax.set_title('Euler and Milstein Convergence')
ax.set_ylabel('Mean Absolute Error')
ax.set_xlabel('Number of Steps (n)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(which = 'both', linewidth = 0.1, color = 'k')
ax.plot(output_euler, label = 'Euler', color = '#193264', marker = 'o')
ax.plot(output_mil, label = 'Milstein', color = '#800000', marker = 'd')
ax.legend();
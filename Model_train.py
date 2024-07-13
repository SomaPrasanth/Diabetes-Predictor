import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def predict(x,w,b):
    result=np.dot(x,w)+b
    return sigmoid(result)

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def compute_cost(x,y,w,b,iterations):
    m=x.shape[0]
    cost=0
    for i in range(m):
        z=np.dot(x[i],w)+b
        f=sigmoid(z)
        cost+=((-y[i]*np.log(f))-((1-y[i])*np.log(1-f)))
    cost/=(m)
   
    return cost

def compute_dervatives(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        z=np.dot(x[i],w)+b
        f=sigmoid(z)
        err=((f-y[i]))
        for j in range(n):
            dj_dw[j]=dj_dw[j]+err*x[i,j]
        dj_db=dj_db+err
    return dj_dw,dj_db        


def compute_gradient(x,y,w,b,alpha,iterations):
    m=x.shape[0]
    cost_his=[]
    while iterations:
        dj_dw,dj_db=compute_dervatives(x,y,w,b)
        w=w-(alpha*dj_dw)
        b=b-(alpha*dj_db)
        cost=compute_cost(x,y,w,b,iterations)
        iterations-=1
        cost_his.append(cost)
        print(f'Cost at iteration{iterations-10001} is {cost}')
    return w,b,cost_his    

        


X_list=[]
df=pd.read_csv('diabetes[1].csv')

a0=list(df.iloc[:,0])
a1=list(df.iloc[:,1])
a2=list(df.iloc[:,2])
a3=list(df.iloc[:,3])
a4=list(df.iloc[:,4])
a5=list(df.iloc[:,5])
a6=list(df.iloc[:,6])
a7=list(df.iloc[:,7])
for i in range(len(a0)):
    temp=[a0[i],a1[i],a2[i],a3[i],a4[i],a5[i],a6[i],a7[i]]
    X_list.append(temp)
X_data=np.array(X_list)
Y_data=np.array(df.iloc[:,8])
W=np.zeros(8)
B=0.0
alpha=0.0000003
iterations=10000
W,B,cost_his=compute_gradient(X_data,Y_data,W,B,alpha,iterations)
print(f'\n\nW={W}\n\nB={B}\n')
iterations=[i+1 for i in range(10000)]
plt.plot(iterations,cost_his)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
print("Enter the following details:")
pregancies=int(input("Enter no. of pregancies:"))
glucose=int(input("Enter glucose level:"))
bp=int(input("Enter blood pressure level:"))
skinthick=int(input("Enter skin thickness:"))
insulin=int(input("Enter insulin level:"))
bmi=float(input("Enter body mass index:"))
dpf=float(input("The 'DiabetesPedigreeFunction' is a function that scores the probability of diabetes based on family history.Enter DPF score:"))
age=int(input("Enter your age:"))

input_array=np.array([pregancies,glucose,bp,skinthick,insulin,bmi,dpf,age])


result=predict(input_array,W,B)
if(result>=0.3):
    print("Your are having diabetes")
elif(result<0.3):
    print("Your not having diabetes")   

pregancies=int(input("Enter no. of pregancies:"))
glucose=int(input("Enter glucose level:"))
bp=int(input("Enter blood pressure level:"))
skinthick=int(input("Enter skin thickness:"))
insulin=int(input("Enter insulin level:"))
bmi=float(input("Enter body mass index:"))
dpf=float(input("The 'DiabetesPedigreeFunction' is a function that scores the probability of diabetes based on family history.Enter DPF score:"))
age=int(input("Enter your age:"))

input_array=np.array([pregancies,glucose,bp,skinthick,insulin,bmi,dpf,age])


result=predict(input_array,W,B)
if(result>=0.3):
    print("Your are having diabetes")
elif(result<0.3):
    print("Your not having diabetes")    
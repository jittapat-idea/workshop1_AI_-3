import os
import numpy as np
import math


#os.system('cls')

# activation function
def sigmoid(x):
    return (math.e**x)/(1+math.e**x)

#
x = np.array([[1,1,1,0,1,1,0],
             [1,0,1,1,0,0,1],
             [1,0,0,0,1,1,0],
             [1,1,0,1,0,0,1],
             [1,1,1,0,1,0,1],
             [1,0,0,1,0,1,0],
             [1,0,1,0,1,0,1],
             [1,1,0,1,0,0,0],
             [1,0,1,0,1,1,1],
             [1,1,0,0,1,1,0]] , dtype=np.float64)

#จำนวนรอบ
iteration = 10000
#learning rate
alpha = 0.05

#ค่า นํ้าหนัก
w =np.array([1,1,1,1,1,1,1] , dtype=np.float64 )

#ค่า bios
b = 1 

#ค่า output (ผู้ใช้ซื้อร้องเท้าหรือไม่?)
y =np.array([0,0,1,0,1,0,1,0,0,1],dtype=np.float64)


def ขาไป(w,x):
    ค่าy = sigmoid(np.matmul(x,w)+b)

    return ค่าy

def ขากลับ(yมีหมวก,x,y):
    n = len(x)
    del_w = (2/n) * np.matmul((yมีหมวก - y) * yมีหมวก*(1-yมีหมวก),x)

    return del_w


def ปรับค่านํ้าหนัก(alpha,w,del_w):
    wใหม่ = w - (alpha * del_w)

    return wใหม่

for _ in range(iteration):
    yมีหมวก = ขาไป(w,x)
    del_w = ขากลับ(yมีหมวก,x,y)
    w = ปรับค่านํ้าหนัก(alpha,w,del_w)

#feature ไหนมีค่า weight มากเเสดงว่า feature นั้นมีผลต่อการตัดสินใจเลือกซื้อร้องเท้า
for i in range(len(w)):
    print(f"Feature{i} = {w[i]}")




# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 16:50
# @Author  : ljf
import numpy as np
import math

# f = x^(y!)%z
def factorial(n):
    # if n==1:
    #     return 1
    # else:
    #     return n*factorial(n-1)
    total = 1
    for i in range(1,n+1):
        total*=i
    return total
def main(x,y,z):
    # return x**(factorial(y))%z
    return x**math.factorial(y)%z
    # 取log变换
    # fac = factorial(y)
    # print(fac)
    # log = fac*np.log2(x)
    # return (2**log)%z
    # return (2**(factorial(y)*np.log2(x)))%z

if __name__ == "__main__":
    print(main(2,1,2))
    print(main(3, 2, 2))
    print(main(1, 100000, 1))
    print(main(1, 100000, 2))
    print(main(99036, 92879, 77028))
    print(main(57582, 1465, 57582))
    print(main(14916, 63624, 37968))
    print(main(48778, 6070, 89146))
# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 10:45
# @Author  : ljf
import numpy as np

def RightPoint(num):

    # 1.判断开方取整之后奇偶性
    sq = int(np.sqrt(num))
    # 奇数返回
    if sq % 2!=0:
        return sq,sq
    # 偶数-1 返回
    else:
        return sq-1,sq-1
def CentrePoint(point):
    return (point+1)//2+1,(point+1)//2+1

def DecideSide(right_w,right_h,remain):
    # 返回在那条边以及当前边所剩的距离
    side = right_h+1
    if remain//side==0:
        n_side = "right_side"
        side_remain = remain
    elif remain//side==1:
        n_side = "top_side"
        side_remain= remain -side
    elif remain // side == 2:
        n_side = "left_side"
        side_remain =remain -side*2
    elif remain // side == 3:
        n_side = "down_side"
        side_remain = remain - side*3

    return  n_side, side_remain
    pass
def NumPoint(right_x,right_y,remain):
    n_side,side_remain = DecideSide(right_x,right_y,remain)

    if remain==0:
        return right_x,right_y
    # 判断在那条边并计算坐标
    if n_side=="right_side":
        right_x += 1
        right_y +=2
        right_x -= (side_remain-1)
    elif n_side == "top_side":
        right_x = 1
        right_y +=2
        right_y -= side_remain
    elif n_side == "left_side":
        right_x = 1
        right_y = 1
        right_y +=side_remain
    elif n_side == "down_side":
        right_x += 2
        right_y = 1
        right_y +=side_remain
    # print("输入数字所在边：{}，所剩距离：{}".format(n_side,side_remain))
    return right_x,right_y

def main(num):
    # 1求出输入数字的最近一层右下角坐标
    right_x ,right_y = RightPoint(num)
    # print("最近一层右下角坐标：[{},{}]".format(right_x,right_y))
    # 2.计算中心坐标（1,1）
    centre_x ,centre_y = CentrePoint(right_x)
    # print("中心点坐标：[{},{}]".format(centre_x,centre_y))
    # 3.计算输入数字的坐标
    remain = num-right_x*right_y
    # print("输入数字从最近一层右下角走的距离：{}".format(remain))
    num_x,num_y = NumPoint(right_x,right_y,remain)
    # print("输入数字的坐标：[{},{}]".format(num_x,num_y))

    distance = np.abs(num_x-centre_x)+np.abs(num_y-centre_y)
    print("数字{}距离中心点[1,1]的曼哈顿距离：{}".format(num,distance))

if __name__ == "__main__":
    main(12)
    main(100000)
    main(2345678)

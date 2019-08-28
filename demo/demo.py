# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 10:45
# @Author  : ljf
import time
import math

class Solution:
    def majorityElement(self, nums):
        count = 1
        num = nums[0]
        for i in range(1,len(nums)):
            if num == nums[i]:
                count += 1
            else:
                if count == 0:
                    num = nums[i]
                else:
                    count -= 1
        return num

if __name__ == "__main__":
    solution = Solution()
    solution.majorityElement([2,2,1,1,1,2,2])



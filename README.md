# DSA
# 1. Two Sum
Problem: Given an array of integers, return indices of the two numbers such that they add up to a specific target.
Approach: Use a hash map to store the difference between the target and the current number, checking if it exists in the map.
Time Complexity: O(n).


def twoSum(nums, target):

    hashmap = {}
    
    for i, num in enumerate(nums):
    
        diff = target - num
        if diff in hashmap:
            return [hashmap[diff], i]
        hashmap[num] = i

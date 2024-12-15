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


# 2. Find the Missing Number in an Array
Problem: An array of size n contains numbers from 0 to n-1. Find the missing number.
Approach: Use XOR or calculate the sum difference.
Time Complexity: O(n).

    def missingNumber(nums):
        n = len(nums)
        return n * (n + 1) // 2 - sum(nums)

# 3. Detect a Cycle in a Linked List
Problem: Check if a linked list contains a cycle.
Approach: Use Floyd’s Cycle Detection (two-pointer method).
Time Complexity: O(n).

    def hasCycle(head):
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

# 4. Merge Two Sorted Arrays
Problem: Merge two sorted arrays without using extra space.
Approach: Start merging from the back of both arrays.
Time Complexity: O(m + n).

    def merge(nums1, m, nums2, n):
        while m > 0 and n > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        nums1[:n] = nums2[:n]

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


# 5. Reverse a Linked List
Problem: Reverse a singly linked list.
Approach: Use iterative or recursive methods.
Time Complexity: O(n).

    def reverseList(head):
        prev, curr = None, head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        return prev

# 6. Valid Parentheses
Problem: Check if a string of parentheses is valid.
Approach: Use a stack to push and pop matching brackets.
Time Complexity: O(n).

    def isValid(s):
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        for char in s:
            if char in mapping:
                top = stack.pop() if stack else '#'
                if mapping[char] != top:
                    return False
            else:
                stack.append(char)
        return not stack

# 7. Implement a Queue using Stacks
Problem: Implement a queue using two stacks.
Approach: Use two stacks: one for enqueue and another for dequeue.
Time Complexity: O(1) amortized.

    class MyQueue:
        def __init__(self):
            self.stack1 = []
            self.stack2 = []
    
        def push(self, x):
            self.stack1.append(x)
    
        def pop(self):
            if not self.stack2:
                while self.stack1:
                    self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
    
        def peek(self):
            if not self.stack2:
                while self.stack1:
                    self.stack2.append(self.stack1.pop())
            return self.stack2[-1]
    
        def empty(self):
            return not self.stack1 and not self.stack2

# 8. Binary Tree Inorder Traversal
Problem: Perform an inorder traversal of a binary tree.
Approach: Use recursion or a stack for iterative traversal.
Time Complexity: O(n).

    def inorderTraversal(root):
        result, stack = [], []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            result.append(root.val)
            root = root.right
        return result

# 9.Lowest Common Ancestor of a Binary Tree
Problem: Find the LCA of two nodes in a binary tree.
Approach: Use recursion to find the split point.
Time Complexity: O(n).

    def lowestCommonAncestor(root, p, q):
        if not root or root == p or root == q:
            return root
        left = lowestCommonAncestor(root.left, p, q)
        right = lowestCommonAncestor(root.right, p, q)
        return root if left and right else left or right

# 10. Search in Rotated Sorted Array
Problem: Search a target in a rotated sorted array.
Approach: Use modified binary search.
Time Complexity: O(log n).

    def search(nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

# 11. Maximum Subarray (Kadane’s Algorithm)
Problem: Find the contiguous subarray with the largest sum.
Approach: Use Kadane’s Algorithm to calculate the maximum sum at each index.
Time Complexity: O(n).

    def maxSubArray(nums):
        max_sum = curr_sum = nums[0]
        for num in nums[1:]:
            curr_sum = max(num, curr_sum + num)
            max_sum = max(max_sum, curr_sum)
        return max_sum

# 12. Rotate Matrix (90 Degrees Clockwise)
Problem: Rotate an n x n 2D matrix in place.
Approach: First transpose the matrix, then reverse each row.
Time Complexity: O(n²).

    def rotate(matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            row.reverse()

# 13. Trapping Rain Water
Problem: Calculate the amount of rainwater that can be trapped between bars.
Approach: Use two-pointer approach to calculate the trapped water at each bar.
Time Complexity: O(n).

    def trap(height):
        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
        water = 0
        while left <= right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        return water
# 14. Set Matrix Zeroes
Problem: If an element in a matrix is 0, set its entire row and column to 0.
Approach: Use the first row and column as markers for zero rows/columns.
Time Complexity: O(m * n).
    
    def setZeroes(matrix):
        rows, cols = len(matrix), len(matrix[0])
        row_zero = False
    
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == 0:
                    matrix[0][c] = 0
                    if r == 0:
                        row_zero = True
                    else:
                        matrix[r][0] = 0
    
        for r in range(1, rows):
            for c in range(1, cols):
                if matrix[0][c] == 0 or matrix[r][0] == 0:
                    matrix[r][c] = 0
    
        if matrix[0][0] == 0:
            for r in range(rows):
                matrix[r][0] = 0
    
        if row_zero:
            for c in range(cols):
                matrix[0][c] = 0






    



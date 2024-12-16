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
# 15. Median of Two Sorted Arrays
Problem: Find the median of two sorted arrays.
Approach: Use binary search to partition the arrays.
Time Complexity: O(log(min(m, n))).

    def findMedianSortedArrays(nums1, nums2):
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        x, y = len(nums1), len(nums2)
        low, high = 0, x
    
        while low <= high:
            partitionX = (low + high) // 2
            partitionY = (x + y + 1) // 2 - partitionX
    
            maxX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
            minX = float('inf') if partitionX == x else nums1[partitionX]
            maxY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
            minY = float('inf') if partitionY == y else nums2[partitionY]
    
            if maxX <= minY and maxY <= minX:
                if (x + y) % 2 == 0:
                    return (max(maxX, maxY) + min(minX, minY)) / 2
                else:
                    return max(maxX, maxY)
            elif maxX > minY:
                high = partitionX - 1
            else:
                low = partitionX + 1

# 16. Longest Increasing Subsequence (LIS)
Problem: Find the length of the longest increasing subsequence.
Approach: Use DP with binary search for optimization.
Time Complexity: O(n log n).

    def lengthOfLIS(nums):
        sub = []
        for num in nums:
            i = bisect.bisect_left(sub, num)
            if i == len(sub):
                sub.append(num)
            else:
                sub[i] = num
        return len(sub)

# 17. Knapsack Problem
Problem: Find the maximum value you can obtain with a weight limit.
Approach: Use a 2D DP table to store subproblem solutions.
Time Complexity: O(n * W), where n is the number of items and W is the weight limit.

    def knapsack(weights, values, W):
        n = len(weights)
        dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    
        for i in range(1, n + 1):
            for w in range(W + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
                else:
                    dp[i][w] = dp[i-1][w]
        return dp[n][W]

# 18. Breadth-First Search (BFS)
Problem: Perform BFS traversal of a graph.
Approach: Use a queue to process nodes level by level.
Time Complexity: O(V + E), where V is vertices and E is edges.

    from collections import deque
    def bfs(graph, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
    
        while queue:
            node = queue.popleft()
            print(node, end=" ")
    
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)


# 19 . Dijkstra’s Algorithm
Problem: Find the shortest path from a source to all other vertices in a weighted graph.
Approach: Use a priority queue (min-heap).
Time Complexity: O((V + E) * log V).

    import heapq
    def dijkstra(graph, start):
        pq = [(0, start)]
        distances = {vertex: float('infinity') for vertex in graph}
        distances[start] = 0
    
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
    
            if current_distance > distances[current_vertex]:
                continue
    
            for neighbor, weight in graph[current_vertex].items():
                distance = current_distance + weight
    
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
    
        return distances

# 20. Topological Sorting
Problem: Perform topological sort of a Directed Acyclic Graph (DAG).
Approach: Use DFS with a stack or Kahn’s Algorithm (BFS-based).
Time Complexity: O(V + E).

    def topologicalSort(graph):
        visited = set()
        stack = []
    
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)
    
        for vertex in graph:
            dfs(vertex)
    
        return stack[::-1]

# 21. Bubble Sort
Repeatedly swap adjacent elements if they are in the wrong order.
Time Complexity: O(n²)
Space Complexity: O(1)

    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j +1] = arr[j+1], arr[j]
        return arr
        
# 22. Selection Sort
Select the smallest element and place it at the beginning.
Time Complexity: O(n²)
Space Complexity: O(1)

    def selection_sort(arr):
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    # Example
    print(selection_sort([64, 25, 12, 22, 11]))

# 23. Insertion Sort
Build the sorted list one item at a time.
Time Complexity: O(n²)
Space Complexity: O(1)

    def insertion_sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    # Example
    print(insertion_sort([12, 11, 13, 5, 6]))

# 24. Merge Sort
Divide and conquer: divide the array into halves, sort each half, and merge them.
Time Complexity: O(n log n)
Space Complexity: O(n)

        def merge_sort(arr):
            if len(arr) > 1:
                mid = len(arr) // 2
                left = arr[:mid]
                right = arr[mid:]
        
                merge_sort(left)
                merge_sort(right)
        
                i = j = k = 0
        
                while i < len(left) and j < len(right):
                    if left[i] < right[j]:
                        arr[k] = left[i]
                        i += 1
                    else:
                        arr[k] = right[j]
                        j += 1
                    k += 1
        
                while i < len(left):
                    arr[k] = left[i]
                    i += 1
                    k += 1
        
                while j < len(right):
                    arr[k] = right[j]
                    j += 1
                    k += 1
            return arr

    # Example
    print(merge_sort([38, 27, 43, 3, 9, 82, 10]))

# 25. Quick Sort
Pick a pivot, partition the array, and recursively sort the partitions.
Time Complexity: O(n log n) (average case), O(n²) (worst case)
Space Complexity: O(log n) (for recursion stack)

    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

    # Example
    print(quick_sort([10, 7, 8, 9, 1, 5]))

# 26. Heap Sort
Build a max heap and repeatedly extract the maximum element.
Time Complexity: O(n log n)
Space Complexity: O(1)

     def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
    
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    def heap_sort(arr):
        n = len(arr)
    
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
    
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            heapify(arr, i, 0)
    
        return arr

    # Example
    print(heap_sort([12, 11, 13, 5, 6, 7]))

# 27. Counting Sort
Count the occurrence of each element and build the sorted array.
Time Complexity: O(n + k), where k is the range of the input.
Space Complexity: O(k)


    def counting_sort(arr):
        max_val = max(arr)
        count = [0] * (max_val + 1)
    
        for num in arr:
            count[num] += 1
    
        sorted_arr = []
        for i, freq in enumerate(count):
            sorted_arr.extend([i] * freq)
    
        return sorted_arr

    # Example
    print(counting_sort([4, 2, 2, 8, 3, 3, 1]))

# 28. Counting Sort
Count the occurrence of each element and build the sorted array.
Time Complexity: O(n + k), where k is the range of the input.
Space Complexity: O(k)

    def counting_sort(arr):
        max_val = max(arr)
        count = [0] * (max_val + 1)
    
        for num in arr:
            count[num] += 1
    
        sorted_arr = []
        for i, freq in enumerate(count):
            sorted_arr.extend([i] * freq)
    
        return sorted_arr

    # Example
    print(counting_sort([4, 2, 2, 8, 3, 3, 1]))
 
# 29.  Radix Sort
Sort based on individual digits, starting from the least significant digit.
Time Complexity: O(nk), where k is the number of digits.
Space Complexity: O(n)

    def counting_sort_radix(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
    
        for i in arr:
            index = (i // exp) % 10
            count[index] += 1
    
        for i in range(1, 10):
            count[i] += count[i - 1]
    
        for i in range(n - 1, -1, -1):
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
    
        for i in range(n):
            arr[i] = output[i]

    def radix_sort(arr):
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            counting_sort_radix(arr, exp)
            exp *= 10
        return arr
    
    # Example
    print(radix_sort([170, 45, 75, 90, 802, 24, 2, 66]))

# 30. Bucket Sort
Distribute elements into buckets, sort each bucket, and merge.
Time Complexity: O(n + k)
Space Complexity: O(n)
    
    def bucket_sort(arr):
        bucket = [[] for _ in range(len(arr))]
    
        for num in arr:
            index = int(len(arr) * num)
            bucket[index].append(num)
    
        for b in bucket:
            b.sort()
    
        sorted_arr = []
        for b in bucket:
            sorted_arr.extend(b)
    
        return sorted_arr
    
    # Example
    print(bucket_sort([0.25, 0.36, 0.58, 0.41, 0.29, 0.22, 0.40]))














    



# leetcode -2

# 4. 寻找两个正序数组的中位数

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 方法1: 朴素解，归并排序之后返回中间的值
        mid = (len(nums1)+len(nums2))
        lst = self.merge(nums1,nums2)
        if mid%2 == 1:
            return lst[mid//2]
        elif mid%2 == 0:
            return (lst[mid//2]+lst[mid//2-1])/2
    
    def merge(self,lst1,lst2):
        p1 = 0
        p2 = 0
        ans = []
        if len(lst1) == 0:
            return lst2
        elif len(lst2) == 0:
            return lst1
        while p1 < len(lst1) and p2 < len(lst2):
            if lst1[p1] < lst2[p2]:
                ans.append(lst1[p1])
                p1 += 1
            else:
                ans.append(lst2[p2])
                p2 += 1
        while p1 < len(lst1):
            ans.append(lst1[p1])
            p1 += 1
        while p2 < len(lst2):
            ans.append(lst2[p2])
            p2 += 1
        return ans
```

```

```



# 5. 最长回文子串

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # 当s长度小于等于1时，直接返回
        if len(s) <= 1:
            return s
        # 动态规划dp
        # dp[i][j]的含义是，s[i:j+1]是否是回文串
        # 先申请足够大小的二维数组，先全部用false填充
        dp = [[False for i in range(len(s))]for k in range(len(s))]
        # 显然j+1>= i才可以，并且 j+1==i时候，只有一个字符，那么它是回文串,那么主对角线可以全部填充True
        for i in range(len(s)):
            dp[i][i] = True
        # 要判断s[i:j+1]是否为回文，需要判断的是s[i+1:j]【对应dp[i+1][j-1]】是否为回文且s[i]==s[j]
        # 画方格图可以判断，需要左下角临近的dp来判断
        # 状态转移方程为 dp[i][j] = (dp[i+1][j-1] and s[i]==s[j])
        # 发现如果只有主对角线尚且无法完成状态转移，那么主对角线右边的平行线需要填充
        # 它的填充为
        for i in range(len(s)-1):
            dp[i][i+1] = (s[i]==s[i+1])
        # 现在可以状态转移了，因为在空位可以又了左下方的数
        # 填充方向为从左到右的纵列
        for j in range(2,len(s)):
            for i in range(0,j-1):
                dp[i][j] = (dp[i+1][j-1] and s[i]==s[j])
        # 此时dp数组建立好了
        # 在每一行里用双指针找T
        # print(dp)
        temp = []
        for every_row in dp:
            left = 0
            right = len(every_row)-1
            while every_row[left] != True:
                left += 1
            while every_row[right] != True:
                right -= 1
            temp.append(right-left)
        # 找到temp的最大差值
        max_gap = max(temp)
        index = temp.index(max_gap)
        return s[index:index+max_gap+1]

```

# 11. 盛最多水的容器

给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器。

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # 双指针，容器缩小，过程中存取最大值
        # 指针移动逻辑为，每次移动低的那一根
        # 理由是，由于木桶原理，储水量被短的那根限定了，缩小只能更窄，那么便不在选这根短的
        # 有一种贪心的感觉
        left = 0
        right = len(height)-1
        max_store = -1 # 初始化一个存水量 任意非正值都可以
        while left < right : # 
            temp_store = (right-left)*min(height[left],height[right])
            max_store = max(max_store,temp_store) # 存取过程中的最大量
            if height[left] < height[right]: # 移动短的
                left += 1
            elif height[left] >= height[right]: #移动短的
                right -= 1
        return max_store
```

# 16. 最接近的三数之和

给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort() # 预先排序
        ans = 0
        gap = 0xffffffff # 预先设定最大差值
        for i in range(len(nums)):
            aim = target - nums[i]
            left = i + 1
            right = len(nums) - 1
            while left < right:
                sum_num = nums[left] + nums[right] + nums[i]
                if abs(sum_num-target) < gap:
                    gap = abs(sum_num - target)
                    ans = sum_num
                if nums[left] + nums[right] == aim: # 如果有相等的，直接返回答案
                    return sum_num
                elif nums[left] + nums[right] < aim:
                    left += 1
                elif nums[left] + nums[right] > aim:
                    right -= 1
        return ans
```



# 18. 四数之和

给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]] ：

0 <= a, b, c, d < n
a、b、c 和 d 互不相同
nums[a] + nums[b] + nums[c] + nums[d] == target
你可以按 任意顺序 返回答案 。

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # 多皮版两数之和,顺序任意
        # 使用集合去重复降低代码复杂度，用空间换时间
        nums.sort() # 预排序
        ans_list = []
        for a in range(len(nums)):
            t1 = target - nums[a]
            for b in range(a+1,len(nums)):
                t2 = t1 - nums[b]
                c = b + 1
                d = len(nums) - 1
                while c < d:
                    if nums[c] + nums[d] == t2: # 注意收集答案之后两指针同时移动
                        ans_list.append((nums[a],nums[b],nums[c],nums[d]))
                        c += 1
                        d -= 1
                    elif nums[c] + nums[d] < t2: # 总和偏小，左指针右移动
                        c += 1
                    elif nums[c] + nums[d] > t2: # 总和偏大，右指针左移
                        d -= 1
        ans_list = set(ans_list) # 去重复
        final = []
        for tp in ans_list:
            final.append(tp)
        return final
```



# 24. 两两交换链表中的节点

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

**你不能只是单纯的改变节点内部的值**，而是需要实际的进行节点交换。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # 思路：分成奇链和偶链，然后合并两链
        # 这方法比较复杂，需要扫两轮
        if head == None:
            return head
        if head.next == None:
            return head
        odd = head
        odd_head = odd
        even = head.next
        even_head = even
        while odd.next != None and even.next != None:
            odd.next = odd.next.next
            odd = odd.next
            even.next = even.next.next
            even = even.next
        # 然后循环弹出头节点,为了避免出环，需要奇数链的尾巴设置为None
        odd.next = None
        cur2 = odd_head # cur2是奇数链
        cur1 = even_head # cur1是偶数链
        # print(cur1,cur2)
        while cur1.next != None: # 处理到倒数第二个节点
            temp1 = cur1.next
            temp2 = cur2.next
            cur1.next = cur2
            cur2.next= temp1
            cur1 = temp1
            cur2 = temp2
        # 奇数可能多一个点，此时cur2有值
        temp1 = cur1.next
        temp2 = cur2.next
        if cur2.next != None:
            cur1.next = cur2
        else:
            cur1.next = cur2
            cur2.next = temp1

        return (even_head)

```

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # 递归：先处理再递归
        # 递归出口为：节点数不足2个点时候
        if head == None or head.next == None:
            return head
        # 否则，取第二个节点为头，对第二个节点之后的节点调用递归
        # 调整第二个节点和第一个节点的关系
        new_head = head.next
        head.next = self.swapPairs(new_head.next) # 原第一个节点的next
        new_head.next = head # 原第二个节点的next指向原第一个节点
        return new_head
```

# 28. 实现 strStr()

实现 strStr() 函数。

给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

 

说明：

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # BF暴力解法
        for i in range(len(haystack) - len(needle) + 1):
            if needle == haystack[i:i+len(needle)]:
                return i
        return -1

```

```python
# 这个方法长串会超时，之后用KMP改进
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # BF暴力解法
        if needle == '': # 空串返回0
            return 0
        if len(needle) > len(haystack): # 子串更长，返回-1
            return -1 
        p1 = 0
        while p1 < len(haystack):
            if haystack[p1] != needle[0]:
                p1 += 1
            if len(haystack)-p1 < len(needle): # 剪枝必备，如果长度都不足了，那么一定找不到
                return -1
            elif haystack[p1] == needle[0]:
                temp_p1 = p1
                p2 = 0
                while p1 < len(haystack) and p2 < len(needle)and haystack[p1] == needle[p2]:
                    p1 += 1
                    p2 += 1
                if p2 == len(needle):
                    return temp_p1
                p1 = temp_p1 + 1
        return -1



```

# 33. 搜索旋转排序数组 + 改编

整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 由于数值无重复
        # 画图辅助，用中线将左右两边分为有序区和无序区域
        # 根据nums[mid],nums[left],target的关系，将其分为四种情况
        # 先根据nums[mid]和nums[left]分成两组。
        # nums[left] < nums[mid]时， 
        #1.在左有序组【进一步：left<target<mid]，2.在右无序组[情况使用else就行]

        # nums[left] > nums[mid]时， 
        #3. 在左无序组【使用else]，4.在右有序组[right>target>mid]
        # 二分查找
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left+right)//2
            # print(mid,'left=',left,'right=',right)
            if target == nums[mid]: # 返回的是索引
                return mid
            if nums[0] <= nums[mid]: # 1+2
                # print('1+2')
                if nums[0] <= target < nums[mid]: # 1
                    right = mid-1
                else: # 2
                    left = mid+1
            elif nums[0] > nums[mid]: # 3+4
                # print('3+4')
                if nums[mid]< target <= nums[len(nums)-1]: # 4
                    left = mid+1
                else: # 3
                    right = mid-1
        return -1
```

面试题 10.03. 搜索旋转数组

搜索旋转数组。给定一个排序后的数组，包含n个整数，但这个数组已被旋转过很多次了，次数不详。请编写代码找出数组中的某个元素，假设数组元素原先是按升序排列的。若有多个相同元素，返回索引值最小的一个。

示例1:

 输入: arr = [15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14], target = 5
 输出: 8（元素5在该数组中的索引）
示例2:

 输入：arr = [15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14], target = 11
 输出：-1 （没有找到）
提示:

arr 长度范围在[1, 1000000]之间

```python
class Solution:
    def search(self, arr: List[int], target: int) -> int:
        # 画图辅助
        # 分为四种基本情况【图上先不考虑有等值】
        # 根据arr[left],arr[mid],arr[right]的大小关系
        # arr[left] <= arr[mid] 为第一类
        # 1. arr[left] <= target < arr[mid]  左有序区
        # 2. else 右无序区
        # arr[left] > arr[mid] # 为第二类
        # 3. else 左无序区
        # 4. arr[mid] < target <= arr[right] 右有序区
        # 考虑搜索过程中有可能等值情况在left和right上，那么先right收缩到和left不相等为止
        left = 0
        right = len(arr)-1
        while left <= right:
            print(arr[left:right+1])
            if arr[left] == target: 
                # 如果left是结果值，直接返回最左边
                return left
            while 0<=right and arr[left] == arr[right]: # 当他们不是target的时候
            # 由于是返回最左边的那个,所以right收缩，
                right -= 1
            mid = (left+right)//2
            if arr[mid] == target: # 注意这一行的修改,由于找左最小值
                right = mid
            elif arr[left] <= arr[mid]: # 1+2
                if arr[left] <= target < arr[mid]: # 1
                    right = mid - 1 # 收缩右边界
                else: # 2
                    left = mid + 1 # 
            elif arr[left] > arr[mid]: # 3+4
                if arr[mid] <= target <= arr[right]: # 4
                    left = mid + 1
                else: # 3
                    right = mid - 1            
        return -1
```

# 47. 全排列 II

给定一个可包含重复数字的序列 `nums` ，**按任意顺序** 返回所有不重复的全排列。

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # 方法一：先全部加入，再过筛去除重复的
        ans = []
        path = []
        n = len(nums)
        def backtracking(lst): # 回溯
            if len(path) == n: # 到达长度之后收集结果
                ans.append(path[:]) # 传值而不是传引用
            for i in lst:
                copylst = lst.copy()
                copylst.remove(i) # 下一次选择时不能选择这次选择的结果
                path.append(i) # 路径选择
                backtracking(copylst) # 在排除了这一次选择的值的新列表中进行回溯
                path.pop() # 撤销路径选择
                
        backtracking(nums)
        # 去重过筛，注意只有不可变元素可以哈希，所以利用元组
        memo = set()
        final = []
        for i in ans:
            if tuple(i) not in memo:
                memo.add(tuple(i))
                final.append(i)
        return final
```



# 56. 合并区间

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 分情况讨论，画图辅助
        # 排序后分成[pre[0],pre[1]],[cur[0],cur[1]]
        # 1. pre[1] < cur[0] 两数组不重叠，直接把cur加入ans
        # 2+3. pre[1] >= cur[0] 两数组重叠，进一步讨论，
        # 2. pre[1] >= cur[0] and pre[1] < cur[1]: # 部分重叠，
        #  ans 弹出前一个，加入[pre[0],cur[1]]
        # 3. pre[1] >= cur[0] and pre[1] <= cur[1]: # 后面的被包裹在内
        #  ans 弹出前一个，加入[pre[0],pre[1] 或者是不变
        intervals.sort(key = lambda x:x[0])
        ans = [intervals[0]]  # 初始化1个pre
        pre = ans[-1]
        for i in range(1,len(intervals)):
            cur = intervals[i]
            if pre[1] < cur[0]:  # 1
                ans.append(cur)
                pre = ans[-1]
            elif pre[1] >= cur[0] and pre[1] < cur[1]: # 2
                ans.pop()
                ans.append([pre[0],cur[1]])
                pre = ans[-1]
            elif pre[1] >= cur[0] and pre[1] <= cur[1]: # 3
                pass
        return (ans)
```

# 62. 不同路径

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 动态规划
        # 初始化数组为
        dp = [[0 for i in range(n+1)]for j in range(m+1)]# 内圈表示m行n列，左上外圈为0
        dp[0][1] = 1 # 设置base
        # dp[i][j]的含义为i行j列的路径数目
        # 状态转移dp[i][j] = dp[i-1][j] + dp[i][j-1]        
        # 含义为能到达第i行第j列的路径为到(他左边的和到他上面的)和
        # 开始填充注意填充原则，先遍历横行，从左到右
        for i in range(1,m+1):
            for j in range(1,n+1):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1] # 返回右下角那个值即可
        

```

# 63. 不同路径 II

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右

网格中的障碍物和空位置分别用 1 和 0 来表示。

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # dp
        # 初始化左上角外圈为0
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0 for i in range(n+1)]for k in range(m+1)]
        # 设置基态 dp[0][1] = 1
        dp[0][1] = 1
        # if grid[i-1][j-1] == 1:  dp[i][j] = 0 当这个位置是障碍物的时候,状态转移为这个
        # 否则状态转移方程为dp[i][j] = dp[i-1][j]+dp[i][j-1] # 如果那两格子不被占据
        for i in range(1,m+1):
            for j in range(1,n+1):
                if obstacleGrid[i-1][j-1] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1] # 返回最后一个
```

# 64. 最小路径和

给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # dp
        # 创建 m+1 行，n+1 行的dp，
        # 外边框:列初始化为0,行初始化为极大值
        m = len(grid)
        n = len(grid[0])
        dp = [[0 for i in range(n+1)]for k in range(m+1)]
        for i in range(len(dp[0])):
            dp[0][i] = 0xffffffff
        for i in range(len(dp)):
            dp[i][0] = 0xffffffff
        dp[1][1] = grid[0][0] # 第一格子赋值为网格第一个数，开始遍历

        # 状态转移为dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i-1][j-1]
        # 遍历顺序为按行遍历
        for i in range(1,m+1):
            for j in range(1,n+1):
                if i == j == 1:
                    continue
                dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i-1][j-1]
        return dp[-1][-1]

```

# 65. 有效数字

有效数字（按顺序）可以分成以下几个部分：

一个 小数 或者 整数
（可选）一个 'e' 或 'E' ，后面跟着一个 整数
小数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
下述格式之一：
至少一位数字，后面跟着一个点 '.'
至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
一个点 '.' ，后面跟着至少一位数字
整数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
至少一位数字
部分有效数字列举如下：

["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]
部分无效数字列举如下：

["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]
给你一个字符串 s ，如果 s 是一个 有效数字 ，请返回 true 。

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        # 去除首位空格
        s = s.strip()
        if len(s) == 0: return False
        # 
        # print(self.isInt(s) , self.isFloat(s) , self.isExp(s))
        return self.isInt(s) or self.isFloat(s) or self.isExp(s)
        
    def isInt(self,t): # 判断传入字符是否是整数
        if len(t) == 0: return False
        t = t.strip() # 去空格
        if len(t) == 0: return False
        if t[0] in "+-": #去符号
            t = t[1:]
        if len(t) == 0: return False
        for n in t:
            if n not in set("0123456789"):
                return False
        return True
    
    def isInt_byNoStrip(self,t): # 给e判断用的
        if len(t) == 0: return False
        if t[0] in "+-": #去符号
            t = t[1:]
        if len(t) == 0: return False
        for n in t:
            if n not in set("0123456789"):
                return False
        return True
    
    def isInt_byNoStrip_NoSymbol(self,t): # 给.判断用的
        if len(t) == 0: return False
        for n in t:
            if n not in set("0123456789"):
                return False
        return True

    def isFloat(self,t): # 判断传入字符是否是小数
        if len(t) == 0: return False
        t = t.strip() # 去空格
        if len(t) == 0: return False
        if t[0] in "+-": #去符号
            t = t[1:]
        if len(t) == 0: return False
        if t[0] == ".": # 3# 检测情况为小数点开头的
            return self.isInt_byNoStrip_NoSymbol(t[1:])
        countPoint = False # 不能有两个小数点
        pointIndex = None
        for i in range(len(t)):
            if t[i] == "." and countPoint == False:
                pointIndex = i 
                countPoint = True
            elif t[i] == "." and countPoint == True:
                return False
            elif t[i] not in set("0123456789"):
                return False
        if pointIndex != None:
            if pointIndex == len(t) - 1: # 小数点是最后一位 # 1
                return self.isInt(t[:-1])
            else:
                return self.isInt_byNoStrip(t[pointIndex+1:]) # 2
    
    def isExp(self,t):
        if len(t) == 0: return False
        t = t.strip() # 去空格
        if len(t) == 0: return False
        if t[0] in "+-": #去符号
            t = t[1:]
        if len(t) == 0: return False
        findE = None
        OnlyE = 0 # 记录是否只有一个e
        for i in range(len(t)):
            if t[i] in "eE":
                findE = i
                OnlyE += 1
            elif t[i] not in set("0123456789+-."):
                return False
            if OnlyE > 1:
                return False  
        if findE != None:
            front = t[:findE] # 前面必须为数字【小数or整数】
            after = t[findE+1:] # 后面必须为整数
            frontValid = self.isInt(front) or self.isFloat(front)
            afterValid = self.isInt_byNoStrip(after)
            return frontValid and afterValid
        elif findE == None:
            return False
```

# 79. 单词搜索

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # 回溯法，一个方向数组，一个已经访问的数组
        m = len(board)
        n = len(board[0])
        booleanVisited = [[False for j in range(n)] for i in range(m)]
        direc = [(-1,0),(+1,0),(0,+1),(0,-1)]

        def isValid(i,j): # 判断坐标是否合法
            return 0 <= i < m and 0 <= j < n

        def dfs(i,j,index): # index来自于word
            if index == len(word) - 1 and board[i][j] == word[index]:
                return True
            if board[i][j] != word[index]:
                return False
            
            result = False # 注意这一行，很精髓

            booleanVisited[i][j] = True # 标记为已访问
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if isValid(new_i,new_j) and not booleanVisited[new_i][new_j]: # 判断是否合法
                    if dfs(new_i,new_j,index+1):
                        result = True
                        break # 注意这个break找到就返回
            booleanVisited[i][j] = False # 取消标记
            return result
        
        for i in range(m):
            for j in range(n):
                if dfs(i,j,0) == True: # 找到一个就返回
                    return True
        return False
```

# 82. 删除排序链表中的重复元素 II

存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。

返回同样按升序排列的结果链表。

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 递归
        # 如果链表节点为1，或者0，return head
        # 否则比较head 和 head.next ，两者要是不一样，递归处理deleteDuplicates(head.next)
        # 如果head和head.next一样，则记录head,利用指针指向第一个不为这个mark值的节点，
        # 然后递归处理这个节点
        if head == None or head.next == None:
            return head
        if head.val != head.next.val:
            head.next = self.deleteDuplicates(head.next)
        elif head.val == head.next.val:
            mark = head
            cur = head
            while cur != None and cur.val == mark.val: # 直到不同位置
                cur = cur.next
            head = self.deleteDuplicates(cur) # 那个不同值，作为递归传入，进行处理

        return head
        
```

# 84. 柱状图中最大的矩形

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 方法1:分治法，基于递归
        # 基于木桶原理：找到数据组中最矮的，然后乘以左右边界。
        # 最坏时间复杂度为O(n^2),递归平均为O(nlogn),平均状况下是一种不错的解法
        # 但是力扣这种魔鬼拦超时的测试用例需要在找最矮的里面搞一个补丁。
        if len(heights) == 0: # 递归边界
            return 0
        idx = self.find_min_index(heights)
        area = heights[idx] * (len(heights))
        leftPart = self.largestRectangleArea(heights[:idx])
        rightPart = self.largestRectangleArea(heights[idx+1:])
        return max(area,leftPart,rightPart)
    
    def find_min_index(self,hList):
        min_num = min(hList)
        rand = random.randint(0,len(hList)-1)
        if hList[rand] == min_num: # 面对测试数据编程。。。 因为他的暴力拦截拦的是全1或者是单调的阶梯型，去掉这两行
            return rand
        for index,val in enumerate(hList):
            if val == min_num:
                return index
```



# 86. 分隔链表

给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # 小于的作为链1，大于等于的作为链2
        smaller_head = ListNode() # 哑节点，小头
        bigger_head = ListNode() # 哑节点，大头
        smaller_cur = smaller_head
        bigger_cur = bigger_head
        cur = head
        while cur != None:
            if cur.val < x: # 进小链
                smaller_cur.next = cur
                smaller_cur = smaller_cur.next
            elif cur.val >= x: # 进大链
                bigger_cur.next = cur
                bigger_cur = bigger_cur.next
            cur = cur.next
        # 这样走完，需要看两者的最后的指针是否是指向正确
        # 走完之后，小链的尾端指向大链的头，注意哑节点
        smaller_cur.next = bigger_head.next # 小链尾指向大链哑节点的下一个点
        bigger_cur.next = None # 大链末尾置None
        return smaller_head.next # 返回小链哑节点的下一位
                
```

# 90. 子集 II

给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        # 方法1: 暴力过筛
        ans = [[]]
        path = []
        nums.sort() # 排序一下
        n = len(nums)
        def backtracking(lst):# 注意这个回溯写法
            if lst == []:
                return 
            path.append(lst[0])
            ans.append(path[:])
            backtracking(lst[1:])
            path.pop()
            backtracking(lst[1:])
        backtracking(nums)

        final = []
        memo = set()
        for i in ans:
            if tuple(i) not in memo:
                memo.add(tuple(i))
                final.append(i)
        return final
            
```

# 113. 路径总和 II

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        # dfs
        ans = [] # 收集答案
        path = [] # 路径
        def dfs(root,targetSum): # 注意，没有说所有节点值为正数，不能剪枝
            if root == None:
                return 
            path.append(root.val) # 做选择
            if root.val == targetSum and root.left == None and root.right == None: # 看是否接收
                ans.append(path[:])
            dfs(root.left,targetSum-root.val)
            dfs(root.right,targetSum-root.val)
            path.pop() # 取消选择
        dfs(root,targetSum) # 开始搜索
        return ans
```

# 120. 三角形最小路径和

给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 大空间dp,空间为On^2
        # dp[i][j] 以第i层，第j个为终点，能拿到的最小值，如果数组下标越界，则赋值极大值
        # 状态转移为 dp[i][j] = nums[i][j] + min(dp[i-1][j-1],dp[i-1][j]) # 注意越界
        n = len(triangle)
        dp = [[0xffffffff for i in range(n)]for k in range(n)]
        nums = triangle
        for i in range(n):
            for j in range(i+1):
                if i - 1 < 0: # 处理越界问题
                    dp[i][j] = nums[i][j]
                elif j-1 < 0:
                    dp[i][j] = nums[i][j] + dp[i-1][j]
                else:
                    dp[i][j] = nums[i][j] + min(dp[i-1][j-1],dp[i-1][j])
        # 返回最后一行的最小值
        return min(dp[-1])
```

# 129. 求根节点到叶节点数字之和

给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的 所有数字之和 。

叶节点 是指没有子节点的节点。

```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        # dfs收集路径，最终把每条路径上的结果加入，由于深度不超过10，直接字符串to数字
        # 节点字符为0～9，更方便处理了
        path = []
        ans = [] # 收集每条路径
        def dfs(root):
            if root == None:
                return 
            path.append(str(root.val))
            if root.left == None and root.right == None: # 叶子收集
                ans.append(int(''.join(path[:])))
            dfs(root.left)
            dfs(root.right)
            path.pop()
        dfs(root) # 直接dfs收集
        return sum(ans) # 返回每条路径的总和
```

# 130. 被围绕的区域

给你一个 `m x n` 的矩阵 `board` ，由若干字符 `'X'` 和 `'O'` ，找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 方法1: 把所有边界上的O替换成# ，之后再替换回来
        m = len(board)
        n = len(board[0])
        queue = []
        extraqueue = []
        extraVisited = set() # 两个访问集合，一个用来复原
        visited = set() # 正常BFS遍历
        for i in range(m):
            for j in range(n):
                if board[i][j] == "O":
                    if i == 0 or i == m-1 or j == 0 or j == n-1:
                        extraqueue.append((i,j))
                        extraVisited.add((i,j))
                    else:
                        queue.append((i,j))
                        visited.add((i,j))
        
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        # BFS
        while len(extraqueue) != 0:
            new_equeue = []
            for i,j in extraqueue:
                board[i][j] = "#"
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and (new_i,new_j) not in extraVisited and board[new_i][new_j] == "O":
                        extraVisited.add((new_i,new_j))
                        new_equeue.append((new_i,new_j))
            extraqueue = new_equeue
        
        while len(queue) != 0:
            new_queue = []
            for i,j in queue:
                board[i][j] = "X"
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and (new_i,new_j) not in visited and board[new_i][new_j] == "O":
                        visited.add((new_i,new_j))
                        new_queue.append((new_i,new_j))
            queue = new_queue
        
        for i,j in extraVisited: # 还原
            board[i][j] = "O"
```

# 131. 分割回文串

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

**回文串** 是正着读和反着读都一样的字符串。

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        # 先预处理dp，dp[i][j] 的含义是s[i:j+1]是否回文串，注意左闭右开
        # 主对角线显然是，补充主对角线的右平行线
        # dp[i][j] 的状态转移是当掐头去尾为回文串且新添加的s[i] == s[j]
        n = len(s)
        dp = [[False for i in range(n)] for k in range(n)]
        for i in range(n): # 填充主对角线
            dp[i][i] = True
        for i in range(n-1): # 填充右平行对角线
            dp[i][i+1] = (s[i] == s[i+1])
        # 开始状态转移，dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
        # 填充顺序为从左到右的纵列
        for j in range(2,n):
            for i in range(0,j-1):
                dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
        # 此时dp可以判断s[i:j+1]是否为回文串

        ans = [] # 收集最终结果
        path = [] # 收集路径

        # 注意这个回溯
        def dfs (i): # 参数为[0~i]左闭右开，是否需要继续搜索
            if i == n: # 到达了右边界，收集答案
                ans.append(path[:])
                return  # 并返回
            for j in range(i,n):
                if dp[i][j] == True: # 为True才有必要做选择，否则剪枝
                    path.append(s[i:j+1]) # 做选择
                    dfs(j+1)
                    path.pop() # 取消选择
        dfs(0) # 开始搜索
        return ans
```

# 132. 分割回文串 II

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是回文。

返回符合要求的 **最少分割次数** 。

```
动态规划

设dp[i]表示的是字符串s中，0~i的子串 分割成回文串所需的【最小分割次数】
则有:

如果0~i的子串本身就是回文串则无需分割，最小分割次数dp[i]=0
如果0~i不是回文串，但存在一个j，使得j+1~i是回文串，则可将其在j处分割视为一次可行分割，而此时前面的0~j还处于未分割状态。
则可有0~i的一种可行分割次数为0~j的分割次数+1（当j+1~i是回文串时），而dp[i]是所有可行分割情况中最小的
则可有如下状态转移方程：
dp[i] = 0      // 本身是回文
dp[i] = min(dp[j]) + 1 //# 本身不是回文串 # 找到分隔字符j，使得s[j+1:i]为回文串。且s[0:j]也是回文串。


```

```python
class Solution:
    def minCut(self, s: str) -> int:
        # 两层dp
        # 第一次dp:dp[i][j] 为 s[i:j+1]是否是回文串【注意左闭右开】
        n = len(s)
        dp = [[False for i in range(n)]for k in range(n)]
        # 主对角线填充，右平行对角线填充
        for i in range(n):
            dp[i][i] = True
        for i in range(n-1):
            dp[i][i+1] = (s[i] == s[i+1])
        # dp转移，dp[i][j] = dp[i+1][j-1] and s[i] == s[j]
        # 从左到右纵列填充
        for j in range(2,n):
            for i in range(0,j-1): # 注意左闭右开
                dp[i][j] = dp[i+1][j-1] and s[i] == s[j]
        # print(dp)

        # 第二次dp为，它是回文串且没有分隔完毕的时候，
        # 如果是回文，则ddp[i] = 0
        # 如果不是回文，则ddp[i] = min(group) + 1 ,其中group 是 ddp[k],k<i 的集合,且dp[i][k] == True
        # 且其中 s[k+1..i] 是一个回文串
        # ddp[0] = 0 # 边界条件,从左扫到最后，返回最后一个值
        ddp = [0xffffffff for i in range(n)] # 初始化填入最大值
        # 只看第一位，显然ddp[0] = 0
        ddp[0] = 0
        for i in range(1,n):
            if dp[0][i] == True: # 本身是回文的时候，填充0
                ddp[i] = 0
            else: # 本身不为0的时候，需要找到前面分隔成回文的最小值，并且+1次分隔次数，
                group = []
                for k in range(i): # j小于i，
                    if dp[k+1][i] == True:
                        group.append(ddp[k])
                ddp[i] = min(group) + 1 # 上面这个for循环一定会有true，因为j+1最后一次会等于i
        # print(ddp)
        return ddp[-1] # 返回最后一个值
```

# 143. 重排链表

给定一个单链表 L 的头节点 head ，单链表 L 表示为：

 L0 → L1 → … → Ln-1 → Ln 
请将其重新排列后变为：

L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …

不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

示例 1:

输入: head = [1,2,3,4]
输出: [1,4,2,3]

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # 思路，快慢指针找到链表的中间
        # 拆成两条链，将第二条链倒序，然后交叉组合头部成为新链
        slow = head
        fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        # 此时slow的下一个节点，分奇偶讨论时
        # 如0，1，2，3 
        # 0，1, 2，|| 3
        # slow.next是 2，，把slow.next 置None，从slow.next开始反转
        # 如0，1，2，3，4
        # 0，1，2 ｜｜ 3，4 ，slow.next是 3，把slow.next 置None，从slow.next开始反转
        store = slow.next
        l2 = store
        slow.next = None
        # 翻转链2
        helper_cur = None
        while l2 != None:
            temp = l2.next
            l2.next = helper_cur
            helper_cur = l2
            l2 = temp
        # 此时helper_cur指向新链头
        cur1 = head
        cur2 = helper_cur
        # 合并两个链表
        #print(cur1)
        #print(cur2)
        # 迭代合并两个链表
        # cur1链条一定大于等于cur2，所以用这个迭代
        while cur1 != None:
            if cur2 != None:
                temp1 = cur1.next  # 暂存cur1下一位
                temp2 = cur2.next  # 暂存cur2下一位
                cur1.next = cur2   # 调整指向
                cur2.next = temp1
                cur1 = temp1
                cur2 = temp2 
            else:
                cur1 = cur1.next

```

# 146. LRU 缓存机制

运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。


进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？

```python
class Node:
    def __init__(self,key = -1,val = -1):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class DQ:  # 手写一个双端队列，带头尾哨兵方便逻辑统一。为了调用方便，和系统内置deque同音，并且方法名设置相同
    # 需要对python的内置数据结构具有一定的了解
    def __init__(self):
        self.header = Node()
        self.tailer = Node()
        self.header.next = self.tailer
        self.tailer.prev = self.header
        self.size = 0

    def appendleft(self, new_node):  # 注意参数是节点类
        temp = self.header.next
        self.header.next = new_node
        new_node.prev = self.header
        new_node.next = temp
        temp.prev = new_node
        self.size += 1

    def popright(self):  # 这里方法名写成popright是为了更明确，实际上pop就是从最右边pop
        temp = self.tailer.prev
        temp.prev.next = self.tailer
        self.tailer.prev = temp.prev
        self.size -= 1
        return temp  # 这里需要这个节点的k,v,返回值是节点类

    def remove(self, node):  # 移除任意节点，注意参数是节点类
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
   
class LRUCache:
# LRU基于哈希表和双向链表
# 基于习惯，把左边当作头部，把右边当作尾部。每次put和get都回把元素提到最左边。自然加入的逻辑是在左边加
# 利用哈希表存储k,v对，put的时候如果达到上限，则把尾巴“挤”出去
    def __init__(self, capacity: int):
        self.cap = capacity
        self.hashmap = dict()
        self.cache = DQ()                

    def get(self, key: int) -> int: # 未成功get返回-1
        # get的逻辑是，找到节点。删除原位置节点。放在最左边
        if key not in self.hashmap: return -1
        elif key in self.hashmap:
            the_node = self.hashmap[key]
            self.cache.remove(the_node)
            self.cache.appendleft(the_node)
            return self.hashmap[key].val
        
    def put(self, key: int, value: int) -> None: 
        # 手写的时候推荐先写方法put，方便理清楚hashmap和cache里面存了什么
        # 放入的逻辑是:
        # 是已经存在的键，无需考虑容量，直接更新，并且将其提到最左边
        if key in self.hashmap:
            the_old_node = self.hashmap[key] # 记录旧节点
            self.hashmap[key] = Node(key,value) # 创建新节点
            self.cache.remove(the_old_node)
            self.cache.appendleft(self.hashmap[key])
        # 如果没有超过容量，则直接在左边放入。
        # 如果超过容量，则删除掉最后的。再放入。删除最后的时候注意处理map
        else:
            if self.cache.size < self.cap:
                new_node = Node(key = key,val = value)
                self.hashmap[key] = new_node
                self.cache.appendleft(new_node)

            elif self.cache.size == self.cap:
                new_node = Node(key = key,val = value)
                self.hashmap[key] = new_node
                the_delete_node = self.cache.popright() # 根据返回值删去对应的map
                self.cache.appendleft(new_node)
                del self.hashmap[the_delete_node.key]
        # 检查用
        # print("")
        # cur = self.cache.header.next
        # while cur.next != None:
        #     print(cur.val,end = '|')
        #     cur = cur.next
        # print()
    
```

# 149. 直线上最多的点数

给你一个数组 `points` ，其中 `points[i] = [xi, yi]` 表示 **X-Y** 平面上的一个点。求最多有多少个点在同一条直线上。

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        # 用标准方程式的解法Ax+By+C = 0
        # 使用向量解法，所有点互不相同。任意取两点，转化为标准向量。并且把符号变成[+,+]或者[+,-]
        # 所有点坐标都是int，为了防止浮点数比较问题，标准向量选取为取都除以最大公约数
        if len(points) == 1:
            return 1
        dic = collections.defaultdict(int)
        for i in range(len(points)): # 收集所有的标准方程式
            for j in range(i+1,len(points)):
                tp = self.calcFunction(points[i],points[j])
                dic[tp] += 1
        maxLength = 0
        # 注意 如果n个点共线，那么两两取点的时候有很多解
        # 例如，5个点。每两个计算一次，有重复计算 4+3+2+1 == 10 （n-1)*n == val*2 #计算这个的解 
        for key in dic:
            val = dic[key]
            length = self.findSolution(val) # 根据总值还原成点点数量
            if length > maxLength:
                maxLength = length
        return maxLength

    def findGCD(self,a,b): # 找到最小公倍数
        while a != 0:
            temp = a
            a = b % a
            b = temp
        return b # 

    def standard(self,x,y): # 标准化也适用于(x,0),(0,y)这一类
        t = self.findGCD(x,y)
        x = x//t
        y = y//t
        if x * y < 0:
            x = abs(x)
            y = -abs(y)
        return (x,y)
    
    def getVector(self,cp1,cp2): # 得到标准化向量
        x = cp2[0] - cp1[0]
        y = cp2[1] - cp1[1]
        return self.standard(x,y) 
    
    def calcFunction(self,cp1,cp2): # 注意这个计算是纸上算的
        A,B = self.getVector(cp1,cp2)
        C = (A*cp1[1]-B*cp1[0])
        return (B,-A,C) # 系数满足 mx+ny+p = 0
    
    def findSolution(self,val): # 求解
        return int((1+sqrt(1+8*val))/2)
```



# 159. 至多包含两个不同字符的最长子串

给定一个字符串 ***s\*** ，找出 **至多** 包含两个不同字符的最长子串 ***t\*** ，并返回该子串的长度。

```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        # 滑动窗口
        window = defaultdict(int)
        left = 0 # 窗口指针
        right = 0
        maxlength = 1 # 需要返回的结果，起码是1
        # 以下逻辑没有用到前缀和极有可能超时
        windowlength = 0
        while right < len(s):
            add_char = s[right]
            right += 1
            window[add_char] += 1
            windowlength += 1
            temp_sum = 0
            if len(window) <= 2: # 更新逻辑在前后都有
                maxlength = max(maxlength,windowlength)
            while left < right and len(window) > 2: # 如果大于2，则收集结果
                delete_char = s[left]
                left += 1
                window[delete_char] -= 1
                windowlength -= 1
                if window[delete_char] == 0: del window[delete_char]
            if len(window) <= 2: # 更新逻辑在前后都有
                maxlength = max(maxlength,windowlength)

        return maxlength
            


```

# 170. 两数之和 III - 数据结构设计

设计一个接收整数流的数据结构，该数据结构支持检查是否存在两数之和等于特定值。

实现 TwoSum 类：

TwoSum() 使用空数组初始化 TwoSum 对象
void add(int number) 向数据结构添加一个数 number
boolean find(int value) 寻找数据结构中是否存在一对整数，使得两数之和与给定的值相等。如果存在，返回 true ；否则，返回 false 。

```python
class TwoSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """        
        # 字典类收集
        self.dict1 = collections.defaultdict(int)
        # 存一个可以findout的,不用重复查
        self.findout = set()

    def add(self, number: int) -> None:
        """
        Add the number to an internal data structure..
        """
        self.dict1[number] += 1

    def find(self, value: int) -> bool:
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        """
        if value in self.findout: # 先查这个有没有找出来过
            return True
        for i in self.dict1: # 否则扫描每一个
            need = value - i
            if self.dict1.get(need) == None: 
                continue # 开始下一轮查找
            elif need != i: # 当need不是正在扫的数的时候
                if self.dict1.get(need) >= 1: 
                    self.findout.add(value)
                    return True
            elif need == i: # 当need是在扫的数的时候
                if self.dict1.get(need) >= 2:
                    self.findout.add(value)
                    return True
        # 扫完了都没有
        return False

# Your TwoSum object will be instantiated and called as such:
# obj = TwoSum()
# obj.add(number)
# param_2 = obj.find(value)
```

# 172. 阶乘后的零

给定一个整数 *n*，返回 *n*! 结果尾数中零的数量。

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        # 列表找规律模拟
        # 5 -> 1
        # 25 -> 5*1 + 1 == 6
        # 125 -> 6*5 + 1 == 31
        # 625 -> 31*5 + 1 == 156
        # 且例如655 被分解为 655 = 625 + 25 + 5
        start = 5
        temp_lst = [(5,1)]
        while start * 5 <= n :
            start *= 5
            temp_couple = temp_lst[-1]
            temp_lst.append((start,temp_couple[1]*5+1))
        count = 0
        while len(temp_lst) != 0: #过筛
           count += ( n // temp_lst[-1][0] ) * temp_lst[-1][1]
           n = n % temp_lst[-1][0]
           temp_lst.pop()
        return count

```

```python
# 数学方法
class Solution:
    def trailingZeroes(self, n: int) -> int:
        # 看其中包含多少个5
        count = 0
        while n >= 5:
            count += n//5
            n //= 5
        return count
```

# 179. 最大数

给定一组非负整数 `nums`，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

**注意：**输出结果可能非常大，所以你需要返回一个字符串而不是整数。

```python
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # nums[i]属于int类，那么在其他语言中拼接的时候注意使用long进行限制
        # 排序时用到了一个这样的思想：
        # [x,y] >= [y,x] ，则选取 xy的顺序，否则选择yx的顺序
        # 即选取俩数的时候，一定要有拼接完之后的数值更大
        # 手写一个快排逻辑
        # 特殊情况判断

        if len(nums) == 0:
            return ''
        if len(nums) == 1:
            return str(nums[0])
        if sum(nums) == 0: # 这一条好坑。。[0,0,0,0]
            return '0'

        def quick_sort(lst): # 传入的是列表引用，内容可以修改
            if len(lst) <= 1: # 只有一个的时候，已经是排序好的
                return
            pivot = str(lst[0]) # 这里选取的是第一个数作为基准
            n = len(lst)
            less = []
            equal = []
            Greater = []
            while lst:
                temp_num = str(lst.pop()) # 字符化处理
                if int(pivot + temp_num) == int(temp_num + pivot):
                    equal.append(temp_num)
                elif int(pivot + temp_num) >= int(temp_num + pivot):
                    Greater.append(temp_num)
                elif int(pivot + temp_num) <= int(temp_num + pivot):
                    less.append(temp_num)
            quick_sort(less)
            quick_sort(Greater)
            while less:
                lst.append(less.pop(0))
            while equal:
                lst.append(equal.pop(0))
            while Greater:
                lst.append(Greater.pop(0))
        quick_sort(nums)
        return ''.join(nums)

```

# 186. 翻转字符串里的单词 II

给定一个字符串，逐个翻转字符串中的每个单词。

```python
class Solution:
    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        # 1. 先快慢指针翻转单个单词
        slow = 0
        fast = 0
        while slow < len(s):
            while fast < len(s) and s[fast] != ' ':
                fast += 1
            left = slow
            right = fast - 1
            while left < right:
                s[left],s[right] = s[right],s[left]
                left += 1
                right -= 1
            slow = fast + 1
            fast += 1
        # 2.全局翻转
        t_left = 0
        t_right = len(s)-1
        while t_left < t_right:
            s[t_left],s[t_right] = s[t_right],s[t_left]
            t_left += 1
            t_right -= 1
```

# 187. 重复的DNA序列

所有 DNA 都由一系列缩写为 'A'，'C'，'G' 和 'T' 的核苷酸组成，例如："ACGAATTCCG"。在研究 DNA 时，识别 DNA 中的重复序列有时会对研究非常有帮助。

编写一个函数来找出所有目标子串，目标子串的长度为 10，且在 DNA 字符串 s 中出现次数超过一次。

```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        # 切片暴搜
        ans = [] # 收集答案
        if len(s) < 10:
            return ans
        ddict = collections.defaultdict(int)
        p = 0
        while p < len(s) - 9:
            ddict[s[p:p+10]] += 1
            p += 1
        for i in ddict:
            if ddict[i] >= 2:
                ans.append(i)
        return ans
```

# 204. 计数质数

统计所有小于非负整数 *`n`* 的质数的数量。

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        # python素数筛,只筛数量
        if n < 2:
            return 0
        if n == 2:
            return 0
        # 先初始化所有数字为True
        grid = [True for i in range(n)]
        grid[0],grid[1] = False,False
        count = 0 # 计数用
        # 筛的上界为 sqrt(n)即可
        upto = math.ceil(sqrt(n))+1 # 注意range的左闭右开
        for index in range(upto):
            if grid[index] == True:
                for multi in range(2,n//index+1): # 两倍以上的数就不要了
                    if index * multi < n:
                        grid[index * multi] = False
        for i in grid:
            if i == True:
                count += 1
        return count


```

# 208. 实现 Trie (前缀树)

Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。

```python
class TrieNode: # 声明一个前缀树节点
    def __init__(self):
        self.children = [None for i in range(26)] # 序号从0～25
        self.isEnd = False # 标志是否是单词

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode() # 传入一个实例


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root # 从根节点扫起
        for char in word: # 检查节点的children是否是非None，需要获取它的索引
            index = (ord(char)-ord("a"))
            if node.children[index] == None: # 如果还没有被创建,则传递给他一个实例化的前缀树节点
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isEnd = True # 扫完了则把当前节点设置为是一个单词

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root # 从根节点扫起，中途不能没有
        for char in word:
            index = (ord(char)-ord("a"))
            if node.children[index] == None: # 如果还没有被创建,返回False
                return False
            node = node.children[index] # 如果中途没有断 则继续找
        return node.isEnd # 


    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root # 从根节点扫起，中途不能没有
        for char in prefix:
            index = (ord(char)-ord("a"))
            if node.children[index] == None: # 如果还没有被创建,返回False
                return False
            node = node.children[index] # 如果中途没有断 则继续找
        return node != None # 如果节点非空，则是True，如果节点空，则是False


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

# 215. 数组中的第K个最大元素

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `**k**` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
        # 使用小根堆筛，维护k大小的小根堆，先把前k个堆化，然后再从之后进行筛选
        # 如果元素大于堆顶，弹出堆顶，再则入堆，如果元素小于堆顶，则不要
        # 考虑使用堆，需要手写堆[之后再写]
        minroot_heap = nums[:k]
        heapq.heapify(minroot_heap) # 堆化
        sieve = nums[k:] # 要过筛的元素
        for i in sieve:
            if i > minroot_heap[0]:
                heapq.heappop(minroot_heap)
                heapq.heappush(minroot_heap,i)
        return (minroot_heap[0])


```

# 233. 数字 1 的个数

给定一个整数 `n`，计算所有小于等于 `n` 的非负整数中数字 `1` 出现的个数。

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        # 数学方法
        length = len(str(n))           
        kp = [0 for i in range(length+1)]
        kp[0] = 0
        for i in range(1,len(kp)):
            kp[i] = 10 * kp[i-1] + 10 ** (i - 1)       
        # 例如 23756 = 20000 + 3000 + 700 + 50 + 6
        # 2 * kp[4] + 10000 + 3 * kp[3] +1000 + 7 * kp[2] + 100 + 6 * kp[1] + 10 + 5 * kp[0] + 1
        # 如果位的值大于1，则 位值*kp[位长-1] + 10 ** (位长-1)
        # 如果位的值小于等于1，则 位值
        count = 0
        times = len(str(n))
        for i in range(times):
            temp_length = len(str(n)) - 1
            the_bit = n // 10 ** temp_length
            times -= 1
            if the_bit > 1: 
                count += the_bit * kp[temp_length] + 10 ** temp_length
            elif the_bit == 1: # 注意这一行的处理。(n - 10 ** temp_length + 1)
                count += kp[temp_length] + (n - 10 ** temp_length + 1)
            elif the_bit == 0:
                pass
            n = n % 10 ** temp_length  
        return count


```

# 236. 二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 递归解法，后序遍历
        if root == None : return None
        if root == p or root == q: return root # 如果根是节点，则返回
        left = self.lowestCommonAncestor(root.left,p,q) # 递归搜索左右子树
        right = self.lowestCommonAncestor(root.right,p,q) 
        # 根据左右子树是否为空分类
        if left == None and right == None: return None # 搜不到
        if left == None and right != None: return right
        if left != None and right == None: return left
        if left != None and right != None: return root
```

# 239. 滑动窗口最大值

给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 使用单调队列，仅仅在窗口大小内使用单调队列,队列包含值
        window = collections.deque()
        mono_queue = collections.deque() # 双端队列
        # 初始化窗口和双端队列，单调队列头部为大值，
        for index,val in enumerate(nums[:k]):
            window.append(val) # 主窗口直接加入即可
            # 单调队列需要比较队尾，队尾的值大于将要入队的值，直接进来，队尾的值小于将要入队的值，弹出队尾。
            if len(mono_queue) == 0:
                mono_queue.append(val)
            elif mono_queue[-1] > val:
                mono_queue.append(val)
                continue
            else:
                while len(mono_queue) != 0 and mono_queue[-1] < val:
                    mono_queue.pop()
                mono_queue.append(val)
        ans = [mono_queue[0]] # 收集答案用
        # 开始扫描全部的窗口
        for val in nums[k:]:
            # print(window,mono_queue)
            e = window.popleft() # 出窗口的元素
            window.append(val) # 入窗口的元素
            if e == mono_queue[0]: # 如果出窗口的元素就是单调队列的头，那么单调队列也要出头
                mono_queue.popleft()
            if len(mono_queue) == 0: # 如果单调队列为空，则直接加入
                mono_queue.append(val)
            elif mono_queue[-1] > val: # 如果满足单调递减，直接加入
                mono_queue.append(val)
            else:
                while len(mono_queue) != 0 and mono_queue[-1] < val: # 否则进行调整直到满足单调递减
                    mono_queue.pop()
                mono_queue.append(val)
            ans.append(mono_queue[0]) # 结果收集为本轮窗口中的单调队列头
        return ans
```

# 243. 最短单词距离

给定一个单词列表和两个单词 word1 和 word2，返回列表中这两个单词之间的最短距离。

示例:
假设 words = ["practice", "makes", "perfect", "coding", "makes"]

输入: word1 = “coding”, word2 = “practice”
输出: 3
输入: word1 = "makes", word2 = "coding"
输出: 1
注意:
你可以假设 word1 不等于 word2, 并且 word1 和 word2 都在列表里。

```python
class Solution:
    def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        # 获取出现过的索引位置
        index1 = []
        index2 = []
        for i in range(len(wordsDict)):
            if wordsDict[i] == word1:
                index1.append(i)
            if wordsDict[i] == word2:
                index2.append(i)
        # 返回两个数组中数的最小差值，不使用暴力搜法
        # 暴力搜的时间复杂度是n**2
        # 排序之后使用双指针的时间复杂度是nlogn
        # 这一题直接暴力
        min_gap = 0xffffffff # 事先声明一个极大值
        for i in index1:
            for j in index2:
                if abs(i-j) < min_gap:
                    min_gap = abs(i-j)
        return min_gap
        

```

```python
class Solution:
    def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        # 一轮扫描法
        # 预先设置俩坐标的初始值为极大值
        index1 = 0xffffffff
        index2 = 0xffffffff
        min_gap = 0xffffffff
        # index1是上一次word1的出现坐标
        # index2是上一次word2的出现坐标
        # 这里有贪心的意思
        for i in range(len(wordsDict)):
            if wordsDict[i] == word1:
                index1 = i
                min_gap = min(min_gap,abs(index1-index2))
            if wordsDict[i] == word2:
                index2 = i
                min_gap = min(min_gap,abs(index1-index2))
        return min_gap
```

# 244. 最短单词距离 II

请设计一个类，使该类的构造函数能够接收一个单词列表。然后再实现一个方法，该方法能够分别接收两个单词 word1 和 word2，并返回列表中这两个单词之间的最短距离。您的方法将被以不同的参数调用 多次。

```python
class WordDistance:

    def __init__(self, wordsDict: List[str]):
        self.lst = wordsDict

    def shortest(self, word1: str, word2: str) -> int:
        min_gap = 0xffffffff
        index1 = 0xffffffff
        index2 = 0xffffffff
        for i,value in enumerate(self.lst):
            if value == word1:
                index1 = i
                min_gap = min(min_gap,abs(index1-index2))
            if value == word2:
                index2 = i
                min_gap = min(min_gap,abs(index1-index2))
        return min_gap

```

# 245. 最短单词距离 III

给定一个单词列表和两个单词 word1 和 word2，返回列表中这两个单词之间的最短距离。

word1 和 word2 是有可能相同的，并且它们将分别表示为列表中两个独立的单词。

示例:
假设 words = ["practice", "makes", "perfect", "coding", "makes"].

输入: word1 = “makes”, word2 = “coding”
输出: 1
输入: word1 = "makes", word2 = "makes"
输出: 3
注意:
你可以假设 word1 和 word2 都在列表里。

```python
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        # 处理两个相同时和两个不同时的逻辑
        if word1 != word2:
            index1 = []
            index2 = []
            p1 = 0
            p2 = 0
            for i,val in enumerate(wordsDict):
                if val == word1:
                    index1.append(i)
                if val == word2:
                    index2.append(i)
            # 初始化指针指向最开始，最小间隔先假设是初始间隔
            min_gap = abs(index1[p1]-index2[p2])
            while p1 < len(index1) and p2 < len(index2):
                temp_gap = abs(index1[p1]-index2[p2])
                min_gap = min(min_gap,temp_gap)
                # 指针移动逻辑，移动较小的那个，使得它和另一个更接近
                if index1[p1] < index2[p2]:
                    p1 += 1
                else:
                    p2 += 1
            return min_gap
        # 处理两个相同时候的逻辑
        elif word1 == word2:
            index = []
            for i,val in enumerate(wordsDict):
                if val == word1:
                    index.append(i)
            p = 1
            min_gap = abs(index[0]-index[1])
            while p < len(index):
                temp_gap = abs(index[p]-index[p-1])
                min_gap = min(min_gap,temp_gap)
                p += 1
            return min_gap

```

# 246. 中心对称数

中心对称数是指一个数字在旋转了 180 度之后看起来依旧相同的数字（或者上下颠倒地看）。

请写一个函数来判断该数字是否是中心对称数，其输入将会以一个字符串的形式来表达数字。

```python
class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        # 模拟
        key = list("69810")
        val = list("96810")
        the_dict = dict(zip(key,val))
        left = 0
        right = len(num) - 1
        while left <= right: # 注意这里是小于等于号
            if num[left] in the_dict and num[right] in the_dict:
                if the_dict[num[left]] == num[right] and the_dict[num[right]] == num[left]:
                    left += 1
                    right -= 1
                else:
                    return False
            else:
                return False
        return True
```

# 247. 中心对称数 II

中心对称数是指一个数字在旋转了 180 度之后看起来依旧相同的数字（或者上下颠倒地看）。

找到所有长度为 n 的中心对称数。

```python
class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        # n限制了长度在1～14以内,注意0不能打头
        # n如果是偶数，直接由镜像法获得
        # n如果是奇数，则最中间的那个只能是0，1，8

        valid_lst = ["0","1","6","8","9"]
        if n == 1: return ["0","1","8"]
        mirror = dict(zip(["0","1","6","8","9"],["0","1","9","8","6"]))
        length = n // 2
        ans = []
        path = []
        def backtracking(lst):
            if len(path) == n//2:
                ans.append(path[:])
                return
            for i in lst:
                if len(path) == 0 and i == "0": continue # 不能以0打头
                path.append(i)
                backtracking(lst)
                path.pop()
        backtracking(valid_lst)
        final = []
        if n % 2 == 0:
            for l in ans:
                temp = ''.join(l)
                for thenum in l[::-1]: # 注意这里的倒序
                    temp += mirror[thenum] # 照镜子
                final.append(temp)
            return final
        elif n % 2 == 1:
            for l in ans: # 奇数其实可以优化成insert三次即可
                temp1 = ''.join(l)+"0"
                temp2 = ''.join(l)+"1"
                temp3 = "".join(l)+"8"
                for thenum in l[::-1]:
                    temp1 += mirror[thenum]
                    temp2 += mirror[thenum]
                    temp3 += mirror[thenum]
                final.append(temp1)
                final.append(temp2)
                final.append(temp3)
            return final
```

# 248. 中心对称数 III

中心对称数是指一个数字在旋转了 180 度之后看起来依旧相同的数字（或者上下颠倒地看）。

写一个函数来计算范围在 [low, high] 之间中心对称数的个数。

```python
# 超级暴力模拟
class Solution:
    def strobogrammaticInRange(self, low: str, high: str) -> int:
        # 闭区间的low和high，且low<=high
        # 当low的长度不等于high的时候，把位于它俩之间的数全部算上。而无需实际的求全排列
        # 当low的长度等于high的时候，直接回溯法求全部的数值
        ans = 0
        for i in range(len(low)+1,len(high)):
            ans += len(self.helper(i))
        start = self.helper(len(low)) # 此时start是一个有很多数的列表，找到第一个大于等于low的数
        end = self.helper(len(high)) # 此时end是一个有很多数的列表，找到第一个严格大于high的数
        # 先不使用二分看能不能暴力搜到
        if len(low) != len(high):
            n1 = 0
            for i in start:
                if int(i) >= int(low):
                    break
                n1 += 1
            n2 = 0
            for i in end:
                if int(i) > int(high):
                    break
                n2 += 1
            return ans + len(start) - n1 + n2
        else:
            ans = 0
            k1 = int(low)
            k2 = int(high)
            for i in start:
                if k1 <= int(i) <= k2:
                    ans += 1
                elif int(i) > k2:
                    break
            return ans
    
    def helper(self, n: int) -> List[str]:
        # n限制了长度在1～14以内,注意0不能打头
        # n如果是偶数，直接由镜像法获得
        # n如果是奇数，则最中间的那个只能是0，1，8
        valid_lst = ["0","1","6","8","9"]
        if n == 1: return ["0","1","8"]
        mirror = dict(zip(["0","1","6","8","9"],["0","1","9","8","6"]))
        length = n // 2
        ans = []
        path = []
        def backtracking(lst):
            if len(path) == n//2:
                ans.append(path[:])
                return
            for i in lst:
                if len(path) == 0 and i == "0": continue # 不能以0打头
                path.append(i)
                backtracking(lst)
                path.pop()
        backtracking(valid_lst)
        final = []
        if n % 2 == 0:
            for l in ans:
                temp = ''.join(l)
                for thenum in l[::-1]: # 注意这里的倒序
                    temp += mirror[thenum] # 照镜子
                final.append(temp)
            return final
        elif n % 2 == 1:
            for l in ans: # 奇数其实可以优化成insert三次即可
                temp1 = ''.join(l)+"0"
                temp2 = ''.join(l)+"1"
                temp3 = "".join(l)+"8"
                for thenum in l[::-1]:
                    temp1 += mirror[thenum]
                    temp2 += mirror[thenum]
                    temp3 += mirror[thenum]
                final.append(temp1)
                final.append(temp2)
                final.append(temp3)
            return final
```



# 249. 移位字符串分组

给定一个字符串，对该字符串可以进行 “移位” 的操作，也就是将字符串中每个字母都变为其在字母表中后续的字母，比如："abc" -> "bcd"。这样，我们可以持续进行 “移位” 操作，从而生成如下移位序列：

"abc" -> "bcd" -> ... -> "xyz"
给定一个包含仅小写字母字符串的列表，将该列表中所有满足 “移位” 操作规律的组合进行分组并返回。

```python
class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        # 先按照长度分组，使用哈希
        # 在按照字母的gap差值模26分组
        l_dict = collections.defaultdict(list)
        for word in strings:
            l_dict[len(word)].append((word,self.analyze(word)))
        # print(l_dict),此时字典中按照长度已经分组完毕，在内层循环的时候创建临时字典
        ans = []
        for key in l_dict: # 在内层循环的时候创建临时字典
            temp_dict = collections.defaultdict(list)
            for cp in (l_dict[key]):
                temp_dict[cp[1]].append(cp[0])
            for value in temp_dict.values():
                ans.append(value)
        return ans

   
    def analyze(self,word): # 分析每个字母的间隔，注意模26
        if len(word) == 1:
            return (0)
        p = 1
        gap_tuple = [] # 返回的时候转成元组
        while p < len(word):
            gap = (ord(word[p])-ord(word[p-1])) % 26
            gap_tuple.append(gap)
            p += 1
        return tuple(gap_tuple)
            
```

# 252. 会议室

给定一个会议时间安排的数组 intervals ，每个会议时间都会包括开始和结束的时间 intervals[i] = [starti, endi] ，请你判断一个人是否能够参加这里面的全部会议。

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        # 需要无重叠
        # 即满足cur[1] <= next[0],遍历到倒数第二个，否则return False
        # 排序
        intervals.sort()
        for i in range(len(intervals)-1):
            if intervals[i][1] > intervals[i+1][0]:
                return False
        return True
            
```

# 253. 会议室 II

给你一个会议时间安排的数组 intervals ，每个会议时间都会包括开始和结束的时间 intervals[i] = [starti, endi] ，为避免会议冲突，同时要考虑充分利用会议室资源，请你计算至少需要多少间会议室，才能满足这些会议安排。

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        # 思路：看同时会有多少个人上会议室
        # 遇到当前时刻有人上会议室则+1，遇到当前时刻有人下会议室则-1
        # 当扫完进会议室的即可，之后人数只会下降
        now_people = 0 # 初始化现在的人数
        max_people = 0 # 初始化最多的人数
        in_meeting = [i[0] for i in intervals]
        out_meeting = [i[1] for i in intervals]
        in_meeting.sort() # 需要排序
        out_meeting.sort() # 需要排序
        p1 = 0
        p2 = 0
        while p1 < len(in_meeting): # 扫描在会议室的
            if out_meeting[p2] <= in_meeting[p1]: # 扫描可能出去的
                while out_meeting[p2] <= in_meeting[p1]:
                    p2 += 1
                    now_people -= 1
            p1 += 1
            now_people += 1
            max_people = max(max_people,now_people)
        return max_people


```

# 255. 验证前序遍历序列二叉搜索树

给定一个整数数组，你需要验证它是否是一个二叉搜索树正确的先序遍历序列。

你可以假定该序列中的数都是不相同的。

```python
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        # 递归，大数据量超时
        # 利用前序分割左右,找到第一个大于根节点的索引，以它为基础分隔左右
        if len(preorder) == 0:
            return True
        n = len(preorder)
        findIndex = len(preorder) # 初始化为没有
        for i in range(1,n): # 
            if preorder[i] > preorder[0]:
                findIndex = i
                break
        # print("left = ",preorder[1:findIndex])
        # print("right = ",preorder[findIndex:])
        left = self.verifyPreorder(preorder[1:findIndex])
        right = self.verifyPreorder(preorder[findIndex:])

        rootVal = preorder[0]
        leftValid = True
        for i in preorder[1:findIndex]:
            if i > rootVal:
                leftValid = False
                break
        rightValid = True
        for i in preorder[findIndex:]:
            if i < rootVal:
                rightValid = False
                break
        return leftValid and rightValid and left and right # 注意这一行
```

```python
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        # 单调栈，这个单调栈需要理解一下
        # 维护的是单调递减栈，当遇到大值的时候，循环弹出，且记录弹出序列的最大值！
        # 之后入栈的也必须全部大于这个最大值并且维持单调递减
        dec_stack = []
        tempMax = -0xffffffff
        for i in preorder:
            # print("tempMax = ",tempMax,'    i = ',i)
            if len(dec_stack) == 0 and i > tempMax:
                dec_stack.append(i)
            elif i < dec_stack[-1] and i > tempMax: # 维持了单调递减，直接加入
                dec_stack.append(i)
            elif i > dec_stack[-1]: 
                while len(dec_stack) > 0 and i > dec_stack[-1]:
                    tempMax = max(tempMax,dec_stack.pop()) # 循环弹出，且记录弹出序列的最大值！
                dec_stack.append(i)
            else:
                return False
        return True

```

# 259. 较小的三数之和

给定一个长度为 n 的整数数组和一个目标值 target，寻找能够使条件 nums[i] + nums[j] + nums[k] < target 成立的三元组  i, j, k 个数（0 <= i < j < k < n）。

```python
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        # 预排序，再处理
        nums.sort()
        ans = 0
        for i in range(len(nums)):
            aim = target - nums[i]
            left = i + 1
            right = len(nums) - 1
            while left < right:
                if nums[left] + nums[right] >= aim: # 数值偏大，需要移动大指针
                     right -= 1
                elif nums[left] + nums[right] < aim: # 数值合理，收集，然后移动
                    ans += right - left # 收集区间内全部的结果，这里是定left。
                    left += 1 # 这一条不能忘了
        return ans
```

# 261. 以图判树

给定从 `0` 到 `n-1` 标号的 `n` 个结点，和一个无向边列表（每条边以结点对来表示），请编写一个函数用来判断这些边是否能够形成一个合法有效的树结构。

```python
class UnionFind:
    def __init__(self,size):
        self.root = [i for i in range(size)]
    
    def find(self,x):
        while x != self.root[x]:
            x = self.root[x]
        return x
    
    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootX] = rootY
            return True # 成功连接
        if rootX == rootY: # 注意这个，如果两者之前已经连接，则gg
            return False
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # 并查集，找是否只有一个根节点,且每个节点只有一个父亲
        UF = UnionFind(n)
        for cp in edges:
            if UF.union(cp[0],cp[1]) == False: # 调用直接包含在了if中，启动if则启动了调用
                return False # union如果之前已经连接，返回的是False
        the_set = set()
        for i in range(n):
            e = UF.find(i)
            the_set.add(e)
        return len(the_set) == 1 # 当经过了父亲数量筛选后只需看最终并查集中根节点数量是不是1即可

```

# 264. 丑数 II

给你一个整数 `n` ，请你找出并返回第 `n` 个 **丑数** 。

**丑数** 就是只包含质因数 `2`、`3` 和/或 `5` 的正整数。

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        # 动态规划+3指针
        dp = [0 for i in range(n+1)] # dp[n]是第n个丑数
        dp[0],dp[1] = 1,1 # 初始化
        p2,p3,p5 = 1,1,1 # 初始化三个指针的初始位置
        # 三个指针的含义是，这个指针乘以对应的后缀值，然后取最小的，为新生成的丑数
        for i in range(2,n+1):
            ugly2,ugly3,ugly5 = dp[p2]*2,dp[p3]*3,dp[p5]*5
            dp[i] = min(ugly2,ugly3,ugly5) # 取最小的那个
            # 注意指针移动逻辑为三个if，因为可能在一次生成中多个指针需要移动
            # 例如生成10的时候，可能是dp[p2]==5,它*2 也可能是 dp[p5]==2,它*5,两者都是10
            # 防止在下一步生成的还是10，所以俩都移动
            if dp[i] == ugly2: p2 += 1
            if dp[i] == ugly3: p3 += 1
            if dp[i] == ugly5: p5 += 1
        return dp[-1]

```

# 266. 回文排列

给定一个字符串，判断该字符串中是否可以通过重新排列组合，形成一个回文字符串。

```python
class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        # 判断奇数个数的字符是不是小于等于1
        ct = collections.Counter(s)
        odd = 0
        for i in ct:
            if ct[i] % 2 == 1:
                odd += 1
        return odd <= 1
```

# 267. 回文排列 II

给定一个字符串 `s` ，返回其通过重新排列组合后所有可能的回文字符串，并去除重复的组合。

如不能形成任何回文排列时，则返回一个空列表。

```python
class Solution:
    def generatePalindromes(self, s: str) -> List[str]:
        # 本题限制了大小为1～16,为a～z
        # 回文的充要条件为奇数个数的字符不大于一个
        d_dt = collections.defaultdict(int)
        for i in s:
            d_dt[i] += 1
        the_odd = None # 默认没有奇数字符
        # 对于里面的所有偶数字符，取一半进行回溯，另一半镜相后添加
        lst = ""
        for key in d_dt:
            if d_dt[key] % 2 == 1 and the_odd == None:
                the_odd = key # 标记奇数次字符
                lst += (key) * ((d_dt[key]-1)//2) # 注意这一行！！！
            elif d_dt[key] % 2 == 1 and the_odd != None:
                return [] # 大于两个奇数字符，返回空列表
            elif d_dt[key] % 2 == 0:
                lst += key * (d_dt[key]//2)
        # 对lst里面的进行回溯和去重复
        cp_lst = [ch for ch in lst]
        back_lst = []
        path = []
        def backtracking(path,choice):
            if len(path) == len(lst):
                back_lst.append(''.join(path[:])) # 收集路径结果
                return # 这个不要忘了
            for i in choice:
                cp = choice.copy()
                cp.remove(i)
                path.append(i)
                backtracking(path,cp)
                path.pop()

        backtracking(path,cp_lst) # 调用回溯
        # 去重复
        back_lst = set(back_lst)
        final = []
        if the_odd == None: # 如果没有奇数次的字符
            for i in back_lst:
                final.append(i+i[::-1])
        elif the_odd != None:
            for i in back_lst:
                final.append(i+the_odd+i[::-1])
        final.sort() # 按照字典序排序一下方便和答案做检查对照
        return final
       
```

# 270. 最接近的二叉搜索树值

给定一个不为空的二叉搜索树和一个目标值 target，请在该二叉搜索树中找到最接近目标值 target 的数值。

注意：

给定的目标值 target 是一个浮点数
题目保证在该二叉搜索树中只会存在一个最接近目标值的数

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        # 二分搜索,路径中更新值,更新逻辑为，小于当前的gap值
        node = root
        gap = abs(root.val-target)
        theval = root.val
        while node != None:
            if abs(node.val-target) < gap:
                gap = abs(node.val-target)
                theval = node.val
            if target > node.val: # 往右边搜
                node = node.right
            elif target < node.val: # 往左边搜
                node = node.left
            elif theval == node.val: # 注意这一条，由于python的浮点数精度对比原则，它能够相等。。。
                return int(node.val)
        return theval      
```

# 279. 完全平方数

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

```python
class Solution:
    def numSquares(self, n: int) -> int:
        # dp[i] 为凑出i需要的最小数量
        # 状态转移为dp[i] = min(组)，组 == (1+dp[i-j**2]),j从1～int(sqrt(i))
        # 设置边界条件0，dp[0] = 0
        dp = [0 for i in range(n+1)] # 申请n+1位
        i = 1
        while i <= n: # i从1开始填充
            up_to = int(sqrt(i)) # 设置上限
            j = 1 # 
            group = []
            while j <= up_to:
                group.append(1+dp[i-j**2])
                j += 1
            dp[i] = min(group)
            i += 1
        return dp[-1] 
```

# 288. 单词的唯一缩写

单词的 缩写 需要遵循 <起始字母><中间字母数><结尾字母> 这样的格式。如果单词只有两个字符，那么它就是它自身的 缩写 。

以下是一些单词缩写的范例：

dog --> d1g 因为第一个字母 'd' 和最后一个字母 'g' 之间有 1 个字母
internationalization --> i18n 因为第一个字母 'i' 和最后一个字母 'n' 之间有 18 个字母
it --> it 单词只有两个字符，它就是它自身的 缩写


实现 ValidWordAbbr 类：

ValidWordAbbr(String[] dictionary) 使用单词字典 dictionary 初始化对象
boolean isUnique(string word) 如果满足下述任意一个条件，返回 true ；否则，返回 false ：
字典 dictionary 中没有任何其他单词的 缩写 与该单词 word 的 缩写 相同。
字典 dictionary 中的所有 缩写 与该单词 word 的 缩写 相同的单词都与 word 相同 。

```python
class ValidWordAbbr:

    def __init__(self, dictionary: List[str]):
        # 记录缩写词
        self.memo = collections.defaultdict(list)
        for word in dictionary:
            if len(word) <= 2:
                if self.memo[word] == []:
                    self.memo[word].append(word)
            else:
                length = str(len(word)-2)
                key = word[0]+length+word[-1]
                self.memo[key].append(word)
    def isUnique(self, word: str) -> bool:
        if len(word) <= 2:
            return True
        length = str(len(word)-2)
        key = word[0]+length+word[-1]
        if key not in self.memo:
            return True
        
        if len(self.memo[key]) != 1: # 长度不为1，一定不独特
            return False
        elif len(self.memo[key]) == 1: # 还需要进一步判断
            return self.memo[key][0] == word
```

# 295. 数据流的中位数

中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。

```python
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        # 即求topK系列的变种
        # 第K个最大的和第K个最小的，K是数据流的半长度,若为总长度奇数，则多出来的那一个丢在小根堆中
        self.size = 0
        self.maxHeap = [] # 存topK小,
        self.minHeap = [] # 存topK大


    def addNum(self, num: int) -> None:
        self.size += 1 
        # 更新逻辑为，要使得minHeap中存的是大数，maxHeap中存的是小数
        # 此时size为奇数，最终在这个位数的数字【不一定还是它本身】要丢在小根堆，先去大根堆逛一圈，然后看它是否符合要求
        # 逛完大根堆的那个数弹出来加入小根堆
        # 
        if self.size % 2 == 1: # 大根堆一进一出，大小不变，小根堆进1
            heapq.heappush(self.maxHeap,-num)
            e = -heapq.heappop(self.maxHeap)
            heapq.heappush(self.minHeap,e)
        # 先去小顶堆逛一圈
        elif self.size % 2 == 0: # 
            heapq.heappush(self.minHeap,num)
            e = heapq.heappop(self.minHeap)
            heapq.heappush(self.maxHeap,-e)

    def findMedian(self) -> float:
        if self.size % 2 == 1: # 奇数大小，返回
            return self.minHeap[0]
        elif self.size % 2 == 0:
            avg = (self.minHeap[0] - self.maxHeap[0])/2
            return avg
```

# 303. 区域和检索 - 数组不可变

给定一个整数数组  nums，求出数组从索引 i 到 j（i ≤ j）范围内元素的总和，包含 i、j 两点。

实现 NumArray 类：

NumArray(int[] nums) 使用数组 nums 初始化对象
int sumRange(int i, int j) 返回数组 nums 从索引 i 到 j（i ≤ j）范围内元素的总和，包含 i、j 两点（也就是 sum(nums[i], nums[i + 1], ... , nums[j])）

```python
class NumArray:
# 前缀和思想,prefix[i]为不包含当前元素的前缀和
    def __init__(self, nums: List[int]):
        self.prefix = [] # 初始化前缀和
        self.nums = nums # 调用
        temp_sum = 0
        for i in nums: # 填充前缀和
            self.prefix.append(temp_sum)
            temp_sum += i

    def sumRange(self, left: int, right: int) -> int:
        ans = self.prefix[right] - self.prefix[left] + self.nums[right]
        return ans

```

# 304. 二维区域和检索 - 矩阵不可变

给定一个二维矩阵 matrix，以下类型的多个请求：

计算其子矩形范围内元素的总和，该子矩阵的左上角为 (row1, col1) ，右下角为 (row2, col2) 。
实现 NumMatrix 类：

NumMatrix(int[][] matrix) 给定整数矩阵 matrix 进行初始化
int sumRegion(int row1, int col1, int row2, int col2) 返回左上角 (row1, col1) 、右下角 (row2, col2) 的子矩阵的元素总和。

```python
class NumMatrix:
# 观察数据量可知，这一题一定不能使用暴力法
# 而通常这种求和问题需要使用到前缀和的思想
# 那么在创建矩阵时候，创建一个记忆了前缀和的矩阵即可
# 如果查询时候的左上角是0，0，返回记忆矩阵的右下角即可
# 如果查询时候的左上角不是0，0，则返回:记忆矩阵的右下角-上长条-左长条+左上角
    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        self.memo = [[0 for j in range(n)] for i in range(m)]
        temp_sum = 0
        for i in range(len(self.memo[0])): # 填充第一横行
            temp_sum += matrix[0][i]
            self.memo[0][i] = temp_sum
        temp_sum = 0
        for i in range(len(self.memo)): # 填充第一纵列
            temp_sum += matrix[i][0]
            self.memo[i][0] = temp_sum
        # self.memo的填充为 self.memo[i][j] = self.memo[i][j-1]+self.memo[i-1][j] + matrix[i][j] - self.memo[i-1][j-1]
        for i in range(1,m):
            for j in range(1,n):
                self.memo[i][j] = self.memo[i][j-1]+self.memo[i-1][j] + matrix[i][j] - self.memo[i-1][j-1]
        # 此时初始化完成。
        # print(self.memo) 打印检验
        
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        # 如果查询时候的左上角是0，0，返回记忆矩阵的右下角即可
        # 题目保证查询坐标合法，无需考虑非法性判断
        if row1 == col1 == 0:
            return self.memo[row2][col2]
        # 如果查询时候的左上角不是0，0，则返回:记忆矩阵的右下角-上长条-左长条+左上角
        if row1 != 0 and col1 != 0:
            ans = self.memo[row2][col2] - self.memo[row1-1][col2] - self.memo[row2][col1-1] + self.memo[row1-1][col1-1]
            return ans
        # 如果查询时候左上角row坐标是0,只需减去左长条
        if row1 == 0 and col1 != 0:
            ans = self.memo[row2][col2] - self.memo[row2][col1-1]
            return ans
        # 如果查询时候左上角col坐标是0，只需减去上长条
        if row1 != 0 and col1 == 0:
            ans = self.memo[row2][col2] - self.memo[row1-1][col2]
            return ans


```

# 313. 超级丑数

超级丑数 是一个正整数，并满足其所有质因数都出现在质数数组 primes 中。

给你一个整数 n 和一个整数数组 primes ，返回第 n 个 超级丑数 。

题目数据保证第 n 个 超级丑数 在 32-bit 带符号整数范围内。

```python
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        # 默认1是一个超级丑数，dp的思路,需要深刻理解丑数里的多指针,参加264
        # 创建一串数组来存这些质数的位置指针,这些指针都初始指向1
        point = [1 for i in range(len(primes))]
        dp = [1 for i in range(n+1)] # dpn的意思是第n个丑数
        # 质数指针指的下标是dp的下标
        for i in range(2,n+1):
            # arr来考虑每一个质数指针乘以它的对应的质数，取最小的那个
            arr = [primes[k]*dp[point[k]] for k in range(len(primes))]
            dp[i] = min(arr)
            for k in range(len(arr)):
                if dp[i] == arr[k]:
                    point[k] += 1
        # print(dp)
        return dp[-1]
```

```python
# 另一种写法
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        # 创建一个primes数组,primes已经递增，无需排序
        primesPoint = [0 for i in range(len(primes))] # 初始化都指向索引0的位置,里面存的都是索引
        # dp
        dp = [1 for i in range(n)]
        for i in range(1,n): # 
            temp = []
            for p in range(len(primesPoint)): # p指向的是这个质数的索引
                temp.append(dp[primesPoint[p]] * primes[p])
            dp[i] = min(temp)
            for p in range(len(primesPoint)):
                if dp[i] == dp[primesPoint[p]] * primes[p]:
                    primesPoint[p] += 1
        return dp[-1]
```

# 314. 二叉树的垂直遍历

给你一个二叉树的根结点，返回其结点按 **垂直方向**（从上到下，逐列）遍历的结果。

如果两个结点在同一行和列，那么顺序则为 **从左到右**。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        # BFS收集之后， 重新添加进结果中
        bfs_ans = []
        if root == None:
            return []
        queue = [(root,0)]
        while len(queue) != 0:
            new_queue = []
            for cp in queue:
                if cp[0] != None:
                    bfs_ans.append([cp[0].val,cp[1]]) # 把索引加进去
                if cp[0].left != None:
                    new_queue.append((cp[0].left,cp[1]-1))
                if cp[0].right != None:
                    new_queue.append((cp[0].right,cp[1]+1))
            queue = new_queue
        # print(ans) # 此时收集到的ans里面，是很多数组，数组为[值，序号]
        bfs_ans.sort(key = lambda x:x[1]) # 用序号排序
        ans = [] # 收集最终结果
        dic = collections.defaultdict(list)
        for i in bfs_ans:
            dic[i[1]].append(i[0])
        for value in dic.values():
            ans.append(value)
        return ans
```

# 318. 最大单词长度乘积

给定一个字符串数组 words，找到 length(word[i]) * length(word[j]) 的最大值，并且这两个单词不含有公共字母。你可以认为每个单词只包含小写字母。如果不存在这样的两个单词，返回 0。

```python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        # 方法1: 初始化每个字符串所对应的哈希表,表中记录的是元素是否出现
        the_dict = collections.defaultdict(list)
        for w in words:
            record_char = [False for i in range(26)]
            for i in w:
                record_char[ord(i)-ord("a")] = True # 只要出现过，则True
            the_dict[w] = record_char
        longest = 0
        # 之后双层for循环两两检查
        for i in range(len(words)):
            for j in range(i,len(words)):
                lst1 = the_dict[words[i]] # 单词1所对应的字母表
                lst2 = the_dict[words[j]] # 单词2所对应的字母表
                for k in range(26):
                    if lst1[k] == True and lst2[k] == True: # 两个同时存在
                        break # 有相同字符则开启下一轮检查
                    if k == 25 and (lst1[k] and lst2[k]) != True: # 检查到了最后一个，并且两个不同时为True
                        longest = max(longest,len(words[i])*len(words[j]))
        return longest
        
```

# 323. 无向图中连通分量的数目

给定编号从 `0` 到 `n-1` 的 `n` 个节点和一个无向边列表（每条边都是一对节点），请编写一个函数来计算无向图中连通分量的数目。

```python
class UnionFind:
    # quick union 实现
    def __init__(self,size):
        self.root = [i for i in range(size)]

    def find(self,x): 
        while x != self.root[x]:
            x = self.root[x]
        return x
    
    def union(self,x,y):
        rootX = self.find(x) # 找到x的根节点
        rootY = self.find(y) # 找到y的根节点
        if rootX != rootY: # 如果俩节点不相等
            self.root[rootY] = rootX # 把x的根节点赋值给Y，即y并入x

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        # 并查集模板题
        # 给出的edge都十分规整，无需处理
        UF = UnionFind(n) # 构建一个并查集
        for cp in edges: # 连接并查集
            UF.union(cp[0],cp[1])
        the_set = set() # 收集根节点，自动去重
        for i in range(len(UF.root)):
            the_set.add(UF.find(i))
        return len(the_set) # 返回根节点数即可
```

# 325. 和等于 k 的最长子数组长度

给定一个数组 `*nums*` 和一个目标值 `*k*`，找到和等于 *`k`* 的最长连续子数组长度。如果不存在任意一个符合要求的子数组，则返回 `0`。

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        max_length = 0 # 初始化最长长度
        # 前缀和，再两数之和
        pre_fix = set()
        temp_sum = 0
        for index,val in enumerate(nums):
            temp_sum += val
            pre_fix.add((index,temp_sum)) # 元组为索引，值        
        ddt = collections.defaultdict(list) # 字典为 值:索引
        for tp in pre_fix:
            ddt[tp[1]].append(tp[0])
        
        # print(ddt)
        if k in ddt: # 如果里面本身有这个值，则取它的索引 + 1
            max_length = max(ddt[k]) + 1 # 索引 + 1

        # 需要找到两个值的差值为 k
        # k = a1 - a2  ,其中k的正负号不明

        for key in ddt:
            target = k + key #  target为需要的值，key为能凑出来的所有值
            if target in ddt: # 需要的值的索引取最小，key取最大
                # target的索引要大于key的索引
                # print(target,key)
                max_length = max(max_length,(max(ddt[target]) - min(ddt[key])))
        return max_length
        
```

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        max_length = 0
        #  前缀和 + hash
        pre_fix = dict()  # k == 值， v为索引
        pre_fix[0] = 0 # 初始化不包含自身这一项的前缀和
        temp_sum = 0
        for i in range(len(nums)):
            temp_sum += nums[i]
            if temp_sum not in pre_fix: # 只需要记录最早出现的那一个就行
                pre_fix[temp_sum] = i + 1
            taget = temp_sum - k
            if taget in pre_fix:
                max_length = max(max_length,i + 1 - pre_fix[taget])    
        return max_length

```



# 328. 奇偶链表

给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        # 提取奇数链和偶数链
        # 设置奇数位置和偶数位置的指针，方便调整指向
        # 处理空链表
        if head == None:
            return head
        odd = head # 奇数表头
        even = head.next # 偶数表头
        even_head = head.next # 为了链接奇偶分链表，需要存下偶链的第一头
        while odd.next != None and even.next != None:
            odd.next = odd.next.next # 跨两步
            odd = odd.next # 移动odd标记
            even.next = even.next.next # 跨两步
            even = even.next # 移动even标记
        odd.next = even_head
        return head
            
```

# 339. 嵌套列表权重和

这题很坑，需要读懂题意和结构。练习读api的题目

```python
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        # 关键在于读懂题意，递归处理
        # 1. 用含有返回值的递归

        def dfs(nL,level):
            temp = 0
            for e in nL: # e的类型要么是值，要么是一个列表，
                if e.isInteger():
                    temp += e.getInteger() * level 
                else:
                    temp += dfs(e.getList(),level+1) # 所以这里需要e.getList() 而不是直接dfs（e,level+1)
            return temp
        
        return dfs(nestedList,1)
```

# 340. 至多包含 K 个不同字符的最长子串

给定一个字符串 ***`s`\*** ，找出 **至多** 包含 *`k`* 个不同字符的最长子串 ***T\***。

```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        # 传统滑动窗口
        left = 0
        right = 0
        window = collections.defaultdict(int)
        max_length = 0
        size = 0 # 其实可以用left和right计算出来，但是懒得算了
        while right < len(s):
            add_char = s[right]
            window[add_char] += 1
            right += 1
            size += 1
            while left < right and len(window) > k: # 收缩
                delete_char = s[left]
                window[delete_char] -= 1
                if window[delete_char] == 0: 
                    del window[delete_char]
                left += 1
                size -= 1
            max_length = max(max_length,size) # 注意收集结果需要在这里收集
        return max_length

```

# 341. 扁平化嵌套列表迭代器

给你一个嵌套的整数列表 nestedList 。每个元素要么是一个整数，要么是一个列表；该列表的元素也可能是整数或者是其他列表。请你实现一个迭代器将其扁平化，使之能够遍历这个列表中的所有整数。

实现扁平迭代器类 NestedIterator ：

NestedIterator(List<NestedInteger> nestedList) 用嵌套列表 nestedList 初始化迭代器。
int next() 返回嵌套列表的下一个整数。
boolean hasNext() 如果仍然存在待迭代的整数，返回 true ；否则，返回 false 。

```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        # 方法1:使用直接初始化的方法,不符合迭代器的思想
        self.lst = []
        
        # 这个的类型是list里面一堆NestedInteger的意思。。。
        def dfs(nL):
            if nL == []:
                return 
            for ls in nL:
                if ls.isInteger() == True:
                    self.lst.append(ls.getInteger())
                elif ls.isInteger() == False:
                    dfs(ls.getList())

        dfs(nestedList)
        self.p = 0
            
    def next(self) -> int:
        val = self.lst[self.p]
        self.p += 1
        return val
    
    def hasNext(self) -> bool:
        return self.p < len(self.lst)
```

```

```



# 343. 整数拆分

给定一个正整数 *n*，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        # dp
        # dp[i]的意思是，当数值为i的时候的乘积最大值
        # 那么dp[i] = max(group(j * dp[i-j])) 其中j<=i-1且大于1
        # 上面这个思路是有问题的，可以选择不截断dp[i-j]的部分，可能它值更大
        # 即 dp[i] = max(group(j * dp[i-j])) 或者是 max(group(j * (i-j)))
        # 所以改写成dp[i] = max(group(j * max(dp[i-j],i-j)))
        # 申请空间为n+1长度,显然dp[1] = 1 ,dp[0]无所谓
        if n == 2: return 1
        if n == 3: return 2
        # 否则，进入递归
        dp = [1 for i in range(n+1)]
        # 初始化一些计算可得的条件值
        dp[1] = 1
        dp[2] = 1
        dp[3] = 2

        for i in range(4,n+1):
            group = []
            for j in range(2,i): 
                temp_max = max((i-j),dp[i-j])
                group.append(j*temp_max)
            if group != []:
                dp[i] = max(group)
        return dp[-1]
```

# 347. 前 K 个高频元素

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 堆化，但是是以字典k-v对建堆
        counters_dict = collections.Counter(nums) # 先建立k-v
        # 以v来建堆，获取的是topK大
        # 维持堆大小为k，由于是小根堆，所以里面存的都是频率小的数，每次挤进来都会把最小的挤出去
        # 扫完之后，那些不满足条件的小的数都被挤出去了，留下来的自然是符合条件的k高频
        min_heap = []
        for key,freq in counters_dict.items():
            heapq.heappush(min_heap,(freq,key))
            if len(min_heap) > k: # 超过堆大小则弹出
                heapq.heappop(min_heap)
        # 结果为min_heap中的key
        ans = []
        for i in min_heap:
            ans.append(i[1])
        return ans # 不要求topK这一部分排序

```

# 357. 计算各个位数不同的数字个数

给定一个**非负**整数 n，计算各位数字都不同的数字 x 的个数，其中 0 ≤ x < 10**n 。

```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        # 只考虑闭区间【0，8】
        if n <= 1:
            return 10**n
        # 排列组合问题
        dp = [0 for i in range(n+1)]
        dp[0] = 0
        dp[1] = 10
        # 最终返回的是sum(dp)
        # dp[i]的含义是n位数里各个位置不相同的数目
        # 为C10i*i! - C9(i-1)*(i-1)!
        for i in range(2,n+1):
            dp[i] = self.calc_combineNum(10,i)*self.factorial(i) - self.calc_combineNum(9,i-1)*self.factorial(i-1)
        return sum(dp)
    # python中math库这俩都有。
    def calc_combineNum(self,a,b): # C a,b 例如 C 10 2 = 45 # 手写组合算法
        up = 1 # 分子
        for i in range(b):
            up *= (a-i)
        down = 1 # 分母
        for i in range(1,b+1):
            down *= i
        return up//down
    
    def factorial(self,t):  #手写阶乘算法
        ans = 1
        for i in range(1,t+1):
            ans *= i
        return ans
```

# 359. 日志速率限制器

请你设计一个日志系统，可以流式接收消息以及它的时间戳。每条 不重复 的消息最多只能每 10 秒打印一次。也就是说，如果在时间戳 t 打印某条消息，那么相同内容的消息直到时间戳变为 t + 10 之前都不会被打印。

所有消息都按时间顺序发送。多条消息可能到达同一时间戳。

实现 Logger 类：

Logger() 初始化 logger 对象
bool shouldPrintMessage(int timestamp, string message) 如果这条消息 message 在给定的时间戳 timestamp 应该被打印出来，则返回 true ，否则请返回 false 。

```python
class Logger:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # 哈希表存储单词，并且存储这个单词这次的时间，打印的时候更新
        self.hash_table = dict()


    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """
        if self.hash_table.get(message) == None: # 之前没有出现过
            self.hash_table[message] = timestamp
            return True
        elif self.hash_table[message] + 10 <= timestamp:
            self.hash_table[message] = timestamp
            return True
        elif self.hash_table[message] + 10 > timestamp:
            return False

```

# 360. 有序转化数组

给你一个已经 排好序 的整数数组 nums 和整数 a、b、c。对于数组中的每一个数 x，计算函数值 f(x) = ax2 + bx + c，请将函数值产生的数组返回。

要注意，返回的这个数组必须按照 升序排列，并且我们所期望的解法时间复杂度为 O(n)。

```python
class Solution:
    def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
        def func(x): 
            return a*x**2 + b*x + c
        if len(nums) == 0:
            return []
        left = 0
        right = len(nums) - 1
        ans = [0 for i in range(len(nums))] # 注意先申请ans数组的做法！！！
        # 否则很难写
        if a >= 0: # 分情况讨论，a大于0，大的值一定在数组端点，ans从右边往左填充
            index = len(ans) - 1
            while left <= right:
                if func(nums[left]) <= func(nums[right]):
                    ans[index] = func(nums[right])
                    right -= 1
                    index -= 1
                else:
                    ans[index] = func(nums[left])
                    left += 1
                    index -= 1
        elif a < 0: # a < 0 ，小的值一定在端点，从小到大填充ans
            index = 0
            while left <= right:
                if func(nums[left]) <= func(nums[right]):
                    ans[index] = func(nums[left])
                    left += 1
                    index += 1
                else:
                    ans[index] = func(nums[right])
                    right -= 1
                    index += 1
        return ans
       
```

# 364. 加权嵌套序列和 II

和339一样，需要读懂题意，然后进行合理的api使用

两次dfs

```python
class Solution:
    def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
        # 和339一样异曲同工，稍微需要变换一下
        # 先找到最大深度
        maxDp = 0
        def findDepth(nL,depth): # 这个也是dfs找深度
            nonlocal maxDp
            maxDp = max(maxDp,depth)
            for e in nL:
                if e.isInteger():
                    pass
                else:
                    findDepth(e.getList(),depth+1)
        findDepth(nestedList,1)

        def dfs(nL,level,maxDepth): # 这个是dfs获取值
            temp = 0
            for e in nL:
                if e.isInteger():
                    temp += e.getInteger() * (maxDepth-level+1)
                else:
                    temp += dfs(e.getList(),level+1,maxDepth)
            return temp
       
        return dfs(nestedList,1,maxDp)
```

# 366. 寻找二叉树的叶子节点

给你一棵二叉树，请按以下要求的顺序收集它的全部节点：

1. 依次从左到右，每次收集并删除所有的叶子节点
2. 重复如上过程直到整棵树为空

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        # 模拟
        # 注意处理树只有一个节点的情况
        ans = []
        path = []
        def dfs(node,parent_node): # 加入了记忆父节点的dfs
            if node == None:
                return
            if node.left == None and node.right == None: # 叶子节点，收集
                path.append(node.val)
                if parent_node != None and node == parent_node.left: # 是左叶子，把左边置空
                    parent_node.left = None
                if parent_node != None and node == parent_node.right:# 是右叶子，把右边置空
                    parent_node.right = None
            parent_node = node
            dfs(node.left,parent_node)
            dfs(node.right,parent_node)
        while root != None :
            if root.left == None and root.right == None:
                break
            path = []
            dfs(root,None) # 根节点的父节点设置为None
            ans.append(path[:])

        path = [root.val] # 还剩下根节点未处理
        ans.append(path[:])
        root = None # 清空树
        return ans
```

# 369. 给单链表加一

用一个 非空 单链表来表示一个非负整数，然后将这个整数加一。

你可以假设这个整数除了 0 本身，没有任何前导的 0。

这个整数的各个数位按照 高位在链表头部、低位在链表尾部 的顺序排列。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def __init__(self):
        self.car = 0
        
    def plusOne(self, head: ListNode) -> ListNode:
        # 用递归来做，低位置在链表尾部
        dummy = ListNode(0) # 加个哑节点，万一首位有进位需要用到
        dummy.next = head
        self.helper(dummy)
        if dummy.val == 1:
            return dummy
        else:
            return dummy.next

    def helper(self,head): # 这个是需要用到的递归函数
        # 注意递归处理层级，使用图方便理解
        if head == None:
            return 
        if head.next == None:
            if head.val < 9:
                head.val += 1
                self.car = 0
            elif head.val == 9:
                head.val = 0
                self.car = 1
            return head
        head.next = self.helper(head.next) # 这一行不能放在最后面，这是处理后面的节点
        # 顺序是：先处理后面的节点，再处理自身节点
        if self.car < 1:
            pass
        elif self.car == 1 and head.val < 9:
            head.val += 1
            self.car = 0
        elif self.car == 1 and head.val == 9:
            head.val = 0
            self.car = 1
        return head

   
```

# 371. 两整数之和

**不使用**运算符 `+` 和 `-` ，计算两整数 `a` 、`b` 之和。

```java
class Solution {
    // 位运算，
    // 分离成进位部分，和非进位部分
    // 作循环，直到非进位部分都归零
    public int getSum(int a, int b) {
        while (b != 0){
            int carry = (a & b) << 1;
            a = a ^ b;
            b = carry;
        }
        return a;
    }
}
```

# 373. 查找和最小的K对数字

给定两个以升序排列的整数数组 nums1 和 nums2 , 以及一个整数 k 。

定义一对值 (u,v)，其中第一个元素来自 nums1，第二个元素来自 nums2 。

请找到和最小的 k 个数对 (u1,v1),  (u2,v2)  ...  (uk,vk) 。

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        # 堆
        if k > len(nums1) * len(nums2): # 越界则锁定为最多数量
            k = len(nums1) * len(nums2)
        min_heap = [] # 小顶堆
        # heap逻辑为从左比到右，先比较和，然后j大的放前面所以设计键为(i+j,j,[i,j])
        for i in (nums1[:k]): # 前k个数一定来自于两组数字的前k*k个，切片无需担心越界
            for j in (nums2[:k]):
                heapq.heappush(min_heap,(i+j,j,[i,j]))    
        ans = [] # 收集结果
        while len(ans) != k:
            ans.append(heapq.heappop(min_heap)[2])
        return ans
```

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        # 堆
        if k > len(nums1) * len(nums2): # 越界则锁定为最多数量
            k = len(nums1) * len(nums2)
        min_heap = [] # 大顶堆
        lst = deque()
        # heap逻辑为从左比到右，先比较和，然后j大的放前面所以设计键为[-(i+j),-(j),[i,j]]
        for i in (nums1[:k]): # 前k个数一定来自于两组数字的前k*k个，切片无需担心越界
            for j in (nums2[:k]):
                lst.append([-(i+j),-(j),[i,j]])
        for tuples in lst:
            heapq.heappush(min_heap,tuples)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        ans = [] # 收集结果
        while len(ans) != k:
            ans.append(heapq.heappop(min_heap)[2])
        return ans[::-1]
```

# 378. 有序矩阵中第 K 小的元素

给你一个 n x n 矩阵 matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        # 其一定在前k行前k列，用k*k过筛，
        # 找第k小，用大顶堆
        seive = []
        for i in matrix[:k]: # 切片无需担心越界
            for num in i[:k]:
                seive.append(-num)
        max_heap = [] # 维护大顶堆大小为k即可
        for i in seive:
            heapq.heappush(max_heap,i)
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        return -max_heap[0]
        
```

# 379. 电话目录管理系统

设计一个电话目录管理系统，让它支持以下功能：

get: 分配给用户一个未被使用的电话号码，获取失败请返回 -1
check: 检查指定的电话号码是否被使用
release: 释放掉一个电话号码，使其能够重新被分配

```python
class PhoneDirectory:

    def __init__(self, maxNumbers: int):
        """
        Initialize your data structure here
        @param maxNumbers - The maximum numbers that can be stored in the phone directory.
        """
        self.free = set()
        self.used = set()
        for i in range(maxNumbers):
            self.free.add(i)
        self.maxsize = maxNumbers

    def get(self) -> int:
        """
        Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available.
        """
        if len(self.free) != 0:
            e = self.free.pop() # 如果集合非空，弹出
            self.used.add(e) # 将弹出的元素加入使用中
            return e
        return -1

    def check(self, number: int) -> bool:
        """
        Check if a number is available or not.
        """
        return True if number in self.free else False # 如果未被使用，返回True,否则返回False

    def release(self, number: int) -> None:
        """
        Recycle or release a number.
        """
        if number in self.used:
            self.used.remove(number) # 注意，remove没有返回值
            self.free.add(number)

```

# 380. O(1) 时间插入、删除和获取随机元素

设计一个支持在平均 时间复杂度 O(1) 下，执行以下操作的数据结构。

insert(val)：当元素 val 不存在时，向集合中插入该项。
remove(val)：元素 val 存在时，从集合中移除该项。
getRandom：随机返回现有集合中的一项。每个元素应该有相同的概率被返回。

```python
class RandomizedSet:
# 数组 + 哈希表
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.arr = [] # 数组内存的都是值
        self.hashtable = {}

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val not in self.hashtable:
            self.hashtable[val] = len(self.arr) # k-v对为 元素值:数组索引
            self.arr.append(val)
            # print("after insert ",self.hashtable,self.arr)
            return True
        else:
            return False

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.hashtable: # 如果它在哈希表中
            index = self.hashtable[val] # 找到他的索引
            # 将其与最后一个元素交换，并且需要得到最后一个元素信息
            last_element = self.arr[-1]
            self.arr[index] = last_element
            # 更新哈希表
            self.hashtable[last_element] = index
            delete_val = self.arr.pop() # 数组弹出
            del self.hashtable[val] # 删除这个值对应的键
            # print("after remove ",self.hashtable,self.arr)
            return True
        return False


    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        i = random.randint(0,len(self.arr)-1)
        return self.arr[i]

```

# 386. 字典序排数

给定一个整数 n, 返回从 1 到 n 的字典顺序。

例如，

给定 n =1 3，返回 [1,10,11,12,13,2,3,4,5,6,7,8,9] 。

请尽可能的优化算法的时间复杂度和空间复杂度。 输入的数据 n 小于等于 5,000,000。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(10)]
        self.isValid = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,i):
        node = self.root
        for index in str(i):
            if node.children[int(index)] == None:
                node.children[int(index)] = TrieNode()
            node = node.children[int(index)]
        node.isValid = True
    
    def insertAll(self,lst):
        for i in lst:
            self.insert(i)
    
    def collect(self): # 收集方法
        node = self.root
        node.isValid = True
        path = []
        ans = []
        def dfs(path,node):
            if node == None:
                return
            if len(path) != 0: # 每次都收集
                ans.append(int("".join(path[:])))             
            for c in range(10):
                path.append(str(c))                
                dfs(path,node.children[c])
                path.pop()

        dfs(path,self.root) # 调用搜索
        return ans
                    

class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        # 字典树
        lst = [str(i) for i in range(1,n+1)] # 创建
        tree = Trie()
        tree.insertAll(lst) # 全插入
        ans = tree.collect() # 收集
        return ans

```

# 392. 判断子序列

给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

进阶：

如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # 双排指针
        # s是应该是较短的那一个
        if len(s) > len(t):
            return False    
        if len(s) == 0: # s为空的时候
            return True    
        p1 = 0
        p2 = 0
        while p1 < len(s):
            while p2 < len(t) and t[p2] != s[p1]:
                p2 += 1
            if p2 == len(t): # 越界了都还没有匹配上
                return False # 待会儿处理
            if t[p2] == s[p1]: # 匹配上了则两者各自+1
                p1 += 1
                p2 += 1
            if p1 == len(s): # 检查s是否扫完了，扫完了则True
                return True
        return False
                
```

# 394. 字符串解码

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

```python
class Solution:
    def decodeString(self, s: str) -> str:
        # 遇到数字压入数字栈，遇到右括号出数字栈
        # 注意数字可能有几位
        num_stack = []
        # 处理数字
        stack_arr = [[]]
        pStack = 0
        ans = []
        p = 0
        while p < len(s):
            if s[p].isdigit(): # 处理多位数
                temp_num = s[p]
                while s[p+1].isdigit(): # 如果下一位还是数字，则加入
                    temp_num += s[p+1]
                    p += 1
                p += 1
                num_stack.append(int(temp_num)) # 收集完全部的数字，丢进数字栈
            elif s[p] == '[':
                stack_arr.append([])
                pStack += 1
                p += 1
            elif s[p] == ']':
                stack_arr[pStack-1] += (stack_arr[pStack]*num_stack.pop())
                stack_arr[pStack] = []
                pStack -= 1
                p += 1
            else:
                stack_arr[pStack].append(s[p])
                p += 1
        return ''.join(stack_arr[0])
```

# 414. 第三大的数

给你一个非空数组，返回此数组中 **第三大的数** 。如果不存在，则返回数组中最大的数。

```python
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        # 利用On的时间复杂度完成，找distinct
        # 会修改原数组
        max1 = max(nums)
        for p in range(len(nums)):
            if nums[p] == max1:  # 将它赋值成-0xffffffff
                nums[p] = -0xffffffff
        max2 = max(nums)
        if max2 == -0xffffffff:
            return max1
        for p in range(len(nums)):
            if nums[p] == max2: # 也将它赋值成-0xffffffff
                nums[p] = -0xffffffff
        max3 = max(nums)
        if max3 == -0xffffffff:
            return max1
        else:
            return max3
```

# 422. 有效的单词方块

给你一个单词序列，判断其是否形成了一个有效的单词方块。

有效的单词方块是指此由单词序列组成的文字方块的 第 k 行 和 第 k 列 (0 ≤ k < max(行数, 列数)) 所显示的字符串完全相同。

注意：

给定的单词数大于等于 1 且不超过 500。
单词长度大于等于 1 且不超过 500。
每个单词只包含小写英文字母 a-z。

```python
class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        # 已经限定了每个单词只包含a-zz
        # 方法1： 补全每一行,先找到指定的kk
        k = -1
        for w in words:
            k = max(k,len(w))
        k = max(k,len(words))
        target_length = k
        while len(words) < k: # 补全竖排
            words.append("")
        for i in range(len(words)):
            need_length = target_length - len(words[i])
            if need_length != 0:
                words[i] += '1'*need_length # 补全横排
        # 开始检查
        for i in range(len(words)):
            for j in range(i,len(words)):
                if words[i][j] != words[j][i]:
                    return False

        return True 
```

# 430. 扁平化多级双向链表

多级双向链表中，除了指向下一个节点和前一个节点指针之外，它还有一个子链表指针，可能指向单独的双向链表。这些子列表也可能会有一个或多个自己的子项，依此类推，生成多级数据结构，如下面的示例所示。

给你位于列表第一级的头节点，请你扁平化列表，使所有结点出现在单级双链表中。

```python
# 纯力扣才能采用的题
"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

class Solution:
    # 思路，使用递归完成
    # 递归思路：对当前节点而言，当前节点的next = recur
    def flatten(self, head: 'Node') -> 'Node':
        # 看图可以发现，如果有child，需要处理child 
        if head == None:
            return head
        cur = head
        # 开始走主链
        while cur != None:
            if cur.child != None : # 需要复杂处理的
                temp = cur.next # 暂存下一个节点
                # 需要扁平化孩子，所以递归调用
                flat = self.flatten(cur.child)
                # 接上cur和扁平后的链表
                cur.next = flat
                flat.prev = cur
                cur.child = None # 这一题规定扁平化完毕，要置child指向None
                # 找到这一条拉链的最后一个节点，让它和temp链接
                tail = cur
                while tail.next != None:
                    tail = tail.next
                # 如果temp不为None,进行链接
                if temp != None: 
                    tail.next = temp # 链接
                    temp.prev = tail
                cur = temp # cur走到主链下一个位置
            elif cur.child == None: # 不需要复杂处理，没有分支链
                cur = cur.next
        return head
# 大神解法：脖子左歪45度，多级链表变成了二叉树，输出先序即可
# 这题还是child当左子树，next当右子树，然后先序遍历方便
```

# 441. 排列硬币

你总共有 n 枚硬币，你需要将它们摆成一个阶梯形状，第 k 行就必须正好有 k 枚硬币。

给定一个数字 n，找出可形成完整阶梯行的总行数。

n 是一个非负整数，并且在32位有符号整型的范围内。

```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        # 即为找到(1+k)*k/2 <= n 中k可以取到的最大值
        # 二分闭区间查找
        left = 1
        right = n
        while left <= right:
            mid = (left+right)//2
            if (1+mid)*mid//2 <= n and (1+mid)*mid//2 > n:
                return mid
            elif (1+mid)*mid//2 <= n: # 数值偏小，需要变大
                left = mid + 1
            elif (1+mid)*mid//2 > n: # 数值偏大，需要变小
                right = mid - 1
        return right
```

# 443. 压缩字符串

给你一个字符数组 chars ，请使用下述算法压缩：

从一个空字符串 s 开始。对于 chars 中的每组 连续重复字符 ：

如果这一组长度为 1 ，则将字符追加到 s 中。
否则，需要向 s 追加字符，后跟这一组的长度。
压缩后得到的字符串 s 不应该直接返回 ，需要转储到字符数组 chars 中。需要注意的是，如果组长度为 10 或 10 以上，则在 chars 数组中会被拆分为多个字符。

请在 修改完输入数组后 ，返回该数组的新长度。

你必须设计并实现一个只使用常量额外空间的算法来解决此问题。

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        # 非常量空间方法
        # 常量级方法使用扫描指针+填充指针
        buffer = []
        for ch in chars:
            if len(buffer) == 0:
                buffer.append([ch,1])
            elif buffer[-1][0] == ch:
                buffer[-1][1] += 1
            elif buffer[-1][0] != ch:
                buffer.append([ch,1])
        s = ""
        for tp in buffer:
            if tp[1] == 1:
                s += tp[0]
            elif tp[1] != 1:
                s += tp[0]+str(tp[1])        
        for i in range(len(s)):  # 注意要修改
            chars[i] = s[i]
        return len(s)
```

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        # 常量空间解法不采取buffer
        p1 = 0 # 扫描指针
        p2 = 0 # 填充指针
        the_char = chars[0] # 额外空间1
        times = 0 # 额外空间1
        while p1 < len(chars):
            if chars[p1] == the_char:
                times += 1
                p1 += 1
            elif chars[p1] != the_char:
                if times == 1:
                    chars[p2] = the_char
                    p2 += 1
                elif times != 1:
                    need  = str(times) # 次数字符化
                    chars[p2] = the_char
                    p2 += 1
                    pt = 0 # 指向字符化后的次数
                    for t in need:
                        chars[p2] = need[pt]
                        p2 += 1
                        pt += 1
                the_char = chars[p1] # 重置
                times = 0
        # 下面这一段是擦屁股用的。。。因为最后一类字符没有收集进答案
        if times == 1:
            chars[p2] = the_char
            p2 += 1
        elif times != 1:
            need  = str(times)
            chars[p2] = the_char
            p2 += 1
            pt = 0
            for t in need:
                chars[p2] = need[pt]
                p2 += 1
                pt += 1
        return p2
```



# 447. 回旋镖的数量

给定平面上 n 对 互不相同 的点 points ，其中 points[i] = [xi, yi] 。回旋镖 是由点 (i, j, k) 表示的元组 ，其中 i 和 j 之间的距离和 i 和 k 之间的距离相等（需要考虑元组的顺序）。

返回平面上所有回旋镖的数量。

```python
class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        # 求到这个点到所有不同距离数，每次选的点作为中心点
        ans = 0 # 收集结果
        for center in points:
            temp_dict = collections.defaultdict(int) # 记录每个距离的点的个数
            for corner in points: # 有一个为0的不影响结果，不加if筛了
                the_distance = self.calc_distance(center,corner)
                temp_dict[the_distance] += 1
            # 所有值大于2的都要两两计算,算排列数
            for value in temp_dict.values():
                if value >= 2:
                    ans += (value * (value - 1))
        return ans

    def calc_distance(self,coord1,coord2): # 计算出平方即可
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

```

# 450. 删除二叉搜索树中的节点

给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

首先找到需要删除的节点；
如果找到了，删除它。
说明： 要求算法时间复杂度为 O(h)，h 为树的高度。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.thePack = []

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        # 1. 没有该节点，不做任何处理
        # 2. 如果该节点是叶子节点，直接删去
        # 3. 如果该节点只有一个孩子节点，把它的父节点和孩子节点连接
        # 4. 如果该节点有两个孩子节点，用它的中序前驱代替自己,再删去
        self.find(root,None,key) 
        node = self.thePack[0][0] # node是要删除的节点
        theparent = self.thePack[0][1] 
        if node == None: # 1.
            return root 
        elif node.left == None and node.right == None: # 2.
            if theparent == None: # 删除的是根节点且全树只有一个节点
                return None
            if theparent.left == node:
                theparent.left = None
            elif theparent.right == node:
                theparent.right = None
        elif node.left == None and node.right != None: # 3.它只有右子树
            if theparent == None: # 是根节点，返回它的右子树
                return node.right 
            if theparent.left == node:
                theparent.left = node.right
            elif theparent.right == node:
                theparent.right = node.right
        elif node.left != None and node.right == None: # 3.它只有左子树
            if theparent == None: # 是根节点，返回它的左子树
                return node.left
            if theparent.left == node:
                theparent.left = node.left
            elif theparent.right == node:
                theparent.right = node.left
        elif node.left != None and node.right != None: # 4. 这个情况极其复杂
            prev_node = self.find_prev(root,None,key)[0] # 找到它的前驱节点
            prev_node_parent = self.find_prev(root,None,key)[1] # 它的前驱节点的父节点
            # print(prev_node)
            prev_node.val,node.val = node.val,prev_node.val # 交换前驱节点的值和这个节点的值
            if prev_node_parent.left != None and prev_node_parent.left.val == key:
                # pre_node一定最多只有一个孩子，它为空也可以传值
                prev_node_parent.left = prev_node.left
            elif prev_node_parent.right != None and prev_node_parent.right.val == key:
                prev_node_parent.right = prev_node.left         
        return root
            
    def find(self,root,parent,val): # 找到这个节点，和这个节点的父母
        if root == None:
            self.thePack.append((root,parent))
            return
        elif val == root.val:
            self.thePack.append((root,parent))
            return 
        elif val > root.val:
            parent = root
            return self.find(root.right,parent,val)
        elif val < root.val:
            parent = root
            return self.find(root.left,parent,val)

    def find_prev(self,root,parent,val): # 一个节点的中序前驱是，先到它的左孩子，之后尽可能的往右走
        self.find(root,None,val)
        the_node,theparent = self.thePack[1][0],self.thePack[1][1]
        if the_node.left != None:
            parent = the_node
            the_node = the_node.left
        while the_node.right != None:
            parent = the_node
            the_node = the_node.right
        return the_node,parent
```

# 454. 四数相加 II

给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。

为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -2^28 到 2^28 - 1 之间，最终结果不会超过 2^31 - 1 。

```python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        # 哈希表。利用前两个做两数之和，后两个做两数之和
        # 再两数之和
        n = len(nums1)
        # 数组长度相等
        hash1 = collections.defaultdict(int)
        for i in range(n): # 穷举排列
            for j in range(n):
                hash1[nums1[i]+nums2[j]] += 1
        hash2 = collections.defaultdict(int)
        for i in range(n): # 穷举排列
            for j in range(n):
                hash2[nums3[i]+nums4[j]] += 1
        ans = 0 # 收集答案
        for aim in hash1: # 两数之和
            if -aim in hash2:
                ans += hash1[aim]*hash2[-aim]
        return ans
```

# 457. 环形数组是否存在循环

存在一个不含 0 的 环形 数组 nums ，每个 nums[i] 都表示位于下标 i 的角色应该向前或向后移动的下标个数：

如果 nums[i] 是正数，向前 移动 nums[i] 步
如果 nums[i] 是负数，向后 移动 nums[i] 步
因为数组是 环形 的，所以可以假设从最后一个元素向前移动一步会到达第一个元素，而第一个元素向后移动一步会到达最后一个元素。

数组中的 循环 由长度为 k 的下标序列 seq ：

遵循上述移动规则将导致重复下标序列 seq[0] -> seq[1] -> ... -> seq[k - 1] -> seq[0] -> ...
所有 nums[seq[j]] 应当不是 全正 就是 全负
k > 1
如果 nums 中存在循环，返回 true ；否则，返回 false 。

```
class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        # 使用一个记忆数组
        for i in range(len(nums)): # 扫描全数组
            start = i # 设置起点
            now = i # 设置当前状态
            path = [] # 收集路径
            symbol = nums[i] # 初值符号
            visited = [False for i in range(len(nums))] # 每次扫描重置记忆数组
            while visited[now] != True:
                # print("now index",now) # 检查用
                visited[now] = True  # 扫描过后把本次扫描的标记置True
                path.append(now)
                if symbol * nums[now] > 0: # 当前值和符号一样，继续走
                    now = (now + nums[now]) % len(nums)
                elif symbol * nums[now] < 0: # 当前值和符号不一样，停，开始下一轮for扫描
                    break
            if now == start and visited[start] == True: # 首尾相同，收集路径答案，大于3即可以返回
                path.append(now)
                if len(path) >= 3:
                    return True
        return False # 遍历完都没有，返回False

```

# 459. 重复的子字符串

给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        # 求它的所有因数,以因数切片,因数需要去掉本身，长度为1不可
        if len(s) == 1:
            return False
        lst = []
        n = len(s)
        for i in range(1,int(sqrt(n))+1):
            if n % i == 0:
                lst.append(i)
                lst.append(n//i)
        lst.remove(n) # 去掉本身
        # 处理之后以因数切片
        for sliceNum in lst:
            sample = s[:sliceNum]
            count = 0
            for every_slice in range(0,len(s),sliceNum):
                if s[every_slice:every_slice+sliceNum] != sample:
                    break
                else:
                    count += 1
            if count == len(s)//sliceNum:
                return True
        return False
```

# 487. 最大连续1的个数 II

给定一个二进制数组，你可以最多将 1 个 0 翻转为 1，找出其中最大连续 1 的个数。

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        # 滑动窗口，定义一个允许的参数
        # 传统写法，right还是在收缩之前更新的写法
        left = 0
        right = 0
        size = 0
        max_size = 0
        count_zero = 0
        while right < len(nums):
            add = nums[right]
            right += 1 # 
            if add != 0:
                size += 1
                max_size = max(max_size,size)
            elif add == 0:
                size += 1
                count_zero += 1
                if count_zero <= 1:
                    max_size = max(max_size,size)
            while left < right and count_zero > 1: # 这一行注意了
                delete = nums[left]
                if delete == 0:
                    count_zero -= 1
                left += 1
                size -= 1
        return max_size
```

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        # 滑动窗口，定义一个允许的参数
        # 注意这个更新策略，遇到0都加进去，而不是考虑这个0是否是第二个0
        # 考虑这个0是否是第二个0的写法很难
        left = 0
        right = 0
        size = 0
        max_size = 0
        extra = 0 # 默认没有激活
        while right < len(nums):
            add_char = nums[right]
            # 注意：这个right的更新放在了情况里：
            if add_char != 0: # 合法情况1
                size += 1 
                max_size = max(max_size,size)
            elif add_char == 0: # 改变一次
                size += 1
                extra += 1
                if extra <= 1:
                    max_size = max(max_size,size)
            while left < right and extra > 1 and add_char == 0: # 收缩到不再消耗额外的次数为止
                # print("left = ",left,"right = ",right)
                delelet_char = nums[left]
                if delelet_char == 0:
                    extra -= 1
                left += 1
                size -= 1
            right += 1 # 注意right放在这里了
        return max_size

```

# 491. 递增子序列

给你一个整数数组 nums ，找出并返回所有该数组中不同的递增子序列，递增子序列中 至少有两个元素 。你可以按 任意顺序 返回答案。

数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        # 回溯
        self.ans = []
        path = []

        def backtracking(path,nums,index):
            if len(path) >= 2:
                self.ans.append(path[:])
            if index == len(nums):
                return    
            # 只能在选过的数字之后选
            for i in range(len(nums)):
                if i >= index:
                    if len(path) == 0 or nums[i] >= path[-1]:
                        path.append(nums[i])
                        backtracking(path,nums,i+1)
                        path.pop()

        backtracking(path,nums,0)
        the_set = set() # 去重复
        final = []
        for i in self.ans:
            if tuple(i) not in the_set:
                the_set.add(tuple(i))
                final.append(i)

        return final
```

# 496. 下一个更大元素 I

给你两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。

请你找出 nums1 中每个元素在 nums2 中的下一个比其大的值。

nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 单调栈问题,nums1是nums2的子集，
        # 直接扫描全部的nums2，看每个位置的下一个更大元素，然后填充答案数组
        # 找下一个更大元素，使用的单调栈需要是单调递减栈，遇到大于的数的时候特殊处理，进一步收集结果
        # 列表中没有重复元素
        ans = [-1 for i in range(len(nums1))]
        stack = []
        dic = dict() # 使用字典找到并且填充
        p = 0
        while p < len(nums2):
            if len(stack) == 0: # 栈空直接加入
                stack.append(nums2[p]) # 加入的是数值
                p += 1
                continue
            if nums2[p] > stack[-1]: # 遇到了需要特殊处理的情况
                while stack != [] and nums2[p] > stack[-1]:
                    e = stack.pop() # pop的是数值,e找到了下一个更大元素
                    dic[e] = nums2[p] # 加入字典，数值e找到了下一个更大元素值             
            stack.append(nums2[p])
            p += 1
        for i in range(len(nums1)):
            if dic.get(nums1[i]): # 如果有对应的键
                ans[i] = dic[nums1[i]] # 填充
            # 否则不填充
        return ans 
```

# 498. 对角线遍历

给定一个含有 M x N 个元素的矩阵（M 行，N 列），请以对角线遍历的顺序返回这个矩阵中的所有元素，对角线遍历如下图所示。

输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]

输出:  [1,2,4,7,5,3,6,8,9]

解释:

```python
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        # 收集每一条往右上走的
        # 根据奇偶翻转
        # 最终加入答案数组
        # mat[i][j] 表示第i行第j列
        ans = []
        # 闭区间边界
        right_bound = len(mat[0])-1 # 有几列
        down_bound = len(mat)-1 # 有几行
        def get_line (i,j): #对第一列的首元素进行取对角线,对最后一行取对角线
            temp_lst = []
            while 0<=i<=down_bound and 0<=j<=right_bound:
                temp_lst.append(mat[i][j])
                i -= 1
                j += 1
            return temp_lst
        count = 0
        for k in range(down_bound): # 第一列首元素，不包括最后一位
            ans += get_line(k,0)[::(-1)**count]
            count += 1
        for k in range(right_bound+1): # 最后一行元素
            ans += get_line(down_bound,k)[::(-1)**count]
            count += 1
        return ans
        
```

# 503. 下一个更大元素 II

给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        ans = [-1 for i in range(len(nums))] # 收集答案
        # 设置单调递减栈，每次遇到大元素的时候进行特殊处理
        # 循环可以直接把nums 拼接成*2的
        n = len(nums)
        nums = nums + nums
        stack = []
        p = 0
        while p < len(nums):
            if len(stack) == 0:
                stack.append([p,nums[p]]) # 加入的是【索引，数值】
                p += 1
                continue
            if nums[p] > stack[-1][1]:
                while stack != [] and nums[p] > stack[-1][1]:
                    e = stack.pop() # 找到了原来栈顶的下一个更大位置
                    if e[0] < n: # 复制之后的数组无需更新
                        ans[e[0]] = nums[p]
            stack.append([p,nums[p]])
            p += 1
        return ans
```

# 510. 二叉搜索树中的中序后继 II

给定一棵二叉搜索树和其中的一个节点 node ，找到该节点在树中的中序后继。如果节点没有中序后继，请返回 null 。

一个节点 node 的中序后继是键值比 node.val 大所有的节点中键值最小的那个。

你可以直接访问结点，但无法直接访问树。每个节点都会有其父节点的引用。节点 Node 定义如下：

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node parent;
}

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Node':
        # 分情况讨论，由于有了parent指针，故不考虑中序遍历
        # 1. 如果他有右孩子，那么中序后继就是它的右孩子的最左叶子节点
        # 2. 如果它没有右孩子，且是双亲节点的左孩子，那么中序后继是它的双亲【爸爸】
        # 3. 如果它没有右孩子，且是双亲节点的右孩子，那么向上遍历，直到有一个节点，它是它双亲的左孩子，返回那个双亲，对于右边的那个节点而言，根节点的双亲是None,如果过程中遇到None,则返回没有后继
        if node.right != None:
            temp_node = node.right
            while (temp_node.left != None):
                temp_node = temp_node.left
            return temp_node
        if node.right == None:
            if node.parent != None:
                parent_1 = node.parent
                if parent_1.left == node:
                    return parent_1
        if node.right == None:
            while node.parent != None:
                if node.parent.right == node:
                    node = node.parent
                elif node.parent.left == node:
                    return node.parent
            return None
```

# 516. 最长回文子序列

给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # dp，dp[i][j]是在子串s[i:j+1]中的最长回文子序列
        # 那么它的状态可以由掐头去尾，掐头，去尾三种形式来确定
        # 掐头去尾的情况下，如果新加入的头和尾相等，显然+2 ：dp[i][j] = dp[i+1][j-1] + 2
        # 如果新加入的头尾不等，只加入，只需要比对加头或加尾前，原序列里谁更长即可 dp[i][j] = max(dp[i+1][j],dp[i][j-1])
        # 注意dp[i][j] 只能由他左边，下边，左下方的数字得出结果
        n = len(s)
        dp = [[0 for j in range(n)] for i in range(n)]
        # 填充对角线dp[i][i] = 1
        for i in range(0,n):
            dp[i][i] = 1
        # 注意dp[i][j] 只能由他左边，下边，左下方的数字得出结果
        # 那么填充顺序为定j移i，从左到右填充纵列,纵列还需要倒序
        for j in range(1,n):
            for i in range(j-1,-1,-1): # 注意左闭右开
                # print(i,j)
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                elif s[i] != s[j]:
                    dp[i][j] = max(dp[i+1][j],dp[i][j-1])
        # 最终要返回的是dp[头:尾] 即dp[0][-1]
        # print(dp) 
        return dp[0][-1]
```

# 518. 零钱兑换 II

给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。

假设每一种面额的硬币有无限个。 

题目数据保证结果符合 32 位带符号整数。

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # 完全背包问题
        # dp[i]为凑到i元的方法数
        # 那么状态转移为dp[i] = sum(dp[i-coin]) coin是coins里面的，越界则返回0
        # 初始化dp为全0,
        # dp[0] = 1因为如果不选任何的硬币，就可以凑出0元。其他的情况还未知，所以都初始化为0即可。
        dp = [0 for i in range(amount+1)]
        dp[0] = 1
        coins.sort() # 排序硬币
        # 填充dp数组
        # 注意遍历顺序,先固定every_coin，再变动i ,求的是组合数
        # 反过来求的是排列数
        for every_coin in coins:
            for i in range(amount+1):           
                if i - every_coin >= 0: # 越界则略过
                    dp[i] += dp[i-every_coin]
        return dp[-1]

```

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # 利用二维dp来解的话，空间复杂度和时间复杂度较高，仅仅作为解法
        # dp[i][j]是考虑用前n个硬币使得价值和为j的方案数
        # dp[0][0] = 1
        dp = [[0 for i in range(amount+1)]for k in range(len(coins)+1)]
        dp[0][0] = 1
        # 状态转移为dp[i][j] = dp[i-1][j-coin*k] 
        # 其中coin为此次新增加的那一个面值的硬币，边界为j//coin
        # 填充顺序为从左到右，从上到下
        coins.sort() # 硬币排序
        for i in range(1,len(coins)+1): # 第一行无需填充
            for j in range(0,amount+1):
                upto = j//coins[i-1] # 看最多能用多少次
                for times in range(upto+1): # 注意左开右闭合，可以是0次
                    dp[i][j] += dp[i-1][j-coins[i-1]*times] # 注意这里是累加！
        return dp[-1][-1]
```

# 525. 连续数组

给定一个二进制数组 `nums` , 找到含有相同数量的 `0` 和 `1` 的最长连续子数组，并返回该子数组的长度。

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        # 把0看作-1，本题变化为和为0的最长子数组
        # 前缀和收集到目前序号之前到总和
        # 每次检查当前值是否在之前出现过，只需要记录当前值第一次出现的位置
        for i in range(len(nums)): # 变化数组
            if nums[i] == 0:
                nums[i] = -1
        pre_dict = collections.defaultdict(int)
        pre_dict[0] = -1 # 注意这一行，初始化为-1使得在之后总和0时很好处理
        temp_sum = 0
        max_length = 0 # 
        for i in range(len(nums)):
            temp_sum += nums[i]
            if temp_sum not in pre_dict: # 只需记录第一次出现的位置
                pre_dict[temp_sum] = i
            elif temp_sum in pre_dict:
                max_length = max(max_length,i - pre_dict[temp_sum])
        return max_length
```

# 526. 优美的排列

假设有从 1 到 N 的 N 个整数，如果从这 N 个数字中成功构造出一个数组，使得数组的第 i 位 (1 <= i <= N) 满足如下两个条件中的一个，我们就称这个数组为一个优美的排列。条件：

第 i 位的数字能被 i 整除
i 能被第 i 位上的数字整除
现在给定一个整数 N，请问可以构造多少个优美的排列？

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        # 回溯
        nums = [i+1 for i in range(n)]
        self.ans = 0 # 记录是否满足条件
        path = []
        def backtracking(path,lst):
            if len(path) == n:
                self.ans += 1
                return 
            for i in lst:
                if i % (len(path)+1) == 0 or (len(path)+1) % i == 0: # 不满足条件则剪枝
                    cp = lst.copy()
                    cp.remove(i)
                    path.append(i)
                    backtracking(path,cp)
                    path.pop()

        backtracking(path,nums) # 开始回溯
        return (self.ans)
```

# 539. 最小时间差

给定一个 24 小时制（小时:分钟 **"HH:MM"**）的时间列表，找出列表中任意两个时间的最小时间差并以分钟数表示。

```python
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        # 字符串转化成普通的分钟数，int返回
        the_minute = [self.toMinute(i) for i in timePoints]
        the_minute.sort()
        min_gap = 0xffffffff # 初始化为极大值
        p = 0 
        # 找时间差的时候，和前一个，后一个都对比
        while p < len(the_minute) - 1:
            prev = (the_minute[p] - the_minute[p-1]) % 1440
            nxt = abs((the_minute[p+1] - the_minute[p]))
            min_gap = min(prev,nxt,min_gap)
            p += 1
        while p < len(the_minute):
            prev = (the_minute[p] - the_minute[p-1]) % 1440
            min_gap = min(min_gap,prev)
            p += 1
        return min_gap
    
    def toMinute(self,time):
        minute = int(time[:2]) * 60 + int(time[3:])
        return minute
```

# 547. 省份数量

有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。

```python
# 使用并查集,本题解的并查集设计方法参考了官方leetcodebook
# https://leetcode-cn.com/leetbook/read/graph/r3yaqt/
# 注意，在并查集中，如果这个节点是根节点，那么父节点是本身【和一般树的定义需要区分开来！】

class UnionFind: # QuickUnion实现。即快速设计union，让更多余的步骤给find去做
    def __init__(self,size):
        self.root = [i for i in range(size)] # 初始化root数组
        # root数组的含义是，当前索引的根节点是谁
    
    # 并查集的核心功能。一个是并，一个是查。都需实现
    def union(self,x,y): # 并
    # 找到需要并的节点的父节点
        rootX = self.find(x) 
        rootY = self.find(y)
        if rootX != rootY: # 两者的父节点不想等，那么需要并，这里统一把y并入x
            self.root[rootY] = rootX
    
    def find(self,x):  #查
    # 由于并的很草率，所以查当然就要复杂一点
        while x != self.root[x]: # 所以是一个循环搜索
            x = self.root[x]
        return x

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        # 利用邻接矩阵来表示无向图
        n = len(isConnected)
        UF = UnionFind(n) # 构建并查集并且初始化
        for i in range(n): 
            for j in range(n):
                if isConnected[i][j] == 1:
                    UF.union(i,j) # 在并查集中链接
        # 此时并查集构建完毕，扫描并查集的root数组，对每个节点调用find,加入到集合中
        the_set = set() # 相当于扫描每个点的最顶级根节点，会自动去重
        for i in range(len(UF.root)): # 扫描
            the_set.add(UF.find(i))
        return len(the_set) # 返回集合中元素个数即可

    
    
```

# 551. 学生出勤记录 I

给你一个字符串 s 表示一个学生的出勤记录，其中的每个字符用来标记当天的出勤情况（缺勤、迟到、到场）。记录中只含下面三种字符：

'A'：Absent，缺勤
'L'：Late，迟到
'P'：Present，到场
如果学生能够 同时 满足下面两个条件，则可以获得出勤奖励：

按 总出勤 计，学生缺勤（'A'）严格 少于两天。
学生 不会 存在 连续 3 天或 3 天以上的迟到（'L'）记录。
如果学生可以获得出勤奖励，返回 true ；否则，返回 false 。

```python
class Solution:
    def checkRecord(self, s: str) -> bool:
    # 模拟
        absent = 0
        late = 0
        for i in s:
            if i == "P":
                late = 0 # 重置
            elif i == "L":
                late += 1
                if late >= 3:
                    return False
            elif i ==  "A":
                late = 0 # 重置
                absent += 1
                if absent >= 2:
                    return False
        return True
```

# 560. 和为K的子数组

给定一个整数数组和一个整数 **k，**你需要找到该数组中和为 **k** 的连续的子数组的个数。

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 要找到连续的，而且需要找到每一个
        count = 0
        # 前缀和 + hash
        pre_dict = collections.defaultdict(list)
        pre_sum = 0
        pre_dict[0] = [0]
        for i in range(len(nums)):
            pre_sum += nums[i]
            pre_dict[pre_sum].append(i) # 要记录每一个，这里选择记录索引，实际上它只是占位的
            target = pre_sum - k # 找是否存在目标值,注意要处理k为0的情况
            if target in pre_dict and target != pre_sum: # 目标存在，记录每一个，所以+上的是len
                count += len(pre_dict[target])
            elif target in pre_dict and target == pre_sum: # 注意要处理k为0的情况
                count += len(pre_dict[target]) - 1 # 排除掉自身
        return count

```

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 简略写的版本
        pre_dict = collections.defaultdict(int)
        temp_sum = 0
        pre_dict[0] = 1  # 注意这一行
        ans = 0
        for i in range(len(nums)):
            temp_sum += nums[i]
            pre_dict[temp_sum] += 1
            target = temp_sum - k
            if target != temp_sum:
                ans += pre_dict[target]
            elif target == temp_sum:
                ans += pre_dict[target] - 1
        return ans
            
```



# 572. 另一棵树的子树

给你两棵二叉树 root 和 subRoot 。检验 root 中是否包含和 subRoot 具有相同结构和节点值的子树。如果存在，返回 true ；否则，返回 false 。

二叉树 tree 的一棵子树包括 tree 的某个节点和这个节点的所有后代节点。tree 也可以看做它自身的一棵子树。

```python
class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        # 子方法则是根据isSameTree来判断是不是一样的子树
        ans = self.pre_order_check(root,subRoot) # 调用遍历检查
        return ans
        
    def pre_order_check(self,node,subRoot):
        if node == None:
            return False
        if self.isSameTree(node,subRoot):
            return True
        left = self.pre_order_check(node.left,subRoot)
        right = self.pre_order_check(node.right,subRoot)  
        return left or right  
          
    def isSameTree(self, p: TreeNode, q: TreeNode): # 检查两数是否相同
        if p == None and q == None:
            return True
        elif p == None and  q != None:
            return False
        elif p != None and q == None:
            return False
        elif p.val != q.val:
            return False
        else:
            return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
```

# 592. 分数加减运算

给定一个表示分数加减运算表达式的字符串，你需要返回一个字符串形式的计算结果。 这个结果应该是不可约分的分数，即最简分数。 如果最终结果是一个整数，例如 2，你需要将它转换成分数形式，其分母为 1。所以在上述例子中, 2 应该被转换为 2/1。

```python
class Solution:
    def fractionAddition(self, expression: str) -> str:
        # 先通分，再化简
        # 两个数的乘积等于这两个数的最大公约数与最小公倍数的积
        # 格式化
        lst = []
        stack = []
        for i in range(len(expression)):
            if i != 0 and (expression[i] == "+" or expression[i] == "-"):
                lst.append(stack)
                stack = []
                if expression[i] == "-":
                    stack.append(expression[i])
            else:
                stack.append(expression[i])
        if len(stack) != 0:
            lst.append(stack)
        # 此时lst已经格式化完毕,省略+号
        
        ans = reduce(self.calc,lst)   # python的reduce语法，超赞！
           
        return "".join(ans)
    
    def calc(self,lst1,lst2): # 两两计算
        cp1 = lst1.copy()
        cp2 = lst2.copy()
        if cp1[0] == "-":
            cp1 = cp1[1:]
        t1 = ''.join(cp1)
        cp1 = t1.split("/")
        if cp2[0] == "-":
            cp2 = cp2[1:]
        t2 = "".join(cp2)
        cp2 = t2.split("/")
        a1 = int(cp1[0])*int(cp2[1])
        a2 = int(cp2[0])*int(cp1[1])
        b = int(cp1[1])*int(cp2[1])
        if lst1[0] != "-" and lst2[0] != "-": # +,+
            a = (a1+a2)
        elif lst1[0] == "-" and lst2[0] != "-":# -,+
            a = (a2-a1)
        elif lst1[0] != "-" and lst2[0] == "-":# + ,-
            a = (a1-a2)
        elif lst1[0] == "-" and lst2[0] == "-": # -,-
            a = -(a1+a2)

        gcd = self.findGCD(a,b)
        a //= gcd
        b //= gcd
        if b < 0:
            a = -a
            b = -b
        return [str(a)+"/"+str(b)]

    
    def findGCD(self,a,b): # 找到最大公约数
        while a != 0:
            temp = a
            a = b % a
            b = temp
        return b
    
    def findLCM(self,a,b): # 找到最小公倍数
        return a*b//(self.findGCD(a,b))
```

# 611. 有效三角形的个数

给定一个包含非负整数的数组，你的任务是统计其中可以组成三角形三条边的三元组个数。

```python
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        count = 0 # 记录合法个数
        # 思路，固定两边找第三边，可以是固定小、中，然后找合理的大边
        # 一定要满足 a + b > c ，在二分搜索中，搜到c可以取到的最大值即可
        # 先排序
        nums.sort()
        temp = []
        for i in nums:
            if i != 0:
                temp.append(i)
        nums = temp
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                left = j
                right = n - 1
                target = nums[i] + nums[j]
                while left <= right: # 找到小于目标值的最大值
                    mid = (left+right)//2
                    if nums[mid] == target: # 值大了，需要缩小
                        right = mid - 1
                    elif nums[mid] > target:
                        right = mid - 1
                    elif nums[mid] < target:
                        left = mid + 1
                # 搜完之后,left - 1 为第一个小于target的索引
                # 那么从j+1~[left-1]有 left - j - 1 个数
                count += left - 1 - j
        return count


```

# 633. 平方数之和

给定一个非负整数 `c` ，你要判断是否存在两个整数 `a` 和 `b`，使得 `a2 + b2 = c` 。

```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        # 双指针，假设a<=b
        a = 0
        b = int(sqrt(c))
        while a <= b:
            if a**2 + b**2 == c:
                return True
            elif a**2 + b**2 > c: # 数大了，b要减少
                b -= 1
            elif a**2 + b**2 < c: # 数小了，a要增大
                a += 1
        # 扫完了都没有结果
        return False

```

# 647. 回文子串

给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # 处理特殊情况
        if len(s) <= 1:
            return len(s)
        # dp
        # dp[i][j]的含义是s[i:j+1]注意区间左闭右开，是否为回文串
        # 先申请足够大的数组，默认全部初始化为false，显然i要小于等于j+1
        dp = [[False for i in range(len(s))]for k in range(len(s))]
        # 主对角线上只有一个字母，显然都是True,填充
        # count记录True的个数
        count = 0
        for i in range(len(s)):
            dp[i][i] = True
            count += 1
        # 子问题为，dp[i][j]要为回文，需要掐头去尾为回文且s[i] == s[j]
        # 掐头去尾为 dp[i+1][j-1]
        # 状态转移方程为：dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
        # 画方格可以发现状态转移需要左下角的数，只有主对角线不够所以填充主对角线右边的平行线
        # 此时只需要判断是否为
        for i in range(len(s)-1):
            dp[i][i+1] = (s[i] == s[i+1])
            if s[i] == s[i+1]: # 计数
                count += 1
        # 此时可以开始状态转移，由于需要左下角的数，那么从左到右，纵列填充
        for j in range(2,len(s)):
            for i in range(0,j-1): #注意左闭右开
                dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
                if dp[i][j] == True:
                    count += 1
        return count
```

# 655. 输出二叉树

在一个 m*n 的二维字符串数组中输出二叉树，并遵守以下规则：

行数 m 应当等于给定二叉树的高度。
列数 n 应当总是奇数。
根节点的值（以字符串格式给出）应当放在可放置的第一行正中间。根节点所在的行与列会将剩余空间划分为两部分（左下部分和右下部分）。你应该将左子树输出在左下部分，右子树输出在右下部分。左下和右下部分应当有相同的大小。即使一个子树为空而另一个非空，你不需要为空的子树输出任何东西，但仍需要为另一个子树留出足够的空间。然而，如果两个子树都为空则不需要为它们留出任何空间。
每个未使用的空间应包含一个空的字符串""。
使用相同的规则输出子树。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def printTree(self, root: TreeNode) -> List[List[str]]:
        # 行数m是高度
        # 列数n是2**m-1
        # 模拟填充
        depthList = []
        def findDepth(root,d):
            if root == None:
                return
            if root.left == None and root.right == None:
                depthList.append(d)
            findDepth(root.left,d+1)
            findDepth(root.right,d+1)

        findDepth(root,1) # 调用
        m = max(depthList)
        n = 2 ** m - 1
        everyLevel = [["" for j in range(n)] for i in range(m)]
        start_index = n // 2
        everyLevel[0][start_index] = str(root.val)

        def bfs(root):
            queue = [(root,start_index)]
            levelNum = 1
            while len(queue) != 0:
                new_queue = []
                for pair in queue:
                    if pair[0].left != None: # 纸上写这个系数更新规则
                        leftIndex = pair[1] - 2**(m-1-levelNum) # 左子树除以2
                        new_queue.append((pair[0].left,leftIndex))
                        everyLevel[levelNum][leftIndex] = str(pair[0].left.val)
                    if pair[0].right != None:
                        rightIndex = pair[1] + 2**(m-1-levelNum)  # 右子树为这                        
                        new_queue.append((pair[0].right,rightIndex))
                        everyLevel[levelNum][rightIndex] = str(pair[0].right.val)
                levelNum += 1
                queue = new_queue

        bfs(root)
        return everyLevel
```

# 670. 最大交换

给定一个非负整数，你**至多**可以交换一次数字中的任意两位。返回你能得到的最大值。

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        # 往后扫描有没有比当前位置的值大的数
        # 如果有的话，选择值最大的那个，且最靠后的那个
        # 用字符串化方便处理
        if num < 10: # 一位处理不了
            return num
        s = list(str(num))
        for i in range(len(s)):
            group = collections.defaultdict(list) # 收集结果，
            for j in range(i+1,len(s)):
                if int(s[i]) < int(s[j]):
                    group[s[j]].append(j)
            # 找到在其后的最大值
            # 如果group为空，开启下一轮循环
            if len(group) == 0:
                continue
            # 否则进行进一步判断
            key = max(group) # 找到键的最大值
            j = max(group[key]) # 找到对应最大值的索引
            s[i],s[j] = s[j],s[i]
            return int("".join(s))
        return int("".join(s))
            
```

# 671. 二叉树中第二小的节点

给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。如果一个节点有两个子节点的话，那么该节点的值等于两个子节点中较小的一个。

更正式地说，root.val = min(root.left.val, root.right.val) 总成立。

给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。

```python
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        # 朴素做法，由于数据量小，先序遍历，排序，第一个不等于首位的值
        # 完全没有用到性质。。。
        temp = [] # 收集结果
        def pre_order(node):
            if node != None:
                temp.append(node.val)
                pre_order(node.left)
                pre_order(node.right)
        pre_order(root)
        if len(temp) < 2: # 都不足两个，肯定-1
            return -1
        temp.sort() # 排序
        mark = temp[0]
        for i in temp:
            if i != mark: # 找到了不等于的那个
                return i # 直接return
        return -1 # 否则return -1

```

```python
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        # 使用到性质的dfs 【先序遍历】
        # 树非空
        # 如果当前节点的值大于等于 ans，那么以当前节点为根的子树中所有节点的值都大于等于 ans，我们就直接回溯，无需对该子树进行遍历。
        ans = -1 # 初始化-1
        mark = root.val
        def dfs(root): # 变量名为，根，标记，可能会返回的答案
            nonlocal ans # 注意这一行，使得ans作用域跨越
            if root == None:
                return 
            if ans != -1 and root.val >= ans: # 这一行的意思是，ans被更新过，且当前节点为根的子树中所有节点的值都大于等于 ans，我们就直接回溯，无需对该子树进行遍历。
                return
            if root.val <= mark: # 正常继续dfs
                dfs(root.left)
                dfs(root.right)
            else:
                ans = root.val
                return 
        dfs(root)
        return ans
```

# 677. 键值映射

实现一个 MapSum 类，支持两个方法，insert 和 sum：

MapSum() 初始化 MapSum 对象
void insert(String key, int val) 插入 key-val 键值对，字符串表示键 key ，整数表示值 val 。如果键 key 已经存在，那么原来的键值对将被替代成新的键值对。
int sum(string prefix) 返回所有以该前缀 prefix 开头的键 key 的值的总和。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.value = None

class TrieTree:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,key,val):
        node = self.root
        for char in key:
            index = (ord(char)-ord("a"))
            if node.children[index] == None: # 如果这个孩子是空的，创建它
                node.children[index] = TrieNode() 
            node = node.children[index]
        node.value = val
        # print(node,node.value)

    def find_prefix(self,prefix):
        node = self.root
        for char in prefix:
            index = (ord(char)-ord("a"))
            if node.children[index] == None: # 如果这个孩子是空的，创建它
                node.children[index] = TrieNode()
            node = node.children[index]
        the_sum = 0
        queue = [node] # BFS搜结果
        while len(queue) != 0:
            new_queue = []
            for every_node in queue:
                # print(every_node,every_node.value)
                if every_node.value != None: # 收集结果
                    the_sum += every_node.value
                for index in range(26):
                    if every_node.children[index] != None:
                        new_queue.append(every_node.children[index])
            queue = new_queue

        return the_sum

class MapSum:
# 前缀树
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.Tree = TrieTree()

    def insert(self, key: str, val: int) -> None:
        self.Tree.insert(key,val)

    def sum(self, prefix: str) -> int:
        ans = self.Tree.find_prefix(prefix)
        return ans


```

# 678. 有效的括号字符串

给定一个只包含三种字符的字符串：（ ，） 和 *，写一个函数来检验这个字符串是否为有效字符串。有效字符串具有如下规则：

任何左括号 ( 必须有相应的右括号 )。
任何右括号 ) 必须有相应的左括号 ( 。
左括号 ( 必须在对应的右括号之前 )。
* 可以被视为单个右括号 ) ，或单个左括号 ( ，或一个空字符串。
一个空字符串也被视为有效字符串。

```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        # 用栈1匹配(和)
        # * 用另一个栈2存储
        # 遇到左括号则入栈1，遇到右括号则先考虑弹出栈1，在栈1不够的情况下或者是弹出栈2
        # 栈始终记录的是索引值
        stack1 = [] # 存储左括号
        stack2 = [] # 存储*
        for i in range(len(s)):
            # print(i,stack1,stack2)
            if s[i] == '(':
                stack1.append(i)
            elif s[i] == ')':
                if len(stack1) != 0:
                    stack1.pop()
                    continue # 开启下一次
                if len(stack2) != 0:
                    stack2.pop()
                elif len(stack2) == 0:
                    return False
            elif s[i] == '*':
                stack2.append(i)
        # 处理完之后，看栈1和栈2的数量
        # 只有当栈1中的所有下标之后都有被*消除，才返回true
        # 当栈1比栈2还长，肯定False
        
        if len(stack1) > len(stack2):
            return False
        while stack1 != []:
            if stack1[-1] < stack2[-1]: # (在*之前
                stack1.pop()
                stack2.pop()
            else:
                return False
        return True

```

# 680. 验证回文字符串 Ⅱ

给定一个非空字符串 `s`，**最多**删除一个字符。判断是否能成为回文字符串。

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            elif s[left] != s[right]:
                break
        return left >= right or self.isPalindrome(s,left+1,right) or self.isPalindrome(s,left,right-1)
    
    def isPalindrome(self,s,p1,p2): #  闭区间
        while p1 < p2: 
            if s[p1] == s[p2]:
                p1 += 1
                p2 -= 1
            elif s[p1] != s[p2]:
                break
        return p1 >= p2
```

# 684. 冗余连接

树可以看成是一个连通且 无环 的 无向 图。

给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间，且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges ，edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。

请找出一条可以删去的边，删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案，则返回数组 edges 中最后出现的边。

```python
class UF: # 并查集，quickFind未优化
    def __init__(self,size):
        self.root = [i for i in range(size)]
    
    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootX] = rootY
    
    def find(self,x):
        while x != self.root[x]:
            x = self.root[x]
        return x
    
    def connect(self,x,y):
        return self.find(x) == self.find(y)

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        # 提示中已经给出了 n == len(edges)
        # 并查集方法只需要找到第一个已经重复的即为答案
        unionFind = UF(len(edges))
        for i in edges:
            if not unionFind.connect(i[0]-1,i[1]-1): # 连接序号都减1
                unionFind.union(i[0]-1,i[1]-1)
            else: # 如果已经连接，返回原始边
                return [i[0],i[1]]
```

# 690. 员工的重要性

给定一个保存员工信息的数据结构，它包含了员工 唯一的 id ，重要度 和 直系下属的 id 。

比如，员工 1 是员工 2 的领导，员工 2 是员工 3 的领导。他们相应的重要度为 15 , 10 , 5 。那么员工 1 的数据结构是 [1, 15, [2]] ，员工 2的 数据结构是 [2, 10, [3]] ，员工 3 的数据结构是 [3, 5, []] 。注意虽然员工 3 也是员工 1 的一个下属，但是由于 并不是直系 下属，因此没有体现在员工 1 的数据结构中。

现在输入一个公司的所有员工信息，以及单个员工 id ，返回这个员工和他所有下属的重要度之和。

```python
"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        # dfs，注意数据结构，id不要求连续
        # 先把员工转成哈希
        graph = collections.defaultdict(list) # 第一位为重要程度，第二位为下属
        for ep in employees:
            graph[ep.id].append(ep.importance)
            graph[ep.id].append(ep.subordinates)

        ans = 0 # 使用nonlocal接受结果
        # 使用不带返回值的dfs
        def dfs(theID): # 传入的是employees[id]
            nonlocal ans
            ans += graph[theID][0]
            for sub in graph[theID][1]:
                dfs(sub)

        dfs(id)
        return ans
```

```python
        # 使用返回值的dfs
        def dfs(theID): # 传入的是employees[id]
            temp = 0
            temp += graph[theID][0]
            for sub in graph[theID][1]:
                temp += dfs(sub)
            return temp
```

# 702. 搜索长度未知的有序数组

给定一个升序整数数组，写一个函数搜索 nums 中数字 target。如果 target 存在，返回它的下标，否则返回 -1。注意，这个数组的大小是未知的。你只可以通过 ArrayReader 接口访问这个数组，ArrayReader.get(k) 返回数组中第 k 个元素（下标从 0 开始）。

你可以认为数组中所有的整数都小于 10000。如果你访问数组越界，ArrayReader.get 会返回 2147483647。

```python
# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
#class ArrayReader:
#    def get(self, index: int) -> int:

class Solution:
    def search(self, reader, target):
        """
        :type reader: ArrayReader
        :type target: int
        :rtype: int
        """
        # 数组长度小于20000，二分法找到其长度
        left = 0
        right = 20000 - 1
        max_val = 2147483647
        while left <= right:
            mid = (left + right)//2
            if reader.get(mid) == max_val: # 说明越界，收缩右边
                right = mid - 1
            elif reader.get(mid) < max_val: # 说明没有越界，收缩左边
                left = mid + 1
            elif reader.get(mid) > max_val: # 说明越界，收缩右边
                right = mid - 1
        # 此时，left是长度,right是我需要的右边界二分法下标
        left = 0
        right = right 
        while left <= right:
            mid = (left + right)//2
            if reader.get(mid) == target:
                return mid
            elif reader.get(mid) > target: # 数值偏大，收缩右边界
                right = mid - 1
            elif reader.get(mid) < target: # 数值偏小，收缩左边界
                left = mid + 1
        return -1 #   没有找到
```

# 703. 数据流中的第 K 大元素

设计一个找到数据流中第 k 大元素的类（class）。注意是排序后的第 k 大元素，不是第 k 个不同的元素。

请实现 KthLargest 类：

KthLargest(int k, int[] nums) 使用整数 k 和整数流 nums 初始化对象。
int add(int val) 将 val 插入数据流 nums 后，返回当前数据流中第 k 大的元素。

```python
class KthLargest:
    # 使用内置堆完成
    def __init__(self, k: int, nums: List[int]):
        # 内置堆是小根堆，先建立大小为k的堆，然后再筛
        self.k = k
        self.min_heap = [i for i in nums[:k]]
        heapq.heapify(self.min_heap) # 堆化
        # 当筛的元素比堆顶大的时候，弹出原顶，加入该数，否则不变
        for i in nums[k:]:
            if i > self.min_heap[0]:
                heapq.heappop(self.min_heap)
                heapq.heappush(self.min_heap,i)
        # 此时已经维护好了堆，堆顶是第k大的元素

    def add(self, val: int) -> int:
        if len(self.min_heap) < self.k: # 当堆还没有满的时候，直接入堆
            heapq.heappush(self.min_heap,val)
        elif val > self.min_heap[0]: # 看是否需要维护堆
            heapq.heappop(self.min_heap)
            heapq.heappush(self.min_heap,val)
        return self.min_heap[0] # 维护完成之后返回


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```

# 708. 循环有序列表的插入

给定循环升序列表中的一个点，写一个函数向这个列表中插入一个新元素 insertVal ，使这个列表仍然是循环升序的。

给定的可以是这个列表中任意一个顶点的指针，并不一定是这个列表中最小元素的指针。

如果有多个满足条件的插入位置，你可以选择任意一个位置插入新的值，插入后整个列表仍然保持有序。

如果列表为空（给定的节点是 null），你需要创建一个循环有序列表并返回这个节点。否则。请返回原先给定的节点。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        # 梳理逻辑，
        # 1. 如果原链表为空，插入一个节点之后，自身头尾相接
        # 2. 如果插入值大于原链表最大值，则插在最大值和最小值之间
        # 3. 如果插入值大于原链表最小值，也插在最大值和最小值之间
        # 4. 如果原链表只有一个节点，那么新建一个节点之后互相指认
        # 5. 正常情况，当cur小于等于目标值且cur.next大于等于目标值
        if head == None: # 原来为空链表
            new_node = Node(val = insertVal)
            new_node.next = new_node
            return new_node
        elif head.next == head: # 即只有一个节点的时候
            new_node = Node(val = insertVal)
            head.next = new_node
            new_node.next = head
            return head
        else: # 节点多于2个
            biggest_node,smallest_node = self.find_biggest_smallest(head)
            # print(biggest_node.val,smallest_node.val,"insert = ",insertVal)
            if insertVal >= biggest_node.val or insertVal <= smallest_node.val:
                new_node = Node(val = insertVal)
                biggest_node.next = new_node
                new_node.next = smallest_node
            elif smallest_node.val < insertVal < biggest_node.val:
                cur1 = smallest_node
                cur2 = cur1.next
                while cur2.val < insertVal:
                    cur1 = cur1.next
                    cur2 = cur2.next
                new_node = Node(val = insertVal)
                cur1.next = new_node
                new_node.next = cur2
            return head

    def find_biggest_smallest(self,node):
        biggest_node = node
        smallest_node = node
        cur = node
        times = 0
        while times < 2:
            if cur == node:
                times += 1
            elif cur != node:
                if cur.val >= biggest_node.val: # 取大于等于号保证更新到的是最后一个最大值
                    biggest_node = cur
                elif cur.val < smallest_node.val: # 取严格小于号保证是第一个最小值
                    smallest_node = cur
            cur = cur.next
        return [biggest_node,smallest_node]
                    
```

# 713. 乘积小于K的子数组

给定一个正整数数组 `nums`和整数 `k` 。

请找出该数组内乘积小于 `k` 的连续的子数组的个数。

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # 滑动窗口
        if k <= 1:
            return 0
        left = 0
        right = 0
        window = 1
        ans = 0
        while right < len(nums):
            add_num = nums[right]
            right += 1
            window *= add_num # 注意，由于right自增了1，所以是到right前一位的累积乘积，是闭区间窗口的右端点的下一个点
            while window >= k and left < right: # 窗口收缩条件
                delete_num = nums[left]
                window /= delete_num
                left += 1  # 注意，由于left自增了1，此时left就是左边界
            ans += (right - left) # 实际上是(right-1 - (left) + 1) ，简化了
        return ans 
```

# 716. 最大栈

设计一个最大栈数据结构，既支持栈操作，又支持查找栈中最大元素。

实现 MaxStack 类：

MaxStack() 初始化栈对象
void push(int x) 将元素 x 压入栈中。
int pop() 移除栈顶元素并返回这个元素。
int top() 返回栈顶元素，无需移除。
int peekMax() 检索并返回栈中最大元素，无需移除。
int popMax() 检索并返回栈中最大元素，并将其移除。如果有多个最大元素，只要移除 最靠近栈顶 的那个。

```python
class MaxStack:
# 主栈和辅助栈
# 主栈存取所有元素
# 辅助栈存取到目前为止的最大元素
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.main_stack = []
        self.helper = [-0xffffffff]

    def push(self, x: int) -> None:
        self.main_stack.append(x)
        if self.helper[-1] <= x:
            self.helper.append(x)

    def pop(self) -> int:
        if self.helper[-1] == self.main_stack[-1]:
            self.helper.pop() 
        e = self.main_stack.pop()
        return e

    def top(self) -> int: 
        return self.main_stack[-1]

    def peekMax(self) -> int:
        return self.helper[-1]

    def popMax(self) -> int:
        remain = []
        while self.main_stack[-1] != self.helper[-1]:
            remain.append(self.main_stack.pop())
        e = self.main_stack.pop()
        self.helper.pop()
        for i in remain[::-1]: # 注意这个remain要倒着加回去
            self.push(i)
        return e
```

# 720. 词典中最长的单词

给出一个字符串数组words组成的一本英语词典。从中找出最长的一个单词，该单词是由words词典中其他单词逐步添加一个字母组成。若其中有多个可行的答案，则返回答案中字典序最小的单词。

若无答案，则返回空字符串。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.isEnd = False
class Trie:
    def __init__(self,lst):
        self.root = TrieNode()
        self.theList = {w:True for w in lst}
    
    def insertWord(self,word):
        node = self.root
        for ch in word:
            index = ord(ch) - ord("a")
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isEnd = True
    
    def search(self,word):
        node = self.root
        path = ""
        for ch in word:
            index = ord(ch) - ord("a")
            if node.children[index] != None and node.children[index].isEnd == True:
                node = node.children[index]
                path += ch
                if path in self.theList:
                    self.theList[path] = False # 不用再查它了
            else:
                return path
        return word

    def insertAll(self,wordsList):
        for word in wordsList:
            self.insertWord(word)
    
class Solution:
    def longestWord(self, words: List[str]) -> str:
        # 一种较为复杂的前缀树实现方案
        words.sort(key = len,reverse = True)
        tree = Trie(words)
        tree.insertAll(words)
        ans_lst = []
        for w in words:
            if tree.theList[w] == True:
                ans_lst.append(tree.search(w))
        # 此时ans_lst收集到的是一堆答案，取最长的，按照字典序排序
        max_length = 0
        for i in ans_lst:
            if len(i) >= max_length:
                max_length = len(i)
        ans = []
        for i in ans_lst:
            if len(i) == max_length:
                ans.append(i)
        ans.sort() # 字典序排序
        return ans[0]
```

# 724. 寻找数组的中心下标

给你一个整数数组 nums ，请计算数组的 中心下标 。

数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。

如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。这一点对于中心下标位于数组最右端同样适用。

如果数组有多个中心下标，应该返回 最靠近左边 的那一个。如果数组不存在中心下标，返回 -1 。

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        # 利用前缀和数组，进行计算
        if len(nums) == 1:
            return 0
        sum_array = [0]
        temp_sum = 0
        for i in nums:
            temp_sum += i
            sum_array.append(temp_sum)
        # print(sum_array)
        # 左边和为sum_array[p],右边和为sum_array[-1] - sum_array[p+1]
        for p in range(len(sum_array)-1):
            if sum_array[p] == sum_array[-1] - sum_array[p+1]:
                return p
        return -1
```

# 725. 分隔链表

给定一个头结点为 root 的链表, 编写一个函数以将链表分隔为 k 个连续的部分。

每部分的长度应该尽可能的相等: 任意两部分的长度差距不能超过 1，也就是说可能有些部分为 null。

这k个部分应该按照在链表中出现的顺序进行输出，并且排在前面的部分的长度应该大于或等于后面的长度。

返回一个符合上述规则的链表的列表。

举例： 1->2->3->4, k = 5 // 5 结果 [ [1], [2], [3], [4], null ]

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def splitListToParts(self, head: ListNode, k: int) -> List[ListNode]:
        # 注意，返回的是链表列表
        ans = []
        # 需要判断每个链表内的长度
        sz = 0
        cur = head
        while cur != None:
            sz += 1
            cur = cur.next
        avg_length = sz // k # 粗略记录每个链表的平均长度
        extra = sz % k # 前几个链表可能需要有附加长度
        aim_length = avg_length + 1 if extra != 0 else avg_length # 初始化每个段落的长度
        cur = head # 从头节点扫起
        now_size = 1
        while cur != None:
            temp_head = cur # 记录下当前节点
            while now_size != (aim_length): # 当前子链表需要达到足够长度
                now_size += 1
                cur = cur.next
            extra -= 1 # 需要附加长度的链表数减1
            ans.append(temp_head) # 添加进答案数组
            temp = cur.next # 断开子链表
            cur.next = None
            cur = temp
            now_size = 1 # 重置大小
            if extra == 0: # 如果不需要附加长度了，目标长度回归标准长度
                aim_length = avg_length

        while len(ans) != k: # 结果不够k位，直接补满
            ans.append(None)               
        return ans
```

```python
class Solution:
    def splitListToParts(self, head: ListNode, k: int) -> List[ListNode]:
        ans = []
        size = 0
        cur = head
        while cur != None:
            size += 1
            cur = cur.next 
        # 然后看每个链需要多长，
        every = size // k
        remain = size % k
        cur = head
        while cur != None:
            ans.append(cur)
            if every > 0:
                for times in range(every-1):
                    cur = cur.next
                if remain > 0:
                    cur = cur.next 
                    remain -= 1
                if cur != None:
                    temp = cur.next
                    cur.next = None 
                    cur = temp
            elif every == 0:
                if remain > 0:
                    remain -= 1
                    temp = cur.next
                    cur.next = None 
                    cur = temp     
        # 不足k位补齐
        diff = k - len(ans)   
        for i in range(diff):
            ans.append(None)
        return ans
```

# 735. 行星碰撞

给定一个整数数组 asteroids，表示在同一行的行星。

对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。

找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。

```python
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        # 栈模拟系列
        # 数值仅仅代表质量，正负号才代表往左往右，数值中包含0
        # 如果星星往右，则直接加入，如果星星往左，则一路碰撞,撞到要么自己没了，要么正数没了
        stack = []
        for i in asteroids:
            if len(stack) == 0:
                stack.append(i)
            elif i >=0 : #
                stack.append(i)
            elif i < 0: 
                # 1. 栈非空，且栈顶往右，入栈元素往左
                while len(stack) > 0 and stack[-1] > 0 and stack[-1] < abs(i):
                    stack.pop()
                if len(stack) > 0 and stack[-1] > 0 and stack[-1] == abs(i): # 自己撞没了
                    stack.pop()
                elif len(stack) == 0:
                    stack.append(i)
                elif stack[-1] <= 0: # 正数撞没了
                    stack.append(i)
                elif stack[-1] > 0: # 有撞不烂的正数，自己反而没了
                    pass
        return stack
```

# 740. 删除并获得点数

给你一个整数数组 nums ，你可以对它进行一些操作。

每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除 所有 等于 nums[i] - 1 和 nums[i] + 1 的元素。

开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        # 打家劫舍换皮版本
        # 选了这个数值之后，不能选择相邻的数值
        # 收益是数值*频率
        # 数值大小不超过1w，使用数组做容器
        max_num = max(nums)
        arr = [0 for i in range(max_num + 1)] # 
        for i in nums:
            arr[i] += 1
        # 此时数组的索引是元素，数组的元素是频率,开始打家劫舍
        # dp[i]的含义是，以元素值i结尾的最大收益
        if len(arr) <= 2: # 即所有元素都是1，那么返回总和就行
            return sum(nums)
        dp = [0 for i in range(max_num + 1)]
        dp[1] = arr[1]*1
        dp[2] = max(arr[1]*1,arr[2]*2)
        for i in range(3,len(dp)):
            dp[i] = max((dp[i-2] + arr[i]*i) ,dp[i-1])
        # print(arr,"\n",dp)
        return dp[-1]
```

# 744. 寻找比目标字母大的最小字母

给你一个排序后的字符列表 letters ，列表中只包含小写英文字母。另给出一个目标字母 target，请你寻找在这一有序列表里比目标字母大的最小字母。

在比较时，字母是依序循环出现的。举个例子：

如果目标字母 target = 'z' 并且字符列表为 letters = ['a', 'b']，则答案返回 'a'

```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        # 二分查找，重写比较器
        # 要查找比目标字母大的最小字母
        left = 0
        right = len(letters) - 1
        while left <= right:
            mid = (left + right) // 2
            if left == right:
                break      
            if ord(letters[mid]) == ord(target): # 要找更大的，收缩左边界
                left = mid + 1
            elif ord(letters[mid]) > ord(target): # 值偏大，收缩右边界
                right = mid - 1
            elif ord(letters[mid]) < ord(target): # 值偏小，收缩左边界
                left = mid + 1
        # 此时根据left进行分类讨论
        if ord(letters[left]) > ord(target):
            return letters[left]
        elif ord(letters[left]) <= ord(target):
            return letters[(left+1)%len(letters)]
```

# 746. 使用最小花费爬楼梯

数组的每个下标作为一个阶梯，第 i 个阶梯对应着一个非负数的体力花费值 cost[i]（下标从 0 开始）。

每当你爬上一个阶梯你都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。

请你找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 0 或 1 的元素作为初始阶梯。

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # 一维dp
        # dp[i]的含义是，到达i前所花费的体力，【两基态除外】
        if len(cost) <= 2:
            return min(cost)
        dp = [0 for i in range(len(cost))]
        dp[0] = 0 # 基态
        dp[1] = 0
        dp += [0xffffffff] # 补充上极大值作为封顶
        for i in range(2,len(dp)):
            dp[i] = min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])
        return dp[-1]

```

# 748. 最短补全词

给定一个字符串牌照 licensePlate 和一个字符串数组 words ，请你找出并返回 words 中的 最短补全词 。

如果单词列表（words）中的一个单词包含牌照（licensePlate）中所有的字母，那么我们称之为 补全词 。在所有完整词中，最短的单词我们称之为 最短补全词 。

单词在匹配牌照中的字母时要：

忽略牌照中的数字和空格。
不区分大小写，比如牌照中的 "P" 依然可以匹配单词中的 "p" 字母。
如果某个字母在牌照中出现不止一次，那么该字母在补全词中的出现次数应当一致或者更多。
例如：licensePlate = "aBc 12c"，那么它由字母 'a'、'b' （忽略大写）和两个 'c' 。可能的 补全词 是 "abccdef"、"caaacab" 以及 "cbca" 。

题目数据保证一定存在一个最短补全词。当有多个单词都符合最短补全词的匹配条件时取单词列表中最靠前的一个。

```python
class Solution:
    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        # 先格式化licensePlate
        template = ""
        for ch in licensePlate.lower():
            if ch.isalpha():
                template += ch
        patternList = [0 for i in range(26)]
        for ch in template:
            patternList[ord(ch)-ord("a")] += 1
        temp_ans = [] # 收集可能的所有值
        for word in words:
            lst1 = self.toList(word)
            count = 0
            for everyBit in self.minusList(lst1,patternList):
                if everyBit < 0:
                    break
                else:
                    count += 1
            if count == 26:
                temp_ans.append(word)
        temp_ans.sort(key = len) # 按照长度排序，由于它是稳定排序，所以返回第一个就行
        return temp_ans[0]

    
    def toList(self,s): # 把s转换成列表
        theList = [0 for i in range(26)]
        for ch in s:
            theList[ord(ch)-ord("a")] += 1
        return theList
    
    def minusList(self,lst1,lst2): # 两个列表做差，只有每一项非负才合法
        if len(lst1) == len(lst2):
            finalList = [lst1[i]-lst2[i] for i in range(len(lst1))]
            return finalList
```



# 760. 找出变位映射

给定两个列表 Aand B，并且 B 是 A 的变位（即 B 是由 A 中的元素随机排列后组成的新列表）。

我们希望找出一个从 A 到 B 的索引映射 P 。一个映射 P[i] = j 指的是列表 A 中的第 i 个元素出现于列表 B 中的第 j 个元素上。

列表 A 和 B 可能出现重复元素。如果有多于一种答案，输出任意一种。

```python
class Solution:
    def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 使用一个used数组查询,这里返回的结果是1-1映射
        n = len(nums1)
        used = [False for i in range(n)]
        ans = []
        for num in nums1:
            for p in range(n):
                if nums2[p] == num and used[p] == False:
                    used[p] = True
                    ans.append(p)
                    break
        return ans
```

# 784. 字母大小写全排列

给定一个字符串`S`，通过将字符串`S`中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        # dfs法
        ans = [] # 收集结果用
        path = [] # 路径记录
        n = len(s)
        s = list(s)
        def dfs(path,lst):
            if len(path) == n:
                ans.append(path[:])
                return 
            if len(lst) == 0:
                return
            if lst[0].isdigit():
                path.append(lst[0])
                dfs(path,lst[1:])
            else:
                cp1 = path.copy()
                cp2 = path.copy()
                cp1.append(lst[0].lower())
                cp2.append(lst[0].upper())
                dfs(cp1,lst[1:])
                dfs(cp2,lst[1:])
        dfs(path,s)
        final = [''.join(i) for i in ans] 
        return final
```

# 787. K 站中转内最便宜的航班

有 n 个城市通过一些航班连接。给你一个数组 flights ，其中 flights[i] = [fromi, toi, pricei] ，表示该航班都从城市 fromi 开始，以价格 pricei 抵达 toi。

现在给定所有的城市和航班，以及出发城市 src 和目的地 dst，你的任务是找到出一条最多经过 k 站中转的路线，使得从 src 到 dst 的 价格最便宜 ，并返回该价格。 如果不存在这样的路线，则输出 -1。

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # 方法1: 动态规划法：
        # 申请二维dp数组，有定义为k+2的横行【经过k个站，最多有k+1条边，加上不经过边，共k+2】
        # ，有0～n-1点的纵列
        INF = float("inf")
        dp = [[INF for j in range(n)] for i in range(k+2)]
        # 第一行，除了src到src，其余的全都是无穷
        dp[0][src] = 0
        # 状态转移方程需要用到边
        # dp[k][u] = min(dp[k][u],dp[k-1][v]+W(v,u))
        # 注意flight[n] = [from_n,to_n,cost]的定义,
        for m in range(1,k+2):
            for i,j,cost in flights:
                # 注意下面这个状态转移，到j是先到i，再看花费
                dp[m][j] = min(dp[m][j],dp[m-1][i]+cost)
        # 返回的是目的地那一列的最小值
        min_cost = [dp[m][dst] for m in range(len(dp))]
        ans = min(min_cost)
        return ans if ans != float('inf') else -1

```

# 789. 逃脱阻碍者

你在进行一个简化版的吃豆人游戏。你从 [0, 0] 点开始出发，你的目的地是 target = [xtarget, ytarget] 。地图上有一些阻碍者，以数组 ghosts 给出，第 i 个阻碍者从 ghosts[i] = [xi, yi] 出发。所有输入均为 整数坐标 。

每一回合，你和阻碍者们可以同时向东，西，南，北四个方向移动，每次可以移动到距离原位置 1 个单位 的新位置。当然，也可以选择 不动 。所有动作 同时 发生。

如果你可以在任何阻碍者抓住你 之前 到达目的地（阻碍者可以采取任意行动方式），则被视为逃脱成功。如果你和阻碍者同时到达了一个位置（包括目的地）都不算是逃脱成功。

只有在你有可能成功逃脱时，输出 true ；否则，输出 false 。

```python
class Solution:
    def escapeGhosts(self, ghosts: List[List[int]], target: List[int]) -> bool:
    # 与其在路上堵着，不如直接在终点等着hhh
        # 鬼不能比自己先到，曼哈顿距离需要严格小于自己到目标的曼哈顿距离
        ghost_manhatum = []
        for p1 in ghosts:
            ghost_manhatum.append(self.calc_manhatum_distance(p1,target))
        my_manhatum = self.calc_manhatum_distance([0,0],target)
        return my_manhatum < min(ghost_manhatum)
    
    def calc_manhatum_distance(self,p1,p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
```

# 791. 自定义字符串排序

字符串S和 T 只包含小写字符。在S中，所有字符只会出现一次。

S 已经根据某种规则进行了排序。我们要根据S中的字符顺序对T进行排序。更具体地说，如果S中x在y之前出现，那么返回的字符串中x也应出现在y之前。

返回任意一种符合条件的字符串T。

```python
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        # 把order作为数字哈希表中，多于的可以放在任何位置
        # 注意，这里把函数签名改了 原来是str:str 。
        # 方法1:哈希映射法 双向映射
        order = list(order)
        the_dict = {val:index for index,val in enumerate(order)}
        mirror_dict = {index:val for index,val in enumerate(order)}
        temp = [] # 有映射的
        unsort = [] # 没有映射的
        for i in range(len(s)):
            if s[i] in the_dict:
                temp.append(the_dict[s[i]])
            else:
                unsort.append(s[i])
        temp.sort() # 排序映射的
        for i in range(len(temp)):
            temp[i] = mirror_dict[temp[i]]
        ans = temp + unsort # 还原
        return ''.join(ans) # 拼接成字符串
```

# 797. 所有可能的路径

给一个有 n 个结点的有向无环图，找到所有从 0 到 n-1 的路径并输出（不要求按顺序）

二维数组的第 i 个数组中的单元都表示有向图中 i 号结点所能到达的下一些结点（译者注：有向图是有方向的，即规定了 a→b 你就不能从 b→a ）空就是没有下一个结点了。

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # DFS 注意这是有向图，无需使用visited
        self.ans = []
        path = [0]
        def dfs(path):
            if path[-1] == len(graph) - 1: # 如果这个路径的最后一个元素是目标，则收集结果
                self.ans.append(path[:])
            # 否则需要找最后一个元素的邻居
            neighbors = graph[path[-1]]
            for neighbor in neighbors: # 把这些邻居填充进去再进行搜索
                path.append(neighbor) # 选择这个邻居
                dfs(path)
                path.pop() # 因为要开启下一轮for循环了，这个邻居得出去了
        dfs(path)
        return self.ans
```

# 812. 最大三角形面积

给定包含多个点的集合，从其中取三个点组成三角形，返回能组成的最大三角形的面积。

```python
class Solution:
    def largestTriangleArea(self, points: List[List[int]]) -> float:
    # 纯暴力
        max_area = -1
        for p1 in points:
            for p2 in points:
                for p3 in points:
                    if p1 != p2 and p2 != p3 and p1 != p3:
                        max_area = max(max_area,self.cacl_area(p1,p2,p3))
        return max_area
    
    def cacl_area(self,p1,p2,p3):
        # 面积计算使用海伦公式
        a = self.cacl_distance(p1,p2)
        b = self.cacl_distance(p2,p3)
        c = self.cacl_distance(p1,p3)
        p = (a+b+c)/2
        if a > 0 and b > 0 and c > 0 and p > 0 and p*(p-a)*(p-b)*(p-c) > 0: # 这一点需要牢记
            area = sqrt(p*(p-a)*(p-b)*(p-c))
            return area
        else:
            return -1
    
    def cacl_distance(self,x,y):
        return sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
        
```

# 814. 二叉树剪枝

给定二叉树根结点 root ，此外树的每个结点的值要么是 0，要么是 1。

返回移除了所有不包含 1 的子树的原二叉树。

( 节点 X 的子树为 X 本身，以及所有 X 的后代。)

```python
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        # 后续遍历递归删除法
        # 因为必须检查完左右之后才能确定本节点是否剪除
        # 注意由于是后续遍历，所以剪的时候是从底下往上剪，参照例2:
        # 最左边下面两个先被剪掉，然后再剪掉倒数第二层的，在剪倒数第二层的时候，它的左右已经空了
        if root == None:
            return None
        root.left = self.pruneTree(root.left) # 左边为修剪过之后的
        root.right = self.pruneTree(root.right) # 右边为修剪过之后的
        # 检查本节点是否能够剪
        if root.left == None and root.right == None and root.val == 0: 
            return None
        return root
```

# 820. 单词的压缩编码

单词数组 words 的 有效编码 由任意助记字符串 s 和下标数组 indices 组成，且满足：

words.length == indices.length
助记字符串 s 以 '#' 字符结尾
对于每个下标 indices[i] ，s 的一个从 indices[i] 开始、到下一个 '#' 字符结束（但不包括 '#'）的 子字符串 恰好与 words[i] 相等
给你一个单词数组 words ，返回成功对 words 进行编码的最小助记字符串 s 的长度 。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.isPath = False
        self.depth = 0 # 当前深度，与单词长度相关

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def reversed_insert(self,word): # 倒序插入
        node = self.root
        for char in word[::-1]: #倒序插入
            parent_depth = node.depth # 获取父节点深度
            index = ord(char) - ord("a")
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node.isPath = True # 父节点是路径的一部分
            node = node.children[index] # 指向孩子
            node.depth = parent_depth + 1 # 孩子的深度是父深度+1
                
    def find_end(self,word): # 倒序搜索，是最长路径的话返回深度+1，否则返回0
        node = self.root
        for char in word[::-1]: #倒序搜索
            index = ord(char) - ord("a")
            node = node.children[index]
        if node.isPath == False: # 它是终点了
            # print(chr(97+index)) 检查用
            return node.depth + 1 # 加上中断用的 # 长度
        elif node.isPath == True: # 它只是子路径的一条的话，不参与计算
            return 0
                
class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        # 不借助语言内置容器的解法。倒置每个单词之后,自己构造字典树
        the_Tree = Trie()
        ans = 0
        # 需要处理掉重复的词,
        words = set(words)
        for word in words: # 添加进字典树
            the_Tree.reversed_insert(word)
        for word in words: # 搜索单词，确认它是否是拥有公共后缀的最长路径。
            ans += the_Tree.find_end(word)
        return ans


```

# 821. 字符的最短距离

给你一个字符串 s 和一个字符 c ，且 c 是 s 中出现过的字符。

返回一个整数数组 answer ，其中 answer.length == s.length 且 answer[i] 是 s 中从下标 i 到离它 最近 的字符 c 的 距离 。

两个下标 i 和 j 之间的 距离 为 abs(i - j) ，其中 abs 是绝对值函数。

```python
class Solution:
    def shortestToChar(self, s: str, c: str) -> List[int]:
        # 线性扫描存下每一个c的位置
        # 有c必在s中
        index_lst = []
        for i in range(len(s)):
            if s[i] == c:
                index_lst.append(i)
        # 用双排指针，对两个列表进行计算
        # 给index_lst前后加上极小和极大值，方便统一语法
        index_lst = [-0xffffffff] + index_lst + [0xffffffff]
        ans = [] # 用于接受答案
        p_index = 0
        for p in range(len(s)): # p是索引
            if s[p] != c:
                pass
            else:
                p_index += 1
            ans.append(min(abs(p-index_lst[p_index]),abs(p-index_lst[p_index+1])))
        return ans
            
```

# 844. 比较含退格的字符串

给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        # 栈思路
        # 利用栈模拟
        stack1 = []
        for i in s:
            if i != '#':
                stack1.append(i)
            elif i == '#':
                if len(stack1) != 0:
                    stack1.pop()
        stack2 = []
        for i in t:
            if i != '#':
                stack2.append(i)
            elif i == '#':
                if len(stack2) != 0:
                    stack2.pop()
        return stack1 == stack2

```

# 865. 具有所有最深节点的最小子树

给定一个根为 root 的二叉树，每个节点的深度是 该节点到根的最短距离 。

如果一个节点在 整个树 的任意节点之间具有最大的深度，则该节点是 最深的 。

一个节点的 子树 是该节点加上它的所有后代的集合。

返回能满足 以该节点为根的子树中包含所有最深的节点 这一条件的具有最大深度的节点。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        self.valid_leaves = [] # 只取最后一个列表
        self.find_valid_leaf(root) # 使用这个方法后，上面那个列表的最后一个是最深的叶子节点
        self.valid_leaves = self.valid_leaves[-1] # 将它调整为真叶子节点
        self.leaves_set = set(self.valid_leaves) # 做集合，查找更快
        ans = self.find_LCA(root)
        return ans

    def find_valid_leaf(self,node):# 深度最大的叶子节点在BFS的最后一行
        # BFS
        if node == None:
            return 
        queue = [node]
        while len(queue) != 0:
            level = []
            new_queue =[] # 给下一层的队列
            for i in queue:
                if i != None:
                    level.append(i) # 收集的是节点
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            self.valid_leaves.append(level)
            queue = new_queue
    
    def find_LCA(self,root):
        if root == None:
            return None
        if root in self.leaves_set:
            return root
        left = self.find_LCA(root.left)
        right = self.find_LCA(root.right)
        if left == None and right == None: return None
        if left != None and right == None: return left
        if left == None and right != None: return right
        if left != None and right != None: return root
        
```

# 875. 爱吃香蕉的珂珂

珂珂喜欢吃香蕉。这里有 N 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 H 小时后回来。

珂珂可以决定她吃香蕉的速度 K （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 K 根。如果这堆香蕉少于 K 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。  

珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。

返回她可以在 H 小时内吃掉所有香蕉的最小速度 K（K 为整数）。

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # bad二分法查找,用了线性搜索强行过
        # left和right为速率
        left = 1
        right = max(piles) # 截止速率为piles中的最大值
        while left <= right:
            mid = (left+right)//2
            temp_hours = self.cacl_Hours(piles,mid)
            if left == mid: # 此时搜索left，left+1……的用时
                h1 = self.cacl_Hours(piles,left)
                h2 = self.cacl_Hours(piles,left+1)
                print("d")
                if h1 > h: # 用时过长
                    start = left + 1
                    while self.cacl_Hours(piles,start) > h:
                        start += 1
                    return start
                elif h1 <= h: # 用时可以接受
                    return left
            if temp_hours == h:
                while self.cacl_Hours(piles,mid) == h: # 这里用线性搜索强行过。。
                    mid -= 1                    
                return mid + 1
            elif temp_hours > h: # 时间过长，速度过慢，需要提速，收缩左边界
                left = mid + 1
            elif temp_hours < h: # 时间过短，速度过快，需要减速，收缩右边界
                right = mid - 1
    
    def cacl_Hours(self,lst,v): # 传入参数为列表，速度v，返回值为时间
        hours = 0
        for i in lst:
            hours += ceil(i/v)
        return hours
```

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 二分法查找
        # left和right为速率
        left = 1
        right = max(piles) # 截止速率为piles中的最大值
        while left <= right: # 闭区间
            mid = (left+right)//2
            judge = self.cacl_Hours(piles,mid,h) #
            if judge: # 可以吃完，还能减慢速度,因为最终值返回的是left，所以没有影响
                right = mid - 1
            elif not judge: # 吃不完，要增加速度
                left = mid + 1
        return left
    
    def cacl_Hours(self,lst,v,limit): # 传入参数为列表，速度v,限制时间，返回值为是否能在限制时间内吃完
        hours = 0
        for i in lst:
            hours += ceil(i/v) # 注意这里向上取整
        return hours <= limit
```

# 881. 救生艇

第 i 个人的体重为 people[i]，每艘船可以承载的最大重量为 limit。

每艘船最多可同时载两人，但条件是这些人的重量之和最多为 limit。

返回载到每一个人所需的最小船数。(保证每个人都能被船载)。

```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        # 排序之后双指针
        # 最重的尽量搭配上最轻的，如果搭配不了，单独给他
        boats = 0
        people.sort()
        left = 0
        right = len(people) - 1
        while left <= right: 
            if left == right: # 指向同一个的时候，直接一条船
                boats += 1
                break
            if people[left] + people[right] <= limit: # 两人都上来
                boats += 1
                left += 1
                right -= 1
            elif people[left] + people[right] > limit: # 只上来胖子
                boats += 1
                right -= 1        
        return boats

```

# 889. 根据前序和后序遍历构造二叉树

返回与给定的前序和后序遍历匹配的任何二叉树。

 pre 和 post 遍历中的值是不同的正整数。

示例：

输入：pre = [1,2,4,5,3,6,7], post = [4,5,2,6,7,3,1]
输出：[1,2,3,4,5,6,7]


提示：

1 <= pre.length == post.length <= 30
pre[] 和 post[] 都是 1, 2, ..., pre.length 的排列
每个输入保证至少有一个答案。如果有多个答案，可以返回其中一个。

```python
class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        # 只要求返回一种可行性答案即可
        # pre = [【1】,2,4,5｜｜3,6,7], post = [4,5,2｜｜6,7,3,【1】]
        # 需要找到左右子树的分界线，扫描到总和
        if len(pre) == len(post) ==  1: # 简写成len(pre) == 1:也行
            return TreeNode(pre[0])
        if len(pre) < 1:
            return None
        gap_index = post.index(pre[1])
        root = TreeNode(pre[0])
        root.left = self.constructFromPrePost(pre[1:gap_index+2],post[:gap_index+1])
        root.right = self.constructFromPrePost(pre[gap_index+2:],post[gap_index+1:-1])
        return root
```

# 919. 完全二叉树插入器

完全二叉树是每一层（除最后一层外）都是完全填充（即，节点数达到最大）的，并且所有的节点都尽可能地集中在左侧。

设计一个用完全二叉树初始化的数据结构 CBTInserter，它支持以下几种操作：

CBTInserter(TreeNode root) 使用头节点为 root 的给定树初始化该数据结构；
CBTInserter.insert(int v)  向树中插入一个新节点，节点类型为 TreeNode，值为 v 。使树保持完全二叉树的状态，并返回插入的新节点的父节点的值；
CBTInserter.get_root() 将返回树的头节点。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class CBTInserter:

    def __init__(self, root: TreeNode):
        # 完全二叉树，使用数组作为容器解决，不用管节点值，只使用节点索引来
        self.TreeArray = [root] # 
        self.BFS(root) # 填充数组，此时每个节点都映射到了数组中

    def insert(self, v: int) -> int: # 返回的是父节点的值
        new_node = TreeNode(v)
        self.TreeArray.append(new_node)
        # 它的父节点的索引为（它的索引-1）//2
        # 如果它的索引是奇数，它是父节点的左孩子，如果它的索引是偶数，它是父节点的右孩子
        the_index = len(self.TreeArray) - 1 # 获取这个节点的索引
        parent_index = (the_index - 1) // 2 # 获取它父节点的索引
        if the_index % 2 == 1:
            self.TreeArray[parent_index].left = self.TreeArray[the_index]
        elif the_index % 2 == 0:
            self.TreeArray[parent_index].right = self.TreeArray[the_index]
        return self.TreeArray[parent_index].val # 返回父节点的值

    def get_root(self) -> TreeNode:
        return self.TreeArray[0]

    def BFS(self,root): # 填充
        queue = [root]
        while len(queue) != 0:
            new_queue = []
            for node in queue:
                if node.left != None:
                    self.TreeArray.append(node.left)
                    new_queue.append(node.left)
                if node.right != None:
                    self.TreeArray.append(node.right)
                    new_queue.append(node.right)
            queue = new_queue
```

# 921. 使括号有效的最少添加

给定一个由 '(' 和 ')' 括号组成的字符串 S，我们需要添加最少的括号（ '(' 或是 ')'，可以在任何位置），以使得到的括号字符串有效。

从形式上讲，只有满足下面几点之一，括号字符串才是有效的：

它是一个空字符串，或者
它可以被写成 AB （A 与 B 连接）, 其中 A 和 B 都是有效字符串，或者
它可以被写作 (A)，其中 A 是有效字符串。
给定一个括号字符串，返回为使结果字符串有效而必须添加的最少括号数。

```python
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        # 最终left需要等于right
        left = 0
        need = 0
        for ch in s:
            if ch == "(":
                left += 1
            elif ch == ")":
                if left >= 1:
                    left -= 1
                elif left == 0: # 没有匹配的左括号，直接加进去
                    need += 1
        # 扫完之后left如果有剩余，都需要补上
        need += left
        return need

```

# 923. 三数之和的多种可能

给定一个整数数组 A，以及一个整数 target 作为目标值，返回满足 i < j < k 且 A[i] + A[j] + A[k] == target 的元组 i, j, k 的数量。

由于结果会非常大，请返回 结果除以 10^9 + 7 的余数。

```python
class Solution:
    def threeSumMulti(self, arr: List[int], target: int) -> int:
        mod = 10 ** 9 + 7
        # 数组排序之后，先转化成元素：频次。进行朴素三数之和
        # 如果三个元素各不相等。ans += 三频次积
        # 如果允许前两个相等，转化成两数之和。然后cn2 * 频次3
        # 如果允许后两个相等，转化成为两数之和 频次 * cn2
        # 如果允许三个相等，cn3
        arr.sort()
        ct = collections.Counter(arr)
        ans = 0
        short_arr = [i for i in ct]
        for i in range(len(short_arr)): # 收集三数不相等的情况
            aim = target - short_arr[i]
            left = i + 1
            right = len(short_arr) - 1
            while left < right:
                if short_arr[left] + short_arr[right] == aim:
                    ans += ct[short_arr[i]]*ct[short_arr[left]]*ct[short_arr[right]]
                    left += 1
                    right -= 1
                elif short_arr[left] + short_arr[right] < aim: # 数值偏小，小指针右移   
                    left += 1
                elif short_arr[left] + short_arr[right] > aim: # 数值偏大，大指针左移
                    right -= 1
        for i in range(len(short_arr)): # 收集有两个相等，第三个不能相等的情况
            if ct[short_arr[i]] < 2:
                continue
            aim = target - short_arr[i]*2
            if aim in ct and aim != short_arr[i]: # 注意这一行，很重要！！！
                n = ct[short_arr[i]]
                ans += ct[aim]*(n*(n-1))//2 # cn2的组合数
        if target % 3 == 0:
            aim = target//3
            if ct[aim] >= 3:
                n = ct[aim]
                ans += (n)*(n-1)*(n-2)//6
        return ans % mod # 结果别忘记取模

```

# 931. 下降路径最小和

给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。

下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。

```python
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        # dp[i][j]表示通过第i行第j列的最小和值
        # 边界条件处理为极大值
        # 申请n行，n列
        n = len(matrix)
        dp = [[0xffffffff for i in range(n)] for k in range(n)]
        # 第一行直接填充
        dp[0] = matrix[0]
        # 状态转移为 dp[i][j] = matrix[i][j] + min(dp[i-1][j-1],dp[i-1][j],dp[i-1][j+1])
        # 注意边界条件,填充顺序为从上到下，从左到右
        for i in range(1,n): # 第一行无需填充
            for j in range(n):
                if j-1 < 0:
                    dp[i][j] = matrix[i][j] + min(dp[i-1][j],dp[i-1][j+1])
                elif j+1 >= n:
                    dp[i][j] = matrix[i][j] + min(dp[i-1][j-1],dp[i-1][j])
                else:
                    dp[i][j] = matrix[i][j] + min(dp[i-1][j-1],dp[i-1][j],dp[i-1][j+1])
        return min(dp[-1]) # 返回最后一行每一个位置为终点时的最小值
```

# 942. 增减字符串匹配

给定只含 "I"（增大）或 "D"（减小）的字符串 S ，令 N = S.length。

返回 [0, 1, ..., N] 的任意排列 A 使得对于所有 i = 0, ..., N-1，都有：

如果 S[i] == "I"，那么 A[i] < A[i+1]
如果 S[i] == "D"，那么 A[i] > A[i+1]

```python
class Solution:
    def diStringMatch(self, s: str) -> List[int]:
        # 从左到右，第一个I所对应的值为0,然后递增
        # 其余的空下来的数字递减填充
        ans = [0xffffffff for i in range(len(s)+1)]
        the_val = 0
        for index,val in enumerate(s):
            if val == "I":
                ans[index] = the_val
                the_val += 1
        another_val = len(ans) - 1
        for i,val in enumerate(ans):
            if ans[i] == 0xffffffff:
                ans[i] = another_val
                another_val -= 1
        return ans
```

# 945. 使数组唯一的最小增量

给定整数数组 A，每次 *move* 操作将会选择任意 `A[i]`，并将其递增 `1`。

返回使 `A` 中的每个值都是唯一的最少操作次数。

```python
class Solution:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        operations = 0 # 记录操作次数
        nums.sort()
        # 堆方法
        # 建立一个大顶堆，如果要进来的数大于堆顶，则直接进来
        # 如果要进来的数小于等于堆顶，则需要做差
        # 注意内置的是小根堆所以逻辑更改为
        # 进来的数小于堆顶，则直接进来
        # 进来的数大于等于堆顶，则需要变化
        max_heap = [-nums[0]]
        for i in nums[1:]:
            if -i < max_heap[0]:
                heapq.heappush(max_heap,-i)
            elif -i >= max_heap[0]:
                gap = abs(-i-max_heap[0]) + 1 # 计算差值
                operations += gap # 计算需要的增量
                heapq.heappush(max_heap,-i-gap)
        return operations
            
```

# 946. 验证栈序列

给定 pushed 和 popped 两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false 。

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        # 已知值都不重复了
        # 按照入栈序列进行模拟
        stack = []
        p = 0
        for i in pushed:
            stack.append(i)
            while len(stack) != 0 and stack[-1] == popped[p]: # 如果此时栈顶和需要的栈顶元素相同，以while弹出
                stack.pop()
                p += 1
        # 注意，扫描完毕之后p最后还+了1，所以返回的是p是否为全长度
        return p == len(pushed) and stack == []
```

# 951. 翻转等价二叉树

我们可以为二叉树 T 定义一个翻转操作，如下所示：选择任意节点，然后交换它的左子树和右子树。

只要经过一定次数的翻转操作后，能使 X 等于 Y，我们就称二叉树 X 翻转等价于二叉树 Y。

编写一个判断两个二叉树是否是翻转等价的函数。这些树由根节点 root1 和 root2 给出。

```python
class Solution:
    def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool:
        # 检查每个节点的父节点的值是否一样即可。而不必去考虑如何翻转
        if root1 == None and root2 != None:
            return False
        elif root1 != None and root2 == None:
            return False
        elif root1 == None and root2 == None:
            return True

        self.memo1 = dict()
        self.memo2 = dict()
        self.memo1[root1.val] = None # 填充根节点初值
        self.memo2[root2.val] = None # 填充根节点初值
        self.find_parent(root1,self.memo1)
        self.find_parent(root2,self.memo2)
        return (self.memo1 == self.memo2)
    
    def find_parent(self,root,memo): # 填充每个节点的父节点
        def dfs(node,parent):
            if node == None:
                return
            if node != None and parent != None:
                memo[node.val] = parent.val
            dfs(node.left,node)
            dfs(node.right,node)
        dfs(root,None)
           
```

# 953. 验证外星语词典

某种外星语也使用英文小写字母，但可能顺序 order 不同。字母表的顺序（order）是一些小写字母的排列。

给定一组用外星语书写的单词 words，以及其字母表的顺序 order，只有当给定的单词在这种外星语中按字典序排列时，返回 true；否则，返回 false。

```python
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        # 将order构建成0～25的映射之后排序。
        # 只有一个单词肯定排序正确
        if len(words) == 1:
            return True
        nums = [i for i in range(26)]
        order_lst = [char for char in order]
        self.alien_dict = dict(zip(order_lst,nums))
        for i in range(1,len(words)): # 开始扫描检查
            if self.check(words[i-1],words[i]) == False:
                return False # 过筛失败
        return True # 过筛成功
    
    def check(self,word1,word2): # 需要检查word1是否小于word2
        # 比较规则是比较第一个不同的字母
        p = 0
        while p < len(word1) and p < len(word2):
            if word1[p] == word2[p]:
                p += 1
            elif self.alien_dict[word1[p]] < self.alien_dict[word2[p]]:
                return True
            elif self.alien_dict[word1[p]] > self.alien_dict[word2[p]]:
                return False
        # 如果word2到头了，word1还没有到头，返回False
        if p == len(word2) and p < len(word1):
            return False
        else: # 包含了1.两者一样，和2.word1是word2的前缀
            return True


```

# 958. 二叉树的完全性检验

给定一个二叉树，确定它是否是一个完全二叉树。

百度百科中对完全二叉树的定义如下：

若设二叉树的深度为 h，除第 h 层外，其它各层 (1～h-1) 的结点数都达到最大个数，第 h 层所有的结点都连续集中在最左边，这就是完全二叉树。（注：第 h 层可能包含 1~ 2h 个节点。）

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCompleteTree(self, root: TreeNode) -> bool:
        # 遍历计数，
        # BFS
        # 由于树非空
        ans2 = [(root,1)] # 收集标号数量结果
        queue = [(root,1)] # 借助队列管理
        while len(queue) != 0:
            level = [] # 收集本层结果
            new_queue = [] # 收集下层需要迭代的值
            for i in queue:
                if i[0] != None:
                    level.append(i[1])
                if i[0].left != None:
                    new_queue.append((i[0].left,2*i[1]))
                if i[0].right != None:
                    new_queue.append((i[0].right,2*i[1]+1))
            ans2 += new_queue
            queue = new_queue
        # 此时ans2的最后一个数的计数属性是不是等于长度
        # 等于则完全 ， 否则则非完全二叉树
        return ans2[-1][1] == len(ans2)

```

# 969. 煎饼排序

给你一个整数数组 arr ，请使用 煎饼翻转 完成对数组的排序。

一次煎饼翻转的执行过程如下：

选择一个整数 k ，1 <= k <= arr.length
反转子数组 arr[0...k-1]（下标从 0 开始）
例如，arr = [3,2,1,4] ，选择 k = 3 进行一次煎饼翻转，反转子数组 [3,2,1] ，得到 arr = [1,2,3,4] 。

以数组形式返回能使 arr 有序的煎饼翻转操作所对应的 k 值序列。任何将数组排序且翻转次数在 10 * arr.length 范围内的有效答案都将被判断为正确。

```python
class Solution:
    def pancakeSort(self, arr: List[int]) -> List[int]:
        # 尬算出非最佳答案。每次找到最大的index。然后让他在最底下
        indexList = []

        def recur(arr,n): # n递减到1
            if n == 1:
                return arr
            index = -1
            for i in range(len(arr)): # 找到最大的那个
                if arr[i] == n:
                    index = i 
                    break
            if index != 0: # 节约次数
                indexList.append(index+1) # 收集第一铲
            temp = arr[0:index+1]
            indexList.append(n) # 手机第二铲
            arr = temp[::-1] + arr[index+1:]
            arr = arr[:n][::-1]+arr[n:]
            return recur(arr,n-1)
        
        arr = recur(arr,len(arr))
        return indexList
```

# 987. 二叉树的垂序遍历

给你二叉树的根结点 root ，请你设计算法计算二叉树的 垂序遍历 序列。

对位于 (row, col) 的每个结点而言，其左右子结点分别位于 (row + 1, col - 1) 和 (row + 1, col + 1) 。树的根结点位于 (0, 0) 。

二叉树的 垂序遍历 从最左边的列开始直到最右边的列结束，按列索引每一列上的所有结点，形成一个按出现位置从上到下排序的有序列表。如果同行同列上有多个结点，则按结点的值从小到大进行排序。

返回二叉树的 垂序遍历 序列。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        # BFS收集之后， 重新添加进结果中
        # 如果同行同列上有多个结点，则按结点的值从小到大进行排序。
        bfs_ans = []
        if root == None:
            return []
        queue = [(root,(0,0))]
        while len(queue) != 0:
            new_queue = []
            for cp in queue:
                if cp[0] != None:
                    bfs_ans.append([cp[0].val,cp[1]]) # 把索引加进去
                if cp[0].left != None:
                    new_queue.append((cp[0].left,(cp[1][0]+1,cp[1][1]-1)))
                if cp[0].right != None:
                    new_queue.append((cp[0].right,(cp[1][0]+1,cp[1][1]+1)))
            queue = new_queue
        # print(ans) # 此时收集到的ans里面，是很多数组，数组为[值，序号]
        bfs_ans.sort(key = lambda x:x[1][1]) # 用序号排序，不会打乱结果
        the_dict = collections.defaultdict(list)
        for cp in bfs_ans: # the_dict里面存的是【值，坐标】
            the_dict[cp[1]].append(cp[0])
        for key in the_dict:
            the_dict[key].sort()
        # print(the_dict)
        dic2 = collections.defaultdict(list) # 【dic2里面存的是值，结果
        for key in the_dict:
            dic2[key[1]]+= the_dict[key]
        # print(dic2)
        ans = []
        for value in dic2.values():
            ans.append(value)
        return ans
        # 这一题调试了挺久
```

# 991. 坏了的计算器

在显示着数字的坏计算器上，我们可以执行以下两种操作：

双倍（Double）：将显示屏上的数字乘 2；
递减（Decrement）：将显示屏上的数字减 1 。
最初，计算器显示数字 X。

返回显示数字 Y 所需的最小操作数。

```python
class Solution:
    def brokenCalc(self, x: int, y: int) -> int:
        # 正面想：x小于y时候，接近y有两种方式，-1 再 乘以2 或者直接乘以2
        # 倒过来想，要从y变成x，则只根据奇偶性质有一种方式
        # 如果y大于x且为奇数，则加一再除以2
        # 如果y小于x则只能加了,为了提高性能，不用y一个一个的加。直接全加上就行
        times = 0
        while y != x:
            if y > x:
                if y > x and y % 2 == 1:
                    y += 1 # 不采取减的方式是因为，如果采取减，则下一轮可能要再接近x的时候重新加回来
                    y = y // 2
                    times += 2
                elif y > x and y % 2 == 0:
                    y = y // 2
                    times += 1
            elif y < x:
                times += (x-y)
                return times
        return times

```

# 993. 二叉树的堂兄弟节点

在二叉树中，根节点位于深度 0 处，每个深度为 k 的节点的子节点位于深度 k+1 处。

如果二叉树的两个节点深度相同，但 父节点不同 ，则它们是一对堂兄弟节点。

我们给出了具有唯一值的二叉树的根节点 root ，以及树中两个不同节点的值 x 和 y 。

只有与值 x 和 y 对应的节点是堂兄弟节点时，才返回 true 。否则，返回 false。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        # 唯一值二叉树,x,y在树中
        ans = []

        def dfs(root,node_val,parent,depth):
            if root == None:
                return 
            if root.val == node_val:
                if parent != None:
                    ans.append((parent.val,depth)) # 收集父节点值和深度即可
                else:
                    ans.append((None,depth))
            parent = root
            dfs(root.left,node_val,parent,depth+1)
            dfs(root.right,node_val,parent,depth+1)

        dfs(root,x,None,0)
        dfs(root,y,None,0)
        # ans存俩元组，(【父节点值】，【深度】)
        if ans[0][0] != ans[1][0] and ans[0][1] == ans[1][1]:
            return True
        return False

```

# 997. 找到小镇的法官

在一个小镇里，按从 1 到 n 为 n 个人进行编号。传言称，这些人中有一个是小镇上的秘密法官。

如果小镇的法官真的存在，那么：

小镇的法官不相信任何人。
每个人（除了小镇法官外）都信任小镇的法官。
只有一个人同时满足条件 1 和条件 2 。
给定数组 trust，该数组由信任对 trust[i] = [a, b] 组成，表示编号为 a 的人信任编号为 b 的人。

如果小镇存在秘密法官并且可以确定他的身份，请返回该法官的编号。否则，返回 -1。

```python
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        # 这一题的意思很绕口，实际上把他们看作节点后，即考虑是否仅存在一个节点入度为n-1，且没有出度
        people = [[0,0] for i in range(n+1)] # 记录入度和出度
        people[0] = [-1,-1] # 0号无效
        # trust[0] 信任 -> trust[1]
        for i in trust:
            people[i[1]][0] += 1 # 被信任的入度+1
            people[i[0]][1] += 1 # 信任别人的出度+1
        count = [] # 默认没有法官
        for index,person in enumerate(people):
            if person[0] == n - 1 and person[1] == 0:
                count.append(index)
        if len(count) == 1:
            return count[0]
        else:
            return -1
               
```

# 1002. 查找常用字符

给定仅有小写字母组成的字符串数组 A，返回列表中的每个字符串中都显示的全部字符（包括重复字符）组成的列表。例如，如果一个字符在每个字符串中出现 3 次，但不是 4 次，则需要在最终答案中包含该字符 3 次。

你可以按任意顺序返回答案。

```python
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        # 对每个词做一个字母哈希表统计
        hash_lst = []
        for w in words:
            the_count = [0 for i in range(26)]
            for char in w:
                the_count[ord(char)-ord("a")] += 1
            hash_lst.append(the_count)
        m = len(words) # 有m行
        ans = []
        for j in range(26): # 取最小值
            min_num = 0xffffffff
            for i in range(m):
                min_num = min(min_num,hash_lst[i][j])
            ans.append(min_num)
        final = ''
        for index in range(26):
            final += chr(97+index)*ans[index]
        final = list(final) # 还原成列表
        return final
            
```

# 1004. 最大连续1的个数 III

给定一个由若干 `0` 和 `1` 组成的数组 `A`，我们最多可以将 `K` 个值从 0 变成 1 。

返回仅包含 1 的最长（连续）子数组的长度。

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        # 允许变k个。k可能为0
        # 传统right的更新
        # 分情况考虑的时候，只要是0就加进来，而不是分为是否是第k+1个0
        # 滑动窗口
        left = 0
        right = 0
        size = 0
        max_size = 0
        countZero = 0
        while right < len(nums):
            add = nums[right]
            right += 1
            if add != 0:
                size += 1
                max_size = max(max_size,size)
            elif add == 0:
                size += 1
                countZero += 1
                if countZero <= k:
                    max_size = max(max_size,size)
            while left < right and countZero > k:
                delete = nums[left]
                if delete == 0:
                    countZero -= 1
                size -= 1
                left += 1
        return max_size
```

# 1008. 前序遍历构造二叉搜索树

返回与给定前序遍历 preorder 相匹配的二叉搜索树（binary search tree）的根结点。

(回想一下，二叉搜索树是二叉树的一种，其每个节点都满足以下规则，对于 node.left 的任何后代，值总 < node.val，而 node.right 的任何后代，值总 > node.val。此外，前序遍历首先显示节点 node 的值，然后遍历 node.left，接着遍历 node.right。）

题目保证，对于给定的测试用例，总能找到满足要求的二叉搜索树。

```python
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        # 递归构建
        # 找到左右的分界线
        if len(preorder) == 0:
            return None
        if len(preorder) == 1:
            return TreeNode(preorder[0])
        # 需要用查找法查到第一个大于根的树的索引
        # 这里用的是线性搜索
        index = len(preorder) # 初始化为最后一个值，如果搜完也没找到那就是这个值
        for i in range(len(preorder)):
            if preorder[i] > preorder[0]:
                index = i
                break
        root = TreeNode(preorder[0])
        root.left = self.bstFromPreorder(preorder[1:index])
        root.right = self.bstFromPreorder(preorder[index:]) # python切片可以越界。
        return root
```

# 1011. 在 D 天内送达包裹的能力

传送带上的包裹必须在 D 天内从一个港口运送到另一个港口。

传送带上的第 i 个包裹的重量为 weights[i]。每一天，我们都会按给出重量的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。

返回能在 D 天内将传送带上的所有包裹送达的船的最低运载能力。

```python
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        all_sum = sum(weights)
        # 伪二分查找，起始重量为货物单个重量的最大值
        # 由于里面存着while循环，所以有线性区间
        # left,right是承载量
        maxWeight = max(weights)
        left = maxWeight
        right = all_sum
        # 过程中不能低于left
        while left <= right:
            mid = (left + right)//2
            d = self.calc_Days(weights,mid)
            if left == mid: # 这一句也不能省略。。
                # 此时left和right最多相差1
                d1 = self.calc_Days(weights,left)
                d2 = self.calc_Days(weights,left+1)
                if d1 <= days: # 如果d1已经满足，则返回left,注意必须是小于等于号
                    return left
                else: # 否则返回right
                    return left+1
            if d == days:
                while self.calc_Days(weights,mid) == days: # 这一句很无奈。。。
                    mid -= 1
                return mid+1
            elif d > days: # 所需时长过长，需要加大容量
                left = mid + 1
            elif d < days: # 所需时长过短，需要减少容量
                right = mid - 1
            
    def calc_Days(self,lst,limit): # 贪心计算容量为limit的时候的需要天数
        days = 1
        tempSum = 0
        for i in lst:
            if tempSum + i <= limit:
                tempSum += i
            else:
                days += 1
                tempSum = i
        return days
        
```

```python
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        # 标准二分法。
        # 起始left为货物中的最大值
        # right初始化为总重量
        left = max(weights)
        right = sum(weights)
        while left <= right: # left ,right 是重量
            mid = (left + right)//2
            judge = self.judgeValid(weights,mid,days) # 注意传入参数
            if judge: # 还能减少承载量
                right = mid - 1
            elif not judge: # 承载量不够，需要增大
                left = mid + 1            
        return left
    
    def judgeValid(self,weights,v,limit): # 传入参数为列表，以v承重，limit为限制
        d = 1
        now = 0
        for i in weights:
            if i + now <= v:
                now += i
            else:
                d += 1
                now = i
        return d <= limit # 时长必须小于等于限制
```



# 1013. 将数组分成和相等的三个部分

给你一个整数数组 arr，只有可以将其划分为三个和相等的 非空 部分时才返回 true，否则返回 false。

形式上，如果可以找出索引 i + 1 < j 且满足 (arr[0] + arr[1] + ... + arr[i] == arr[i + 1] + arr[i + 2] + ... + arr[j - 1] == arr[j] + arr[j + 1] + ... + arr[arr.length - 1]) 就可以将数组三等分。

```python
class Solution:
    def canThreePartsEqualSum(self, arr: List[int]) -> bool:
        # 神奇前缀和,包含本位置的前缀和
        # 补丁版本
        presum = []
        temp_sum = 0
        for i in arr:
            temp_sum += i
            presum.append(temp_sum)
        if temp_sum % 3 != 0:
            return False
        if temp_sum == 0: # 打补丁
            times = 0
            for i in presum:
                if i == 0:
                    times += 1
            if times >= 3: return True
            else: return False
        t1 = temp_sum // 3
        t2 = t1 * 2
        valid1 = [False,-1]
        valid2 = [False,-1]
        for i in range(len(presum)):
            if presum[i] == t1 and valid1[0] == False: 
                valid1[0] = True
                valid1[1] = i
            elif presum[i] == t2: # 这个可以尽量靠右
                valid2[0] = True
                valid2[1] = i
        return valid1[0]==valid2[0] and valid1[1] < valid2[1] and valid2[1] != len(presum) - 1

```



# 1022. 从根到叶的二进制数之和

给出一棵二叉树，其上每个结点的值都是 0 或 1 。每一条从根到叶的路径都代表一个从最高有效位开始的二进制数。例如，如果路径为 0 -> 1 -> 1 -> 0 -> 1，那么它表示二进制数 01101，也就是 13 。

对树上的每一片叶子，我们都要找出从根到该叶子的路径所表示的数字。

返回这些数字之和。题目数据保证答案是一个 32 位 整数。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumRootToLeaf(self, root: TreeNode) -> int:
        # dfs,收集路径
        ans = []
        path = []
        def dfs(node):
            if node == None:
                return 
            path.append(node.val) # 做选择
            if node.left == None and node.right == None:
                ans.append(path[:])
            dfs(node.left)
            dfs(node.right)
            path.pop() # 取消选择

        dfs(root) # 调用dfs
        def toBinary(lst): # 将每条路径转换成数字
            num = 0
            bit = 0
            while lst:
                num += lst.pop()*2**bit
                bit += 1
            return num

        the_final_sum = 0
        for every in ans:
            the_final_sum += toBinary(every)
        return the_final_sum

```

# 1046. 最后一块石头的重量

有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出两块 最重的 石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        # 维护堆,要弹出的是最重的，维护大根堆，
        max_heap = []
        for i in stones:
            max_heap.append(-i)
        heapq.heapify(max_heap) # 堆化
        while len(max_heap) >= 2: # 只要里面还有俩石头就弹
            stone1 = heapq.heappop(max_heap)
            stone2 = heapq.heappop(max_heap)
            new_stone = -abs(stone1-stone2)
            if new_stone != 0:
                heapq.heappush(max_heap,new_stone)
        if len(max_heap) == 0: return 0
        else: return -max_heap[0]

```

# 1051. 高度检查器

学校打算为全体学生拍一张年度纪念照。根据要求，学生需要按照 非递减 的高度顺序排成一行。

排序后的高度情况用整数数组 expected 表示，其中 expected[i] 是预计排在这一行中第 i 位的学生的高度（下标从 0 开始）。

给你一个整数数组 heights ，表示 当前学生站位 的高度情况。heights[i] 是这一行中第 i 位学生的高度（下标从 0 开始）。

返回满足 heights[i] != expected[i] 的 下标数量 。

```python
class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        # 纯排序法，和排序之后的比较
        copy_arr = heights.copy()
        copy_arr.sort()
        count = 0
        for i in range(len(heights)):
            if heights[i] != copy_arr[i]:
                count += 1
        return count
```

```python
class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        # 桶排序法,取最大值减小桶设置，不需要设置到100
        max_h = max(heights)
        bucket = [0 for i in range(max_h+1)] # 索引为高度，没有身高为0的，空置
        for h in heights:
            bucket[h] += 1 # 对应身高人数+1
        p = 1
        p_h = 0
        count = 0
        while p <= max_h:
            if bucket[p] == 0:
                p += 1
            else:
                if heights[p_h] != p: # 这个人的身高不和桶索引一样
                    count += 1
                p_h += 1
                bucket[p] -= 1
        return count

```

# 1079. 活字印刷

你有一套活字字模 `tiles`，其中每个字模上都刻有一个字母 `tiles[i]`。返回你可以印出的非空字母序列的数目。

**注意：**本题中，每个活字字模只能使用一次。

```python
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        # set去重，回溯
        ans = set()
        path = []
        index = 0
        length = len(tiles)
        tiles = list(tiles)

        def backtracking(path,choice): # 
            if len(choice) == 0:
                return
            for ch in choice:
                cp = choice.copy()
                cp.remove(ch) # 需要的是全排列，则需要用拷贝剩余可选方案的方法
                path.append(ch)
                ans.add("".join(path[:]))
                backtracking(path,cp)
                path.pop()

        backtracking(path,tiles)
        return len(ans)

```

# 1085. 最小元素各数位之和

给你一个正整数的数组 A。

然后计算 S，使其等于数组 A 当中最小的那个元素各个数位上数字之和。

最后，假如 S 所得计算结果是 奇数 ，返回 0 ；否则请返回 1。

```python
class Solution:
    def sumOfDigits(self, nums: List[int]) -> int:
        # 模拟
        num = str(min(nums))
        the_sum = 0
        for i in num:
            the_sum += int(i)
        return 0 if the_sum % 2 == 1 else 1
```

# 1094. 拼车

假设你是一位顺风车司机，车上最初有 capacity 个空座位可以用来载客。由于道路的限制，车 只能 向一个方向行驶（也就是说，不允许掉头或改变方向，你可以将其想象为一个向量）。

这儿有一份乘客行程计划表 trips[][]，其中 trips[i] = [num_passengers, start_location, end_location] 包含了第 i 组乘客的行程信息：

必须接送的乘客数量；
乘客的上车地点；
以及乘客的下车地点。
这些给出的地点位置是从你的 初始 出发位置向前行驶到这些地点所需的距离（它们一定在你的行驶方向上）。

请你根据给出的行程计划表和车子的座位数，来判断你的车是否可以顺利完成接送所有乘客的任务（当且仅当你可以在所有给定的行程中接送所有乘客时，返回 true，否则请返回 false）。

```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        # 按照上车，下车做哈希表映射？
        up = collections.defaultdict(int) # 记录上车人数 时间:人数
        down = collections.defaultdict(int) # 记录下车人数 时间:人数
        for cp in trips:
            up[cp[1]] += cp[0]
            down[cp[2]] -= cp[0]
        travel = collections.defaultdict(int)
        for key in up:
            if key in down:
                travel[key] += down[key] # 注意本身down是负数
                del down[key]
            travel[key] += up[key]
        travel.update(down)
        # 现在已经整理成为按照时间点上下车的了
        # 求最大值即可
        sort_travel = []
        for key in travel:
            sort_travel.append([key,travel[key]])
        sort_travel.sort()
        now = 0
        max_people = -1
        for cp in sort_travel:
            now += cp[1]
            max_people = max(max_people,now)
        return capacity >= max_people

```

# 1099. 小于 K 的两数之和

给你一个整数数组 nums 和整数 k ，返回最大和 sum ，满足存在 i < j 使得 nums[i] + nums[j] = sum 且 sum < k 。如果没有满足此等式的 i,j 存在，则返回 -1 。

```python
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        left = 0
        right = len(nums)-1
        max_sum = -1
        nums.sort() # 排序一下
        while left < right:
            if nums[left] + nums[right] >= k: # 大于等于，则right要缩小
                right -= 1
            elif nums[left] + nums[right] < k: # 否则记录答案之后增大
                max_sum = max(max_sum, nums[left] + nums[right])
                left += 1
        return max_sum

```

# 1056. 易混淆数

给定一个数字 N，当它满足以下条件的时候返回 true：

原数字旋转 180° 以后可以得到新的数字。

如 0, 1, 6, 8, 9 旋转 180° 以后，得到了新的数字 0, 1, 9, 8, 6 。

2, 3, 4, 5, 7 旋转 180° 后，得到的不是数字。

易混淆数 (confusing number) 在旋转180°以后，可以得到和原来不同的数，且新数字的每一位都是有效的。

```python
class Solution:
    def confusingNumber(self, n: int) -> bool:
        # 模拟
        if n == 0:
            return False
        dic = dict(zip([0, 1, 6, 8, 9],[0, 1, 9, 8, 6]))
        s = str(n)[::-1]
        temp = []
        for ch in s:
            if int(ch) not in dic: return False
            else:
                temp.append(dic[int(ch)])
        # 还需要去掉前导0，其实只要有前导0就一定不相等
        p = 0
        while temp[p] == 0:
            p += 1
        final = []
        # 此时p指向非0
        ans = ''
        while p < len(temp):
            ans += str(temp[p])
            p += 1
        return ans != str(n)
```

# 1061. 按字典序排列最小的等效字符串

给出长度相同的两个字符串：A 和 B，其中 A[i] 和 B[i] 是一组等价字符。举个例子，如果 A = "abc" 且 B = "cde"，那么就有 'a' == 'c', 'b' == 'd', 'c' == 'e'。

等价字符遵循任何等价关系的一般规则：

自反性：'a' == 'a'
对称性：'a' == 'b' 则必定有 'b' == 'a'
传递性：'a' == 'b' 且 'b' == 'c' 就表明 'a' == 'c'
例如，A 和 B 的等价信息和之前的例子一样，那么 S = "eed", "acd" 或 "aab"，这三个字符串都是等价的，而 "aab" 是 S 的按字典序最小的等价字符串

利用 A 和 B 的等价信息，找出并返回 S 的按字典序排列最小的等价字符串。

```python
# 未优化并查集
class UF:
    def __init__(self,size):
        self.root = [i for i in range(size)]
    
    def find(self,x):
        while x != self.root[x]:
            x = self.root[x]
        return x 
    
    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootX] = rootY
    
class Solution:
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        # 格式化转为列表
        lst1 = [ord(ch)-ord("a") for ch in s1]
        lst2 = [ord(ch)-ord("a") for ch in s2]

        n = len(lst1)

        theUnionFind = UF(26) # 参数为26

        for i in range(n):
            theUnionFind.union(lst1[i],lst2[i])
        
        theGroup = collections.defaultdict(list)

        for i in range(26): # 注意参数是26
            mark = theUnionFind.find(i)
            theGroup[mark].append(i)
            # theGroup[i].append(mark)
        
        for i in range(26):
            theGroup[mark].sort() # 排序成最小的
        
        for key in range(26):
            if theGroup.get(key) == None:
                theGroup[key] = []

        for key in theGroup:
            for element in theGroup[key]:
                theGroup[element] = theGroup[key]

        ans = ""
        for ch in baseStr:
            index = ord(ch) - ord("a")
            tempCh = chr(theGroup[index][0]+97)
            ans += tempCh
        return ans
```

# 1064. 不动点

给定已经按 **升序** 排列、由不同整数组成的数组 `arr`，返回满足 `arr[i] == i` 的最小索引 `i`。如果不存在这样的 `i`，返回 `-1`。

```python
class Solution:
    def fixedPoint(self, arr: List[int]) -> int:
        # 线性搜索法
        for i in range(len(arr)):
            if arr[i] == i:
                return i 
        return -1
```

```python
class Solution:
    def fixedPoint(self, arr: List[int]) -> int:
        # 二分法
        # 如果中点的数字小于索引，往右边搜
        # 如果中点的数字大于索引，往左边搜
        # 如果中点的数字等于索引，还得往左边搜,搜左边界
        left = 0
        right = len(arr) - 1
        while left <= right: # while内打补丁的方法
            mid = (left + right) // 2
            # print(mid,arr[mid],"~~",arr[left:right+1]) 检查用
            if left == right:
                if arr[mid] == mid:
                    return mid
                else:
                    break
            if mid == arr[mid]:
                right = mid
            elif arr[mid] < mid :
                left = mid + 1
            elif arr[mid] > mid :
                right = mid - 1
        return -1
```

```python
class Solution:
    def fixedPoint(self, arr: List[int]) -> int:
        # 二分法
        # 如果中点的数字小于索引，往右边搜
        # 如果中点的数字大于索引，往左边搜
        # 如果中点的数字等于索引，还得往左边搜,搜左边界
        # 不在while内打补丁
        left = 0
        right = len(arr) - 1
        while left <= right: # while内打补丁的方法
            mid = (left + right) // 2
            # print(mid,arr[mid],"~~",arr[left:right+1]) 检查用
            if mid == arr[mid]:
                right = mid - 1
            elif arr[mid] < mid :
                left = mid + 1
            elif arr[mid] > mid :
                right = mid - 1
        # 检查left
        if left >= len(arr):
            return -1
        if left == arr[left]:
            return left
        return -1
```

# 1100. 长度为 K 的无重复字符子串

给你一个字符串 `S`，找出所有长度为 `K` 且不含重复字符的子串，请你返回全部满足要求的子串的 **数目**。

```python
class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        # 固定窗口大小的滑动窗口
        if k > len(s):
            return 0
        # 初始化窗口,固定化窗口大小
        left = 0
        right = k
        window = collections.defaultdict(int)
        for char in s[:k]:
            window[char] += 1
        count = 0
        if len(window) == k: # 判断初始化窗口是否有效
            count += 1
        while right < len(s):
            # print(window)
            add_char = s[right]
            delete_char = s[left]
            window[add_char] += 1
            window[delete_char] -= 1
            if window[delete_char] == 0:
                del window[delete_char]
            if len(window) == k:
                count += 1
            left += 1
            right += 1
        return count
```

# 1111. 有效括号的嵌套深度

【题目描述过于晦涩难懂。。。】

给你一个「有效括号字符串」 seq，请你将其分成两个不相交的有效括号字符串，A 和 B，并使这两个字符串的深度最小。

不相交：每个 seq[i] 只能分给 A 和 B 二者中的一个，不能既属于 A 也属于 B 。
A 或 B 中的元素在原字符串中可以不连续。
A.length + B.length = seq.length
深度最小：max(depth(A), depth(B)) 的可能取值最小。 
划分方案用一个长度为 seq.length 的答案数组 answer 表示，编码规则如下：

answer[i] = 0，seq[i] 分给 A 。
answer[i] = 1，seq[i] 分给 B 。
如果存在多个满足要求的答案，只需返回其中任意 一个 即可。

```python
class Solution:
    def maxDepthAfterSplit(self, seq: str) -> List[int]:
        depth = []
        left = 0
        # 记录括号的原本深度
        for ch in seq:
            if ch == "(":
                left += 1
                depth.append(left)
            elif ch == ")":
                depth.append(left)
                left -= 1
        halfDepth = max(depth)//2 # 大于这个深度的分为一组
        # 小于等于这个深度的分为一组
        ans = []
        for i in depth:
            if i > halfDepth:
                ans.append(1)
            else:
                ans.append(0)
        return ans
```

# 1119. 删去字符串中的元音

给你一个字符串 `S`，请你删去其中的所有元音字母（ `'a'`，`'e'`，`'i'`，`'o'`，`'u'`），并返回这个新字符串。

```python
class Solution:
    def removeVowels(self, s: str) -> str:
        the_set = set("aeiou")
        ans = [] # 用列表存
        for i in s:
            if i not in the_set:
                ans.append(i)
        return ''.join(ans)
```

# 1122. 数组的相对排序

给你两个数组，arr1 和 arr2，

arr2 中的元素各不相同
arr2 中的每个元素都出现在 arr1 中
对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1 的末尾。

```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        # 双向映射法，调用了排序api和两次映射
        the_map = {val:index for index,val in enumerate(arr2)}
        mirror_map = {index:val for index,val in enumerate(arr2)}
        seive = []
        temp = []
        for i in arr1:
            if i not in the_map:
                seive.append(i)
            else:
                temp.append(the_map[i])
        seive.sort()
        temp.sort()
        ans = []
        for i in temp:
            ans.append(mirror_map[i])
        ans += seive
        return ans
```

# 1123. 最深叶节点的最近公共祖先

给你一个有根节点的二叉树，找到它最深的叶节点的最近公共祖先。

回想一下：

叶节点 是二叉树中没有子节点的节点
树的根节点的 深度 为 0，如果某一节点的深度为 d，那它的子节点的深度就是 d+1
如果我们假定 A 是一组节点 S 的 最近公共祖先，S 中的每个节点都在以 A 为根节点的子树中，且 A 的深度达到此条件下可能的最大值。

```python
class Solution:
    def lcaDeepestLeaves(self, root: TreeNode) -> TreeNode:
        self.valid_leaves = [] # 初始化一个收集列表
        self.find_valid_leaves(root) # 调用收集方法
        self.valid_leaves = self.valid_leaves[-1] # 把合法的叶子节点传入
        self.valid_set = set(self.valid_leaves) # 把它做成集合方便查询
        ans = self.find_LCA(root) # 调用找LCA的方法
        return ans
        
    
    def find_valid_leaves(self,root):  # BFS搜到的最后一层是最深的
        if root == None:
            return None
        queue = [root]
        while len(queue) != 0:
            level = [] # 收集本层结果
            new_queue = [] # 放置下一层将要搜索的队列
            for i in queue:
                if i != None:
                    level.append(i)
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            self.valid_leaves.append(level)
            queue = new_queue
    
    def find_LCA(self,root): #
        if root == None: return None
        if root in self.valid_set: return root
        left = self.find_LCA(root.left)
        right = self.find_LCA(root.right)
        if left == None and right == None: return None
        if left != None and right == None: return left
        if left == None and right != None: return right
        if left != None and right != None: return root
                    
```

# 1133. 最大唯一数

给你一个整数数组 `A`，请找出并返回在该数组中仅出现一次的最大整数。

如果不存在这个只出现一次的整数，则返回 -1。

```python
class Solution:
    def largestUniqueNumber(self, nums: List[int]) -> int:
        # 排序，然后，直接Counter查表
        nums.sort(reverse = True)
        Counter1 = collections.Counter(nums)
        ans = -1
        for i in nums:
            if Counter1[i] == 1:
                ans = i
                return ans
        return ans
```

# 1134. 阿姆斯特朗数

假设存在一个 k 位数 N，其每一位上的数字的 k 次幂的总和也是 N，那么这个数是阿姆斯特朗数。

给你一个正整数 N，让你来判定他是否是阿姆斯特朗数，是则返回 true，不是则返回 false。

```python
class Solution:
    def isArmstrong(self, n: int) -> bool:
        # 字符串化
        s = str(n)
        k = len(s)
        sum_num = 0
        for i in s:
            sum_num += int(i)**k
        return n == sum_num
```

# 1135. 最低成本联通所有城市

想象一下你是个城市基建规划者，地图上有 N 座城市，它们按以 1 到 N 的次序编号。

给你一些可连接的选项 conections，其中每个选项 conections[i] = [city1, city2, cost] 表示将城市 city1 和城市 city2 连接所要的成本。（连接是双向的，也就是说城市 city1 和城市 city2 相连也同样意味着城市 city2 和城市 city1 相连）。

返回使得每对城市间都存在将它们连接在一起的连通路径（可能长度为 1 的）最小成本。该最小成本应该是所用全部连接代价的综合。如果根据已知条件无法完成该项任务，则请你返回 -1。

```python
class UF:
    def __init__(self,size):
        self.root = [i for i in range(size)]
    
    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY: # 如果两者不等
            self.root[rootX] = rootY

    def find(self,x):
        while x != self.root[x]:
            x = self.root[x]
        return x
    
    def isConnect(self,x,y): # 判断是否成环
        return self.find(x) == self.find(y)

class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        # UF + Kruskal算法
        # 最小生成树需要n-1条链接
        count = 0 # 记录链接数量
        connections.sort(key = lambda x:x[2]) # 权重排序,Kruskal算法的精髓
        UFSet = UF(n)
        cost = 0 # 记录花费
        # 用并查集的时候无需担心达到了n-1条的时候会有孤立的“岛”
        for pair in connections:
            x = pair[0]-1
            y = pair[1]-1
            if not UFSet.isConnect(x,y): # 未成环
                UFSet.union(x,y) # 
                cost += pair[2]
                count += 1
            if count == n - 1:  # 有了n-1条之后，退出
                return cost
        return -1
       
```

# 1143. 最长公共子序列

给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 朴素动态规划解法
        # 一般而言，子序列问题都可以用二维dp解决
        # dp[i][j]的含义是，一个横着摆，一个竖着摆，以竖着摆到第i个字符结尾和以横着摆的第j个字符结尾的LCS
        # 先初始化为全0,t1竖着摆，t2横着摆，注意左闭右开
        dp = [[0 for j in range(len(text2)+1)] for i in range(len(text1)+1)]
        for j in range(1,len(text2)+1):
            for i in range(1,len(text1)+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                elif text1[i-1] != text2[j-1]:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        # print(dp)
        return dp[-1][-1]
```

# 1150. 检查一个数是否在数组中占绝大多数

给出一个按 非递减 顺序排列的数组 nums，和一个目标数值 target。假如数组 nums 中绝大多数元素的数值都等于 target，则返回 True，否则请返回 False。

所谓占绝大多数，是指在长度为 N 的数组中出现必须 超过 N/2 次。

```python
class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        # 长度超过一半则必为中值
        # 而且一定在1/4 和 3/4 处还是那个值
        # 逻辑对奇偶性处理有点难写
        # 直接暴力
        ct = collections.Counter(nums)
        n = len(nums)
        if ct[target] > n/2:
            return True
        else:
            return False

```

# 1165. 单行键盘

我们定制了一款特殊的力扣键盘，所有的键都排列在一行上。

我们可以按从左到右的顺序，用一个长度为 26 的字符串 keyboard （索引从 0 开始，到 25 结束）来表示该键盘的键位布局。

现在需要测试这个键盘是否能够有效工作，那么我们就需要个机械手来测试这个键盘。

最初的时候，机械手位于左边起第一个键（也就是索引为 0 的键）的上方。当机械手移动到某一字符所在的键位时，就会在终端上输出该字符。

机械手从索引 i 移动到索引 j 所需要的时间是 |i - j|。

当前测试需要你使用机械手输出指定的单词 word，请你编写一个函数来计算机械手输出该单词所需的时间。

```python
class Solution:
    def calculateTime(self, keyboard: str, word: str) -> int:
        # 哈希
        dict1 = {}
        for i in range(len(keyboard)):
           dict1[keyboard[i]] = i # 记录字母对应的索引
        word = keyboard[0] + word # 统一语法，把第一个字母直接加在word里
        sum_gap = 0
        for i in range(1,len(word)):
            sum_gap += abs(dict1[word[i]]-dict1[word[i-1]]) # 注意移动距离带绝对值
        return sum_gap
```

# 1175. 质数排列

请你帮忙给从 1 到 n 的数设计排列方案，使得所有的「质数」都应该被放在「质数索引」（索引从 1 开始）上；你需要返回可能的方案总数。

让我们一起来回顾一下「质数」：质数一定是大于 1 的，并且不能用两个小于它的正整数的乘积来表示。

由于答案可能会很大，所以请你返回答案 模 mod 10^9 + 7 之后的结果即可。

```python
class Solution:
    def numPrimeArrangements(self, n: int) -> int:
        # 先筛出n以内prime的数量，然后对这个数量进行全排列。
        # 然后用n减去这个数得到全排列。两个全排列数字相乘
        if n == 1:
            return 1
        primes = self.countPrimes(n)
        no_primes = n - primes
        # print(self.count_num(primes),self.count_num(no_primes))
        return (self.count_num(primes) * self.count_num(no_primes)) % (10**9+7)
        
    def count_num(self,p): # 传入数p，得到它的阶乘
        if p <= 2:
            return p
        dp = [-1 for i in range(p+1)] 
        dp[0] = 1
        i = 1
        while i <= p:
            dp[i] = i * dp[i-1]
            i += 1
        return dp[-1]
        
    def countPrimes(self, n: int) -> int:
        # python素数筛,只筛数量
        n += 1
        if n < 2:
            return 0
        if n == 2:
            return 0
        # 先初始化所有数字为True
        grid = [True for i in range(n)]
        grid[0],grid[1] = False,False
        count = 0 # 计数用
        # 筛的上界为 sqrt(n)即可
        upto = math.ceil(sqrt(n))+1 # 注意range的左闭右开
        for index in range(upto):
            if grid[index] == True:
                for multi in range(2,n//index+1): # 两倍以上的数就不要了
                    if index * multi < n:
                        grid[index * multi] = False
        for i in grid:
            if i == True:
                count += 1
        return count
```

# 1180. 统计只含单一字母的子串

给你一个字符串 `S`，返回只含 **单一字母** 的子串个数。

```python
class Solution:
    def countLetters(self, s: str) -> int:
        # 滑动窗口
        window = defaultdict(int)
        right = 0
        times = 0 # 收集答案
        s = s + '1' # 补充上右边的墙，使其可以收缩
        while right < len(s):
            add_char = s[right]
            right += 1
            window[add_char] += 1
            if len(window) > 1:
                delete_char = s[right-1] # 刚刚加进来的那个字符抛弃
                del window[delete_char]
                for i in window.values():
                    times += (1+i)*i//2
                window = defaultdict(int) # 重置窗口
                window[add_char] += 1 # 把刚刚那个字符加回来
        return times
```

# 1190. 反转每对括号间的子串

给出一个字符串 s（仅含有小写英文字母和括号）。

请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

注意，您的结果中 不应 包含任何括号。

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        # 思路：每遇见一个左括号开辟一个栈，每遇见一个右括号，把最后这个栈倒到前一个栈中，最后所有元素会在第一个栈聚集
        # 返回第一个栈链接到字符串即可
        stack_arr = [[]] # 这是一个栈数组
        pStack = 0 # 初始化栈数组指针
        for i in s:
            if i == "(":# 每遇见一个左括号开辟一个栈
                stack_arr.append([]) 
                pStack += 1
            elif i == ")": # 每遇见一个右括号，把最后这个栈倒到前一个栈中
                while stack_arr[pStack] != []:
                    stack_arr[pStack-1].append(stack_arr[pStack].pop())
                pStack -= 1
            else:
                stack_arr[pStack].append(i)
        return ''.join(stack_arr[0])
```

# 1196. 最多可以买到的苹果数量

楼下水果店正在促销，你打算买些苹果，arr[i] 表示第 i 个苹果的单位重量。

你有一个购物袋，最多可以装 5000 单位重量的东西，算一算，最多可以往购物袋里装入多少苹果。

```python
class Solution:
    def maxNumberOfApples(self, arr: List[int]) -> int:
        # 排序,贪心
        ans = 0
        the_sum = 0
        arr.sort()
        for i in arr:
            if the_sum + i <= 5000:
                the_sum += i
                ans += 1
            else:
                break
        return ans

```

```java
class Solution {
    public int maxNumberOfApples(int[] arr) {
        Arrays.sort(arr);
        int temp = 0;
        int ans = 0;
        for(int i:arr){
            if (i + temp <= 5000) {
                ans += 1;
                temp += i;
            }
            else break;
        }
        return ans;
    }
}
```



# 1198. 找出所有行中最小公共元素

给你一个矩阵 mat，其中每一行的元素都已经按 严格递增 顺序排好了。请你帮忙找出在所有这些行中 最小的公共元素。

如果矩阵中没有这样的公共元素，就请返回 -1。

```python
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        # 申请数组，为矩阵的行数，它最初指向所有行的第一个，
        # 多排指针
        points = [0 for i in range(len(mat))] # 这个数组存取的所有行的指针
        max_cur = 0 # 记录最大指针的位置，有一个越界则返回
        while max_cur < len(mat[0]):
            # 指针移动逻辑，先判断当前指向的所有值是否相等
            # 如果相等则返回
            # 如果不等，找到最大的那个，所有不等于最大的那个的指针移动
            max_value_lst = []
            for num,row in enumerate(mat):
                max_value_lst.append(mat[num][points[num]]) # 这一行的指针那一个
            max_value = max(max_value_lst)
            count = 0
            for i in max_value_lst:
                if i == max_value:
                    count += 1
            if count == len(mat):
                return max_value
            for num,row in enumerate(mat):
                if mat[num][points[num]] < max_value:
                    points[num] += 1
            max_cur = max(points)
        return -1
```

# 1209. 删除字符串中的所有相邻重复项 II

给你一个字符串 s，「k 倍重复项删除操作」将会从 s 中选择 k 个相邻且相等的字母，并删除它们，使被删去的字符串的左侧和右侧连在一起。

你需要对 s 重复进行无限次这样的删除操作，直到无法继续为止。

在执行完所有删除操作后，返回最终得到的字符串。

本题答案保证唯一。

```python
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        # 栈模拟，想要解决空间，则只能一轮扫描过
        # 存储的是 字符和字符次数
        stack = []
        for char in s:
            if len(stack) == 0:
                stack.append([char,1])
            elif char != stack[-1][0]:
                stack.append([char,1])
            elif char == stack[-1][0]:
                stack[-1][1] += 1
            if stack[-1][1] == k:
                stack.pop()
        # 最终结果还原
        ans = ''
        for cp in stack:
            ans += cp[0] * cp[1]
        return ans
```

# 1213. 三个有序数组的交集

给出三个均为 **严格递增排列** 的整数数组 `arr1`，`arr2` 和 `arr3`。

返回一个由 **仅** 在这三个数组中 **同时出现** 的整数所构成的有序数组。

```python
class Solution:
    def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
        p1,p2,p3 = 0,0,0
        ans = [] # 收集答案
        # 有一个越界就停止搜索
        # 三个数值都一样则加入，否则找出最大者，不是最大的那一个或者那两个都要移动
        while p1 < len(arr1) and p2 < len(arr2) and p3 < len(arr3):
            if arr1[p1] == arr2[p2] == arr3[p3]:
                ans.append(arr1[p1])
                p1 += 1
                p2 += 1
                p3 += 1
            else:
                max_temp = max(arr1[p1],arr2[p2],arr3[p3])
                if arr1[p1] != max_temp:
                    p1 += 1
                if arr2[p2] != max_temp:
                    p2 += 1
                if arr3[p3] != max_temp:
                    p3 += 1
        return ans
 
```

# 1214. 查找两棵二叉搜索树之和

给出两棵二叉搜索树，请你从两棵树中各找出一个节点，使得这两个节点的值之和等于目标值 `Target`。

如果可以找到返回 `True`，否则返回 `False`。

```python
class Solution:
    def twoSumBSTs(self, root1: TreeNode, root2: TreeNode, target: int) -> bool:
        # 把两数变成两列表，没有用到BST的性质
        lst1 = []
        lst2 = []
        self.toList(root1,lst1)
        self.toList(root2,lst2)
        ct1 = collections.Counter(lst1)
        ct2 = collections.Counter(lst2)
        for i in ct1:
            need = target - i
            if need in reversed(ct2):
                return True
        return False 
    
    def toList(self,node,lst): # 将数变成中序列表的函数
        if node == None:
            return 
        self.toList(node.left,lst)
        lst.append(node.val)
        self.toList(node.right,lst)

```

```python
class Solution:
    def twoSumBSTs(self, root1: TreeNode, root2: TreeNode, target: int) -> bool:
        # 把两数变成两列表，一个从小找，一个从大找，找两轮，两轮都找不到则false
        lst1 = []
        lst2 = []
        self.toList(root1,lst1)
        self.toList(root2,lst2)
        p1 = 0
        p2 = len(lst2)-1
        while p1 < len(lst1) and p2 >= 0: # 第一轮搜
            if lst1[p1] + lst2[p2] == target:
                return True
            elif lst1[p1] + lst2[p2] > target: # 大指针左移动
                p2 -= 1
            elif lst1[p1] + lst2[p2] < target: # 小指针右移
                p1 += 1
        p1 = len(lst1) - 1
        p2 = 0
        while p1 >= 0 and p2 < len(lst2): # 第二轮搜
            if lst1[p1] + lst2[p2] == target:
                return True
            elif lst1[p1] + lst2[p2] > target: # 大指针左移动
                p1 -= 1
            elif lst1[p1] + lst2[p2] < target: # 小指针右移
                p2 += 1
        # 两轮都搜不到
        return False
    
    def toList(self,node,lst): # 将数变成中序列表的函数
        if node == None:
            return 
        self.toList(node.left,lst)
        lst.append(node.val)
        self.toList(node.right,lst)

```

# 1222. 可以攻击国王的皇后

在一个 8x8 的棋盘上，放置着若干「黑皇后」和一个「白国王」。

「黑皇后」在棋盘上的位置分布用整数坐标数组 queens 表示，「白国王」的坐标用数组 king 表示。

「黑皇后」的行棋规定是：横、直、斜都可以走，步数不受限制，但是，不能越子行棋。

请你返回可以直接攻击到「白国王」的所有「黑皇后」的坐标（任意顺序）。

```python
class Solution:
    def queensAttacktheKing(self, queens: List[List[int]], king: List[int]) -> List[List[int]]:
        # 把queens变成set,提高查询效率，然后king以8个方向进行搜索
        the_set = set(tuple(i) for i in queens)
        direc_lst = [(-1,-1),(-1,0),(-1,+1),(0,-1),(0,+1),(+1,-1),(+1,0),(+1,+1)]
        ans = [] # 收集结果
        
        def dfs(king,direc):
            i = king[0]
            j = king[1]
            while 0 <= i <= 7 and 0 <= j <= 7:
                i += direc[0]
                j += direc[1]
                if (i,j) in the_set:
                    ans.append([i,j])
                    break # 收集到最近的一个就停止
        
        for direc in direc_lst:
            dfs(king,direc) # 调用搜索方法
        return ans
            
```

# 1228. 等差数列中缺失的数字

有一个数组，其中的值符合等差数列的数值规律，也就是说：

在 0 <= i < arr.length - 1 的前提下，arr[i+1] - arr[i] 的值都相等。
我们会从该数组中删除一个 既不是第一个 也 不是最后一个的值，得到一个新的数组  arr。

给你这个缺值的数组 arr，请你帮忙找出被删除的那个数。

```python
class Solution:
    def missingNumber(self, arr: List[int]) -> int:
        # 由首项和尾项获取公差，其实可以用二分法做
        thesum = (arr[0] + arr[-1]) * (len(arr) + 1) //2 
        return thesum - sum(arr)
```

# 1232. 缀点成线

在一个 XY 坐标系中有一些点，我们用数组 coordinates 来分别记录它们的坐标，其中 coordinates[i] = [x, y] 表示横坐标为 x、纵坐标为 y 的点。

请你来判断，这些点是否在该坐标系中属于同一条直线上，是则返回 true，否则请返回 false。

```python
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        # 分为斜率是否存在来考虑
        if len(coordinates) == 2:
            return True
        
        dy = coordinates[1][1] - coordinates[0][1]
        dx = coordinates[1][0] - coordinates[0][0]
        if dx == 0:
            mark = coordinates[0][0]
            for coord in coordinates:
                if coord[0] != mark:
                    return False
            return True
        
        elif dx != 0:
            x0 = coordinates[0][0]
            y0 = coordinates[0][1]

            for coord in coordinates:
                if dy * (coord[0] - x0) + dx * y0 != coord[1] * dx:
                    return False
            return True
```

# 1248. 统计「优美子数组」

给你一个整数数组 nums 和一个整数 k。

如果某个 连续 子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。

请返回这个数组中「优美子数组」的数目。

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        # 奇数看作1，偶数看作0
        # k已知大于等于1
        # 前缀和+哈希求法 哈希存的键是包含当前位置的前缀和，值存的是个数。在这个位置及之前，这个前缀和出现过几次
        # 前缀和存的是包含本位置的前缀和
        pre_dict = collections.defaultdict(int)
        pre_sum = 0
        pre_dict[0] = 1  # 注意这一行
        ans = 0
        for i in range(len(nums)):
            if nums[i] % 2 != 0:
                pre_sum += 1
            pre_dict[pre_sum] += 1 # 包含这个数，在这个数及之前的奇数个数 key == presum 
            target = pre_sum - k 
            if target in pre_dict:
                ans += pre_dict[target]
        return ans
```

# 1258. 近义词句子

给你一个近义词表 synonyms 和一个句子 text ， synonyms 表中是一些近义词对 ，你可以将句子 text 中每个单词用它的近义词来替换。

请你找出所有用近义词替换后的句子，按 字典序排序 后返回。

```python
class Solution:
    def generateSentences(self, synonyms: List[List[str]], text: str) -> List[str]:
        # 搞一个setgroup进行回溯？
        choice = text.split(" ")
        # 由于一个单词可能有多个同义词分布在不同的小对里，所以每次添加同义词需要检查
        group_set = []
        for tp in synonyms:
            action = True # 防止重复扫
            for i in range(len(group_set)):
                if tp[0] in group_set[i]:
                    group_set[i].add(tp[1])
                    action = False
                    break
                elif tp[1] in group_set[i]:
                    group_set[i].add(tp[0])
                    action = False
                    break  
            if action: 
                temp_set = set()
                temp_set.add(tp[0])
                temp_set.add(tp[1])
                group_set.append(temp_set)
        #########
        record = [] # 记录被并掉的y
        for x in range(len(group_set)): # 打补丁防止乱序
            for y in range(x+1,len(group_set)):
                if len(group_set[x].intersection(group_set[y])) != 0: # 说明两者有重复
                    for e in group_set[y]:
                        group_set[x].add(e)
                        record.append(y)
        temp = []
        for i in range(len(group_set)):
            if i not in record:
                temp.append(group_set[i])
        group_set = temp
        ##########
        synonyms_set = set()# 只有在这个表里面才进行替换
        for tp in synonyms:
            synonyms_set.add(tp[0])
            synonyms_set.add(tp[1])
        for i in range(len(choice)):
            if choice[i] in synonyms_set:
                for ind in range(len(group_set)):
                    if choice[i] in group_set[ind]:
                        choice[i] = ind
                        break
        # 此时choice需要被替换的词变成了序号，然后把groupSet里面的内容变成排序的列表
        cp_groupSet = [list(i) for i in group_set]
        for i in range(len(cp_groupSet)): # 字典序排序
            cp_groupSet[i].sort()        
        #print(choice)
        path = []
        ans = []
        #print(cp_groupSet)
        def dfs(lst,path): # 选择列表，路径
            if len(lst) == 0:
                ans.append(path[:])
                return
            if str(lst[0]).isdigit(): # 需要替换
                replace = cp_groupSet[lst[0]]
                for w in replace:
                    path.append(w)
                    dfs(lst[1:],path)
                    path.pop()
            else:
                path.append(lst[0])
                dfs(lst[1:],path)
                path.pop() # 这一个不能省略
        
        dfs(choice,path)
        
        final = [' '.join(every) for every in ans]
        return final
```

# 1261. 在受污染的二叉树中查找元素

给出一个满足下述规则的二叉树：

root.val == 0
如果 treeNode.val == x 且 treeNode.left != null，那么 treeNode.left.val == 2 * x + 1
如果 treeNode.val == x 且 treeNode.right != null，那么 treeNode.right.val == 2 * x + 2
现在这个二叉树受到「污染」，所有的 treeNode.val 都变成了 -1。

请你先还原二叉树，然后实现 FindElements 类：

FindElements(TreeNode* root) 用受污染的二叉树初始化对象，你需要先把它还原。
bool find(int target) 判断目标值 target 是否存在于还原后的二叉树中并返回结果。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class FindElements:

    def __init__(self, root: TreeNode):
        root.val = 0
        self.inTree = set() # 用哈希表存着，由于调用find次数可能很多，哈希表应该是最优的
        # 如果调用次数很少或者只有一次，使用二分搜索
        self.inTree.add(0)
        self.refresh(root)

    def refresh(self,node): # 还原二叉树
        if node == None:
            return
        self.inTree.add(node.val)
        if node.left != None:
            node.left.val = 2*node.val + 1
        if node.right != None:
            node.right.val = 2*node.val + 2
        self.refresh(node.left)
        self.refresh(node.right)
        
    def find(self, target: int) -> bool:
        return target in self.inTree

# Your FindElements object will be instantiated and called as such:
# obj = FindElements(root)
# param_1 = obj.find(target)
```

# 1265. 逆序打印不可变链表

给您一个不可变的链表，使用下列接口逆序打印每个节点的值：

ImmutableListNode: 描述不可变链表的接口，链表的头节点已给出。
您需要使用以下函数来访问此链表（您 不能 直接访问 ImmutableListNode）：

ImmutableListNode.printValue()：打印当前节点的值。
ImmutableListNode.getNext()：返回下一个节点。
输入只用来内部初始化链表。您不可以通过修改链表解决问题。也就是说，您只能通过上述 API 来操作链表。

```python
# """
# This is the ImmutableListNode's API interface.
# You should not implement it, or speculate about its implementation.
# """
# class ImmutableListNode:
#     def printValue(self) -> None: # print the value of this node.
#     def getNext(self) -> 'ImmutableListNode': # return the next node.

class Solution:
    def printLinkedListInReverse(self, head: 'ImmutableListNode') -> None:
        # 调用api和熟悉递归即可
        if head.getNext() == None:
            head.printValue()
            return 
        node = head.getNext()
        self.printLinkedListInReverse(node)
        head.printValue()
        
```

# 1282. 用户分组

有 n 位用户参加活动，他们的 ID 从 0 到 n - 1，每位用户都 恰好 属于某一用户组。给你一个长度为 n 的数组 groupSizes，其中包含每位用户所处的用户组的大小，请你返回用户分组情况（存在的用户组以及每个组中用户的 ID）。

你可以任何顺序返回解决方案，ID 的顺序也不受限制。此外，题目给出的数据保证至少存在一种解决方案。

```python
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        # 尬算法，数据量小，直接尬算
        groupSizeList = collections.defaultdict(list) # 键为大小
        n = len(groupSizes)
        ans = []
        for i in range(n):
            aimSize = groupSizes[i]
            groupSizeList[aimSize].append(i)
            if aimSize == len(groupSizeList[aimSize]): # 达到给定长度，重置
                ans.append(groupSizeList[aimSize])
                groupSizeList[aimSize] = [] # 重置
        return ans
        
```

# 1315. 祖父节点值为偶数的节点和

给你一棵二叉树，请你返回满足以下条件的所有节点的值之和：

该节点的祖父节点的值为偶数。（一个节点的祖父节点是指该节点的父节点的父节点。）
如果不存在祖父节点值为偶数的节点，那么返回 0 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        # 两次dfs，一次给所有节点先找到爹，一次根据条件收集节点值
        self.parent_dict = dict()
        self.parent_dict[root] = None # 根节点没有父节点
        self.find_parent(root) # 调用找爹方法
        self.the_node_value_sum = 0 
        self.judge_grandparent(root) # 调用判定爷爷是否合法方法
        return self.the_node_value_sum
    
    def find_parent(self,node): # 找爹
        if node == None:
            return 
        if node.left != None:
            self.parent_dict[node.left] = node
        if node.right != None:
            self.parent_dict[node.right] = node
        self.find_parent(node.left)
        self.find_parent(node.right)
    
    def judge_grandparent(self,node): # 判断爷爷是否合法，合法把孙子的值加进去
        if node == None:
            return
        if self.parent_dict[node] != None and self.parent_dict[self.parent_dict[node]] != None:
            if self.parent_dict[self.parent_dict[node]].val % 2 == 0:
                self.the_node_value_sum += node.val
        self.judge_grandparent(node.left)
        self.judge_grandparent(node.right)


```

# 1317. 将整数转换为两个无零整数的和

「无零整数」是十进制表示中 不含任何 0 的正整数。

给你一个整数 n，请你返回一个 由两个整数组成的列表 [A, B]，满足：

A 和 B 都是无零整数
A + B = n
题目数据保证至少有一个有效的解决方案。

如果存在多个有效解决方案，你可以返回其中任意一个。

```python
class Solution:
    def getNoZeroIntegers(self, n: int) -> List[int]:
        lst = [1,n-1]
        # 题目保证有解，则直接while不设限的循环
        while '0' in str(lst[0]) or '0' in str(lst[1]):
            lst[0] += 1
            lst[1] -= 1
        return lst
            
```

# 1325. 删除给定值的叶子节点

给你一棵以 root 为根的二叉树和一个整数 target ，请你删除所有值为 target 的 叶子节点 。

注意，一旦删除值为 target 的叶子节点，它的父节点就可能变成叶子节点；如果新叶子节点的值恰好也是 target ，那么这个节点也应该被删除。

也就是说，你需要重复此过程直到不能继续删除。

```python
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        # 一轮先序遍历存储所有等于目标值的节点，还需要存储这些节点的父节点
        self.findParent = dict()
        self.BFS_lst = []
        if root.val == target:
            self.BFS_lst.append([root,None])

        def bfs(node,val): # BFS倒着搜可以一轮解决
            queue = [node]
            while len(queue) != 0:
                new_queue = []
                level = []
                for node in queue:
                    if node.left != None:
                        if node.left.val == val:
                            level.append([node.left,node]) # 顺序为【节点，父节点】
                        new_queue.append(node.left)
                    if node.right != None:
                        if node.right.val == val:
                            level.append([node.right,node]) # 顺序为【节点，父节点】    
                        new_queue.append(node.right)               
                self.BFS_lst += level
                queue = new_queue
        
        bfs(root,target)
        for cp in self.BFS_lst[::-1]:
            self.findParent[cp[0]] = cp[1]
                           
        # 开始删除叶子节点
        for node in self.findParent:
            if self.findParent[node] == None: # 针对根节点
                pass
            elif node.left == node.right == None: # 是叶子节点才执行
                if node == self.findParent[node].left:
                    self.findParent[node].left = None
                elif node == self.findParent[node].right:
                    self.findParent[node].right = None
        # 对根节点进行特殊处理
        if root.val == target and root.left == None and root.right == None:
            root = None         
        return root

```

```python
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        # 后续遍历,递归
        if root == None:
            return
        root.left = self.removeLeafNodes(root.left,target)
        root.right = self.removeLeafNodes(root.right,target)
        if root.left == None and root.right == None and root.val == target:
            return None
        return root
```

# 1332. 删除回文子序列

给你一个字符串 s，它仅由字母 'a' 和 'b' 组成。每一次删除操作都可以从 s 中删除一个回文 子序列。

返回删除给定字符串中所有字符（字符串为空）的最小删除次数。

「子序列」定义：如果一个字符串可以通过删除原字符串某些字符而不改变原字符顺序得到，那么这个字符串就是原字符串的一个子序列。

「回文」定义：如果一个字符串向后和向前读是一致的，那么这个字符串就是一个回文。

```python
class Solution:
    def removePalindromeSub(self, s: str) -> int:
        # 脑筋急转弯，最多只需要两次，为删掉全部的b和全部的a
        # 本身是回文删除1次
        # 本身不是回文删除两次
        # 空串为0次
        if len(s) == 0:
            return 0
        if s == s[::-1]:
            return 1
        else:
            return 2
```

# 1347. 制造字母异位词的最小步骤数

给你两个长度相等的字符串 s 和 t。每一个步骤中，你可以选择将 t 中的 任一字符 替换为 另一个字符。

返回使 t 成为 s 的字母异位词的最小步骤数。

字母异位词 指字母相同，但排列不同（也可能相同）的字符串。

```python
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        # 两字符长度相等，每次优先替换掉非重复部分。
        # 使用数组记录
        arr1 = [0 for i in range(26)]
        for char in s: # 增长
            index = ord(char) - ord("a")
            arr1[index] += 1
        for char in t: # 抵消
            index = ord(char) - ord("a")
            arr1[index] -= 1
        # 其中只要不为0的则为不相同的字符，一定是偶数个，一次替换可以消俩个
        the_sum = 0
        for i in arr1:
            the_sum += abs(i)
        return the_sum // 2
```

# 1361. 验证二叉树

二叉树上有 n 个节点，按从 0 到 n - 1 编号，其中节点 i 的两个子节点分别是 leftChild[i] 和 rightChild[i]。

只有 所有 节点能够形成且 只 形成 一颗 有效的二叉树时，返回 true；否则返回 false。

如果节点 i 没有左子节点，那么 leftChild[i] 就等于 -1。右子节点也符合该规则。

注意：节点没有值，本问题中仅仅使用节点编号。

```python
class UF:
    def __init__(self,size):
        self.root = [i for i in range(size)]
    
    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootX] = rootY
    
    def find(self,x):
        while x != self.root[x]:
            x = self.root[x]
        return x
    
    def isConnect(self,x,y):
        return self.find(x) == self.find(y)

class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        # 略去所有-1，根节点不在孩子列表，
        # 有效二叉树，有一个节点不被计入,其余节点记录且只被记录一次,切不产生自环
        countList = [0 for i in range(n)]
        onlySet = set()
        for i in range(n):
            onlySet.add(i)

        for t in leftChild:
            if t != -1:
                if t in onlySet:
                    onlySet.remove(t)
                else:
                    return False

        for t in rightChild:
            if t != -1:
                if t in onlySet:
                    onlySet.remove(t)
                else:
                    return False
        # 检查自环            
        UF1 = UF(n)
        for i in range(len(leftChild)):
            if leftChild[i] != -1:
                if not UF1.isConnect(i,leftChild[i]):
                    UF1.union(i,leftChild[i])
                else:
                    return False
        UF2 = UF(n)
        for i in range(len(rightChild)):
            if rightChild[i] != -1:
                if not UF2.isConnect(i,rightChild[i]):
                    UF2.union(i,rightChild[i])
                else:
                    return False

        # print(onlySet)
        # 最终里面只能剩下一个，并且检查自环
        return len(onlySet) == 1 # 有且只有一个


```

# 1363. 形成三的最大倍数

给你一个整数数组 digits，你可以通过按任意顺序连接其中某些数字来形成 3 的倍数，请你返回所能得到的最大的 3 的倍数。

由于答案可能不在整数数据类型范围内，请以字符串形式返回答案。

如果无法得到答案，请返回一个空字符串。

```python
# 垃圾补丁版
class Solution:
    def largestMultipleOfThree(self, digits: List[int]) -> str:
        # 分类讨论
        # 1. 把所有元素按照频次来考虑
        ct = collections.Counter(digits)
        # 存储模xx的余量，以二元列表存，元素在前，频次在后
        mod0,mod1,mod2 = [],[],[] 
        for key,val in ct.items():
            if key % 3 == 0:
                mod0.append([str(key),val])
            elif key % 3 == 1:
                mod1.append([str(key),val])
            elif key % 3 == 2:
                mod2.append([str(key),val])
        # 倒序
        mod0.sort(reverse = True)
        mod1.sort(reverse = True)
        mod2.sort(reverse = True)
        # 记录余2的频次总量和余1的频次总量
        final = [] # 
        m1 = ""
        for cp in mod1:
            m1 += str(cp[0])*cp[1]
        m2 = ""
        for cp in mod2:
            m2 += str(cp[0])*cp[1]
        if len(m1) == len(m2): # 则直接三者合并成串
            digits.sort(reverse = True)
            ans = ''
            for ch in digits:
                ans += str(ch)
            if ans[0] == "0" and len(ans) > 1: return "0"
            else: return ans
        s = sum(digits)
        if s % 3 == 1:
            # 则删除mod1中1个的或者删除2个mod2中的最小数
            if len(mod1) != 0:
                mod1[-1][1] -= 1
            elif len(mod1) == 0:
                if mod2[-1][1] >= 2:
                    mod2[-1][1] -= 2
                elif mod2[-1][1] == 1:
                    mod2[-1][1] -= 1
                    mod2[-2][1] -= 1
        elif s % 3 == 2:
            # 则删除mod2中1个的或者删除2个mod1中的最小数
            mod1,mod2 = mod2,mod1 # 仅交换，以便于复用上面代码
            if len(mod1) != 0:
                mod1[-1][1] -= 1
            elif len(mod1) == 0:
                if mod2[-1][1] >= 2:
                    mod2[-1][1] -= 2
                elif mod2[-1][1] == 1:
                    mod2[-1][1] -= 1
                    mod2[-2][1] -= 1
        final = mod0 + mod1 + mod2
        final.sort(reverse = True)
        ans = ''
        for cp in final:
            ans += cp[0] * cp[1]
        if len(ans) == 0: return ""
        if ans[0] == "0" and len(ans) > 1: return "0"
        return ans
```



# 1385. 两个数组间的距离值

给你两个整数数组 arr1 ， arr2 和一个整数 d ，请你返回两个数组之间的 距离值 。

「距离值」 定义为符合此距离要求的元素数目：对于元素 arr1[i] ，不存在任何元素 arr2[j] 满足 |arr1[i]-arr2[j]| <= d 。

```python
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        # 非排序做法
        # 排序直接二分判断都可以，
        ans = 0 # 收集答案
        for i in arr1:
            count = 0
            for j in arr2: # 看是否筛完了
                if abs(i-j) > d:
                    count += 1
                else: # 否则直接中断这一层for
                    break
            if count == len(arr2): # 如果每个都大于，则最终结果加一
                ans += 1
        return ans
```

# 1403. 非递增顺序的最小子序列

给你一个数组 nums，请你从中抽取一个子序列，满足该子序列的元素之和 严格 大于未包含在该子序列中的各元素之和。

如果存在多个解决方案，只需返回 长度最小 的子序列。如果仍然有多个解决方案，则返回 元素之和最大 的子序列。

与子数组不同的地方在于，「数组的子序列」不强调元素在原数组中的连续性，也就是说，它可以通过从数组中分离一些（也可能不分离）元素得到。

注意，题目数据保证满足所有约束条件的解决方案是 唯一 的。同时，返回的答案应当按 非递增顺序 排列。

```python
class Solution:
    def minSubsequence(self, nums: List[int]) -> List[int]:
        # 排序后贪心
        nums.sort()
        part_sum = sum(nums)
        ans = []
        temp_sum = 0
        for i in range(-1,-len(nums)-1,-1): # 倒序常用的序号for循环
            if temp_sum <= part_sum:
                temp_sum += nums[i]
                part_sum -= nums[i]
                ans.append(nums[i])
            else: # 加上这一行提升性能
                break
        return ans

```

# 1408. 数组中的字符串匹配

给你一个字符串数组 words ，数组中的每个字符串都可以看作是一个单词。请你按 任意 顺序返回 words 中是其他单词的子字符串的所有单词。

如果你可以删除 words[j] 最左侧和/或最右侧的若干字符得到 word[i] ，那么字符串 words[i] 就是 words[j] 的一个子字符串。

```python
class Solution:
    def stringMatching(self, words: List[str]) -> List[str]:
        # 纯暴力法。。
        ans = []
        for w1 in words:
            for w2 in words:
                if w1 != w2:
                    if w1 in w2:
                        ans.append(w1)
        final = set(ans)
        ans = [i for i in final]
        return ans
```



# 1419. 数青蛙

给你一个字符串 croakOfFrogs，它表示不同青蛙发出的蛙鸣声（字符串 "croak" ）的组合。由于同一时间可以有多只青蛙呱呱作响，所以 croakOfFrogs 中会混合多个 “croak” 。请你返回模拟字符串中所有蛙鸣所需不同青蛙的最少数目。

注意：要想发出蛙鸣 "croak"，青蛙必须 依序 输出 ‘c’, ’r’, ’o’, ’a’, ’k’ 这 5 个字母。如果没有输出全部五个字母，那么它就不会发出声音。

如果字符串 croakOfFrogs 不是由若干有效的 "croak" 字符混合而成，请返回 -1 。

```python
class Solution:
    def minNumberOfFrogs(self, croakOfFrogs: str) -> int:
        # 先筛合法性
        ct = [0,0,0,0,0]
        for i in croakOfFrogs:
            if i == "c": ct[0] += 1
            elif i == "r": ct[1] += 1
            elif i == "o": ct[2] += 1
            elif i == "a": ct[3] += 1
            elif i == "k": ct[4] += 1
            for ind in range(1,5): # 依序性需要频次递减
                if ct[ind] > ct[ind-1]:
                    return -1
        mark = ct[0]
        for i in ct: # 检测所有字母是不是相等
            if i != mark:
                return -1            
        # 合法之后筛青蛙数目,ct计数中会出现的最大值。一旦ct内的数全大于1了，则全部减1.
        # 注意要依序
        ct = [0,0,0,0,0]
        frogs = 0
        for i in croakOfFrogs:
            if i == "c": ct[0] += 1
            elif i == "r": ct[1] += 1
            elif i == "o": ct[2] += 1
            elif i == "a": ct[3] += 1
            elif i == "k": ct[4] += 1
            t = 0
            for times in ct:
                if times >= 1:
                    t += 1
            if t == 5:
                for i in range(len(ct)):
                    ct[i] -= 1
            frogs = max(max(ct),frogs)
        return frogs
```

# 1426. 数元素

给你一个整数数组 arr， 对于元素 x ，只有当 x + 1 也在数组 arr 里时，才能记为 1 个数。

如果数组 arr 里有重复的数，每个重复的数单独计算。

```python
class Solution:
    def countElements(self, arr: List[int]) -> int:
        # 排序，然后使用Counter计数
        arr.sort()
        ct = collections.Counter(arr)
        ans = 0 # 返回结果
        for i in ct: # 这里自动是用键迭代
            if ct.get(i+1) != None:
                ans += ct[i] # ans加上这个数的频次
        return ans
```

# 1427. 字符串的左右移

给定一个包含小写英文字母的字符串 s 以及一个矩阵 shift，其中 shift[i] = [direction, amount]：

direction 可以为 0 （表示左移）或 1 （表示右移）。
amount 表示 s 左右移的位数。
左移 1 位表示移除 s 的第一个字符，并将该字符插入到 s 的结尾。
类似地，右移 1 位表示移除 s 的最后一个字符，并将该字符插入到 s 的开头。
对这个字符串进行所有操作后，返回最终结果。

```python
class Solution:
    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        # python可以使用切片的话这一题就是语法练习题。。。
        length = len(s) # 取模提升大数字的左右移的效率
        for i in shift:
            count = i[1] % length
            if i[0] == 0: # 左移
                for i in range(count):
                    s = s[1:] + s[0]
            elif i[0] == 1: # 右移
                for i in range(count):
                    s = s[-1] + s[:-1]
        return s
```

# 1430. 判断给定的序列是否是二叉树从根到叶的路径

给定一个二叉树，我们称从根节点到任意叶节点的任意路径中的节点值所构成的序列为该二叉树的一个 “有效序列” 。检查一个给定的序列是否是给定二叉树的一个 “有效序列” 。

我们以整数数组 arr 的形式给出这个序列。从根节点到任意叶节点的任意路径中的节点值所构成的序列都是这个二叉树的 “有效序列” 。

```python
class Solution:
    def isValidSequence(self, root: TreeNode, arr: List[int]) -> bool:
        # dfs搜索出所有的路径，一一比对
        ans = []
        path = []
        def dfs(node): 
            if node == None:
                return 
            path.append(node.val)
            if node.left == None and node.right == None:
                ans.append(path[:])
            dfs(node.left)
            dfs(node.right)
            path.pop()
        dfs(root)
        for i in ans:  # 比对
            if i == arr:
                return True
        return False
            
```

# 1438. 绝对差不超过限制的最长连续子数组

给你一个整数数组 nums ，和一个表示限制的整数 limit，请你返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差必须小于或者等于 limit 。

如果不存在满足条件的子数组，则返回 0 。

```python
class mono_decrease: # 单调递减队列
    def __init__(self):
        self.window_max = collections.deque()
    
    def append(self,val):
        if len(self.window_max) == 0 or self.window_max[-1] >= val:
            self.window_max.append(val)
        elif len(self.window_max) != 0:
            if self.window_max[-1] < val: # 循环弹出
                while len(self.window_max) > 0 and self.window_max[-1] < val:
                    self.window_max.pop()
                self.window_max.append(val)
    
    def getMax(self):
        if len(self.window_max) != 0:
            return self.window_max[0]
    
    def popleft(self):
        if len(self.window_max) != 0:
            return self.window_max.popleft()

class mono_increase: # 单调递增队列
    def __init__(self):
        self.window_min = collections.deque()
    
    def append(self,val):
        if len(self.window_min) == 0 or self.window_min[-1] <= val:
            self.window_min.append(val)
        elif len(self.window_min) != 0:
            if self.window_min[-1] > val: # 循环弹出
                while len(self.window_min) > 0 and self.window_min[-1] > val:
                    self.window_min.pop()
                self.window_min.append(val)
    
    def getMin(self):
        if len(self.window_min) != 0:
            return self.window_min[0]

    def popleft(self):
        if len(self.window_min) != 0:
            return self.window_min.popleft()

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        # 滑动窗口
        # 需要O(1)的时间找到最大值和最小值,单调队列需要双端队列支持
        main_queue = collections.deque() # 主队列
        winmax = mono_decrease() # 单调队列，维护单调递减，找最大值则直接从左头部出队
        winmin = mono_increase() # 单调队列，维护单调递增，找最小值则直接从左头部出队
        max_size = 0
        size = 0
        left = 0
        right = 0
        while right < len(nums):
            add_num = nums[right]
            right += 1
            main_queue.append(add_num)
            winmax.append(add_num)
            winmin.append(add_num)
            size += 1
            if abs(winmax.getMax() - winmin.getMin()) <= limit: # 符合要求的时候收集
                max_size = max(size,max_size)
            while left < right and abs(winmax.getMax() - winmin.getMin()) > limit:
                delete_num = nums[left]
                left += 1
                size -= 1
                e = main_queue.popleft() # 需要获取主窗口元素，来决定winmax，winmin是否被弹出
                if e == winmax.getMax():
                    winmax.popleft()
                if e == winmin.getMin():
                    winmin.popleft()
        return max_size
```

# 1447. 最简分数

给你一个整数 `n` ，请你返回所有 0 到 1 之间（不包括 0 和 1）满足分母小于等于 `n` 的 **最简** 分数 。分数可以以 **任意** 顺序返回。

```python
class Solution:
    def simplifiedFractions(self, n: int) -> List[str]:
        # 利用最大公约数筛
        # 分母是1～n ,分子是1～n-1
        ansList = []
        for down in range(2,n+1):
            for up in range(1,down): 
                temp = self.theyield(up,down)
                if temp != None:             
                    ansList.append(temp)
        return ansList
    
    def findGcd(self,a,b): # 找最大公约数
        while a != 0:
            temp = a
            a = b % a
            b = temp
        return b
    
    def theyield(self,m,n): # 生成m/n
        if self.findGcd(m,n) == 1:
            return str(m)+"/"+str(n)
        else:
            return None
```

# 1457. 二叉树中的伪回文路径

给你一棵二叉树，每个节点的值为 1 到 9 。我们称二叉树中的一条路径是 「伪回文」的，当它满足：路径经过的所有节点值的排列中，存在一个回文序列。

请你返回从根到叶子节点的所有路径中 伪回文 路径的数目。

```python
class Solution:
    def pseudoPalindromicPaths (self, root: TreeNode) -> int:
        # 普通暴力搜索超时【dfs+检查函数】
        # 需要改写成带记忆的搜索
        self.ans = 0 # 收集最终结果
        self.dic = collections.defaultdict(int) # 记忆字典
        def dfs(root,dic):
            if root == None:
                return 
            copy_dic = dic.copy() # 为了防止重复添加，先把传入的字典直接拷贝
            copy_dic[root.val] += 1 # 然后每次深搜时候只需要加一个值
            if root.left == None and root.right == None: # 到了叶子节点再进行check
                if self.check(copy_dic) <= 1:  # 检查的是新字典
                    self.ans += 1
            dfs(root.left,copy_dic)
            dfs(root.right,copy_dic)
        dfs(root,self.dic) # 调用dfs，然后填充ans
        return self.ans
    def check(self,dic): # 传入字典进行检查
        the_num = 0
        for cnt in dic.values():
            the_num += cnt % 2
        return the_num

```

# 1469. 寻找所有的独生节点

二叉树中，如果一个节点是其父节点的唯一子节点，则称这样的节点为 “独生节点” 。二叉树的根节点不会是独生节点，因为它没有父节点。

给定一棵二叉树的根节点 root ，返回树中 所有的独生节点的值所构成的数组 。数组的顺序 不限 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getLonelyNodes(self, root: TreeNode) -> List[int]:
        self.only_child = [] # 收集结果
        self.find_the_only(root) # 调用先序遍历
        return self.only_child # 返回结果列表，无需顺序
    
    def find_the_only(self,node):
        if node == None:
            return
        # 先序dfs搜结果，注意添加的是值
        if node.left != None and node.right == None:
            self.only_child.append(node.left.val)
        if node.right != None and node.left == None:
            self.only_child.append(node.right.val)
        self.find_the_only(node.left)
        self.find_the_only(node.right)
```

# 1474. 删除链表 M 个节点之后的 N 个节点

给定链表 head 和两个整数 m 和 n. 遍历该链表并按照如下方式删除节点:

开始时以头节点作为当前节点.
保留以当前节点开始的前 m 个节点.
删除接下来的 n 个节点.
重复步骤 2 和 3, 直到到达链表结尾.
在删除了指定结点之后, 返回修改过后的链表的头节点.

进阶问题: 你能通过就地修改链表的方式解决这个问题吗?

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteNodes(self, head: ListNode, m: int, n: int) -> ListNode:
        fast = head
        step = 1
        while fast != None:
            while fast != None and step != m:
                fast = fast.next 
                step += 1
            slow = fast
            step = 1 # 重置
            step2 = 0
            while slow != None and step2 != n:
                slow = slow.next
                step2 += 1
            if fast != None and slow != None:
                fast.next = slow.next
                fast = fast.next 
            elif fast != None and slow == None: # 注意这一行，如果slow都没有顺利走完，那么fast直接壮士断腕
                fast.next = None
                fast = fast.next
        return head
```

# 1484. 克隆含随机指针的二叉树

给你一个二叉树，树中每个节点都含有一个附加的随机指针，该指针可以指向树中的任何节点或者指向空（null）。

请返回该树的 深拷贝 。

该树的输入/输出形式与普通二叉树相同，每个节点都用 [val, random_index] 表示：

val：表示 Node.val 的整数
random_index：随机指针指向的节点（在输入的树数组中）的下标；如果未指向任何节点，则为 null 。
该树以 Node 类的形式给出，而你需要以 NodeCopy 类的形式返回克隆得到的树。NodeCopy 类和Node 类定义一致。

```python
# Definition for a binary tree node.
# class Node:
#     def __init__(self, val=0, left=None, right=None, random=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.random = random
class Solution:
    def copyRandomBinaryTree(self, root: 'Node') -> 'NodeCopy':
        self.dict1 = {None:None}
        # 先存下所有的树节点
        self.pre_order1(root) # 此时dict里面有所有的树的复制节点
        self.pre_order2(root) # 此时dict里面所有的节点建立链接
        return self.dict1[root]
        
    def pre_order1(self,node): # 第一遍把所有节点复制加入到字典中
        if node == None:
            return
        self.dict1[node] = NodeCopy(node.val)
        self.pre_order1(node.left)
        self.pre_order1(node.right)
    
    def pre_order2(self,node): # 第二遍把字典中的节点连接好
        if node == None:
            return 
        self.dict1[node].random = self.dict1[node.random]
        self.dict1[node].left = self.dict1[node.left]
        self.dict1[node].right = self.dict1[node.right]
        self.pre_order2(node.left)
        self.pre_order2(node.right)
        
```

# 1490. 克隆 N 叉树

给定一棵 N 叉树的根节点 root ，返回该树的深拷贝（克隆）。

N 叉树的每个节点都包含一个值（ int ）和子节点的列表（ List[Node] ）。

class Node {
    public int val;
    public List<Node> children;
}
N 叉树的输入序列用层序遍历表示，每组子节点用 null 分隔（见示例）。

进阶：你的答案可以适用于克隆图问题吗？

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def cloneTree(self, root: 'Node') -> 'Node':
        # 哈希表克隆
        # 第一轮扫描先创建每个新的树节点
        self.the_dict = {None:None} # 加入一个初始的None键
        self.child_dict = {None:None}
        self.BFS1(root)
        # print(self.the_dict)
        # print(self.child_dict)    
        # 第二轮扫描链接各个节点
        self.BFS2(root)       
        return self.the_dict[root] # 返回用根找到的字典中的节点
    
    def BFS1(self,root): # 第一轮BFS，先创建每个新的树节点
        if root == None:
            return 
        queue = [root]
        self.the_dict = {root:Node(root.val)} # 事先预备根的键，因为在队列中它没有加进来
        while len(queue) != 0: # 注意数据结构，孩子存在列表里
            new_queue = []
            for node in queue:
                childrenlst = []
                for i in node.children:
                    if i != None:
                        self.the_dict[i] = Node(i.val)
                        childrenlst.append(self.the_dict[i])
                self.child_dict[node] = childrenlst
            for node in queue:
                for i in node.children:
                    new_queue.append(i)
            queue = new_queue

    def BFS2(self,root): # 第二轮BFS，链接每个节点的关系
        if root == None:
            return 
        queue = [root]
        while len(queue) != 0: # 注意数据结构，孩子存在列表里
            new_queue = []
            for node in queue:
                self.the_dict[node].children = self.child_dict[node]
            for node in queue:
                for i in node.children:
                    new_queue.append(i)
            queue = new_queue
        

```

# 1493. 删掉一个元素以后全为 1 的最长子数组

给你一个二进制数组 nums ，你需要从中删掉一个元素。

请你在删掉元素的结果数组中，返回最长的且只包含 1 的非空子数组的长度。

如果不存在这样的子数组，请返回 0 。

```python
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        # 全1必须删一个，其余情况常规滑动窗口
        sum_num = sum(nums)
        if sum_num == len(nums):
            return sum_num - 1
        # 滑动窗口
        left = 0
        right = 0
        size = 0
        max_size = 0
        count_zero = 0
        while right < len(nums):
            add = nums[right]
            right += 1
            if add != 0:
                size += 1
                max_size = max(max_size,size)
            elif add == 0:
                # 注意这里不能有size的变化
                count_zero += 1
                if count_zero <= 1:
                    max_size = max(max_size,size)
            while left < right and count_zero > 1:
                delete = nums[left]
                if delete == 0: # 注意这一段，只有删1的时候才有size的变化
                    count_zero -= 1
                elif delete != 0: 
                    size -= 1
                left += 1
        return max_size

```

# 1524. 和为奇数的子数组数目

给你一个整数数组 `arr` 。请你返回和为 **奇数** 的子数组数目。

由于答案可能会很大，请你将结果对 `10^9 + 7` 取余后返回。

```python
class Solution:
    def numOfSubarrays(self, arr: List[int]) -> int:
        # 子数组算法使用前缀和可能可以踩线过
        ans = 0
        mod = 10**9 + 7
        pre_dict = collections.defaultdict(int)
        pre_dict[0] = 1 # 记录偶数 # 注意这一行初始化
        pre_dict[1] = 0 # 记录奇数
        temp_sum = 0
        # 判断当前数是奇数还是偶数，如果是偶数，则只需要找之前有多少个奇数
        # 如果是奇数，则只需要找之前有多少个偶数
        for i in range(len(arr)):
            temp_sum += arr[i]
            if temp_sum % 2 == 0:
                pre_dict[0] += 1
                ans += pre_dict[1]
            elif temp_sum % 2 == 1:
                pre_dict[1] += 1
                ans += pre_dict[0]
        return ans % mod # 注意模一下

```

# 1525. 字符串的好分割数目

给你一个字符串 s ，一个分割被称为 「好分割」 当它满足：将 s 分割成 2 个字符串 p 和 q ，它们连接起来等于 s 且 p 和 q 中不同字符的数目相同。

请你返回 s 中好分割的数目。

```python
class Solution:
    def numSplits(self, s: str) -> int:
      # 滑动窗口
        leftwindow = collections.defaultdict(int)
        rightwindow = collections.defaultdict(int)
        gap = 0
        ans = 0
        # 初始化rightwindow
        for i in s:
            rightwindow[i] += 1
        while gap < len(s): # 移动分界线
            thechar = s[gap]
            leftwindow[thechar] += 1
            rightwindow[thechar] -= 1
            if rightwindow[thechar] == 0: del rightwindow[thechar]
            if len(leftwindow) == len(rightwindow):
                ans += 1
            gap += 1
        return ans
```

# 1534. 统计好三元组

给你一个整数数组 arr ，以及 a、b 、c 三个整数。请你统计其中好三元组的数量。

如果三元组 (arr[i], arr[j], arr[k]) 满足下列全部条件，则认为它是一个 好三元组 。

0 <= i < j < k < arr.length
|arr[i] - arr[j]| <= a
|arr[j] - arr[k]| <= b
|arr[i] - arr[k]| <= c
其中 |x| 表示 x 的绝对值。

返回 好三元组的数量 。

```python
class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        # 暴力统计
        n = len(arr)
        count = 0
        for i in range(n):
            for j in range(i+1,n):
                for k in range(j+1,n):
                    if abs(arr[i]-arr[j]) <= a and abs(arr[j]-arr[k]) <= b and abs(arr[i]-arr[k]) <= c:
                        count += 1
        return count
```



# 1539. 第 k 个缺失的正整数

给你一个 **严格升序排列** 的正整数数组 `arr` 和一个整数 `k` 。

请你找到这个数组里第 `k` 个缺失的正整数。

```python
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        # 利用一个数组指针来判定
        # 给出一个从1开始的自增变量
        arr = arr + [0xffffffff] # 加一个极大值尾部使得语法统一并且无需判定越界
        start = 1
        p = 0
        while k > 0:
            if start < arr[p]: # 如果那个数不在数组里，缺失了，那么
                k -= 1
            elif start == arr[p]:
                p += 1
            start += 1
        return start-1

```

# 1570. 两个稀疏向量的点积

给定两个稀疏向量，计算它们的点积（数量积）。

实现类 SparseVector：

SparseVector(nums) 以向量 nums 初始化对象。
dotProduct(vec) 计算此向量与 vec 的点积。
稀疏向量 是指绝大多数分量为 0 的向量。你需要 高效 地存储这个向量，并计算两个稀疏向量的点积。

进阶：当其中只有一个向量是稀疏向量时，你该如何解决此问题？

```python
class SparseVector:
    def __init__(self, nums: List[int]):
        # 采取哈希表
        self.n = len(nums)
        self.MyVec = collections.defaultdict(int)
        # 存储方式为存储【索引-值】，值为0则不存取
        for i,value in enumerate(nums):
            if value != 0:
                self.MyVec[i] = value

    # Return the dotProduct of two sparse vectors
    # 注意这个dotProduct的使用是v1.dotProduct(v2)
    # 参数vec是'SparseVector'类，即这个问题给出的类
    def dotProduct(self, vec: 'SparseVector') -> int:
        the_ans = 0
        for i in range(self.n):
            if vec.MyVec[i] != 0:
                the_ans += self.MyVec[i] * vec.MyVec[i]
        return the_ans
        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)
```

# 1583. 统计不开心的朋友

给你一份 n 位朋友的亲近程度列表，其中 n 总是 偶数 。

对每位朋友 i，preferences[i] 包含一份 按亲近程度从高到低排列 的朋友列表。换句话说，排在列表前面的朋友与 i 的亲近程度比排在列表后面的朋友更高。每个列表中的朋友均以 0 到 n-1 之间的整数表示。

所有的朋友被分成几对，配对情况以列表 pairs 给出，其中 pairs[i] = [xi, yi] 表示 xi 与 yi 配对，且 yi 与 xi 配对。

但是，这样的配对情况可能会是其中部分朋友感到不开心。在 x 与 y 配对且 u 与 v 配对的情况下，如果同时满足下述两个条件，x 就会不开心：

x 与 u 的亲近程度胜过 x 与 y，且
u 与 x 的亲近程度胜过 u 与 v
返回 不开心的朋友的数目 。

```python
class Solution:
    def unhappyFriends(self, n: int, preferences: List[List[int]], pairs: List[List[int]]) -> int:
        # 图论题，这题最麻烦的是看懂题目意思
        graph = [[0 for j in range (n)] for i in range(n)]
        # graph[i][j] 是 i 对 j 的偏爱程度
        for i in range(len(preferences)): # 图的构建
            pref = n - 1
            for people in preferences[i]:
                graph[i][people] = pref
                pref -= 1
        friends = [True for i in range(n)] # 初始化每个人都开心
        for cp1 in pairs: # 这个比对逻辑很复杂。。。
            x = cp1[0]
            y = cp1[1]
            for cp2 in pairs:
                u = cp2[0]
                v = cp2[1]
                if x == u and y == v: # 不和自身比较
                    continue 
                if graph[x][u] > graph[x][y] and graph[u][x] > graph[u][v]:
                    friends[x] = False
                x,y = y,x # 交换顺序比对
                if graph[x][u] > graph[x][y] and graph[u][x] > graph[u][v]:
                    friends[x] = False
                u,v = v,u # 交换顺序比对
                if graph[x][u] > graph[x][y] and graph[u][x] > graph[u][v]:
                    friends[x] = False
                x,y = y,x # 交换顺序比对
                if graph[x][u] > graph[x][y] and graph[u][x] > graph[u][v]:
                    friends[x] = False

        count = 0 # 注意一个人最多不开心一次
        for i in friends:
            if i == False:
                count += 1
        return count
```

# 1584. 连接所有点的最小费用

给你一个points 数组，表示 2D 平面上的一些点，其中 points[i] = [xi, yi] 。

连接点 [xi, yi] 和点 [xj, yj] 的费用为它们之间的 曼哈顿距离 ：|xi - xj| + |yi - yj| ，其中 |val| 表示 val 的绝对值。

请你返回将所有点连接的最小总费用。只有任意两点之间 有且仅有 一条简单路径时，才认为所有点都已连接。

```python
class UF: # 并查集判断环是否存在 
    def __init__(self,size):
        self.root = [i for i in range(size)]

    def union(self,x,y): # 并
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY: # 把y并入x
            self.root[self.find(y)] = self.find(x)

    def find(self,x): # 查
        while x != self.root[x]:
            x = self.root[x]
        return x

    def connected(self,x,y): # 看是否连接
        return self.find(x) == self.find(y)

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # 已知所有点两两不同
        # 根据所有点遍历找到最短距离
        # 如果只有一个点 直接返回
        # Kruskal算法
        if len(points) <= 1:
            return 0
        edges = []
        cost = 0
        for ind1,cp1 in enumerate(points): # 两两计算
            for ind2,cp2 in enumerate(points):
                if ind1 > ind2:
                    tp = [[],[],0]
                    temp_val = self.calc_value(cp1,cp2)
                    tp[0] = ind1
                    tp[1] = ind2
                    tp[2] = temp_val
                    edges.append(tp)
                else:
                    break 
        # 此时合成了点和带权边,带权边排序
        edges.sort(key = lambda x:x[2]) # 按照权重排序
        # 调用并查集
        union_find = UF(len(edges)+1) # 参数为边数目
        n = len(points)
        times = 0
        for e in edges: # kruscal 算法只需要连接N-1次
            if times < n - 1:
                if not union_find.connected(e[0],e[1]): 
                    union_find.union(e[0],e[1])
                    cost += e[2]
                    times += 1
            else:
                break
        return cost

    def calc_value(self,cp1,cp2):
        return abs(cp1[0]-cp2[0]) + abs(cp1[1]-cp2[1])
```

# 1592. 重新排列单词间的空格

给你一个字符串 text ，该字符串由若干被空格包围的单词组成。每个单词由一个或者多个小写英文字母组成，并且两个单词之间至少存在一个空格。题目测试用例保证 text 至少包含一个单词 。

请你重新排列空格，使每对相邻单词之间的空格数目都 相等 ，并尽可能 最大化 该数目。如果不能重新平均分配所有空格，请 将多余的空格放置在字符串末尾 ，这也意味着返回的字符串应当与原 text 字符串的长度相等。

返回 重新排列空格后的字符串 。

```python
class Solution:
    def reorderSpaces(self, text: str) -> str:
        # 模拟
        spaceCount = 0
        wordNum = 0
        t = text.split(" ")
        wordList = []
        for w in t:
            if w != "":
                wordNum += 1
                wordList.append(w)
        for ch in text:
            if ch == " ":
                spaceCount += 1
        # 空格插在单词间隙中，所以需要插入的空格数目为
        if wordNum == 1: # 则所有空格丢在最后
            ans = ''.join(wordList) + " "*spaceCount
            return ans 
        need = (spaceCount)//(wordNum-1)
        remain = spaceCount%(wordNum-1) # 丢在最后面
        needSpace = need*" "
        ans = needSpace.join(wordList)+remain*" "
        return ans
```

# 1602. 找到二叉树中最近的右侧节点

给定一棵二叉树的根节点 `root` 和树中的一个节点 `u` ，返回与 `u` **所在层**中**距离最近**的**右侧**节点，当 `u` 是所在层中最右侧的节点，返回 `null` 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findNearestRightNode(self, root: TreeNode, u: TreeNode) -> TreeNode:
        # BFS借助队列管理,树非空
        ans = [] # 收集每层结果,这里其实用不到
        queue = [root]
        while len(queue) != 0:
            level = [] # 收集本层
            new_queue = [] # 下一层结果
            for i in queue:
                if i != None:
                    level.append(i)
                    new_queue.append(i.left)
                    new_queue.append(i.right)
            for i,node in enumerate(level): # 检查u是否在此层中
                if node == u:
                    if i < len(level) - 1:
                        return level[i+1]
                    else:
                        return None
                    break
            queue = new_queue
        # 循环后level中一定有u
```

# 1624. 两个相同字符之间的最长子字符串

给你一个字符串 s，请你返回 两个相同字符之间的最长子字符串的长度 ，计算长度时不含这两个字符。如果不存在这样的子字符串，返回 -1 。

子字符串 是字符串中的一个连续字符序列。

```python
class Solution:
    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        # 哈希表，记录元素和索引,模拟
        record = collections.defaultdict(list)
        for index,val in enumerate(s):
            record[val].append(index)
        ans = -1 # 初始化结果为-1 
        for lst in record.values():
            if len(lst) >= 2:
                ans = max(ans,lst[-1] - lst[0] - 1)
        return ans

```

# 1644. 二叉树的最近公共祖先 II

给定一棵二叉树的根节点 root，返回给定节点 p 和 q 的最近公共祖先（LCA）节点。如果 p 或 q 之一不存在于该二叉树中，返回 null。树中的每个节点值都是互不相同的。

根据维基百科中对最近公共祖先节点的定义：“两个节点 p 和 q 在二叉树 T 中的最近公共祖先节点是后代节点中既包括 p 又包括 q 的最深节点（我们允许一个节点为自身的一个后代节点）”。一个节点 x 的后代节点是节点 x 到某一叶节点间的路径中的节点 y。 

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # 这一题最重要的收获是递归逻辑里，函数要分离，防止调用递归的时候使用了内层的函数
    # 且千万注意函数功能分离时，名称的改写
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.valid = 0 # 判断合法节点数量
        self.dfs(root,p) # 搜索p是不是合法的
        self.dfs(root,q)  # 搜索q是不合法的    
        if self.valid != 2:  # 如果没有两个合法的，直接返回null
            return None
        else:
            return self.find_LCA(root,p,q) # 否则开始真正的搜索
        
    def dfs(self,node,target_node): # 搜索节点是否合法，先序遍历即可
        if node == None:
            return
        if node == target_node:
            self.valid += 1
        self.dfs(node.left,target_node)
        self.dfs(node.right,target_node)
    
    # 之前采取的通用后序遍历找LCA的解法
    def find_LCA(self,root,p,q):
        if root == None: return None
        if root == p or root == q: return root
        left = self.find_LCA(root.left,p,q)
        right = self.find_LCA(root.right,p,q)
        if left == None and right == None: return None
        if left != None and right == None: return left
        if left == None and right != None: return right
        if left != None and right != None: return root
```

# 1646. 获取生成数组中的最大值

给你一个整数 n 。按下述规则生成一个长度为 n + 1 的数组 nums ：

nums[0] = 0
nums[1] = 1
当 2 <= 2 * i <= n 时，nums[2 * i] = nums[i]
当 2 <= 2 * i + 1 <= n 时，nums[2 * i + 1] = nums[i] + nums[i + 1]
返回生成数组 nums 中的 最大 值。

```python
class Solution:
    def getMaximumGenerated(self, n: int) -> int:
        # 纯模拟
        if n == 0 :
            return 0
        elif n == 1 or n == 2:
            return 1
        nums = [0 for i in range(n+1)]
        nums[1] = 1
        for i in range(1,n+1):
            if 2 <= 2 * i <= n:
                nums[2*i] = nums[i]
            if 2 <= 2 * i + 1 <=n:
                nums[2 * i + 1] = nums[i] + nums[i + 1]
        return max(nums)
```

# 1647. 字符频次唯一的最小删除次数

如果字符串 s 中 不存在 两个不同字符 频次 相同的情况，就称 s 是 优质字符串 。

给你一个字符串 s，返回使 s 成为 优质字符串 需要删除的 最小 字符数。

字符串中字符的 频次 是该字符在字符串中的出现次数。例如，在字符串 "aab" 中，'a' 的频次是 2，而 'b' 的频次是 1 。

```python
class Solution:
    def minDeletions(self, s: str) -> int:
        # 都是小写字母，那么使用数组记录频数
        arr = [0 for i in range(26)]
        for i in s:
            index = ord(i) - ord("a")
            arr[index] += 1
        # 记录频率最大值和次大值，如果最大值和次大值不相等，则弹出最大值，它无需处理
        # 如果次大值有多个，只保留一个次大值，其余的全部值都下降
        lst = []
        for n in arr:
            if n != 0:
                lst.append(n)
        lst.sort(reverse = True) # 排序后，左边为最大值。只保留一个相同值
        # 只允许0相同
        if len(lst) <= 1:
            return 0 # 无需调整
        lst = lst + [0]
        ans = 0
        # 现在变为处理一个最多长度为26的数组，数组中可能有重复数，允许减小数值，使得所有正数各不相同。
        # 即类似于处理[5,3,3,2,2,1] 通过缩小数值，使得所有正数都不相同
        max_heap = [-i for i in lst]
        heapq.heapify(max_heap)
        while max_heap[0] != 0:
            e = heapq.heappop(max_heap)
            if e != max_heap[0]:
                pass
            elif e == max_heap[0]:
                ans += 1
                heapq.heappush(max_heap,e+1) # 注意这里用的是+号，因为堆中全是负数
        return ans
```

# 1650. 二叉树的最近公共祖先 III

给定一棵二叉树中的两个节点 p 和 q，返回它们的最近公共祖先节点（LCA）。

每个节点都包含其父节点的引用（指针）。Node 的定义如下：

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node parent;
}
根据维基百科中对最近公共祖先节点的定义：“两个节点 p 和 q 在二叉树 T 中的最近公共祖先节点是后代节点中既包括 p 又包括 q 的最深节点（我们允许一个节点为自身的一个后代节点）”。一个节点 x 的后代节点是节点 x 到某一叶节点间的路径中的节点 y。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        # 有了父节点指针，即变成找找两链表的第一个公共节点,不破坏树
        # 或者是搜到p,q按深度做k-v对，去比对完成，找到深度最大的那个
        lst1 = []
        cur1 = p
        while cur1 != None:
            lst1.append(cur1)
            cur1 = cur1.parent
        lst2 = []
        cur2 = q
        while cur2 != None:
            lst2.append(cur2)
            cur2 = cur2.parent
        lst1 = lst1[::-1]
        lst2 = lst2[::-1]
        # print(lst1,lst2)
        p1 = 0
        p2 = 0
        cur1 = lst1[p1]
        cur2 = lst2[p2]
        # 这是一定有公共前缀的链表,找到第一个不相同的，返回前一个
        while p1 < len(lst1) and p2 < len(lst2):
            cur1 = lst1[p1]
            cur2 = lst2[p2]
            # print(cur1,cur2)
            if cur1 == cur2:
                p1 += 1
                p2 += 1
            else:
                break
        # print(lst1[p1-1])
        return lst1[p1-1]
            
```

# 1656. 设计有序流

有 n 个 (id, value) 对，其中 id 是 1 到 n 之间的一个整数，value 是一个字符串。不存在 id 相同的两个 (id, value) 对。

设计一个流，以 任意 顺序获取 n 个 (id, value) 对，并在多次调用时 按 id 递增的顺序 返回一些值。

实现 OrderedStream 类：

OrderedStream(int n) 构造一个能接收 n 个值的流，并将当前指针 ptr 设为 1 。
String[] insert(int id, String value) 向流中存储新的 (id, value) 对。存储后：
如果流存储有 id = ptr 的 (id, value) 对，则找出从 id = ptr 开始的 最长 id 连续递增序列 ，并 按顺序 返回与这些 id 关联的值的列表。然后，将 ptr 更新为最后那个  id + 1 。
否则，返回一个空列表。

```python
class OrderedStream:
# 读懂题意模拟即可
    def __init__(self, n: int):
        self.ptr = 0 # 初始化位置指针
        self.arr = [None for i in range(n)] # 初始化储存数组

    def insert(self, idKey: int, value: str) -> List[str]:
        self.arr[idKey-1] = value # 填充
        ans = [] # 接收答案
        if self.arr[self.ptr] != None: # 位置指向的元素不为空，则一路收集
            while self.ptr < len(self.arr) and self.arr[self.ptr] != None:
                ans.append(self.arr[self.ptr])
                self.ptr += 1
        return ans
```

# 1676. 二叉树的最近公共祖先 IV

给定一棵二叉树的根节点 root 和 TreeNode 类对象的数组（列表） nodes，返回 nodes 中所有节点的最近公共祖先（LCA）。数组（列表）中所有节点都存在于该二叉树中，且二叉树中所有节点的值都是互不相同的。

我们扩展二叉树的最近公共祖先节点在维基百科上的定义：“对于任意合理的 i 值， n 个节点 p1 、 p2、...、 pn 在二叉树 T 中的最近公共祖先节点是后代中包含所有节点 pi 的最深节点（我们允许一个节点是其自身的后代）”。一个节点 x 的后代节点是节点 x 到某一叶节点间的路径中的节点 y。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
        # 做个集合查表，分离解法
        self.set1 = set(nodes)
        return self.find_LCA(root,nodes)
    
    def find_LCA(self,root,nodes):
        if root == None: return None
        if root in self.set1: return root # 分离进行查询
        left = self.find_LCA(root.left,nodes)
        right = self.find_LCA(root.right,nodes)
        if left == None and right == None: return None
        if left != None and right == None: return left
        if left == None and right != None: return right
        if left != None and right != None: return root

```

# 1736. 替换隐藏数字得到的最晚时间

给你一个字符串 time ，格式为 hh:mm（小时：分钟），其中某几位数字被隐藏（用 ? 表示）。

有效的时间为 00:00 到 23:59 之间的所有时间，包括 00:00 和 23:59 。

替换 time 中隐藏的数字，返回你可以得到的最晚有效时间。

```python
class Solution:
    def maximumTime(self, time: str) -> str:
        # 仔细分析
        # 前两位为??时候，赋值23
        # 第一位为？ 第二位在0～3，则赋值2，第二位在4～9，赋值1
        # 第二位为？ 第一位为0～1 赋值9，第一位为2，赋值3

        # 后两位为??，赋值59
        # 第一位为? 直接赋值5
        # 第二位为？ 直接赋值9
        temp_time = list(time) #先列表化
        if temp_time[0] == temp_time[1] == '?':
            temp_time[0] = '2'
            temp_time[1] = '3'
        if temp_time[0] == '?' and temp_time[1] != '?':
            if temp_time[1] in '0123':
                temp_time[0] = '2'
            elif temp_time[1] in '456789':
                temp_time[0] = '1'
        if temp_time[0] != '?' and temp_time[1] == '?':
            if temp_time[0] in '01':
                temp_time[1] = '9'
            elif temp_time[0] == '2':
                temp_time[1] = '3'
        if temp_time[3] == '?':
            temp_time[3] = '5'
        if temp_time[4] == '?':
            temp_time[4] = '9'
        ans = ''.join(temp_time)
        return ans 
```

# 1740. 找到二叉树中的距离

给定一棵二叉树的根节点 root 以及两个整数 p 和 q ，返回该二叉树中值为 p 的结点与值为 q 的结点间的 距离 。

两个结点间的 距离 就是从一个结点到另一个结点的路径上边的数目。

```python
class Solution:
    def __init__(self):
        self.ans = [] # 收集深度

    def findDistance(self, root: TreeNode, p: int, q: int) -> int:
        # 找到他们的最近公共祖先,再以最近公共祖先去分别搜索p,q，求深度之和
        if p == q:
            return 0
        start = self.findLCA(root,p,q)
        self.dfs(start,p,0)
        self.dfs(start,q,0)
        return sum(self.ans)

    def dfs(self,root,val,depth):
        if root == None:
            return 
        if root.val == val:
            self.ans.append(depth)
            return
        left = self.dfs(root.left,val,depth+1)
        right = self.dfs(root.right,val,depth+1)
        
    
    def findLCA(self,root,p,q):
        if root == None:
            return None
        if root.val == p or root.val == q:
            return root
        left = self.findLCA(root.left,p,q)
        right = self.findLCA(root.right,p,q)
        # p 和 q 一定在树中
        if left != None and right != None:
            return root
        if left == None and right != None:
            return right
        if left != None and right == None:
            return left
```

# 1743. 从相邻元素对还原数组

存在一个由 n 个不同元素组成的整数数组 nums ，但你已经记不清具体内容。好在你还记得 nums 中的每一对相邻元素。

给你一个二维整数数组 adjacentPairs ，大小为 n - 1 ，其中每个 adjacentPairs[i] = [ui, vi] 表示元素 ui 和 vi 在 nums 中相邻。

题目数据保证所有由元素 nums[i] 和 nums[i+1] 组成的相邻元素对都存在于 adjacentPairs 中，存在形式可能是 [nums[i], nums[i+1]] ，也可能是 [nums[i+1], nums[i]] 。这些相邻元素对可以 按任意顺序 出现。

返回 原始数组 nums 。如果存在多种解答，返回 其中任意一个 即可。

```python
class Solution:
    def restoreArray(self, adjacentPairs: List[List[int]]) -> List[int]:
        # 建立默认里面装的是列表的字典
        # 键值互相转换，填充两次
        # 此题保证有解，无需考虑可行性
        dict_1 = defaultdict(list)
        for i in adjacentPairs:
            dict_1[i[0]].append(i[1])
            dict_1[i[1]].append(i[0])
        start = None
        for i in dict_1.keys():
            if len(dict_1[i]) == 1:
                start = i
                break
        ans = [start,dict_1[start][0]]
        next_ = dict_1[dict_1[start][0]]
        while len(next_) != 1: # 
            if next_[0] == ans[-2]: #此行很关键
                ans.append(next_[1])
                next_ = dict_1[next_[1]]
            elif next_[1] == ans[-2]:
                ans.append(next_[0])
                next_ = dict_1[next_[0]]
        return ans

```

# 1753. 移除石子的最大得分

你正在玩一个单人游戏，面前放置着大小分别为 `a`、`b` 和 `c` 的 **三堆** 石子。

每回合你都要从两个 **不同的非空堆** 中取出一颗石子，并在得分上加 `1` 分。当存在 **两个或更多** 的空堆时，游戏停止。

给你三个整数 `a` 、`b` 和 `c` ，返回可以得到的 **最大分数** 。

```python
# 方法1: 贪心解法
class Solution:
    def maximumScore(self, a: int, b: int, c: int) -> int:
        # 贪心，每一次取非0的最小的和最大的
        lst = [a,b,c]
        lst.sort()
        points = 0
        while lst[0] != 0:
            lst[0] -= 1
            lst[2] -= 1
            points += 1
            lst.sort()
        # 然后取到使得第二个位置为0
        points += lst[1]
        return points
```

```python
# 方法2:数学解法
class Solution:
    def maximumScore(self, a: int, b: int, c: int) -> int:
        # 贪心，每一次取非0的最小的和最大的
        lst = [a,b,c]
        lst.sort()
        a,b,c = lst[0],lst[1],lst[2] # 排序使得a<=b<=c
        if a+b <= c: # 这时候把a,b取空就是最优解
        		return a + b
        else: # 利用c把a,b取到尽量平衡，最终a,b差不超过1.用地板除即可
        		return (a+b+c)//2
        	        
```

# 1754. 构造字典序最大的合并字符串

给你两个字符串 word1 和 word2 。你需要按下述方式构造一个新字符串 merge ：如果 word1 或 word2 非空，选择 下面选项之一 继续操作：

如果 word1 非空，将 word1 中的第一个字符附加到 merge 的末尾，并将其从 word1 中移除。
例如，word1 = "abc" 且 merge = "dv" ，在执行此选项操作之后，word1 = "bc" ，同时 merge = "dva" 。
如果 word2 非空，将 word2 中的第一个字符附加到 merge 的末尾，并将其从 word2 中移除。
例如，word2 = "abc" 且 merge = "" ，在执行此选项操作之后，word2 = "bc" ，同时 merge = "a" 。
返回你可以构造的字典序 最大 的合并字符串 merge 。

长度相同的两个字符串 a 和 b 比较字典序大小，如果在 a 和 b 出现不同的第一个位置，a 中字符在字母表中的出现顺序位于 b 中相应字符之后，就认为字符串 a 按字典序比字符串 b 更大。例如，"abcd" 按字典序比 "abcc" 更大，因为两个字符串出现不同的第一个位置是第四个字符，而 d 在字母表中的出现顺序位于 c 之后。

```python
class Solution:
    def largestMerge(self, word1: str, word2: str) -> str:
        # 奇奇怪怪的归并排序？这个题不简单
        # 正向合并
        change = False # 打的补丁，俩字符串一样的时候用下面的比较逻辑会出问题
        if word1 == word2:
            word1 += "a" # 随便补个，最后除去
            change = True
        p1 = 0
        p2 = 0
        ans = ""
        while p1 < len(word1) and p2 < len(word2):
            # 不同的时候取大的，相同的时候根据后面的字符进行选择
            if word1[p1] < word2[p2]:
                ans += word2[p2]
                p2 += 1
            elif word1[p1] > word2[p2]:
                ans += word1[p1]
                p1 += 1
            elif word1[p1] == word2[p2]: # !最关键点：找到后面的第一个非公共前缀
                temp1 = p1 + 1
                temp2 = p2 + 1
                while temp1 < len(word1) and temp2 < len(word2) and word1[temp1] == word2[temp2]:
                    temp1 += 1
                    temp2 += 1

                # 此时两个temp指向空或者指向不同的字符
                if temp1 < len(word1) and temp2 < len(word2): # 
                    # 谁指的大，移动谁的p
                    if word1[temp1] > word2[temp2]:
                        ans += word1[p1]
                        p1 += 1
                    else:
                        ans += word2[p2]
                        p2 += 1
                elif temp1 == len(word1) and temp2 < len(word2):
                    # 与标准word1[p1] 相比
                    if word2[temp2] < word1[p1]:
                        ans += word1[p1]
                        p1 += 1
                    else:
                        ans += word2[p2]
                        p2 += 1
                elif temp1 < len(word1) and temp2 == len(word2):
                    if word1[temp1] < word2[p2]:
                        ans += word2[p2]
                        p2 += 1
                    else:
                        ans += word1[p1]
                        p1 += 1
                else: # 两者都为空，全加了
                    ans += word1[p1]
                    ans += word2[p2]
                    p1 += 1
                    p2 += 1

        while p1 < len(word1):
            ans += word1[p1]
            p1 += 1
        while p2 < len(word2):
            ans += word2[p2]
            p2 += 1
        if change:
            return ans[:-1]
        return ans 
```

```python
class Solution:
    def largestMerge(self, word1: str, word2: str) -> str:
        # 利用python自带api,实时比较
        q1 = deque(word1)
        q2 = deque(word2)
        ans = ""
        while len(q1) != 0 and len(q2) != 0:
            if q1 > q2:
                ans += (q1.popleft())
            else:
                ans += (q2.popleft())
        while len(q1) != 0:
            ans += q1.popleft()
        while len(q2) != 0:
            ans += q2.popleft()
        return ans
```

# 1796. 字符串中第二大的数字

给你一个混合字符串 `s` ，请你返回 `s` 中 **第二大** 的数字，如果不存在第二大的数字，请你返回 `-1` 。

**混合字符串** 由小写英文字母和数字组成。

```python
class Solution:
    def secondHighest(self, s: str) -> int:
        # 这个求的是第二大的数字，不是第二小
        # 数字是取不同的数字
        ct = [0 for i in range(10)]
        for i in s:
            if i.isdigit():
                ct[int(i)] += 1
        valid = False
        for i in range(len(ct)-1,-1,-1):
            if ct[i] != 0 and valid == False:
                valid = True
            elif ct[i] != 0 and valid == True:
                return i
        return -1       

```

# 1804. 实现 Trie （前缀树） II

前缀树（trie ，发音为 "try"）是一个树状的数据结构，用于高效地存储和检索一系列字符串的前缀。前缀树有许多应用，如自动补全和拼写检查。

实现前缀树 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 将字符串 word 插入前缀树中。
int countWordsEqualTo(String word) 返回前缀树中字符串 word 的实例个数。
int countWordsStartingWith(String prefix) 返回前缀树中以 prefix 为前缀的字符串个数。
void erase(String word) 从前缀树中移除字符串 word 。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.words_num = 0
        self.isWord = False
        
class Trie:

    def __init__(self):
        self.root = TrieNode() # 实例化一个根节点

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            index = ord(char) - ord("a")
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isWord = True
        node.words_num += 1


    def countWordsEqualTo(self, word: str) -> int:
        node = self.root
        for char in word:
            index = ord(char) - ord("a")
            if node.children[index] == None:
                return 0 # 搜索中途就断了
            node = node.children[index]
        if node.isWord: # 正好有公共部分，且是单词
            return node.words_num
        elif node.isWord == False: # 虽然有公共部分，但是不是单词
            return 0


    def countWordsStartingWith(self, prefix: str) -> int:
        node = self.root 
        for char in prefix:
            index = ord(char) - ord("a")
            if node.children[index] == None:
                return 0 
            node = node.children[index]
        queue = [node] # 开启BFS搜索结果
        the_sum = 0
        while len(queue) != 0:
            new_queue = []
            for every_node in queue:
                if every_node.isWord:
                    the_sum += every_node.words_num                  
                for index in range(26):
                    if every_node.children[index] != None:
                        new_queue.append(every_node.children[index])
            queue = new_queue
        return the_sum
            
    def erase(self, word: str) -> None:
        node = self.root
        for char in word:
            index = ord(char) - ord("a")
            node = node.children[index]       
        node.words_num -= 1 # 移除只减少一个
        if node.words_num == 0:
            node.isWord = False
```

# 1807. 替换字符串中的括号内容

给你一个字符串 s ，它包含一些括号对，每个括号中包含一个 非空 的键。

比方说，字符串 "(name)is(age)yearsold" 中，有 两个 括号对，分别包含键 "name" 和 "age" 。
你知道许多键对应的值，这些关系由二维字符串数组 knowledge 表示，其中 knowledge[i] = [keyi, valuei] ，表示键 keyi 对应的值为 valuei 。

你需要替换 所有 的括号对。当你替换一个括号对，且它包含的键为 keyi 时，你需要：

将 keyi 和括号用对应的值 valuei 替换。
如果从 knowledge 中无法得知某个键对应的值，你需要将 keyi 和括号用问号 "?" 替换（不需要引号）。
knowledge 中每个键最多只会出现一次。s 中不会有嵌套的括号。

请你返回替换 所有 括号对后的结果字符串。

```python
class Solution:
    def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
        # 模拟，这一题有小陷阱，只有括号里的才需要换
        lst = [] # 格式化
        for i in s:
            if i == "(":
                lst.append(" ")
                lst.append(i)
            elif i == ")":
                lst.append(i)
                lst.append(" ")
            else:
                lst.append(i)
        temp = "".join(lst)
        temp = temp.strip() # 脱去左右空格,里面还会剩下一些空格。。
        temp = temp.split(" ")
        needSet = set()   # 这里面留存的是带括号的
        for w in temp:
            if len(w) != 0 and w[0] == "(": # 这里需要保证len(w) != 0 是因为里面还有可能有空串
                needSet.add(w)
        aimDict = dict()
        for pair in knowledge:
            aimDict["("+pair[0]+")"] = pair[1]
        for w in range(len(temp)):
            if temp[w] in aimDict:
                temp[w] = aimDict[temp[w]]
            elif temp[w] not in aimDict and temp[w] in needSet:
                temp[w] = "?"
        return ''.join(temp)
```

# 1826. 有缺陷的传感器

实验室里正在进行一项实验。为了确保数据的准确性，同时使用 两个 传感器来采集数据。您将获得2个数组 sensor1 and sensor2，其中 sensor1[i] 和 sensor2[i] 分别是两个传感器对第 i 个数据点采集到的数据。

但是，这种类型的传感器有可能存在缺陷，它会导致 某一个 数据点采集的数据（掉落值）被丢弃。

数据被丢弃后，所有在其右侧的数据点采集的数据，都会被向左移动一个位置，最后一个数据点采集的数据会被一些随机值替换。可以保证此随机值不等于掉落值。

举个例子, 如果正确的数据是 [1,2,3,4,5] ， 此时 3 被丢弃了, 传感器会返回 [1,2,4,5,7] (最后的位置可以是任何值, 不仅仅是 7).
可以确定的是，最多有一个 传感器有缺陷。请返回这个有缺陷的传感器的编号 （1 或 2）。如果任一传感器 没有缺陷 ，或者 无法 确定有缺陷的传感器，则返回 -1 。

```python
class Solution:
    def badSensor(self, sensor1: List[int], sensor2: List[int]) -> int:
        # 双指针讨论。当对比到的值不一样的时候，进行两次比对
        # 第一次比对，s1移位，和s2比对，如果在结束之前都没有断匹配，则2是可以有缺陷的
        # 第二次比对，s2移，和s1比对，如果在结束之前都没有断匹配，则1是可以有缺陷的
        # 如果1，2都有可能，则返回-1
        n = len(sensor1)
        p = 0
        active1 = False 
        while p < n:
            if sensor1[p] != sensor2[p]:
                p1 = p + 1
                p2 = p + 1
                memo = p
                active1 = True 
                break 
            p += 1
        
        if not active1:
            return -1

        if active1:
            judge1 = True 
            while p1 < n and p < n:
                if sensor1[p1] != sensor2[p]:
                    judge1 = False 
                    break 
                p1 += 1
                p += 1
        
        p = memo  # 复原p
        if active1:
            judge2 = True 
            while p2 < n and p < n:
                if sensor1[p] != sensor2[p2]:
                    judge2 = False 
                    break 
                p2 += 1
                p += 1
        
        if judge1 and judge2:
            return -1
        if judge1:
            return 2
        if judge2:
            return 1
```

# 1836. 从未排序的链表中移除重复元素

给定一个链表的第一个节点 `head` ，找到链表中所有出现**多于一次**的元素，并删除这些元素所在的节点。

返回删除后的链表。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
        # 利用哈希表存储所有值对应的节点
        dummy = ListNode(0)
        dummy.next = head
        memo = defaultdict(list)
        cur = head
        while cur != None:
            memo[cur.val].append(cur)
            cur = cur.next
        cur1 = dummy
        cur2 = head # 主指针
        while cur2 != None:
            if len(memo[cur2.val]) >= 2: # 需要删除cur2
                temp = cur2.next
                cur1.next = temp
                cur2 = temp
                continue
            else: # cur2不用删除，
                cur1 = cur1.next
                cur2 = cur2.next
        return dummy.next

```

# 1863. 找出所有子集的异或总和再求和

一个数组的 异或总和 定义为数组中所有元素按位 XOR 的结果；如果数组为 空 ，则异或总和为 0 。

例如，数组 [2,5,6] 的 异或总和 为 2 XOR 5 XOR 6 = 1 。
给你一个数组 nums ，请你求出 nums 中每个 子集 的 异或总和 ，计算并返回这些值相加之 和 。

注意：在本题中，元素 相同 的不同子集应 多次 计数。

数组 a 是数组 b 的一个 子集 的前提条件是：从 b 删除几个（也可能不删除）元素能够得到 a 。

```python
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        # 回溯获取全部子集
        ans = []
        path = []
        def backtracking(path,nums):
            if len(path) != 0:
                ans.append(path[:])
            for i,n in enumerate(nums):
                path.append(n)
                backtracking(path,nums[i+1:])
                path.pop()       
        backtracking(path,nums) # 调用回溯
        # 此时ans填充完毕
        count_sum = 0
        for tp in ans:
            start = 0
            for i in tp:
                start ^= i
            count_sum += start
        return count_sum
```

# 1874. 两个数组的最小乘积和

给定两个长度相等的数组a和b，它们的乘积和为数组中所有的a[i] * b[i]之和，其中0 <= i < a.length。

比如a = [1,2,3,4]，b = [5,2,3,1]时，它们的乘积和为1*5 + 2*2 + 3*3 + 4*1 = 22
现有两个长度都为n的数组nums1和nums2，你可以以任意顺序排序nums1，请返回它们的最小乘积和。

```python
class Solution:
    def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
        # 这一题是数学概念换皮
        # 虽然只说允许排列nums1，但是可以构建nums1和nums2的唯一映射使得最小化
        # 所以拿nums2也重排并无影响
        # 并且由于元素都是正数，有逆序和 <= 乱序和 <= 正序和
        nums1.sort()
        nums2.sort(reverse = True)
        sum_num = 0
        for i in range(len(nums1)):
            sum_num += nums1[i] * nums2[i]
        return sum_num

```

# 1910. 删除一个字符串中所有出现的给定子字符串

给你两个字符串 s 和 part ，请你对 s 反复执行以下操作直到 所有 子字符串 part 都被删除：

找到 s 中 最左边 的子字符串 part ，并将它从 s 中删除。
请你返回从 s 中删除所有 part 子字符串以后得到的剩余字符串。

一个 子字符串 是一个字符串中连续的字符序列。

```python
class Solution:
    def removeOccurrences(self, s: str, part: str) -> str:
        # 栈匹配解法,O(n^2)级别
        stack = [] # 双端队列没有切片！
        n = len(part)
        for ch in s:
            stack.append(ch) # 先加入
            while len(stack) != 0 and "".join(stack[len(stack)-n:len(stack)]) == part: # 再检查，注意是循环检查
                tempString = "".join(stack[len(stack)-n:len(stack)])
                if tempString == part:
                    for i in range(n):
                        stack.pop()
        return "".join(stack)

```

# 1913. 两个数对之间的最大乘积差

两个数对 (a, b) 和 (c, d) 之间的 乘积差 定义为 (a * b) - (c * d) 。

例如，(5, 6) 和 (2, 7) 之间的乘积差是 (5 * 6) - (2 * 7) = 16 。
给你一个整数数组 nums ，选出四个 不同的 下标 w、x、y 和 z ，使数对 (nums[w], nums[x]) 和 (nums[y], nums[z]) 之间的 乘积差 取到 最大值 。

返回以这种方式取得的乘积差中的 最大值 。

```python
class Solution:
    def maxProductDifference(self, nums: List[int]) -> int:
        # 这一题所有数字非负已经暴降难度了
        # 最大俩数乘 - 最小两数乘
        nums.sort()
        return (nums[-1]*nums[-2]) - (nums[0]*nums[1])
```

# 1925. 统计平方和三元组的数目

一个 平方和三元组 (a,b,c) 指的是满足 a2 + b2 = c2 的 整数 三元组 a，b 和 c 。

给你一个整数 n ，请你返回满足 1 <= a, b, c <= n 的 平方和三元组 的数目。

```java
class Solution {
    public int countTriples(int n) {
        int count = 0;
        for(int i = 1;i <= n;i += 1){
            for (int j = 1; j <= n; j += 1){
                for (int k = Math.max(i,j); k <= n; k += 1){
                    if (Math.pow(i,2) + Math.pow(j,2) == Math.pow(k,2)){
                        count += 1;
                    }
                }
            }
        }
        return count;
    }
}
```

```python
class Solution:
    def countTriples(self, n: int) -> int:
        count = 0
        target = set() # 记录下 1 ～ n 的全部元素和
        for i in range(1,n+1):
            target.add(i ** 2)

        for a in range(1,n+1):
            for b in range(1,n+1):
                if a ** 2 + b ** 2 in target:
                    count += 1

        return count
```

# 1930. 长度为 3 的不同回文子序列

给你一个字符串 s ，返回 s 中 长度为 3 的不同回文子序列 的个数。

即便存在多种方法来构建相同的子序列，但相同的子序列只计数一次。

回文 是正着读和反着读一样的字符串。

子序列 是由原字符串删除其中部分字符（也可以不删除）且不改变剩余字符之间相对顺序形成的一个新字符串。

例如，"ace" 是 "abcde" 的一个子序列。

```python
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        # 一个收集答案
        # 然后分割窗口，看窗口前后是否有一样的字符
        # 常数很大的O(n)
        if len(s) <= 2:
            return 0
        elif len(s) == 3:
            if s[0] == s[2]:
                return 1
            else:
                return 0
        ans_set = set()
        window1 = collections.defaultdict(int) # 增长窗口
        window2 = collections.defaultdict(int) # 缩小窗口
        for i in s:
            window2[i] += 1
        p = 0 # p指向的字母为中心字母
        # 注意窗口移动策略。
        while p < len(s):
            window2[s[p]] -= 1
            if window2[s[p]] == 0: del window2[s[p]]
            for k in window1:
                if k in window2:
                    ans_set.add(k+s[p]+k)
            window1[s[p]] += 1
            p += 1
        return len(ans_set)
```

# 1933. 判断字符串是否可分解为值均等的子串

一个字符串的所有字符都是一样的，被称作等值字符串。

举例，"1111" 和 "33" 就是等值字符串。
相比之下，"123"就不是等值字符串。
规则：给出一个数字字符串s，将字符串分解成一些等值字符串，如果有且仅有一个等值子字符串长度为2，其他的等值子字符串的长度都是3.

如果能够按照上面的规则分解字符串s，就返回真，否则返回假。

子串就是原字符串中连续的字符序列。

```python
class Solution:
    def isDecomposable(self, s: str) -> bool:
        # 模拟法。其实不需要栈，一个列表而已
        if len(s) % 3 != 2:
            return False
        stack = []
        for i in s:
            if len(stack) == 0:
                stack.append([i,1])
            elif len(stack) != 0:
                if i == stack[-1][0] and stack[-1][1] < 3:
                    stack[-1][1] += 1
                elif i == stack[-1][0] and stack[-1][1] == 3:
                    stack.append([i,1])
                elif i != stack[-1][0]:
                    stack.append([i,1])
        count3 = 0
        count2 = 0
        for cp in stack:
            if cp[1] == 3:
                count3 += 1
            elif cp[1] == 2:
                count2 += 1
        return count2 == 1 and count3 == len(stack)-1
```

# 1935. 可以输入的最大单词数

键盘出现了一些故障，有些字母键无法正常工作。而键盘上所有其他键都能够正常工作。

给你一个由若干单词组成的字符串 text ，单词间由单个空格组成（不含前导和尾随空格）；另有一个字符串 brokenLetters ，由所有已损坏的不同字母键组成，返回你可以使用此键盘完全输入的 text 中单词的数目。

```python
class Solution:
    def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
        # 处理成单词列表
        word_lst = text.split(' ')
        # 缺省字典
        dict1 = defaultdict(int)
        for i in brokenLetters:
           dict1[i] += 1
        count = 0 # 接受答案
        for word in word_lst:
            for char_index in range(len(word)):
                if dict1[word[char_index]] == 1:
                    break
                elif dict1[word[char_index]] == 0 and char_index == len(word)-1: # 这一串的最后一个也经过了筛选
                    count += 1
        return count
                    
```

# 1940. 排序数组之间的最长公共子序列

给定一个由整数数组组成的数组arrays，其中arrays[i]是严格递增排序的，返回一个表示所有数组之间的最长公共子序列的整数数组。

子序列是从另一个序列派生出来的序列，删除一些元素或不删除任何元素，而不改变其余元素的顺序。

```python
class Solution:
    def longestCommomSubsequence(self, arrays: List[List[int]]) -> List[int]:
        # 多排指针，创建一个存储每一行指针位置的数组,初始化都指向0
        points = [0 for i in range(len(arrays))]
        bound = [len(i) for i in arrays]
        path = [] # 收集结果
        # 有一个越界则结束查找
        # 指针移动逻辑，都指向相同值则加入，否则移动所有不是最大值的指针
        def check (points,bound): # 检查是否有指针越界了
            for i in range(len(points)):
                if points[i] >= bound[i]:
                    return False
            return True

        while check(points,bound):
            # 比较所有当前指的值
            now_value_lst = []
            for num,row in enumerate(arrays):
                now_value_lst.append(row[points[num]]) # 把每一行的对应值加进去
            max_value = max(now_value_lst)
            count = 0
            for i in now_value_lst: # 检查是否都指向同一个值
                if i == max_value:
                    count += 1
            if count == len(arrays): # 如果都指向同一个值，收集结果
                path.append(max_value)
                for i in range(len(points)): # 执行完毕之后，所有指针往后移动
                    points[i] += 1 
                continue # 注意这一条
            for num,row in enumerate(arrays):
                if row[points[num]] < max_value: # 只要指向的不是最大值，就移动指针
                    points[num] += 1
        return path
```

# 1945. 字符串转化后的各位数字之和

给你一个由小写字母组成的字符串 s ，以及一个整数 k 。

首先，用字母在字母表中的位置替换该字母，将 s 转化 为一个整数（也就是，'a' 用 1 替换，'b' 用 2 替换，... 'z' 用 26 替换）。接着，将整数 转换 为其 各位数字之和 。共重复 转换 操作 k 次 。

例如，如果 s = "zbax" 且 k = 2 ，那么执行下述步骤后得到的结果是整数 8 ：

转化："zbax" ➝ "(26)(2)(1)(24)" ➝ "262124" ➝ 262124
转换 #1：262124 ➝ 2 + 6 + 2 + 1 + 2 + 4 ➝ 17
转换 #2：17 ➝ 1 + 7 ➝ 8
返回执行上述操作后得到的结果整数。

```python
class Solution:
    def getLucky(self, s: str, k: int) -> int:
        # s为小写字母 
        temp_str = '' # 初步转换
        for i in s:
            temp_str += str((ord(i)-96))
        if k == 0:
            return int(temp_str)
        the_sum = 0
        while k > 0: # 开始正式转换
            k -= 1
            for i in temp_str:
                the_sum += int(i)
            temp_str = str(the_sum)
            the_sum = 0 # 清空
        return int(temp_str)
```

# 1941. 检查是否所有字符出现次数相同

给你一个字符串 s ，如果 s 是一个 好 字符串，请你返回 true ，否则请返回 false 。

如果 s 中出现过的 所有 字符的出现次数 相同 ，那么我们称字符串 s 是 好 字符串。

```python
class Solution:
    def areOccurrencesEqual(self, s: str) -> bool:
        # counter计数
        count_dict = collections.Counter(s)
        for i in s:
            if count_dict[i] != count_dict[s[0]]:
                return False
        return True
```

# 1952. 三除数

给你一个整数 n 。如果 n 恰好有三个正除数 ，返回 true ；否则，返回 false 。

如果存在整数 k ，满足 n = k * m ，那么整数 m 就是 n 的一个 除数 。

```python
class Solution:
    def isThree(self, n: int) -> bool:
        if n <= 3:
            return False
        t = int(sqrt(n))
        count = 0
        ans = []
        for i in range(1,t+1):
            if n % i == 0:
                if i != n//i:                 
                    ans.append(i)
                    ans.append(n//i)
                else:
                    ans.append(i)
        return len(ans) == 3
            
```

# 1953. 你可以工作的最大周数

给你 n 个项目，编号从 0 到 n - 1 。同时给你一个整数数组 milestones ，其中每个 milestones[i] 表示第 i 个项目中的阶段任务数量。

你可以按下面两个规则参与项目中的工作：

每周，你将会完成 某一个 项目中的 恰好一个 阶段任务。你每周都 必须 工作。
在 连续的 两周中，你 不能 参与并完成同一个项目中的两个阶段任务。
一旦所有项目中的全部阶段任务都完成，或者仅剩余一个阶段任务都会导致你违反上面的规则，那么你将 停止工作 。注意，由于这些条件的限制，你可能无法完成所有阶段任务。

返回在不违反上面规则的情况下你 最多 能工作多少周。

```python
class Solution:
    def numberOfWeeks(self, milestones: List[int]) -> int:
        # 数学思路
        # 找到最多的那个
        # 最多的那个如果 大于 剩下来的*2 + 1 则返回剩下来的*2 + 1
        # 否则全部可以完成
        max_value = max(milestones)
        remain = sum(milestones) - max_value
        return min(sum(milestones),remain * 2 + 1)
```

# 1957. 删除字符使字符串变好

一个字符串如果没有 三个连续 相同字符，那么它就是一个 好字符串 。

给你一个字符串 s ，请你从 s 删除 最少 的字符，使它变成一个 好字符串 。

请你返回删除后的字符串。题目数据保证答案总是 唯一的 。

```python
class Solution:
    def makeFancyString(self, s: str) -> str:
        # 使用一条队列记录 元素：频率
        queue = []
        for i in s:
            if len(queue) == 0:
                queue.append([i,1])
            elif i == queue[-1][0]:
                queue[-1][1] += 1
            elif i != queue[-1][0]:
                queue.append([i,1])
        ans = ''
        for cp in queue:
            if cp[1] >= 2:
                cp[1] = 2
            ans += cp[0] * cp[1]
        return ans
```

```python
class Solution:
    def makeFancyString(self, s: str) -> str:
        # 直接模拟
        ans = ""
        for i in s:
            if len(ans) < 2:
                ans += i
            elif len(ans) >= 2 and i == ans[-1] and i == ans[-2]: # 注意这里别写错成s了
                pass
            else:
                ans += i
        return ans
```



# 1961. 检查字符串是否为数组前缀

给你一个字符串 s 和一个字符串数组 words ，请你判断 s 是否为 words 的 前缀字符串 。

字符串 s 要成为 words 的 前缀字符串 ，需要满足：s 可以由 words 中的前 k（k 为 正数 ）个字符串按顺序相连得到，且 k 不超过 words.length 。

如果 s 是 words 的 前缀字符串 ，返回 true ；否则，返回 false 。

```python
class Solution:
    def isPrefixString(self, s: str, words: List[str]) -> bool:
        # 这一题要求按顺序相连，直接检查
        ans = ''
        for i in words:
            ans += i
            if ans == s:
                return True
        return False
```

# 5839. 移除石子使总数最小

给你一个整数数组 piles ，数组 下标从 0 开始 ，其中 piles[i] 表示第 i 堆石子中的石子数量。另给你一个整数 k ，请你执行下述操作 恰好 k 次：

选出任一石子堆 piles[i] ，并从中 移除 floor(piles[i] / 2) 颗石子。
注意：你可以对 同一堆 石子多次执行此操作。

返回执行 k 次操作后，剩下石子的 最小 总数。

floor(x) 为 小于 或 等于 x 的 最大 整数。（即，对 x 向下取整）。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/remove-stones-to-minimize-the-total
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```python
class Solution:
    def minStoneSum(self, piles: List[int], k: int) -> int:
        # api使用问题，使用内置堆，大根堆
        max_heap = [-i for i in piles]
        heapq.heapify(max_heap) # 堆化
        # 执行k次操作 
        for i in range(k):
            e = heapq.heappop(max_heap) # 弹出堆顶
            e = e // 2 # 它完成了floor
            heapq.heappush(max_heap,e)
        return -sum(max_heap) # 返回总和的负数
```

# 5832. 构造元素不等于两相邻元素平均值的数组

给你一个 下标从 0 开始 的数组 nums ，数组由若干 互不相同的 整数组成。你打算重新排列数组中的元素以满足：重排后，数组中的每个元素都 不等于 其两侧相邻元素的 平均值 。

更公式化的说法是，重新排列的数组应当满足这一属性：对于范围 1 <= i < nums.length - 1 中的每个 i ，(nums[i-1] + nums[i+1]) / 2 不等于 nums[i] 均成立 。

返回满足题意的任一重排结果。

```python
class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
    # 贪心配对
        nums.sort()
        n = len(nums)
        l1 = nums[:n//2]
        l2 = nums[n//2:]
        # l1 <= l2
        p = 0
        ans = []
        while p < len(l1):
            ans.append(l2[p])
            ans.append(l1[p])
            p += 1
        if p < len(l2):
            ans.append(l2[p])
        return ans
            
```

# 5834. 使用特殊打字机键入单词的最少时间

有一个特殊打字机，它由一个 圆盘 和一个 指针 组成， 圆盘上标有小写英文字母 'a' 到 'z'。只有 当指针指向某个字母时，它才能被键入。指针 初始时 指向字符 'a' 。


每一秒钟，你可以执行以下操作之一：

将指针 顺时针 或者 逆时针 移动一个字符。
键入指针 当前 指向的字符。
给你一个字符串 word ，请你返回键入 word 所表示单词的 最少 秒数 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/minimum-time-to-type-word-using-special-typewriter
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```python
class Solution:
    def minTimeToType(self, word: str) -> int:
        if len(word) == 1:
            return self.find_gap(word[0],"a") + 1
        else:
            p = 1
            t = self.find_gap(word[0],"a") + 1
            while p < len(word):
                t += self.find_gap(word[p-1],word[p]) + 1
                p += 1
        return t
                    
    def find_gap(self,ch1,ch2):
        ch1 = ord(ch1)-ord("a")
        ch2 = ord(ch2)-ord("a")
        if ch1 == ch2:
            return 0
        elif ch1 != ch2:
            t1 = (ch1-ch2)%26
            t2 = (ch2-ch1)%26
            return min(t1,t2)
```

# 5835. 最大方阵和

给你一个 n x n 的整数方阵 matrix 。你可以执行以下操作 任意次 ：

选择 matrix 中 相邻 两个元素，并将它们都 乘以 -1 。
如果两个元素有 公共边 ，那么它们就是 相邻 的。

你的目的是 最大化 方阵元素的和。请你在执行以上操作之后，返回方阵的 最大 和。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/maximum-matrix-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```python
class Solution:
    def maxMatrixSum(self, matrix: List[List[int]]) -> int:
    # 贪心，可以经过有限次转换尽量消除负数
        neg = 0
        neglst = [] # 记录负数
        poslst = [] # 记录正数
        n = len(matrix)
        zero = 0
        thesum = 0
        for i in range(n):
            for j in range(n):
                thesum += matrix[i][j]
                if matrix[i][j] < 0:
                    neg += 1
                    neglst.append(matrix[i][j])
                elif matrix[i][j] == 0:
                    zero += 1
                else:
                    poslst.append(matrix[i][j])
        if zero >= 1: # 多于1个0，可以把所有负数都转正
            ans = sum(neglst)*(-2) + thesum
        elif neg%2 == 0: # 偶数个负数，可以吧所有负数都转正
            ans = sum(neglst)*(-2) + thesum
        elif neg%2 == 1:
            # 找到正数，负数两者里面绝对值最小的，转成负数
            if len(poslst) == 0: # 没有正数
                pmin = None
                # 找到负数的最大值
                a = max(neglst)
                return sum(neglst)*(-1) + 2 * a
            elif len(neglst) == 0: # 没有负数
                return thesum
            else:
                a = min(abs(max(neglst)),min(poslst))
                ans = sum(neglst)*(-2) + thesum - 2*a
        return ans
            
```

# 5843. 作为子字符串出现在单词中的字符串数目

给你一个字符串数组 patterns 和一个字符串 word ，统计 patterns 中有多少个字符串是 word 的子字符串。返回字符串数目。

子字符串 是字符串中的一个连续字符序列。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/number-of-strings-that-appear-as-substrings-in-word
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```python
class Solution:
    def numOfStrings(self, patterns: List[str], word: str) -> int:
        count = 0
        for ch in patterns:
            if ch in word:
                count += 1
        return count
```

# 5844. 数组元素的最小非零乘积

给你一个正整数 p 。你有一个下标从 1 开始的数组 nums ，这个数组包含范围 [1, 2p - 1] 内所有整数的二进制形式（两端都 包含）。你可以进行以下操作 任意 次：

从 nums 中选择两个元素 x 和 y  。
选择 x 中的一位与 y 对应位置的位交换。对应位置指的是两个整数 相同位置 的二进制位。
比方说，如果 x = 1101 且 y = 0011 ，交换右边数起第 2 位后，我们得到 x = 1111 和 y = 0001 。

请你算出进行以上操作 任意次 以后，nums 能得到的 最小非零 乘积。将乘积对 109 + 7 取余 后返回。

注意：答案应为取余 之前 的最小值。

```python
class Solution:
    def minNonZeroProduct(self, p: int) -> int:
        # 贪心
        # 尽量出现1，和2**p - 2,和一个无法被处理的2**p - 1
        mod = 10 ** 9 + 7
        if p == 1:
            return 1
        temp = pow(2,p) - 1
        ans = pow(temp-1,(temp-1)//2,mod) * temp % mod
        return ans
```

# 5850. 找出数组的最大公约数

给你一个整数数组 `nums` ，返回数组中最大数和最小数的 **最大公约数** 。

两个数的 **最大公约数** 是能够被两个数整除的最大正整数。

```python
class Solution:
    def findGCD(self, nums: List[int]) -> int:
        a = max(nums)
        b = min(nums)
        the_ans = 1
        for i in range(1,b+1):
            if a%i == 0 and b % i == 0:
                the_ans = i
        return the_ans
```

# [5851. 找出不同的二进制字符串](https://leetcode-cn.com/problems/find-unique-binary-string/)

给你一个字符串数组 nums ，该数组由 n 个 互不相同 的二进制字符串组成，且每个字符串长度都是 n 。请你找出并返回一个长度为 n 且 没有出现 在 nums 中的二进制字符串。如果存在多种答案，只需返回 任意一个 即可。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/find-unique-binary-string
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```python
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        the_set = set(nums)
        n = len(nums)
        max_num = 2 ** n 
        stringList = []
        for i in range(0,max_num+1):
            temp = bin(i)[2:]
            gap = n - len(temp)
            temp = gap*"0" + temp
            stringList.append(temp)
        for i in stringList:
            if i not in the_set:
                return i
```

# [5854. 学生分数的最小差值](https://leetcode-cn.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/)

给你一个 下标从 0 开始 的整数数组 nums ，其中 nums[i] 表示第 i 名学生的分数。另给你一个整数 k 。

从数组中选出任意 k 名学生的分数，使这 k 个分数间 最高分 和 最低分 的 差值 达到 最小化 。

返回可能的 最小差值 。

```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        #  滑动窗口，固定窗口大小为k
        nums.sort() # 预先排序
        window = nums[:k]
        def calc(window):
            tempMax = max(window)
            tempMin = min(window)
            minGap = abs(tempMax-tempMin)
            return minGap
        ans = calc(window)
        for i in nums[k:]:
            window.append(i)
            window.pop(0)
            ans = min(ans,calc(window))
        return ans
```

# [5855. 找出数组中的第 K 大整数](https://leetcode-cn.com/problems/find-the-kth-largest-integer-in-the-array/)

给你一个字符串数组 nums 和一个整数 k 。nums 中的每个字符串都表示一个不含前导零的整数。

返回 nums 中表示第 k 大整数的字符串。

注意：重复的数字在统计时会视为不同元素考虑。例如，如果 nums 是 ["1","2","2"]，那么 "2" 是最大的整数，"2" 是第二大的整数，"1" 是第三大的整数。

```python
class Solution:
    def kthLargestNumber(self, nums: List[str], k: int) -> str:
        # 方法1:不讲道理法。python支持大数。。
        nums.sort(key = int,reverse = True)
        return nums[k-1]
```

```python
class Solution:
    def kthLargestNumber(self, nums: List[str], k: int) -> str:
        # 方法2，长度不同的时候，长的大
        # 长度相同的时候，比较字典序即可
        def compareTo(s1,s2): # 快排比较器
            if len(s1) == len(s2):
                if s1 < s2 :
                    return -1
                elif s1 == s2:
                    return 0
                elif s1 > s2:
                    return 1
            elif len(s1) > len(s2):
                return 1
            else:
                return -1
            
        def quickSort(lst): # 这里调整了一下，直接是从大到小的排序了
            if len(lst) <= 1:
                return 
            pivot = lst[random.randint(0,len(lst)-1)]
            L = []
            E = []
            G = []
            while len(lst) != 0:
                ele = lst.pop()
                if compareTo(pivot,ele) < 0:
                    L.append(ele)
                elif compareTo(pivot,ele) == 0:
                    E.append(ele)
                elif compareTo(pivot,ele) > 0:
                    G.append(ele)
            quickSort(L)
            quickSort(G)
            while len(L) != 0:
                lst.append(L.pop(0))
            while len(E) != 0:
                lst.append(E.pop())
            while len(G) != 0:
                lst.append(G.pop(0))
            
        quickSort(nums)        

        return nums[k-1]
    

```


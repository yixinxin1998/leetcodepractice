# <剑指Offer 2> leetcode

# 剑指 Offer 03. 数组中重复的数字

找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
      ## 哈希法
        dict1 = collections.defaultdict(int)
        for i in nums:
            if dict1[i] == 1:
                return i # 遇到重复直接返回
            else:
                dict1[i] += 1
```

# 剑指 Offer 04. 二维数组中的查找

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        # 较为高效的搜索，非双二分
        # 以右上角为根节点，以树逻辑进行查找
        if matrix == []:
            return False
        row_start,row_end = -1,len(matrix) # 行边界
        col_start,col_end = -1,len(matrix[0])   #列边界
        i = 0
        j = len(matrix[0])-1
        while row_start<i<row_end and col_start<j<col_end:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
        return False
```

# 剑指 Offer 05. 替换空格

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        if len(s) == 0:
            return s
        s = list(s) # python中利用列表处理
        p = 0 
        while p < len(s):
            if s[p] == ' ':
                s[p] = '%20'
            p += 1
        ans = ''.join(s) # 重新组合成字符串
        return ans
```

# 剑指 Offer 06. 从尾到头打印链表

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        ans = []
        if head == None:
            return ans
        cur = head
        while cur != None:
            temp = [cur.val] 
            ans = temp + ans # 这里直接加在了头部，借助python列表容器的特性
            cur = cur.next
        return ans
```

# 剑指 Offer 07. 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。

假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder) == 0:
            return None
        else:
            gap = inorder.index(preorder[0]) # 它的效果是二分查找到分隔左右子树的点
            root = TreeNode(preorder[0])
            # 递归赋值
            root.left = self.buildTree(preorder[1:gap+1],inorder[:gap])
            root.right = self.buildTree(preorder[gap+1:],inorder[gap+1:])
            return root
```

# 剑指 Offer 09. 用两个栈实现队列

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

```python
class CQueue:

    def __init__(self):
        self.s1 = [] # 主栈
        self.s2 = [] # 辅助栈

    def appendTail(self, value: int) -> None:
        self.s1.append(value)

    def deleteHead(self) -> int:
        if len(self.s1) == 0:
            return -1
        while self.s1 != []:
            self.s2.append(self.s1.pop())
        e = self.s2.pop()
        while self.s2 != []:
            self.s1.append(self.s2.pop())
        return e
```



# 剑指 Offer 10- I. 斐波那契数列

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

```python
class Solution:
    def fib(self, n: int) -> int:
        return Solution.cacl(n)%1000000007
    def cacl(n,a=0,b=1): # 递归，做好初始值，n决定递归次数
        if n == 0 :
            return a
        else:
            return Solution.cacl(n-1,b,a+b)
```

# 剑指 Offer 10- II. 青蛙跳台阶问题

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

```python
class Solution:
    def numWays(self, n: int) -> int:
        def fib(n,a=1,b=1):
            if n == 0:
                return a
            else:
                return fib(n-1,b,a+b)
        ans = fib(n)
        return ans % 1000000007
```

# 剑指 Offer 11. 旋转数组的最小数字

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。

```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        # 先考虑未发生旋转的情况
        # 此时最左边小于最右边【且只有在初始状态满足这个条件的时候它是未旋转，直接返回首项】
        # 相同的情况下见while循环中的else
        if len(numbers) == 0: # 排除长度为空的
            return []
        nums = numbers
        left = 0
        right = len(nums)-1
        if nums[left] < nums[right]:
            return nums[left]
        # 画图辅助，分为左排序数组和右排序数组，最小值是右排序的第一个值
        # 开始二分查找，闭区间查找
        # situation1 :中值要么比有效范围内的最左边小，中值处于右排序，收缩右排序
        # situation2 :要么比有效范围内的最左边大，中值处于左排序，收缩左排序
        # situation3 :要么无法判断它处于哪一个区间，只能普通的非二分收缩,left，mid，right指的数相等的时候，只能开始普通线性查询
        while left  < right : # left最后退出循环时和right相等
            mid = (left+right)//2
            if nums[mid] == nums[left] == nums[right]: # ss3 
                return min(nums[left:right+1])
            # print('mid_index',mid,'left_index',left,'right_index',right)
            # 循环中，left会指向最小值
            if nums[mid] < nums[left]: # s1
                right = mid
            elif nums[mid] > nums[left]: # s2
                left = mid
            else: # 
                left += 1
        return nums[left]
```

# 剑指 Offer 12. 矩阵中的路径

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

 

例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。

```python
# 方法1：超时未优化版本
# 问题出在backtracking的逻辑中，在它的for循环里，即便找到了符合条件的。也不会立马返回

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # 一个逻辑矩阵，一个方向数组
        # 回溯法
        m = len(board)
        n = len(board[0])
        booleanMat = [[False for j in range(n)] for i in range(m)] # 标记是否访问
        direc = [(-1,0),(+1,0),(0,-1),(0,+1)]
        ansList = []

        def backtracking(board,i,j,index): # index是word的index
            if index == len(word)-1 and board[i][j] == word[index]:
                ansList.append(True)
                return 
            elif board[i][j] == word[index]:
                booleanMat[i][j] = True # 标记为访问
                for di in direc:
                    next_i = i + di[0]
                    next_j = j + di[1]
                    if self.judgeValid(next_i,next_j,m,n) and not booleanMat[next_i][next_j]:
                        backtracking(board,next_i,next_j,index+1)
                booleanMat[i][j] = False # 取消标记
            
        for i in range(m): # 对每个格子都搜
            for j in range(n):     
                backtracking(board,i,j,0)
                if len(ansList) != 0 and ansList[-1] == True: # 搜到一个符合的就返回
                    return True
        return False

                                  
    def judgeValid(self,i,j,limitM,limitN): # 判断是否越界
        if 0 <= i < limitM and 0 <= j < limitN:
            return True
        else:
            return False


```

```python
# 完美剪枝版本，可以通过
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # 一个逻辑矩阵，一个方向数组
        # 回溯法
        m = len(board)
        n = len(board[0])
        booleanMat = [[False for j in range(n)] for i in range(m)] # 标记是否访问
        direc = [(-1,0),(+1,0),(0,-1),(0,+1)]

        def dfs(i,j,index):  #index是word的index
        # 为了防止超时，必须剪枝。所以用的是直接return的版本而不是收集全部结果逐一判断
            if index == len(word) - 1 and word[index] == board[i][j]:
                return True
            if word[index] != board[i][j]:
                return False
            booleanMat[i][j] = True
            result = False # 默认为False,
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if self.judgeValid(new_i,new_j,m,n) and not booleanMat[new_i][new_j]: # 判断是否合法
                    if dfs(new_i,new_j,index+1):
                        result = True 
                        break # 注意这个break
            booleanMat[i][j] = False
            return result
                        
        for i in range(m): # 对每个格子都搜
            for j in range(n):     
                if dfs(i,j,0):
                    return True
                
        return False

                                  
    def judgeValid(self,i,j,limitM,limitN): # 判断是否越界
        if 0 <= i < limitM and 0 <= j < limitN:
            return True
        else:
            return False

```

# 剑指 Offer 13. 机器人的运动范围
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        # 创建一个逻辑数组，判断是否可达
        booleanVisited = [[False for j in range(n)] for i in range(m)]
        direc = [(+1,0),(-1,0),(0,+1),(0,-1)] # 方向数组

        def limited(a,b,c): # i,j为坐标，k为限制
            if not (0 <= a < m and 0 <= b < n):
                return False
            tempSum = 0
            for num in str(a):
                tempSum += int(num)
            for num in str(b):
                tempSum += int(num)
            return tempSum <= k 
        
        def dfs(i,j):
            if limited(i,j,k) and not booleanVisited[i][j]: # 如果没有被限制
                booleanVisited[i][j] = True # 变更为访问过,防止重复走
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    dfs(new_i,new_j)
            else:
                return 

        count = 0 # 记录合法个数
        dfs(0,0)
        for i in range(m):
            for j in range(n):
                if booleanVisited[i][j] == True:
                    count += 1
                    
        return count
```

# 剑指 Offer 14- I. 剪绳子

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        # dp，这一题的递归有陷阱
        # dp[i] = dp[i-j]*j # 1<=j<i,如果分隔的前i-j的长度的需要裁剪
        # 但是可能不需要裁剪，因为dp[i-j]有可能还不如i-j大
        # 所以正确的状态转移为 dp[i] = max(group),group == (max(i-j,dp[i-j])*j)
        # 先申请dp长度为n+1
        dp = [1 for i in range(n+1)]
        # 显然dp[1]的边界条件赋值为1比较合适
        # dp[2] = 1
        for i in range(2,n+1):
            group = []
            for j in range(1,i//2+1): # 这一步其实需要证明可以收缩，其标准条件为range(1,i)
                max_temp = max(i-j,dp[i-j])
                group.append(max_temp*j)
            if group != []:
                dp[i] = max(group) # 取其中的最大值
        return dp[-1] # 此时

```

# 剑指 Offer 14- II. 剪绳子 II

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        # 数学思路，当n>=5的时候，尽量以3分隔
        # 3*(n-3) >= 2 * (n-2)
        # 注意n=2,3的时候的处理
        if n == 2:
            return 1
        if n == 3:
            return 2 # 因为必须要切
        MOD = (10**9 + 7)
        # 1. n % 3 == 0 直接返回 3 ** times
        # 2. n % 3 == 1 回退一个3，那一部分拆成2*2,总体为 3 ** (times-1) * 4
        # 3. n % 3 == 2 返回 3 ** times * 2
        # 结果注意取模
        times = n // 3
        if n % 3 == 0: return (3 ** times) % MOD
        if n % 3 == 1: return (3 ** (times-1) * 4) % MOD
        if n % 3 == 2: return (3 ** times * 2) % MOD

```

# 剑指 Offer 15. 二进制中1的个数

请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        # python的位运算的语法熟悉
        count = 0
        for i in range(32): # 一共32位
            count += (n>>i)&1
        return count
```

# 剑指 Offer 16. 数值的整数次方

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        # 快速幂，不需要考虑大数问题
        # 递归解决
        # 注意处理负数问题和超小数的问题,0.5^(-123456)次方这一类
        # 且这里认为0的0次方为1
        def submethod(n): # 传入n即可
            if n == 0:
                return 1
            y = submethod(n//2)
            if abs(y) <= 2**-64:
                return 0
            if n%2 == 1:
                return x * y * y
            elif n%2 == 0:
                return y*y
        if n >= 0: return submethod(n)
        elif n < 0: 
            if submethod(-n) != 0:
                return 1/submethod(-n)
            else:
                return inf
```

# 剑指 Offer 17. 打印从1到最大的n位数

输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        # 对于python而言无需考虑数的越界等等一系列问题
        # 对于java而言不能使用Bigdecimal的话还需要考虑打印不出错和不越界的问题
        # 对于非python，需要使用字符串来代替结果
        ans = []
        up = 10**n
        for i in range(1,up):
            ans.append(i)
        return ans

```

# 剑指 Offer 18. 删除链表的节点

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

**注意：**此题对比原题有改动

```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        # 设置哑节点，统一删除语法
        # 需要处理空链表
        # 需要处理值不在其中的链表
        if head == None:
            return 
        dummy = ListNode(-1)
        dummy.next = head
        cur1 = dummy
        cur2 = dummy.next
        while cur2 != None and cur2.val != val:  # 如果链表中存在节点值
            cur1 = cur1.next
            cur2 = cur2.next
        # 循环完之后，cur2指向要删除的节点，直接cur1.next指向cur2.next
        if cur2 != None: # 如果链表中存在节点值
            cur1.next = cur2.next
        if cur2 == None: # 如果链表不存在节点值，啥也不干
            pass
        return dummy.next
```

# 剑指 Offer 19. 正则表达式匹配

请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # s纵，p横
        # 二维dp,外加一圈边界
        m = len(s)
        n = len(p)
        dp = [[False for j in range(n+1)] for i in range(m+1)]
        # dp[i][j]的含义是，到s的第i个字母与到p的第j个字母，是否匹配，
        # base dp[0][0] = True 
        # base dp[0][j]: dp[0][j] = dp[0][j - 2] 且 p[j - 1] = '*'
        #  首行 s 为空字符串，因此当 p 的偶数位为 * 时才能够匹配（即让 p 的奇数位出现 0 次，保持 p 是空字符串）。因此，循环遍历字符串 p ，步长为 2（即只看偶数位）。
        dp[0][0] = True 
        for j in range(2,n+1,1): # 跨度是偶数可以使用break，跨度是奇数不可以
            dp[0][j] = dp[0][j-2] and (p[j-1] == '*') # 且当前字符得是*，只要一个断层其实后面的就不需要了
            # if p[j-1] != "*":
            #     break
        # print(dp)
        # 状态转移方程。
        # s新加入一个，p新加入一个。分情况讨论。
        # 如果加入的不是*
        # 1.dp[i - 1][j - 1] 且 s[i - 1] = p[j - 1]
        # 2.dp[i - 1][j - 1] 且 p[j - 1] = '.'

        #
        # 如果加入的是*
        # 1.*0次 dp[i][j - 2]： 即将字符组合  a*【丢掉两位】 看作出现 0 次时，能否匹配；
        # 2.dp[i - 1][j] 且 s[i - 1] = p[j - 2]: 即让字符 p[j - 2] 多出现 1 次时，能否匹配；
        # 3.dp[i - 1][j] 且 p[j - 2] = '.': 即让字符 '.' 多出现 1 次时，能否匹配
        # 解释一下2，3：
        # dp[i][j] = dp[i - 1][j - 2] 且 (s[i - 1] == p[j - 2])
        #而 dp[i - 1][j - 2] = dp[i - 1][j]，相对于 dp[i - 1][j - 2] 来说，这里的 *(也就是 p[j - 1]) 匹配了 0 次
        #两者合并即为：dp[i][j] = (dp[i - 1][j]) && (s[i - 1] == p[j - 2])

        for i in range(1,m+1):
            for j in range(1,n+1):
                if p[j-1] != "*":
                    if s[i-1] == p[j-1] and dp[i-1][j-1]: dp[i][j] = True 
                    if p[j-1] == "." and dp[i-1][j-1] : dp[i][j] = True 
                elif p[j-1] == "*":
                    if dp[i][j-2]: dp[i][j] = True
                    if dp[i-1][j] and s[i-1] == p[j-2]: dp[i][j] = True
                    if dp[i-1][j] and p[j-2] == ".": dp[i][j] = True 

        return dp[-1][-1]
```

```go
func isMatch(s string, p string) bool {
    m,n := len(s),len(p)
    dp := make([][]bool,m+1)
    for i := 0; i < m+1; i++ {
        dp[i] = make([]bool,n+1)
    }
    // 初始化
    dp[0][0] = true
    for j := 2;j < n+1; j++ {
        dp[0][j] = (dp[0][j-2] && p[j-1] == '*')
    }

    for i := 1; i < m + 1; i ++ {
        for j := 1; j < n + 1; j ++ {
            if p[j-1] != '*'{
                state1 := dp[i-1][j-1] && (s[i-1] == p[j-1])
                state2 := dp[i-1][j-1] && (p[j-1] == '.')
                dp[i][j] = state1 || state2
            } else if p[j-1] == '*' {
                state1 := dp[i][j-2]
                state2 := dp[i-1][j] && (s[i-1] == p[j-2])
                state3 := dp[i-1][j] && (p[j-2] == '.')
                dp[i][j] = state1 || state2 || state3
            }
        }
    }
    return dp[m][n]
}
```

```python
# 简略版
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m = len(s)
        n = len(p)
        dp = [[False for j in range(n+1)] for i in range(m+1)]
        dp[0][0] = True 
        for j in range(2,n+1,2):
            dp[0][j] = dp[0][j-2] and (p[j-1] == "*")
        
        for i in range(1,m+1):
            for j in range(1,n+1):
                if p[j-1] != "*":
                    state1 = (dp[i-1][j-1] and s[i-1] == p[j-1])
                    state2 = (dp[i-1][j-1] and p[j-1] == ".")
                    dp[i][j] = state1 or state2
                elif p[j-1] == "*":
                    state1 = dp[i][j-2]
                    state2 = (dp[i-1][j] and s[i-1] == p[j-2])
                    state3 = (dp[i-1][j] and p[j-2] == ".")
                    dp[i][j] = state1 or state2 or state3 
        
        return dp[-1][-1]
```



# 剑指 Offer 20. 表示数值的字符串

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。

数值（按顺序）可以分成以下几个部分：

若干空格
一个 小数 或者 整数
（可选）一个 'e' 或 'E' ，后面跟着一个 整数
若干空格
小数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
下述格式之一：
至少一位数字，后面跟着一个点 '.'
至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
一个点 '.' ，后面跟着至少一位数字
整数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
至少一位数字
部分数值列举如下：

["+100", "5e2", "-123", "3.1416", "-1E-16", "0123"]
部分非数值列举如下：

["12e", "1a3.14", "1.2.3", "+-5", "12e+5.4"]

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
    
    def isInt_byNoStrip_NoSymbol(self,t): # 给.判断用的，不能带符号
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

# 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        # 双指针，扫描指针 + 填充指针
        fast = 0 # 扫描指针
        slow = 0 # 填充指针奇数
        # 是奇数往前丢，是偶数不动
        while fast < len(nums):
            if nums[fast]%2 == 1:
                nums[fast],nums[slow] = nums[slow],nums[fast]
                fast += 1
                slow += 1
            elif nums[fast]%2 == 0:
                fast += 1
        return nums
```

# 剑指 Offer 22. 链表中倒数第k个节点

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
      # 快慢指针法，先让快的走k步，之后同时走，那么快的到达终点之后慢的距离终点自然为k
        fast = head
        count = 1
        while count != k+1:
            fast = fast.next
            count += 1
        cur = head
        while fast != None:
            fast = fast.next
            cur = cur.next
        return cur
```

# 剑指 Offer 24. 反转链表

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # 不借助其他辅助结构，只调整链表指针解决
        # 双指针的过程画图比较清晰
        # 注意cur1的变更逻辑，实际上cur1一直指着第一个节点
        # 从逻辑距离上来说，cur1和cur2是越来越远，cur1的逻辑距离是没有变动的
        cur1 = None # ListNode(-1)不可以
        cur2 = head
        while cur2 != None:
            temp = cur2.next # 存下一个要指的节点
            cur2.next = cur1
            cur1 = cur2
            cur2 = temp
        return cur1
```

# 剑指 Offer 25. 合并两个排序的链表

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 搞个哑节点，然后迭代
        dummy = ListNode()
        cur = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = l1
                l1 = l1.next
                cur = cur.next
            else:
                cur.next = l2
                l2 = l2.next
                cur = cur.next
        cur.next = l1 if l1 else l2
        return dummy.next
```

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 递归不用哑节点
        #特判：如果有一个链表为空，返回另一个链表
        #如果l1节点值比l2小，下一个节点应该是l1，应该return l1，在return之前，指定l1的下一个节点应该是l1.next和l2俩链表的合并后的头结点
        #如果l1节点值比l2大，下一个节点应该是l2，应该return l2，在return之前，指定l2的下一个节点应该是l1和l2.next俩链表的合并后的头结点

        if l1 == None or l2 == None:
            return l1 if l1 else l2
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        elif l2.val < l1.val:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2             
```

# 剑指 Offer 26. 树的子结构

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

​	 3
​	/ \

   4   5
  / \
 1   2
给定的树 B：

   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        # 为了防止超时，需要及时终止
        if A == None or B == None: # 两者其中之一为空
            return False
        if A.val == B.val: # 两者值相等，进入子方法判断
            judge = self.isSimilar(A,B)
            if judge == True:
                return self.isSimilar(A,B)
            else: # 两者比对失败，继续走
                leftPart = self.isSubStructure(A.left,B)
                rightPart = self.isSubStructure(A.right,B)
                return leftPart or rightPart # 左右有一个为True即可
        else: # 值不相等，判断A的左右和B去比较
            leftPart = self.isSubStructure(A.left,B)
            rightPart = self.isSubStructure(A.right,B)
            return leftPart or rightPart # 左右有一个为True即可
    
    def isSimilar(self,rootX,rootY): # 判断结构是否相似
        if rootX == None and rootY == None: # 两者都为空，是True
            return True
        elif rootX != None and rootY == None: # Y已经被比对完了
            return True
        elif rootX == None and rootY != None: # X被比对完了，Y还没有
            return False

        if rootX.val == rootY.val:
            leftpart = self.isSimilar(rootX.left,rootY.left)  
            rightpart = self.isSimilar(rootX.right,rootY.right)
            return leftpart and rightpart # 这个要是and
        elif rootX.val != rootY.val:
            return False 

```



# 剑指 Offer 27. 二叉树的镜像

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

​	 4

   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

​	 4

   /   \
  7     2
 / \   / \
9   6 3   1



```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        # 递归法，很朴素的对称递归
        if root == None:
            return 
        root.left,root.right = root.right,root.left
        self.mirrorTree(root.left)
        self.mirrorTree(root.right)
        return root 
```



# 剑指 Offer 28. 对称的二叉树

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

​	1

   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

​	1

   / \
  2   2
   \   \
   3    3

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        #两个子树互为镜像当且仅当：
        #1.两个子树的根节点值相等；
        #2.第一棵子树的左子树和第二棵子树的右子树互为镜像，且第一棵子树的右子树和第二棵子树的左子树互为镜像；
        def submethod(L,R): # 子方法比较左右子节点是否相等
            if L == None and R == None:
                return True
            elif L == None and R != None:
                return False
            elif L != None and R == None:
                return False
            elif L.val != R.val: ## 这一行别忘了
                return False
            elif L.val == R.val:
                return submethod(L.left,R.right) and submethod(L.right,R.left)
        
        if root == None:
            return True
        return submethod(root.left,root.right)
```

# 剑指 Offer 29. 顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # 考虑好边界条件问题
        if matrix == []:
            return []
        row_start = 0 # 限制区
        col_start = -1 # 限制区
        row_end = len(matrix) # 限制区
        col_end = len(matrix[0]) # 限制
        size = len(matrix)*len(matrix[0])
        ans = [] # 收集结果
        # 起点
        i = 0
        j = -1 # 注意这个起点赋值是为了统一语法
        while len(ans) < size:
            ########################################
            j += 1
            while j < col_end:
                ans.append(matrix[i][j])
                j += 1
            # 回退到没超过的边界，并且记录下一次边界
            j -= 1 
            col_end -= 1
            ########################################
            i += 1
            while i < row_end:
                ans.append(matrix[i][j])
                i += 1
            # 回退到没超过的边界，并且记录下一次边界
            i -= 1
            row_end -= 1
            ########################################
            j -= 1
            while j > col_start:
                ans.append(matrix[i][j])
                j -= 1
            # 回退到没超过的边界，并且记录下一次边界
            j += 1
            col_start += 1
            ########################################
            i -= 1
            while i > row_start:
                ans.append(matrix[i][j])
                i -= 1
            i += 1 
            row_start += 1
            # 检查用 print(ans)
        return ans[:size] # 防止之后的扫描中加入了过多的元素，所以切片截断
```

# 剑指 Offer 30. 包含min函数的栈

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

```
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        # 双栈法
        # 一个栈是正常栈
        # 另一个栈是存到目前为止的最小值
        self.stack1 = []
        self.stack2 = []
        self.min_num = 0xffffffff # 初始化int类型最大值


    def push(self, x: int) -> None:
        self.stack1.append(x)
        if x < self.min_num:
            self.min_num = x
        self.stack2.append(self.min_num)

    def pop(self) -> None:
        if len(self.stack1) >= 0:
            self.stack1.pop()
            self.stack2.pop()
        if len(self.stack1) == 0:
            self.min_num = 0xffffffff # 当栈排空时，要重新把值初始化
        else:
            self.min_num = self.stack2[-1] # 栈弹出时，要更新min指标


    def top(self) -> int:
        if len(self.stack1)  >= 0:
            e = self.stack1[-1]
            return e


    def min(self) -> int:
        if len(self.stack2) >= 0:
            e = self.stack2[-1]
            return e

```

# 剑指 Offer 31. 栈的压入、弹出序列

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

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

# 剑指 Offer 32 - I. 从上到下打印二叉树

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        ans = [] # 收集结果
        # BFS
        if root == None:
            return []
        queue = [root]
        while len(queue) != 0:
            new_queue = []
            for i in queue:
                if i != None:
                    ans.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            queue = new_queue
        return ans
```

# 剑指 Offer 32 - II. 从上到下打印二叉树

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
      	# BFS,队列辅助管理
        ans = [] # 存储结果
        if root == None:
            return ans
        queue = [root]
        
        while len(queue) != 0:
            level = [] # 每层存入的节点
            new_queue = [] #扫完本次之后下一次的预备扫描队列
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            ans.append(level) # 收集本层
            queue = new_queue # 开启下一层

        return ans
```

# 剑指 Offer 32 - III. 从上到下打印二叉树

请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        sequence = -1 # 作为标志辅助正反添加
        level_num = 1 # 作为标志辅助正反添加
        # BFS
        ans = [] # 收集结果
        if root == None:
            return []
        queue = [root]
        while len(queue) != 0:
            level = []
            new_queue = []
            for i in queue:
                if i != None:
                    level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            level_num += 1
            ans.append(level[::(sequence**level_num)])
            queue = new_queue
        return ans

```

# 剑指 Offer 33. 二叉搜索树的后序遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 `true`，否则返回 `false`。假设输入的数组的任意两个数字都互不相同。

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        # 其实就是判断序列化的列表是不是二叉搜索树，
        # 按照左右根的思路来判断，是否能递归分成[左【均小于】｜｜右【均大于】｜｜根]的情况
        # 返回值为索引
        if len(postorder) <= 1 :
            return True
        mark = len(postorder)-1 # 初始化为最后一个数的索引
        for i in range(len(postorder)-1):
            if postorder[i] < postorder[-1]:
                pass
            else:
                mark = i  # 标记的是第一个大于最后一个数的索引
                break #这个很重要，如果有多个，防止覆盖
        for i in range(mark,len(postorder)): # 检查后面的数是不是也均大于
            if postorder[i] < postorder[-1]: # 不能再有小于的
                return False 
        # mark # 禁受住了两轮筛选             
        return self.verifyPostorder(postorder[:mark]) and self.verifyPostorder(postorder[mark:-1])
        
```

# 剑指 Offer 34. 二叉树中和为某一值的路径

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        ans = [] # 收集结果
        path = [] # 路径
        def dfs(root,target,path): # 没有说值都是正数，不能减枝
            if root == None:
                return 
            path.append(root.val) # 做选择
            if root.val == target and root.left == None and root.right == None:
                ans.append(path[:])
            dfs(root.left,target-root.val,path)
            dfs(root.right,target-root.val,path)
            path.pop() # 取消选择
        dfs(root,target,path) # 开始深度搜索
        return ans
```

# 剑指 Offer 35. 复杂链表的复制

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        # 方法1:用哈希法
        # 由于random存的是指针而不是值
        # 第一轮先收集所有节点
        cur = head
        index = 1
        hash_map = dict()
        hash_map[None] = None # 由于会有random的空指针
        while cur != None:
            hash_map[cur] = Node(cur.val)
            cur = cur.next
        for i in hash_map:
            if hash_map[i] != None:
                hash_map[i].next = hash_map[i.next]
                hash_map[i].random = hash_map[i.random]
        return hash_map[head]


```

```python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        # 采用倍增法完成
        # 使用常量级别的空间
        # 处理空链表
        if head == None:
            return head
        # 第一步，把每个链表的节点复制一遍，且用next作为逻辑链接，不能破坏random
        cur = head
        while cur != None:
            temp = cur.next # 存下来下一个节点
            copy_node = Node(cur.val) # 新建复制一个节点
            cur.next = copy_node
            copy_node.next = temp
            cur = cur.next.next
        # 第二步，用奇偶标记，所有奇标记的random给偶数标记
        odd = head
        even = head.next
        while odd.next != None and even.next != None : # 处理到剩最后一组
            # 不能是，因为不能引用原节点：even.random = odd.random
            if odd.random != None: # 这一行很关键防止出错
                even.random = odd.random.next
            odd = odd.next.next
            even = even.next.next 
        if odd.random != None: # 最后一组特殊判断
            even.random = odd.random.next
        # 第三步，奇偶拆链
        even = head.next
        odd = head
        new_head = even
        while odd.next.next != None: # 处理到剩下最后一组
            odd.next = odd.next.next
            even.next = even.next.next
            odd = odd.next
            even = even.next
        odd.next = None # 最后一组特殊判断
        even.next = None
        return new_head

```

# 剑指 Offer 36. 二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        # 中序遍历之后，根据中序遍历列表进行left,right指针调整
        lst = []
        def inOrder(node):
            if node != None:
                inOrder(node.left)
                lst.append(node)
                inOrder(node.right)
        inOrder(root)
        if len(lst) == 0: # 如果树为空，直接返回
            return 
        head = lst[0]
        p = 0
        while p < len(lst): # 取模运算很巧妙,方便统一语法
            lst[p].right = lst[(p+1)%len(lst)]
            lst[p].left = lst[(p-1)%len(lst)]
            p += 1
        return head

```

# 剑指 Offer 37. 序列化二叉树

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 1.先序遍历法
class Codec:
    def __init__(self):
        self.serials = ""

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        # 先序遍历序列化方法
        def dfs(root):
            if root == None:
                self.serials += "#,"
                return 
            self.serials += str(root.val)
            self.serials += ","
            dfs(root.left)
            dfs(root.right)
        
        dfs(root) # 然后需要去掉最后面的","
        self.serials = self.serials[:-1]
        return (self.serials)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        data = data.split(",") # data接受的是上一个函数的返回值
        def decodes(nodes): # 注意传的时候是传的引用
            if len(nodes) == 0:
                return

            e = nodes.pop(0)
            if e != "#":
                root = TreeNode(e)
                root.left = decodes(nodes)
                root.right = decodes(nodes)
                return root
            else:
                return None

        root = decodes(data)
        return root
        
```

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        # 后序遍历法
        ans = ""
        def dfs(root):
            nonlocal ans
            if root == None:
                ans += "#,"
                return 
            dfs(root.left)
            dfs(root.right)

            val = root.val
            ans += str(val)
            ans += ","
        dfs(root)
        # 也要去掉最后一个","
        return ans[:-1]

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        # 传入参数为上一个函数的返回值
        data = data.split(",")
        def decode(nodes): # 传入参数为列表，需要构建根
            if len(nodes) == 0:
                return 
            e = nodes.pop()
            if e != "#":
                root = TreeNode(e)
                root.right = decode(nodes) # 注意这个顺序要反着来，因为取nodes数据是从右到左
                root.left = decode(nodes)
                return root 
            else:
                return None
        
        root = decode(data)
        return root 
```

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
# 层序遍历法
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        ans = ""
        queue = [root]
        while len(queue) != 0:
            new_queue = []
            for node in queue:
                if node == None:
                    ans += "#,"
                else:
                    e = str(node.val)
                    ans += e
                    ans += ","
                    if node.left != None:
                        new_queue.append(node.left)
                    else:
                        new_queue.append(None)
                    if node.right != None:
                        new_queue.append(node.right)
                    else:
                        new_queue.append(None)
            queue = new_queue
        return ans[:-1]
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        # 传入数据为上一函数返回值
        data = data.split(",")
        if len(data) == 1: # 树空,只有一个#
            return None
        # 否则第一个元素就是根节点
        root = TreeNode(data.pop(0)) # 注意这里是pop左边
        queue = [root] # 队列中存的都是父节点
        while len(data) != 0:
            e = queue.pop(0) # 注意这里是pop左边，弹出的是当前处理的节点，接下来解决它的左右孩子
            value = data.pop(0) # 注意这里是pop左边
            if value != "#":
                leftNode = TreeNode(value)
                e.left = leftNode
                queue.append(leftNode)
            else:
                e.left = None
            value = data.pop(0) #注意这里是pop左边
            if value != "#":
                rightNode = TreeNode(value)
                e.right = rightNode
                queue.append(rightNode)
            else:
                e.right = None 
        return root 

```

# 剑指 Offer 38. 字符串的排列

输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        s = list(s) # 转换成列表方便处理
        # 回溯法
        ans = [] # 存储结果
        stack = [] # 路径
        n = len(s)
        def backtracking(s,stack): # 选择列表，选择路径
            if len(stack) == n: # 收集结果
                change = ''.join(stack) # 将路径转化成字符串作为合法结果
                ans.append(change)
            # 做选择
            p = 0
            while p < len(s):
                temp_lst = s.copy() # 需要一个副本，因为不能对原列表进行更改
                element = temp_lst.pop(p)
                stack.append(element) # 做选择
                backtracking(temp_lst,stack)
                stack.pop() # 取消选择
                p += 1
        backtracking(s,stack)
        ans = set(ans) # set去重复
        result = []
        for i in ans:
            result.append(i)
        return result
```

# 剑指 Offer 39. 数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]
```

# 剑指 Offer 40. 最小的k个数

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        # 借助python的内置的堆化
        # 思路1:找出最小的k个数，内置堆为小根堆，那么每次弹出堆顶即可，弹出k次
        heapq.heapify(arr) # 注意，arr已经被堆化，它没有返回值，是直接改变了arr   
        ans = [] # 存储结果
        if k > len(arr):
            return []
        while k > 0:
            ans.append(heapq.heappop(arr)) # 必须在堆化后的数组上使用heapq.heappop()
            k -= 1
        return ans
```

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        # 借助python的内置的堆化
        # 思路2: 找到最小的k个数，用大根堆的方法有，先把前k个堆化
        # 然后之后对每一个，如果比堆顶大或者相等，则不管，如果比堆顶小，大顶弹出，该数加入
        # 由于python内置的是小根堆，那么实现的时候先全部乘以-1
        # 由于使用了负数逻辑，那么判别的时候是，比堆顶小或者相等，则不管，如果比堆顶大，则弹出顶，加入数
        # 特殊处理
        if len(arr) < k:
            return []
        if k == 0:
            return []
        temp_arr = [-1*i for i in arr]
        max_heap = temp_arr[:k]
        heapq.heapify(max_heap)
        for i in temp_arr[k:]:
            if i > max_heap[0]:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap,i)
        # 此时还都是负数，要复原
        ans = [-1*i for i in max_heap]
        return ans

```

# 剑指 Offer 41. 数据流中的中位数

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。

```
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
        # 先去小顶堆逛一圈,逛完小根堆的那个数弹出来加入大根堆
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



# 剑指 Offer 42. 连续子数组的最大和

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 贪心，遇到和小于0就丢弃，从下一个开始重新构建子数组
        max_sum = -0xffffffff # 初始化一个极小值
        temp_sum = 0
        p = 0
        while p < len(nums):
            temp_sum += nums[p]
            if temp_sum < 0:
                temp_sum = 0
            else:
                max_sum = max(max_sum,temp_sum) # 更新最大值
            p += 1
        # 如果始终没有更新值，即数组中每一个数都小于0，则返回最大值
        if max_sum == -0xffffffff:
            return max(nums)
        else:
            return max_sum
```

# 剑指 Offer 43. 1～n 整数中 1 出现的次数

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

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
            if the_bit > 1: 
                count += the_bit * kp[temp_length] + 10 ** temp_length
            elif the_bit == 1: # 注意这一行的处理。(n - 10 ** temp_length + 1)
                count += kp[temp_length] + (n - 10 ** temp_length + 1)
            elif the_bit == 0:
                pass
            n = n % 10 ** temp_length  
        return count
```



# 剑指 Offer 44. 数字序列中某一位的数字

数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。

请写一个函数，求任意第n位对应的数字。【n是int范围内】

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        # 数学思想
        # 先确定是几位数，然后贴出字符串找第几位
        # 1位数的有10个
        # 2位数的有90个 * 2个字符
        # 3位数的有900个 * 3个字符
        # 4位数的有9000个 * 4个字符
        # k位数的有 9* 10**(k-1) * k个字符 【k不等于1】
        # 根据n确定是几位数
        # 如果 n 小于 10 直接返回：
        if n < 10:
            return n
        # 否则进一步处理
        k = 1 # 代表位数
        store = n - 10 # 存下这个数，利用store判断位数
        while store >= 0:
            k += 1
            store -= 9* 10**(k-1) * k
        # 此时k已经指明是几位数。store表明这个数和100………………0的距离
        # store还原成前一步的正数
        print(store)
        store += 9* 10**(k-1) * k
        print(store)
        # 这个数字除以k，商为从100………………0的第几位数，余数为是这个数字的第几位
        a = store // k # 商
        b = store % k # 余数
        print(a,b)
        # start 用来计算是哪一个具体数字
        start = 10**(k-1)+(a)
        print(start)
        return int(str(start)[b])
```

# 剑指 Offer 45. 把数组排成最小的数

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        # nums[i]属于int类，那么在其他语言中拼接的时候注意使用long进行限制
        # 排序时用到了一个这样的思想：
        # [x,y] >= [y,x] ，则选取 xy的顺序，否则选择yx的顺序
        # 即选取俩数的时候，一定要有拼接完之后的数值更大
        # 手写一个快排逻辑
        # 特殊情况处理
        if len(nums) == 0:
            return ''
        if len(nums) == 1:
            return str(nums[0])
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
            # 注意这个必须pop(0),因为less相对有序了，反过来pop()是有问题的
            while less:
                lst.append(less.pop(0))
            while equal:
                lst.append(equal.pop(0))
            while Greater:
                lst.append(Greater.pop(0))
        quick_sort(nums)
        return ''.join(nums[::-1])

```

# 剑指 Offer 46. 把数字翻译成字符串

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

```python
class Solution:
    def translateNum(self, num: int) -> int:
        # 字符串化处理
        # dp[i] 为到这一位时候翻译的可能数目
        # 状态转移为dp[i] = dp[i-1]  + 如果i-1位为1或2且这一位为0，1，2，3，4,5【dp[i-2]】
        # i用索引表示
        num = str(num)
        n = len(num)
        if n <= 1: # 一位的只能是1
            return 1
        dp = [0 for i in range(len(num))]
        dp[0] = 1 # 初始化为1
        dp[1] = 1 + (1 if 10 <= int(num[:2]) <= 25 else 0) # 看情况初始化第二位
        for i in range(2,n): # 填充dp
            if (num[i-1] == '2') and 0<= int(num[i]) <= 5: # 2开头只能收012345
                dp[i] = dp[i-1] + dp[i-2]
            elif (num[i-1] == '1'): # 1开头全收了
                dp[i] = dp[i-1] + dp[i-2]
            else:
                dp[i] = dp[i-1] 
        return dp[-1]

```

# 剑指 Offer 47. 礼物的最大价值

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        # dp
        # 初始化数组为 m+1 行， n+1列
        # 且外圈为0，按照从左到右填充
        # 状态转移方程为dp[i][j] = max(dp[i-1][j],dp[i][j-1])+grid[i-1][j-1]
        if grid == []:
            return 0
        m = len(grid)
        n = len(grid[0])
        dp = [ [0 for i in range(n+1)] for k in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                dp[i][j] = max(dp[i-1][j],dp[i][j-1])+grid[i-1][j-1]

        return dp[-1][-1] # 返回最后一个

```

# 剑指 Offer 49. 丑数

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

**说明:** 

1. `1` 是丑数。
2. `n` **不超过**1690。

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        # 起点从1开始
        # 维护好一个数组，其中存取的是已经排序的丑数
        # 动态规划，三指针的思路
        # 指针永远指向合格的丑数，三个不同效果的走的速度不一样
        # 画图辅助理解
        if n == 0:
            return 
        dp = [0 for i in range(n+1)] # dp[n]的意思是第n个偶数
        # 初始化dp[0],dp[1] = 1,1
        dp[0], dp[1] = 1, 1
        t2, t3, t5 = 1, 1, 1 # 指向dp[1]的索引,他们指向的值是最小的乘以指定数之后即大于已经生成的当前最好一个丑数
        # 最后一个丑数一定由之前的丑数得来
        for i in range(2,n+1):
            M2, M3, M5 = dp[t2]*2, dp[t3]*3 , dp[t5]*5
            dp[i] = min(M2,M3,M5)
            # 必须是3个if不能是elif,注意指针逻辑
            # 例如生成10的时候，可能是dp[t2]==5,它*2==10 也可能是 dp[t5]==2,它*5==10，
            # 如果有一个指针没有更新，那么下一次还会是10,导致整体都偏慢而无法确定准确的第n个丑数
            # 【用elif的话，要得到正确结果的解决办法申请足够大的数组去重从小到大排序】
            if dp[i] == M2: t2 += 1
            if dp[i] == M3: t3 += 1
            if dp[i] == M5: t5 += 1
        return dp[-1]

            
```

# 剑指 Offer 50. 第一个只出现一次的字符

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        frequent = collections.Counter(s) # 这建造出来的是个字典，但是是无序的
        for char in s:
            if frequent[char] == 1:
                return char
        return ' '
```

# 剑指 Offer 51. 数组中的逆序对

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        # 归并排序

        def mergeSort(left,right): # 传入参数为左右坐标,闭区间
            if left >= right:
                return 0 # 返回数值为逆序数

            mid = (left + right)//2            
            count = mergeSort(left,mid)+mergeSort(mid+1,right) # 俩返回值相加
            # print(left,mid,mid+1,right)
            p1 = left
            p2 = mid+1
            p3 = 0 # 【主数组填充指针】
            temp = [0 for i in range(right-left+1)]
            while p1 <= mid and p2 <= right: # p1是前半截，p2是后半截
                if nums[p1] <= nums[p2]: # 正常顺序
                    temp[p3] = nums[p1]
                    p1 += 1
                    p3 += 1
                elif nums[p1] > nums[p2]: # 逆序
                    temp[p3] = nums[p2]
                    p2 += 1
                    p3 += 1
                    count += (mid-p1+1) # 注意这一条。代表那一坨都逆序了
            # 不知道哪边有剩余
            while p1 <= mid: 
                temp[p3] = nums[p1]
                p1 += 1
                p3 += 1
            while p2 <= right:
                temp[p3] = nums[p2]
                p2 += 1
                p3 += 1
                count += (mid-p1+1)
            nums[left:right+1] = temp # 注意这一条
            return count
        
        l = 0
        r = len(nums)-1
        ans = mergeSort(l,r)
        return ans
```



# 剑指 Offer 53 - I. 在排序数组中查找数字 I

统计一个数字在排序数组中出现的次数。

示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
示例 2:

输入: nums = [5,7,7,8,8,10], target = 6
输出: 0

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 二分查找左边界和二分查找右边界
        # 左闭、右闭区间查找
        # 先找左边界
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left+right)//2
            if nums[mid] == target:  # 如果中间值是目标，需要收缩右边界
                right = mid - 1
            elif nums[mid] > target: # 如果中间值大于目标，需要收缩右边界
                right = mid - 1
            elif nums[mid] < target: # 如果中间值小于目标，需要收缩左边界
                left = mid + 1
        # 这样找完之后返回left
        mark1 = left # 存储left
        # 再找右边界
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left+right)//2
            if nums[mid] == target:  # 如果中间值是目标，需要收缩左边界
                left = mid + 1
            elif nums[mid] > target: # 如果中间值大于目标，需要收缩右边界
                right = mid - 1
            elif nums[mid] < target: # 如果中间值小于目标，需要收缩左边界
                left = mid + 1
        # 这样找完之后存储right
        mark2 = right
        # 最终返回的结果为 mark2 - mark1 + 1
        return mark2 - mark1 + 1

```

# 剑指 Offer 52. 两个链表的第一个公共节点

输入两个链表，找出它们的第一个公共节点。

如果两个链表没有交点，返回 null.
在返回结果后，两个链表仍须保持原有的结构。
可假定整个链表结构中没有循环。
程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 要求仅仅用O(1)内存
        # 只能用双指针
        # 最多只会换航道一次！
        node1 = headA
        node2 = headB
        while node1 != node2:
            node1 = node1.next if node1 else headB
            node2 = node2.next if node2 else headA
        return node1
```



# 剑指 Offer 54. 二叉搜索树的第k大节点

给定一棵二叉搜索树，请找出其中第k大的节点。

```python
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        # 直接中序遍历然后返回下标为len(ans)-k
        ans = []
        def submethod(node):
            if node == None:
                return
            submethod(node.left)
            ans.append(node.val)
            submethod(node.right)
        submethod(root)
        return ans[len(ans)-k] 
```

# 剑指 Offer 55 - I. 二叉树的深度

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0
        else:
            return max(self.maxDepth(root.left),self.maxDepth(root.right))+1
```

# 剑指 Offer 55 - II. 平衡二叉树

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        # 递归法
        if root == None:
            return True
        return abs(self.deepth(root.left)-self.deepth(root.right))<=1 and self.isBalanced(root.left) and self.isBalanced(root.right)
               
    def deepth(self,root):#先写个深度求解方法
        if root == None:
            return 0
        else:
            return max(self.deepth(root.left),self.deepth(root.right)) + 1
```

# 剑指 Offer 56 - I. 数组中数字出现的次数

一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        # 分治法的思想，首先先全部异或
        # 找到全部异或值中不为1的那一位，记为t
        # 以t和所有nums的值对应位进行运算，按照0和1分成两组，
        # 两组分别异或得到最终结果
        a = 0
        for i in nums:
            a ^= i
        mark = 0 # 把最低位视为第0位，找到那个1的位
        for i in range(32):
            if (a>>i)&1 == 1:
                mark = i
                break
        group1 = []
        group2 = []
        for i in nums: #分组
            if (i>>mark)&1 == 1:
                group1.append(i)
            else:
                group2.append(i)
        p = 0
        q = 0
        for i in group1:
            p ^= i
        for i in group2:
            q ^= i
        return [p,q]
```

# 剑指 Offer 56 - II. 数组中数字出现的次数 II

在一个数组 `nums` 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 对每一位的bit求和，然后模3
        ans = 0
        for i in range(32):
            bit = 0
            for num in nums:
                bit += (num>>i)&1 # 右移位取末位和1取值
            bit = bit%3
            ans += bit*2**(i) # 对收集的结果及时转化成对应的十进制
        return ans      
```



# 剑指 Offer 57. 和为s的两个数字

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 双指针两数之和
        left = 0
        right = len(nums)-1
        while left < right: #两者不能相等
            if nums[left] + nums[right] == target:
                return [nums[left],nums[right]]
            elif nums[left] + nums[right] > target: #值大了，右边左移
                right -= 1
            elif nums[left] + nums[right] < target: #否则左边右移
                left += 1
        return []
```

# 剑指 Offer 57 - II. 和为s的连续正数序列

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        # 数学思路
        # 从i开始，如果(p+ q)*(q-p+1)/2 == target:并且p<q,且长度大于2，收集
        # 解方程，定p求q，q是整数才收集
        ans = []
        p = 1
        q = (-1+(1-(4*(p-p*p-2*target)))**0.5)/2
        while p < q:
            if q == int(q):
                q = int(q)
                lst = [p+i for i in range(q-p+1)]
                ans.append(lst)
            p += 1
            q = (-1+(1-(4*(p-p*p-2*target)))**0.5)/2
        return ans

```

# 剑指 Offer 58 - I. 翻转单词顺序

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # 首先先将其格式化处理，去除多余空格，利用栈
        # 当栈顶是空格时，即将入栈元素不能是空格,先给栈加一个预空格统一语法
        stack = ' '
        for i in s:
            if stack[-1] != ' ':
                stack += i
            elif stack[-1] == ' ':
                if i == ' ':
                    pass
                elif i != ' ':
                    stack += i
        lst = stack[1:].split(' ') 
        if lst[-1] == '':
            lst = lst[:-1] #去掉尾巴的‘’
        # lst[::-1]然后链接成字符串
        lst = lst[::-1]
        ans = ' '.join(lst)
        return ans
```

# 剑指 Offer 59 - I. 滑动窗口的最大值

给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if nums == []:
            return []
        # 使用单调队列，队列头是大值，所以要单调递减队列
        window = collections.deque()
        mono_d_queue = collections.deque()
        # 初始化队列和窗口
        for val in nums[:k]:
            window.append(val) # 窗口直接加入
            if len(mono_d_queue) == 0:
                mono_d_queue.append(val)
            elif mono_d_queue[-1] >= val: # 如果满足单调递减，直接加入
                mono_d_queue.append(val)
            elif mono_d_queue[-1] < val: # 需要调整
                while len(mono_d_queue) != 0 and mono_d_queue[-1] < val:
                    mono_d_queue.pop() # 从尾巴弹出那些破坏单调的值
                mono_d_queue.append(val)
        ans = [mono_d_queue[0]] # 收集答案，把初始化完毕的结果先放进去
        for val in nums[k:]:
            e = window.popleft() # 取窗口的最左边的值
            if e == mono_d_queue[0]:
                mono_d_queue.popleft() # 如果出窗口的值也是大值，则大值需要弹出
            window.append(val) # 窗口直接加入

            if len(mono_d_queue) == 0:
                mono_d_queue.append(val)
            elif mono_d_queue[-1] >= val: # 如果满足单调递减，直接加入
                mono_d_queue.append(val)
            elif mono_d_queue[-1] < val: # 需要调整
                while len(mono_d_queue) != 0 and mono_d_queue[-1] < val:
                    mono_d_queue.pop() # 从尾巴弹出那些破坏单调的值
                mono_d_queue.append(val)
            # 收集本轮窗口的最大值,在队列头
            ans.append(mono_d_queue[0])
        return ans
```

# 剑指 Offer 59 - II. 队列的最大值

请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

```python
class MaxQueue:
    # 双队列，主队列和辅队列。这里都采用双端队列
    # 主队列是正常队列
    # 辅队列是单调队列，由于从队头需要弹出最大值，所以是单调递减队列
    def __init__(self):
        self.main_queue = collections.deque()
        self.help_queue = collections.deque()

    def max_value(self) -> int:
        if len(self.help_queue) == 0:
            return -1
        return self.help_queue[0]
        
    def push_back(self, value: int) -> None:
        # 主队列入队直接入
        self.main_queue.append(value)
        # 辅助队列需要保证单调递减。所以从后往前扫，如果尾巴小于入队元素，则从后端弹出
        if len(self.help_queue) == 0: # 辅助列为空，直接入队
            self.help_queue.append(value)
            return 
        # 所以从后往前扫，如果尾巴小于入队元素，则从后端弹出
        while len(self.help_queue) >= 1 and self.help_queue[-1] < value:
            self.help_queue.pop()
        self.help_queue.append(value)
        return 

    def pop_front(self) -> int:
        if len(self.main_queue) == 0:
            return -1
        e = self.main_queue.popleft() # 获取弹出元素
        # 如果弹出元素等于辅助队列队头，则辅助队列也要弹出
        if e == self.help_queue[0]:
            self.help_queue.popleft()
        return e

```

# 剑指 Offer 60. n个骰子的点数

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

 

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        # 组合问题
        # python纯模拟会超时 6 ** 11在3E，减半也在1.5E
        # 该方法超时
        lst = [0 for i in range(6*n+1)] # 预先声明长度
        def choice(n,theSum):
            if n == 0:
                lst[theSum] += 1
                return 
            for a in range(1,7):
                theSum += a
                choice(n-1,theSum)
                theSum -= a
        
        choice(n,0)
        sumAll = sum(lst[n:])
        ans = [i/sumAll for i in lst[n:]]
        return ans

```

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        # 组合问题
        # 该方法不会超时
        lst = [0 for i in range(6*n-(n-1))] # 预先声明长度，初始化为全0
        # 动态规划，已知
        # 第n次为[.n.]
        # 则第n+1次为[.n.]叠6次，每次叠的时候向右滑动
        for i in range(6): # 基态6个1
            lst[i] = 1
        for t in range(n-1):
            base = lst.copy()
            for offset in range(1,6): # 
                for i in range(len(lst)): # 叠加
                    if (i+offset) < len(lst):
                        lst[i+offset] += base[i]
        # 此时lst为总频次
        allSum = sum(lst)
        ans = [i/allSum for i in lst]
        return ans

```



# 剑指 Offer 61. 扑克牌中的顺子

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        # 这个题中 10 JQKA 不是顺子
        # 数学思想
        # 顺子中不能有重复的牌
        # 关键思想！
        # 最关键的除了0之外，最大值和最小值的差距小于5
        max_num = -1 # 注意这个初始化
        min_num = 14 # 注意这个初始化
        repeat = set()
        for i in nums:
            if i == 0: # 跳过0
                continue
            max_num = max(max_num,i)
            min_num = min(min_num,i)
            if i not in repeat: repeat.add(i)
            elif i in repeat: return False # 有重复则跳出，0在第一行被跳出了
        return max_num-min_num < 5

```

# 剑指 Offer 62. 圆圈中最后剩下的数字

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

```python
# 约瑟夫环问题
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        # 数学问题
        # 先分析情况，以01234为例，把它变做ABCDE，已知最后活下来的是D。看它每一轮的索引
        # 以n=5 m=3 为例子，删除的是第3个 f(n,m) = 索引
        # 第一轮 ：ABCDE  D的索引为3    f(5,3) = 3
        # 第二轮： DEAB   D的索引为0    f(4,3) = 0
        # 第三轮： BDE    D的索引为1    f(3,3) = 1
        # 第四轮： BD     D的索引为1    f(2,3) = 1
        # 第五轮： D      D的索引为0    f(1,3) = 0
        # 删去为(m-1)%当前长度
        # 索引变化规律为最后一次一定为0，你不用管其他人是死是活
        
        # 如何倒推索引呢 从第五轮推到原始轮
        # 以第一轮第二轮为例
        # 第一轮  A   B   C   D   E   
        # 第二轮              D   E    A   B
        # 变化：              D   E  | A   B   (C)
        
        # 设第1轮时候 D 的坐标为 u ,下一轮的坐标为 v
        # 则可知坐标变化关系 (u - m % (第一轮长度)) % (第一轮长度) = v
        # 解方程：
        # (u - m % n ) % n = v  
        # (u - m % n) = kn + v
        # u = kn + v + m%n [由于u需要在范围内]
        # u == [kn + v + m%n] % n == v + m % n
        # 即 u = (v + m%n) % n
        # 即 u = (v + m) % n
        
        # 移项： u = v + m %（第一轮长度）注意： 【第一轮长度 == 第二轮长度+1】
        # 即 f(n,m) = (f(n-1,m) + m % n) % n
        # 即 f(n,m) = (f(n-1,m) + m) % n
        
        # 递归终点为n == 1 此时f(1,m) == 0
        # f[n+1] = (f(n) + m )%(n+1)
        
        # 正式代码如下：
        s = 0 # 表示开始
        for i in range(1,n):
            s = (s + m) % (i+1)
        return s
        
```

# 剑指 Offer 63. 股票的最大利润

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 贪心法
        # 扫描，第一轮扫描，到目前为止的，存在过的最低值是多少
        # 第二轮扫描，假设今天卖出，可以获利多少
        # 返回最大获利
        if len(prices) == 0:
            return 0
        min_point = []
        p = 0
        min_num = 0xffffffff # 初始化一个极大值
        while p < len(prices):
            if min_num > prices[p]:
                min_num = prices[p]
            min_point.append(min_num)
            p += 1
        p = 0
        max_profit = -0xffffffff # 初始化一个极小值
        while p < len(min_point):
            max_profit = max(max_profit,prices[p]-min_point[p])
            p += 1
        return max_profit


```

# 剑指 Offer 65. 不用加减乘除做加法

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

```java
class Solution {
    // 思路： 使用位运算
    // 且考虑，用异或来保留没有进位的加和
    // 用与运算来保留一共进位多少
    // 补码运算在考虑正负数的时候，会自动丢弃越出32位的那一位，不需要担心符号位
    public int add(int a, int b) {
        // 用java的位置运算
        while (b != 0){
            int carry = (a&b) << 1; // 用与运算来确定哪些位置有进位，注意要进位要左移位
            a = a ^ b; // 把b的非进位和全部加到a里面
            b = carry; // 剩下来的如果不是0，则循环，赋值给bb，下一轮再加进去都是
        }
        return a;

    }
}
```

# 剑指 Offer 64. 求1+2+…+n

求 `1+2+...+n` ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

```python
class Solution:
    def sumNums(self, n: int) -> int:
        # 求和公式为(1+n)n/2
        # 其中/2可以用位运算解决
        # 1+n的位运算比较好解决
        # 乘法的位运算需要解决

        # 或者使用递归来解决
        # 一般而言，递归出口是用条件判断语句
        # 但是由于这一题条件判断语句被ban了，所以需要另寻出路
        # 力扣提升了递归栈的深度，所以可以不用处理栈溢出
        # and 短路判断
        # 当前者不满足时候，如为0时，则退出递归
        return n and n + self.sumNums(n-1)
```

# 剑指 Offer 66. 构建乘积数组

给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。

```python
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        if len(a) == 0:
            return []
        # 前缀和后缀思想
        L = [1]
        i = 0
        temp = a[0]
        while i < len(a)-1: # 从左往右扫出前缀
            L.append(temp)
            i += 1
            temp *= a[i]

        R = collections.deque() # 用了双端队列辅助，如果不用的话在收集结果时候吧R的下标改写成length-i
        R.append(1)
        i = -1
        temp = a[-1]
        while i > -len(a): # 从右边往左扫出后缀
            R.appendleft(temp)
            i -= 1
            temp *= a[i]

        ans = [] # 收集结果
        for i in range(len(L)):
            ans.append(L[i]*R[i])
        return ans
```

# 剑指 Offer 67. 把字符串转换成整数

```python
class Solution:
    def strToInt(self, s: str) -> int:
        # 梳理逻辑，这里把函数签名改成了s，防止和python的冲突
        if len(s) == 0: return 0
        # 1. 丢弃无用的开头空格
        p = 0
        while p < len(s) and s[p] == " ":
            p += 1
        # 2.处理尾巴
        p2 = len(s) - 1 # 处理尾巴
        while p2 >= 0 and s[p2] not in set("0123456789"):
            p2 -= 1
        s = s[p:p2+1]
        if len(s) == 0: return 0 # 掐头去尾为空之后，则返回0
        # 处理符号,# 处理字母开头
        symbol = 1
        if s[0] == "-":
            symbol = -1
            s = s[1:]
        elif s[0] == "+":
            s = s[1:]
        elif s[0] not in set("0123456789"): 
            return 0
        # 去符号后看首位是否还是符号
        if s[0] in "+-":
            return 0
        low = - 2**31
        up = 2**31 - 1
        # 中间还不能有小数点,等非数字,有的话截去
        record = len(s) # 初始化为全长-1
        for i in range(len(s)):
            if s[i] not in set("0123456789"):
                record = i
                break # 记录到第一个就break
        s = s[:record]
        if len(s) == 0: return 0 # 每次切片操作都需要检查是否为空
        ans = symbol * int(s) # 加上符号位
        if ans < low:
            return low
        elif ans > up:
            return up
        return ans
            
```

# 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 如果两者都小于root，递归到左子树
        # 如果两者都大于root，递归到右子树
        # 如果两者一个大于root一个小于root，返回root
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left,p,q)
        elif p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right,p,q)
        else:
            return root
```

# 剑指 Offer 68 - II. 二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # 不使用额外结构的递归解法
        # 分情况讨论，如果p,q之一是root,另一个在它的子树下，返回root
        # 这个写法优秀在p,q可以适配到不在树中的时候
        if root == None: return None
        if root == p or root == q : return root
        left = self.lowestCommonAncestor(root.left,p,q) # 递归搜左右子树
        right = self.lowestCommonAncestor(root.right,p,q)
        # 
        # 根据right和left是否有返回值分类
        if left == None and right == None: return 
        if left != None and right == None: return left
        if left == None and right != None: return right
        if left != None and right != None: return root
```


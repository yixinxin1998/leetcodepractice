# [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

```

```

# [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

```

```

# 913. 猫和老鼠

```

```

# 第一周：11.29 - 12.05 【五题】

## [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```python
# 动态规划，注意状态的设置，推荐用文字表达清楚
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 需要提升一个维度
        # dp[][][0]表示没有完成过交易
        # dp[][][1]表示完成过一次交易
        # dp[][][2]表示完成过两次交易
        # dp[0][][]表示当前不持有股票的最大收益
        # dp[1][][]表示当前持有股票的最大收益

        n = len(prices)
        dp = [[[0 for k in range(3)] for j in range(n)] for i in range(2)]
        
        # 分析状态 dp[][i][]中i表示天数，那么一共有 6种状态
        # dp[0][i][0]表示当前不持有股票，没有完成过交易
        # dp[0][i][1]表示当前不持有股票，完成过一次交易
        # dp[0][i][2]表示当前不持有股票，完成过两次交易
        # dp[1][i][0]表示当前持有股票，没有完成过交易
        # dp[1][i][1]表示当前持有股票，完成过一次交易
        # dp[1][i][2]表示当前持有股票，完成过两次交易

        # 初始化i=0时候的值
        dp[0][0][0] = 0
        dp[0][0][1] = -0xffffffff # 不存在这个状态
        dp[0][0][2] = -0xffffffff # 不存在这个状态
        dp[1][0][0] = -prices[0] 
        dp[1][0][1] = -0xffffffff
        dp[1][0][2] = -0xffffffff 

        # 开始dp,对照描述进行dp状态转移
        # dp[0][i][0]表示当前不持有股票，没有完成过交易
        # dp[0][i][1]表示当前不持有股票，完成过一次交易
        # dp[0][i][2]表示当前不持有股票，完成过两次交易
        # dp[1][i][0]表示当前持有股票，没有完成过交易
        # dp[1][i][1]表示当前持有股票，完成过一次交易
        # dp[1][i][2]表示当前持有股票，完成过两次交易 # 这一行没意义
        for i in range(1,n):
            dp[0][i][0] = dp[0][i-1][0]
            dp[0][i][1] = max(dp[0][i-1][1],dp[1][i-1][0]+prices[i]) # 要么继承前一天，要么来自于持有股票没有完成交易的那一天
            dp[0][i][2] = max(dp[0][i-1][2],dp[1][i-1][1]+prices[i]) # 要么继承前一天，要么来自于持有股票且完成一次交易
            dp[1][i][0] =  max(dp[1][i-1][0],0-prices[i])
            dp[1][i][1] = max(dp[1][i-1][1],dp[0][i-1][1]-prices[i])
            # dp[1][i][2] = 没意义
        # print(dp)
        return max(dp[0][n-1][1],dp[0][n-1][2],0)
```

## [265. 粉刷房子 II](https://leetcode-cn.com/problems/paint-house-ii/)

假如有一排房子，共 n 个，每个房子可以被粉刷成 k 种颜色中的一种，你需要粉刷所有的房子并且使其相邻的两个房子颜色不能相同。

当然，因为市场上不同颜色油漆的价格不同，所以房子粉刷成不同颜色的花费成本也是不同的。每个房子粉刷成不同颜色的花费是以一个 n x k 的矩阵来表示的。

例如，costs[0][0] 表示第 0 号房子粉刷成 0 号颜色的成本花费；costs[1][2] 表示第 1 号房子粉刷成 2 号颜色的成本花费，以此类推。请你计算出粉刷完所有房子最少的花费成本。

注意：

所有花费均为正整数。

```python
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        # 复杂度 n*(k^2)
        n = len(costs)
        k = len(costs[0])
        dp = [[0 for j in range(n)] for i in range(k)]
        # 初始化
        for i in range(k):
            dp[i][0] = costs[0][i]

        for j in range(1,n):
            group = []
            for i in range(k):
                group.append(dp[i][j-1])
            for i in range(k):
                memo = group[i] # 记录一下
                group[i] = 0xffffffff # 设置为极大值，不会挑选到它
                dp[i][j] = min(group) + costs[j][i]
                group[i] = memo # 还原
        
        # 求最后一列的最小值
        ans = []
        for i in range(k):
            ans.append(dp[i][n-1])
        return min(ans)
    
    # 优化的时候可以对min优化，记录最大值和次大值。这样不需要在每次内层min的时候再耗费一层k的复杂度
```

## [786. 第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/) 【需要补刷】

给你一个按递增顺序排序的数组 arr 和一个整数 k 。数组 arr 由 1 和若干 素数  组成，且其中所有整数互不相同。

对于每对满足 0 < i < j < arr.length 的 i 和 j ，可以得到分数 arr[i] / arr[j] 。

那么第 k 个最小的分数是多少呢?  以长度为 2 的整数数组返回你的答案, 这里 answer[0] == arr[i] 且 answer[1] == arr[j] 。

```python
# 纯暴力，python
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        themap = dict()
        n = len(arr)
        for i in range(n):
            for j in range(i+1,n):
                themap[(arr[i],arr[j])] = arr[i]/arr[j]
        
        lst = []
        for key in themap:
            lst.append([key,themap[key]])
        
        lst.sort(key = lambda x:x[1])
        # print(lst[k-1])
        return list(lst[k-1][0])
```

## [982. 按位与为零的三元组](https://leetcode-cn.com/problems/triples-with-bitwise-and-equal-to-zero/) 【需要补刷位运算算法】

给定一个整数数组 A，找出索引为 (i, j, k) 的三元组，使得：

0 <= i < A.length
0 <= j < A.length
0 <= k < A.length
A[i] & A[j] & A[k] == 0，其中 & 表示按位与（AND）操作符。

```python
class Solution:
    def countTriplets(self, nums: List[int]) -> int:
        # i,j,k是可以相等的
        # 先两两预处理出一串结果
        temp = []
        n = len(nums)
        for i in range(n):
            for j in range(n):
                temp.append(nums[i]&nums[j])
        
        ct1 = collections.Counter(nums)
        ct2 = collections.Counter(temp)
        
        ans = 0

        for key1 in ct1:
            for key2 in ct2:
                if key1&key2 == 0:
                    ans += ct1[key1]*ct2[key2]
        return ans
```

```python
class Solution:
    def countTriplets(self, A: List[int]) -> int:
        mem = [0] * 65536
        mask = (1 << 16) - 1 # 获得全1掩码
        
        # 标记数字能够令哪些数字相与变成0，也就是遍历所有的0所在的位置组成的数字
        for num in A:
            mk = mask ^ num # 用异或，mk为和num的所有的不同位
            i = mk # 起始赋值
            while i:
                mem[i] += 1
                # 这一步是关键，位运算找出所有的满足条件的数字
                i = (i - 1) & mk
            # 数字0肯定能够相与变成0
            mem[0] += 1
        res = 0

        for n1 in A:
            for n2 in A:
                res += mem[n1 & n2]

        return res
```

## [1553. 吃掉 N 个橘子的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-eat-n-oranges/) 【需要补dij算法的解法】

厨房里总共有 n 个橘子，你决定每一天选择如下方式之一吃这些橘子：

吃掉一个橘子。
如果剩余橘子数 n 能被 2 整除，那么你可以吃掉 n/2 个橘子。
如果剩余橘子数 n 能被 3 整除，那么你可以吃掉 2*(n/3) 个橘子。
每天你只能从以上 3 种方案中选择一种方案。

请你返回吃掉所有 n 个橘子的最少天数。

```python
class Solution:
    def minDays(self, n: int) -> int:
        # 记忆化递归+贪心，不能使用dp，数量级原因dp肯定不过
        memo = dict()
        memo[1] = 1
        memo[2] = 2
        memo[3] = 2

        # 贪心
        def recur(n):
            if n in memo:
                return memo[n]           
            # 先达到最接近2的数
            s2 = recur(n//2) + 1
            s2 += (n%2)
            s3 = recur(n//3) + 1
            s3 += (n%3)
            memo[n] = min(s2,s3)
            return memo[n]
                    
        return recur(n)
```

# 第二周：12.06-12.12 【六题】

## [305. 岛屿数量 II](https://leetcode-cn.com/problems/number-of-islands-ii/) 

给你一个大小为 m x n 的二进制网格 grid 。网格表示一个地图，其中，0 表示水，1 表示陆地。最初，grid 中的所有单元格都是水单元格（即，所有单元格都是 0）。

可以通过执行 addLand 操作，将某个位置的水转换成陆地。给你一个数组 positions ，其中 positions[i] = [ri, ci] 是要执行第 i 次操作的位置 (ri, ci) 。

返回一个整数数组 answer ，其中 answer[i] 是将单元格 (ri, ci) 转换为陆地后，地图中岛屿的数量。

岛屿 的定义是被「水」包围的「陆地」，通过水平方向或者垂直方向上相邻的陆地连接而成。你可以假设地图网格的四边均被无边无际的「水」所包围。

```python
class UF:
    def __init__(self,size):
        self.root = [i for i in range(size)]
        self.count = 0 # 记录联通分量的个数
    
    def find(self,x):
        while x != self.root[x]:
            x = self.root[x]
        return x 

    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootX] = rootY 
            self.count -= 1 

    
    def isConnect(self,x,y):
        return self.find(x) == self.find(y)


class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        # 非优化并查集，每次添加前后检查是否把上下左右联通，
        # 最初，全是水，positions可能有重复
        ufset = UF(m*n)
        limit = m*n
        # 对应的index为: index = x*n + y 
        ans = []
        islandSet = set()
        for x,y in positions:
            index = x*n+y 
            if index in islandSet:  # 注意这一行，去重复
                ans.append(ufset.count)
                continue 
            ufset.count += 1 # 添加一个联通分量，再进行检查
            islandSet.add(index)
            # print(ufset.count)
            if index-n >= 0 and index-n in islandSet: # 往上
                if not ufset.isConnect(index,index-n):
                    ufset.union(index,index-n)
            if index+n < limit and index+n in islandSet: # 往下
                if not ufset.isConnect(index,index+n):
                    ufset.union(index,index+n)
            if y > 0 and index-1 in islandSet: # 往左
                if not ufset.isConnect(index,index-1):
                    ufset.union(index,index-1)
            if y < n-1 and index+1 in islandSet: # 往右
                if not ufset.isConnect(index,index+1):
                    ufset.union(index,index+1)
            ans.append(ufset.count)
        return ans

```

## [660. 移除 9](https://leetcode-cn.com/problems/remove-9/) 【数学】

从 1 开始，移除所有包含数字 9 的所有整数，例如 9，19，29，……

这样就获得了一个新的整数数列：1，2，3，4，5，6，7，8，10，11，……

给定正整数 n，请你返回新数列中第 n 个数字是多少。1 是新数列中的第一个数字。

```python
class Solution:
    def newInteger(self, n: int) -> int:
        # 写出需要的数字
        # 0， 1， 2， 3， 4， 5， 6， 7， 8
        # 10，11，12，13，14，15，16，17，18
        # 20，21，22，23，24，25，26，27，28
        # 。。。
        # 80，81，82，83，84，85，86，87，88
        # 100，101，102，103，104，105，106，107，108
        # 每一行是9个，即九进制数
        # 手写一个进制转换
        lst = []
        while n != 0:
            remain = n%9 # 
            n //= 9
            lst.append(remain)
        # 最终结果倒序，转int
        lst = lst[::-1]
        ans = int("".join(str(e) for e in lst))
        return ans
```

## [149. 直线上最多的点数](https://leetcode-cn.com/problems/max-points-on-a-line/) 【hash + 确定哈希key】

给你一个数组 `points` ，其中 `points[i] = [xi, yi]` 表示 **X-Y** 平面上的一个点。求最多有多少个点在同一条直线上。

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        # 理清思路：任意两个点都可以用直线的标准形式获得
        # 但是标准形式中 Ax+By+C = 0 的系数其实是不一定的，为了使得该三元组有代表性，需要使得三个数均为整数。
        # 且规定A>=0, 先令A == y1-y2 ; B == x2-x1 ; 找到它们的绝对值的最大公因数，再都除以这个gcd，再代入求CC
        def getGCD(a,b):
            while a != 0:
                temp = a
                a = b % a
                b = temp
            return b # 

        def normalize(p1,p2): # 传入两个点
            x1,y1 = p1
            x2,y2 = p2 
            A = y1-y2 
            B = x2-x1 
            if A < 0: 
                A = -A 
                B = -B 
            if A == 0: # 注意这一段，直接令B==1，防止B，C没有除以GCD
                B = abs(B)
                return (0,1,-1*y1)

            gcd = getGCD(abs(A),abs(B))
            A //= gcd 
            B //= gcd 
            return (A,B,-A*x1-B*y1)
        
        recordDict = collections.defaultdict(int) # key是(A,B,C)三元组
        n = len(points)
        for i in range(n):
            for j in range(i+1,n):
                p1 = points[i]
                p2 = points[j]
                key = normalize(p1,p2)
                recordDict[key] += 1

        
        # 其中key的最大值为 (Ck2) == max(key),反求k
        maxVal = 0
        for key in recordDict:
            maxVal = max(maxVal,recordDict[key])
        
        def findSolution(val): # 求解
            return int((1+sqrt(1+8*val))/2)

        return findSolution(maxVal)

```

## [1402. 做菜顺序](https://leetcode-cn.com/problems/reducing-dishes/) 【数学】

一个厨师收集了他 n 道菜的满意程度 satisfaction ，这个厨师做出每道菜的时间都是 1 单位时间。

一道菜的 「喜爱时间」系数定义为烹饪这道菜以及之前每道菜所花费的时间乘以这道菜的满意程度，也就是 time[i]*satisfaction[i] 。

请你返回做完所有菜 「喜爱时间」总和的最大值为多少。

你可以按 任意 顺序安排做菜的顺序，你也可以选择放弃做某些菜来获得更大的总和。

```python
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        # 预先排序
        satisfaction.sort()
        # 利用顺序和最大暴力算一波
        if satisfaction[0] >= 0:
            return sum((i+1)*satisfaction[i] for i in range(len(satisfaction)))
        # 否则先预先算出全部的和sum((i+1)*satisfaction[i] for i in range(len(satisfaction)))
        comp = sum((i+1)*satisfaction[i] for i in range(len(satisfaction)))
        # 1a + 2b + 3c + 4d -> b + 2c + 3d -> c + 2d -> d 
        # 为减去abcd，减去bcd，减去cd，减去d
        sigSum = sum(satisfaction)
        biggest = comp
        n = len(satisfaction)
        p = 0
        while p != n:
            comp -= sigSum
            biggest = max(biggest,comp)
            sigSum -= satisfaction[p]
            p += 1
        return biggest
```

## [689. 三个无重叠子数组的最大和](https://leetcode-cn.com/problems/maximum-sum-of-3-non-overlapping-subarrays/) 【尬枚举做法，周末补强】

给你一个整数数组 nums 和一个整数 k ，找出三个长度为 k 、互不重叠、且 3 * k 项的和最大的子数组，并返回这三个子数组。

以下标的数组形式返回结果，数组中的每一项分别指示每个子数组的起始位置（下标从 0 开始）。如果有多个结果，返回字典序最小的一个。

```python
class Solution:
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        # 先预处理,每一位为包含自身的往后数k个数的和
        n = len(nums)
        lst = []
        ss = sum(nums[:k])
        for i in range(k,n):
            lst.append(ss)
            ss -= nums[i-k]
            ss += nums[i]
        lst.append(ss)
        lst = lst + [0]*(n-len(lst))

        # 其中枚举规则为类似与接雨水，枚举在k个之前的出现过的最大值和左边
        tmax = lst[0]
        tind = 0
        preMax = [-1 for i in range(n)]
        for i in range(n):
            if lst[i] > tmax:
                tmax = lst[i]
                tind = i
            preMax[i] = [tmax,tind]
        # 然后切割后k个舍去，并且偏移前k个
        preMax = [0]*k + preMax[:n-k]

        # 继续枚举
        tmax = lst[n-1]
        tind = n-1
        postMax = [-1 for i in range(n)]
        for i in range(n-1,-1,-1):
            if lst[i] >= tmax: # 注意这里是大于等于，因为要使得字典序最小，相等的时候也要更新
                tmax = lst[i]
                tind = i 
            postMax[i] = [tmax,tind]
        # 然后切割前k个，填充后k个
        postMax = postMax[k:]+[0]*k

        # print(lst)
        # print(preMax)
        # print(postMax)
        # 对k~n-k枚举
        ans = [-1,-1,-1]
        biggest = -1
        for i in range(k,n-k):
            if preMax[i][0]+postMax[i][0]+lst[i] > biggest:
                biggest = preMax[i][0]+postMax[i][0]+lst[i]
                ans = [preMax[i][1],i,postMax[i][1]]
        return ans
```

```go
func maxSumOfThreeSubarrays(nums []int, k int) []int {
    // 先预处理pre
    ss := sum(nums[:k])
    n := len(nums)
    lst := make([]int,n,n)
    for i:=k;i<n;i++ {
        lst[i-k] = ss
        ss -= nums[i-k]
        ss += nums[i]
    }
    lst[n-k] = ss
    // fmt.Println(lst)
    
    sw1 := make([][]int,n,n)
    tmax := lst[0]
    tind := 0
    for i:=0;i<n-k;i++ {
        if lst[i] > tmax {
            tmax = lst[i]
            tind = i
        }
        sw1[i+k] = []int{tmax,tind}
    } 
    
    sw3 := make([][]int,n,n)
    tmax = lst[n-1]
    tind = n-1
    for i:=n-1;i>k-1;i-- {
        if lst[i] >= tmax {
            tmax = lst[i]
            tind = i
        }
        sw3[i-k] = []int{tmax,tind}
    }
    
    //fmt.Println(sw1)
    //fmt.Println(sw3)
     
    ans := []int{-1,-1,-1}
    biggest := -1
    for i:=k;i<n-k;i++ {
        if sw1[i][0]+sw3[i][0]+lst[i] > biggest {
            biggest = sw1[i][0]+sw3[i][0]+lst[i]
            ans = []int{sw1[i][1],i,sw3[i][1]}
        }
    }
    return ans
}

func sum(arr []int) int {
    t := 0
    for _,v := range arr {
        t += v
    }
    return t
}
```

## [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/) 【后序遍历的递归】

给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        # 带有后序遍历性质的dfs
        m,n = len(matrix),len(matrix[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        grid = [[0 for j in range(n)] for i in range(m)]
        # 对每个位置进行后续遍历更新
        def dfs(i,j):
            if visited[i][j]:
                return grid[i][j]
            visited[i][j] = True 
            group = [0]
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and matrix[new_i][new_j] > matrix[i][j]:
                    group.append(dfs(new_i,new_j))
            grid[i][j] = max(group)+1
            return grid[i][j]
        
        for i in range(m):
            for j in range(n):
                dfs(i,j)
        
        ans = 1
        for line in grid:
            ans = max(ans,max(line))
        return ans
```

# 第三周：12.13-12.19 【七题】

## [剑指 Offer II 114. 外星文字典](https://leetcode-cn.com/problems/Jf1JuT/)

现有一种使用英语字母的外星文语言，这门语言的字母顺序与英语顺序不同。

给定一个字符串列表 words ，作为这门语言的词典，words 中的字符串已经 按这门新语言的字母顺序进行了排序 。

请你根据该词典还原出此语言中已知的字母顺序，并 按字母递增顺序 排列。若不存在合法字母顺序，返回 "" 。若存在多种可能的合法字母顺序，返回其中 任意一种 顺序即可。

字符串 s 字典顺序小于 字符串 t 有两种情况：

在第一个不同字母处，如果 s 中的字母在这门外星语言的字母顺序中位于 t 中字母之前，那么 s 的字典顺序小于 t 。
如果前面 min(s.length, t.length) 字母都相同，那么 s.length < t.length 时，s 的字典顺序也小于 t 。

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:

        # 建图,有向图，用临接表，拓扑排序
        tt = set("".join(words))

        graph = collections.defaultdict(list)
        indegree = collections.defaultdict(int)

        # 激活全部
        for ch in tt:
            graph[ch]
            indegree[ch]
        # 两两比较
        p = 1
        n = len(words)
        while p < n:
            w1 = words[p-1]
            w2 = words[p]
            p1 = 0 # 指针扫字符串用
            p2 = 0 # 指针扫字符串用
            state = False # 默认没有找到拓扑序
            while p1 < len(w1) and p2 < len(w2):
                # 补丁,特殊情况，如果前者长于后者且后者是前者的前缀，则直接返回“”
                if len(w1) > len(w2) and w1[:len(w2)] == w2: return ""
                if w1[p1] == w2[p2]:
                    p1 += 1
                    p2 += 1
                elif w1[p1] != w2[p2]: # 说明找到了不同，且w1的字典序要小，加入图构建拓扑序列
                    graph[w1[p1]].append(w2[p2])
                    indegree[w2[p2]] += 1
                    state = True # 标志找到了拓扑序
                    break 

            p += 1
        
        ans = []
        # bfs找拓扑序
        queue = []
        visited = set()
        for temp in indegree:
            if indegree[temp] == 0: # 入度为0
                queue.append(temp)
        
        # print(graph)
        
        while len(queue) != 0:
            new_queue = []
            for node in queue:
                if node in visited: # 不存在拓扑序
                    return ""
                visited.add(node)
                ans.append(node)
                for neigh in graph[node]:
                    indegree[neigh] -= 1
                    if indegree[neigh] == 0:
                        new_queue.append(neigh)
            queue = new_queue

        if len(ans) != len(tt): # 不是所有字母都排序了
            return ""

        return "".join(ans)
```

## [269. 火星词典](https://leetcode-cn.com/problems/alien-dictionary/) 【难点在于获取字典序的排序逻辑】

同上

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        # 拓扑排序，找到排序的依据【相邻词】
        tt = set("".join(words))
        graph = collections.defaultdict(list)
        indegree = collections.defaultdict(int)
        for ch in tt:
            graph[ch] # 激活
            indegree[ch] # 激活
        
        p = 1
        n = len(words)
        while p < n:
            w1 = words[p-1]
            w2 = words[p]
            p1 = 0
            p2 = 0
            while p1 < len(w1) and p2 < len(w2):
                # 补丁：如果前一个单词比后面长，且后面的单词是前面单词的前缀，则直接返回空
                if len(w1) > len(w2) and w1[:len(w2)] == w2: return ""
                if w1[p1] == w2[p2]:
                    p1 += 1
                    p2 += 1
                elif w1[p1] != w2[p2]:
                    graph[w1[p1]].append(w2[p2])
                    indegree[w2[p2]] += 1
                    break 
            p += 1
        
        # 拓扑排序
        queue = []
        ans = []
        visited = set()
        # 所有入度为0的点加入队列
        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)
        
        while len(queue) != 0:
            new_queue = []
            for node in queue:
                if node in visited:
                    return ""
                visited.add(node)
                ans.append(node)
                for neigh in graph[node]:
                    indegree[neigh] -= 1
                    if indegree[neigh] == 0:
                        new_queue.append(neigh)
            queue = new_queue 
        
        if len(ans) == len(tt):
            return "".join(ans)
        else:
            return ""
```

## [980. 不同路径 III](https://leetcode-cn.com/problems/unique-paths-iii/)

在二维网格 grid 上，有 4 种类型的方格：

1 表示起始方格。且只有一个起始方格。
2 表示结束方格，且只有一个结束方格。
0 表示我们可以走过的空方格。
-1 表示我们无法跨越的障碍。
返回在四个方向（上、下、左、右）上行走时，从起始方格到结束方格的不同路径的数目。

每一个无障碍方格都要通过一次，但是一条路径中不能重复通过同一个方格。

```python
class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        # 回溯带路径暴搜，记录当前长度
        m,n = len(grid),len(grid[0])
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        visited = [[False for j in range(n)] for i in range(m)]
        memo = set() # key是字符化的路径
        needLength = 0
        start = None
        end = None 
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    needLength += 1
                elif grid[i][j] == 1:
                    start = [i,j]
                elif grid[i][j] == 2:
                    end = [i,j]
        

        def dfs(i,j,path,nowLength):
            if i == end[0] and j == end[1]:
                if nowLength == needLength+1: # 由于算了头部
                    memo.add(tuple(path))
                    return 

            visited[i][j] = True 
            nowLength += 1
            path.append((i,j))
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and (grid[new_i][new_j] == 0 or grid[new_i][new_j] == 2):
                    dfs(new_i,new_j,path,nowLength)
            path.pop()
            nowLength -= 1
            visited[i][j] = False 
        
        dfs(start[0],start[1],[],0)

        # print(memo)
        return len(memo)
```

## [630. 课程表 III](https://leetcode-cn.com/problems/course-schedule-iii/) 【比较难】

这里有 n 门不同的在线课程，按从 1 到 n 编号。给你一个数组 courses ，其中 courses[i] = [durationi, lastDayi] 表示第 i 门课将会 持续 上 durationi 天课，并且必须在不晚于 lastDayi 的时候完成。

你的学期从第 1 天开始。且不能同时修读两门及两门以上的课程。

返回你最多可以修读的课程数目。

```python
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key = lambda x:x[1]) # 预先排序
        # 迭代扫描，参照hint
        mytotal = 0 # 记录我现在所用到的总时间
        timelimit = 0 # 扫描到的时候更新截止日期
        ans = 0
        # 尝试将扫描到的课程加入，如果没有超过此时允许的截止日期，ans++
        # 否则把之前占用时间最长的弹出，替换成现在的这个，最多允许弹出一次【因为弹两次还不如不加】
        theHeap = [] # 大顶堆
        for dur,last in courses:
            timelimit = max(timelimit,last)
            if mytotal + dur <= timelimit: # 如果没有超时限，加入
                mytotal += dur 
                ans += 1
                heapq.heappush(theHeap,-dur)
            else:
                if len(theHeap) != 0 and -theHeap[0] > dur: # 如果超时限了，且最大的比当前费时间
                    temp = -heapq.heappop(theHeap)
                    mytotal -= temp # 反悔一下,腾出时间
                    ans -= 1
                    if mytotal + dur <= timelimit:
                        mytotal += dur 
                        ans += 1
                        heapq.heappush(theHeap,-dur)
                    # else: # 不如不反悔 ，这一段注释掉也对，加上也对
                    #     ans += 1 # 还原
                    #     heapq.heappush(theHeap,-temp)
                    #     mytotal += temp 
        # print(theHeap)        
        return ans
```

## [483. 最小好进制](https://leetcode-cn.com/problems/smallest-good-base/) 【数学思维，周末补强】

对于给定的整数 n, 如果n的k（k>=2）进制数的所有数位全为1，则称 k（k>=2）是 n 的一个好进制。

以字符串的形式给出 n, 以字符串的形式返回 n 的最小好进制。

```python
class Solution:
    def smallestGoodBase(self, n: str) -> str:
        # 假设 n 可以转化成为 (111...11)k ； k进制下s+1位
        # 有 n = 1 + k**1 + k**2 + k**s
        # 目的是找到最小的k
        # (k+1)**s > n # 二项式定理得， 且 n > k**s 
        # 那么 k+1 > n**(1/s) > k
        # 枚举s,s越大，k越小，所以s从64开始枚举，代入后其整数部分即为k，验证 (1-k**(s+1))/(1-k)是否==n即可
        n = int(n)
        # 长度是s+1，s至少为1，但s为1的时候方程：k+1 > n**(1/s) > k 失效，
        # 为了方便边界讨论，n-1一定是符合条件的解，如果迭代搜索过程中没有更新，则返回该值
        ans = str(n-1)
        for s in range(64,1,-1): # 枚举到长度为3即可
            k = math.floor(pow(n,1/s))
            if k == 1: continue # k不能等于1
            if (1-k**(s+1)) == n * (1-k): # 移项提升精度判定,注意k不能等于1,找到立刻返回,否则它会被n-1顶替掉
                ans = k
                return str(ans)
        return str(ans)
```

## [1610. 可见点的最大数目](https://leetcode-cn.com/problems/maximum-number-of-visible-points/)【用到了atan2(y,x)】

给你一个点数组 points 和一个表示角度的整数 angle ，你的位置是 location ，其中 location = [posx, posy] 且 points[i] = [xi, yi] 都表示 X-Y 平面上的整数坐标。

最开始，你面向东方进行观测。你 不能 进行移动改变位置，但可以通过 自转 调整观测角度。换句话说，posx 和 posy 不能改变。你的视野范围的角度用 angle 表示， 这决定了你观测任意方向时可以多宽。设 d 为你逆时针自转旋转的度数，那么你的视野就是角度范围 [d - angle/2, d + angle/2] 所指示的那片区域。

```python
# 官解
class Solution:
    def visiblePoints(self, points: List[List[int]], angle: int, location: List[int]) -> int:
        sameCnt = 0
        polarDegrees = []
        for p in points:
            if p == location:
                sameCnt += 1
            else:
                polarDegrees.append(atan2(p[1] - location[1], p[0] - location[0]))
        polarDegrees.sort()

        n = len(polarDegrees)
        polarDegrees += [deg + 2 * pi for deg in polarDegrees]

        degree = angle * pi / 180
        maxCnt = max((bisect_right(polarDegrees, polarDegrees[i] + degree) - i for i in range(n)), default=0)
        return maxCnt + sameCnt

```

```python
class Solution:
    def visiblePoints(self, points: List[List[int]], angle: int, location: List[int]) -> int:
        # 预处理points
        for i in range(len(points)):
            points[i][0] -= location[0]
            points[i][1] -= location[1]
        
        # 注意精度问题
        zeroCount = 0
        angleDict = collections.defaultdict(int)
        for x,y in points:
            if x == 0 and y == 0:
                zeroCount += 1
                continue 
            if x == 0 and y > 0:
                angleDict[90] += 1
                angleDict[450] += 1
                continue 
            elif x == 0 and y < 0:
                angleDict[-90] += 1
                angleDict[270] += 1
                continue               
            ag = atan2(y,x) # atan2会根据x,y的符号返回-pai,pai的象限角,不能简单的用atan，因为它对(-,-)返回的值和++无法简单区分
            ag = math.degrees(ag)
            angleDict[ag] += 1
            angleDict[ag+360] += 1
        
        pairs = [[key,angleDict[key]] for key in angleDict]
        pairs.sort()
        
        maxSum = 0

        for index in range(len(pairs)):
            if pairs[index][0] > 180:
                break
            p = bisect.bisect_right(pairs,[angle+pairs[index][0],0xffffffff])
            tempSum = 0
            for i in range(index,p):
                tempSum += pairs[i][1]
            if tempSum > maxSum:
                maxSum = tempSum

        return maxSum + zeroCount
```

## [5959. 使数组 K 递增的最少操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-the-array-k-increasing/)【抄了个LIS模板】

给你一个下标从 0 开始包含 n 个正整数的数组 arr ，和一个正整数 k 。

如果对于每个满足 k <= i <= n-1 的下标 i ，都有 arr[i-k] <= arr[i] ，那么我们称 arr 是 K 递增 的。

比方说，arr = [4, 1, 5, 2, 6, 2] 对于 k = 2 是 K 递增的，因为：
arr[0] <= arr[2] (4 <= 5)
arr[1] <= arr[3] (1 <= 2)
arr[2] <= arr[4] (5 <= 6)
arr[3] <= arr[5] (2 <= 2)
但是，相同的数组 arr 对于 k = 1 不是 K 递增的（因为 arr[0] > arr[1]），对于 k = 3 也不是 K 递增的（因为 arr[0] > arr[3] ）。
每一次 操作 中，你可以选择一个下标 i 并将 arr[i] 改成任意 正整数。

请你返回对于给定的 k ，使数组变成 K 递增的 最少操作次数 。

```python
class Solution:
    def kIncreasing(self, arr: List[int], k: int) -> int:
        n = len(arr)
        # 读懂题意,分LIS 最长上升子序列
        tempList = []
        for start in range(k):
            lst = []
            for t in range(start,len(arr),k):
                lst.append(arr[t])
            tempList.append(lst)
        
        # 需要将tempList里面的每一组变成上升序列【可以相等,求总操作数】
        # 这里面的每一组互不干扰
        # 找到每一组中最长的LIS，结果为组全长-LIS，最终累加

        def lengthOfLIS(nums: [int]) -> int:
            tails, res = [0] * len(nums), 0
            for num in nums:
                i, j = 0, res
                while i < j:
                    m = (i + j) // 2
                    if tails[m] <= num: i = m + 1 # 如果要求非严格递增，将此行 '<' 改为 '<=' 即可。
                    else: j = m
                tails[i] = num
                if j == res: res += 1
            return res
        
        count = 0
        for lst in tempList:
            count += len(lst)-lengthOfLIS(lst)
        return count
```

# 第四周：12.20 - 12.26 【四题】

## [1547. 切棍子的最小成本](https://leetcode-cn.com/problems/minimum-cost-to-cut-a-stick/)【区间dp】

```python
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        # 区间dp模版题
        # 先根据切点分组
        cuts += [0,n]
        cuts.sort()
        value = []
        p = 1
        while p < len(cuts):
            value.append(cuts[p]-cuts[p-1])
            p += 1
        # 然后区间dp，当作石子合并问题
        # dp[i][j]表示闭区间合并【i～j】的最小成本
        # dp[i][i] = 0
        n = len(value) # 这里改变了n的赋值，习惯
        dp = [[0xffffffff for j in range(n)] for i in range(n)]
        for i in range(n):
            dp[i][i] = 0 # base

        # 状态转移: dp[i][j] = min(dp[i][j],dp[i][k]+dp[k+1][j]+sum(i~j)) k从i ~ j-1的闭区间

        for length in range(2,n+1): # 长度是[2,n]的闭区间
            i = 0
            j = i+length-1
            # print(i,j,length)
            while j < n:
                for k in range(i,j):
                    dp[i][j] = min(dp[i][j],dp[i][k]+dp[k+1][j]+sum(value[i:j+1]))
                i += 1
                j += 1
        return dp[0][-1]
```

```python
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        # 区间dp模版题
        # 先根据切点分组
        cuts += [0,n]
        cuts.sort()
        value = []
        p = 1
        while p < len(cuts):
            value.append(cuts[p]-cuts[p-1])
            p += 1
        # 然后区间dp，当作石子合并问题
        # dp[i][j]表示闭区间合并【i～j】的最小成本
        # dp[i][i] = 0
        n = len(value) # 这里改变了n的赋值，习惯
        dp = [[0xffffffff for j in range(n)] for i in range(n)]
        for i in range(n):
            dp[i][i] = 0 # base

        # 状态转移: dp[i][j] = min(dp[i][j],dp[i][k]+dp[k+1][j]+sum(i~j)) k从i ~ j-1的闭区间

        # 前缀和优化sum
        preSum = [0 for i in range(n+1)]
        pre = 0
        for i in range(n):
            pre += value[i]
            preSum[i+1] = pre 

        for length in range(2,n+1): # 长度是[2,n]的闭区间
            i = 0
            j = i+length-1
            # print(i,j,length)
            while j < n:
                for k in range(i,j):
                    dp[i][j] = min(dp[i][j],dp[i][k]+dp[k+1][j]+preSum[j+1]-preSum[i]) # 注意这里
                i += 1
                j += 1
        return dp[0][-1]
            
```

## [1044. 最长重复子串](https://leetcode-cn.com/problems/longest-duplicate-substring/)【需要补强RK算法】

给你一个字符串 s ，考虑其所有 重复子串 ：即，s 的连续子串，在 s 中出现 2 次或更多次。这些出现之间可能存在重叠。

返回 任意一个 可能具有最长长度的重复子串。如果 s 不含重复子串，那么答案为 "" 。

```python
# 方法1: 纯暴力算法
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        ans = ""
        for i in range(len(s)): # 枚举起点
            while s[i:i+len(ans)+1] in s[i+1:]:
                ans = s[i:i+len(ans) + 1]
        return ans
```

## [1745. 回文串分割 IV](https://leetcode-cn.com/problems/palindrome-partitioning-iv/) 【区间dp+枚举断点】

给你一个字符串 s ，如果可以将它分割成三个 非空 回文子字符串，那么返回 true ，否则返回 false 。

当一个字符串正着读和反着读是一模一样的，就称其为 回文字符串 。

```python
class Solution:
    def checkPartitioning(self, s: str) -> bool:
        # 预处理，然后根据dp数组来考虑计数True，当前位置往后有多少个True的数量
        n = len(s)
        dp = [[False for j in range(n)] for i in range(n)]
        for i in range(n):
            dp[i][i] = True 
        for i in range(n-1):
            dp[i][i+1] = (s[i] == s[i+1])
        
        # 状态转移: dp[i][j] = dp[i+1][j-1] and s[i] == s[j] 掐头去尾
        for j in range(2,n):
            for i in range(j-1):
                dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
        
        # 然后直接双重for闭区间枚举
        for i in range(0,n-1):
            for j in range(i+1,n-1):
                if dp[0][i] == True and dp[i+1][j] == True and dp[j+1][n-1] == True:
                    return True 
        return False 
```

## [810. 黑板异或游戏](https://leetcode-cn.com/problems/chalkboard-xor-game/)【博弈论】

黑板上写着一个非负整数数组 nums[i] 。Alice 和 Bob 轮流从黑板上擦掉一个数字，Alice 先手。如果擦除一个数字后，剩余的所有数字按位异或运算得出的结果等于 0 的话，当前玩家游戏失败。 (另外，如果只剩一个数字，按位异或运算得到它本身；如果无数字剩余，按位异或运算结果为 0。）

并且，轮到某个玩家时，如果当前黑板上所有数字按位异或运算结果等于 0，这个玩家获胜。

假设两个玩家每步都使用最优解，当且仅当 Alice 获胜时返回 true。

```python
class Solution:
    def xorGame(self, nums: List[int]) -> bool:
        # 博弈论仔细分析
        # 如果是偶数个数字，Alice一定不会输，因为所有的nums大于0
        # 如果是奇数个数字，如果开场就是nums抑或和为0，则Alice必胜，如果不可以， 则递归到了B遇到了必胜情况
        if len(nums) % 2 == 0:
            return True 
        t = 0
        for n in nums:
            t ^= n 
        if t == 0:
            return True 
        else:
            return False 
```

# 第五周： 12.27 - 1.02 【五题】

## [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/) 【dp】

给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。

字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）

题目数据保证答案符合 32 位带符号整数范围。

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # 类似于编辑距离中的删除操作，加哨兵位
        m,n = len(s),len(t)
        if len(s) < len(t): # s更短直接不需要匹配
            return 0
        # dp[i][j]表示s的前i个和t的前j个匹配上的总数目
        # 状态转移为： 如果两者最后一个匹配上了
        # 那么，如果使用最后一个进行匹配 dp[i][j] += dp[i-1][j-1]
        # 如果不使用最后一个进行匹配 dp[i][j] += dp[i-1][j]
        # 合并为： dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
        # 如果两者最后一个没有匹配上，那么继承自dp[i-1][j]
        dp = [[0 for j in range(n+1)] for i in range(m+1)]
        # 注意初始化
        # dp[i][0]表示前i个最多可以和空串匹配上的次数，显然都为1
        # dp[0][j]表示不使用s，能和t匹配上的次数，显然都为0
        for i in range(m+1):
            dp[i][0] = 1
        
        for i in range(1,m+1):
            for j in range(1,n+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[-1][-1]
```

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # 记忆化搜索
        memo = dict()
        # 初始化

        def recur(i,j): # s的前i个和t的前j个
            if (i,j) in memo:
                return memo[(i,j)]
            if j == 0:
                return 1
            if i == 0:
                return 0
            if s[i-1] == t[j-1]:
                state1 = recur(i-1,j)
                state2 = recur(i-1,j-1)
                memo[(i,j)] = state1 + state2 
            else:
                state1 = recur(i-1,j)
                memo[(i,j)] = state1 
            return memo[(i,j)]
        
        return recur(len(s),len(t))
```

## [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/) 【dp + dfs】

给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。

说明：

分隔时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # 以[]代表False，如果匹配上，则填充序号
        n = len(s)
        dp = [[] for j in range(n+1)]
        dp[0] = [True]
        for i in range(1,n+1):
            for w in wordDict:
                if i - len(w) >= 0 and s[i-len(w):i] == w and dp[i-len(w)]:
                    dp[i].append(i-len(w))
        # print(dp)
        # 然后倒序搜索路径
        # 根据路径切割,以路径序号建图
        now = n 
        queue = [dp[-1]]
        graph = collections.defaultdict(list)
        for i in range(1,n+1):
            if len(dp[i]) >= 1:
                for each in dp[i]:
                    graph[i].append(each)
        # dfs搜序号
        pList = []
        def dfs(i,path):
            if i == 0:
                path.append(i)
                pList.append(path[:])
                path.pop()
                return 
            path.append(i)
            for neigh in graph[i]:
                dfs(neigh,path)
            path.pop()
        
        dfs(n,[])
        # print(pList)
        ans = []
        for e in pList:
            e = e[::-1]
            temp = []
            for i in range(1,len(e)):
                temp.append(s[e[i-1]:e[i]])
            ans.append(" ".join(temp))
        return ans
```

## [472. 连接词](https://leetcode-cn.com/problems/concatenated-words/) 【参照了题解】

给你一个 不含重复 单词的字符串数组 words ，请你找出并返回 words 中的所有 连接词 。

连接词 定义为：一个完全由给定数组中的至少两个较短单词组成的字符串。

```python
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.isEnd = False 

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,word):
        node = self.root 
        for ch in word:
            if node.children.get(ch) == None:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.isEnd = True 
    
    def dfs(self,word,index):
        if index == len(word):
            return True 
        node = self.root 
        for i in range(index,len(word)):
            node = node.children.get(word[i])
            if node == None:
                return False 
            if node.isEnd and self.dfs(word,i+1):
                return True 
        return False 
    

class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words.sort(key = len)
        ans = []
        theTrie = Trie()
        for w in words:
            if w == "":
                continue 
            if theTrie.dfs(w,0):
                ans.append(w)
            else:
                theTrie.insert(w)
        return ans
```

## [164. 最大间距 ](https://leetcode-cn.com/problems/maximum-gap/) 「桶排序」

给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。

如果数组元素个数小于 2，则返回 0。

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        # 1.排序做法
        nums.sort()
        if len(nums) < 2:
            return 0 
        
        maxGap = 0
        for p in range(1,len(nums)):
            if nums[p] - nums[p-1] > maxGap:
                maxGap = nums[p] - nums[p-1]
        return maxGap
```

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        # 桶排序
        # 桶排序需要确定桶长度和桶数量,这里假设桶数量为元素个数
        if len(nums) < 2:
            return 0
        n = len(nums)
        maxN,minN = max(nums),min(nums)
        # 画图,n个点,其中间隔为(maxN - minN) / (n - 1)
        # 例如1,3,5,7 ;需要gap为2
        # 例如1,3,5,8 ;需要gap也为2
        # 注意桶排序区间的最右端可取值不要求是数组中存在的真实最大值,所以导致了桶数量需要+1
        # 数放到一个个桶里面，不断更新更大的（后一个桶内元素的最小值 - 前一个桶内元素的最大值），最后就得到了答案。
        # 确定数的索引的方法: (n - minN) / d
        d = max(1,math.floor((maxN - minN) / (n - 1))) 
        cnt = (maxN-minN)//d + 1 # 
        bucket = [[] for i in range(cnt)]

        for n in nums:
            loc = (n-minN)//d # 向下取整数就行
            bucket[loc].append(n)
        # print(d , cnt)
        # print(bucket)
        # 将每个桶更新为桶中最大最小值,比较相邻桶隔壁的最大的和本地的最小的比
        for i in range(len(bucket)):
            if len(bucket[i]) > 0:
                bmax = max(bucket[i])
                bmin = min(bucket[i])
                bucket[i] = [bmin,bmax]
            else:
                bucket[i] = [bucket[i-1][1],bucket[i-1][1]]
        # print(bucket)
        ans = 0
        for i in range(1,len(bucket)):
            pre = bucket[i-1][1]
            now = bucket[i][0]
            if pre != None and now != None:
                ans = max(ans,now-pre)
        return ans
           
```

## [390. 消除游戏 ](https://leetcode-cn.com/problems/elimination-game/)

列表 arr 由在范围 [1, n] 中的所有整数组成，并按严格递增排序。请你对 arr 应用下述算法：

从左到右，删除第一个数字，然后每隔一个数字删除一个，直到到达列表末尾。
重复上面的步骤，但这次是从右到左。也就是，删除最右侧的数字，然后剩下的数字每隔一个删除一个。
不断重复这两步，从左到右和从右到左交替进行，直到只剩下一个数字。
给你整数 n ，返回 arr 最后剩下的数字。

```python
class Solution:
    def lastRemaining(self, n: int) -> int:
        # 参照三叶的题解
        # 对称性 f(n) + f'(n) = n + 1
        # 递归性 f(n) = f'(n//2) * 2
        # base : f(1) = f'(1) = 1 
        # 解函数方程得到递推关系: f(n) = (n//2 + 1 - f(n//2)) * 2
        # 
        
        def recur(n):
            if n == 1:
                return 1 
            else:
                return (n//2 + 1 - recur(n//2)) * 2
        
        return recur(n)
```

# 第六周 01.02 - 01.09 【五题】

## [381. O(1) 时间插入、删除和获取随机元素 - 允许重复](https://leetcode-cn.com/problems/insert-delete-getrandom-o1-duplicates-allowed/)

设计一个支持在平均 时间复杂度 O(1) 下， 执行以下操作的数据结构。

注意: 允许出现重复元素。

insert(val)：向集合中插入元素 val。
remove(val)：当 val 存在时，从集合中移除一个 val。
getRandom：从现有集合中随机获取一个元素。每个元素被返回的概率应该与其在集合中的数量呈线性相关。

```python
# 哈希表+数组
# 数组存元素,哈希表存索引,但是存的是索引list
class RandomizedCollection:

    def __init__(self):
        self.arr = []
        self.record = collections.defaultdict(list)

    def insert(self, val: int) -> bool:
        # 重复的返回False
        if self.record[val] == []:
            self.arr.append(val)
            self.record[val].append(len(self.arr)-1) # 加入一个索引
            return True
        else:
            self.arr.append(val)
            self.record[val].append(len(self.arr)-1) # 加入一个索引
            return False             

    def remove(self, val: int) -> bool:
        if self.record[val] == []:
            return False 

        elif self.record[val] != []:
            index = self.record[val].pop()
            # 将最后一个元素换到它的位置上
            last = self.arr[-1] # 找到最后一个元素
            self.arr[index] = last
            # print('remove index = ',len(self.arr)-1)
            # print(self.record[last])
            # self.printNow()
            self.record[last].append(index) # 注意这两条的先后顺序
            self.record[last].remove(len(self.arr)-1) # 移除它
            self.arr.pop()
           
            return True 


    def getRandom(self) -> int:
        index = random.randint(0,len(self.arr)-1)
        return self.arr[index]

    def printNow(self): # 打印检查用的
        print('self.record = ',self.record)
        print('self.arr = ',self.arr)
        return 

```

## [2117. 一个区间内所有数乘积的缩写](https://leetcode-cn.com/problems/abbreviating-the-product-of-a-range/)【难度降低了，变成了纯阅读理解题】

给你两个正整数 left 和 right ，满足 left <= right 。请你计算 闭区间 [left, right] 中所有整数的 乘积 。

由于乘积可能非常大，你需要将它按照以下步骤 缩写 ：

统计乘积中 后缀 0 的数目，并 移除 这些 0 ，将这个数目记为 C 。
比方说，1000 中有 3 个后缀 0 ，546 中没有后缀 0 。
将乘积中剩余数字的位数记为 d 。如果 d > 10 ，那么将乘积表示为 <pre>...<suf> 的形式，其中 <pre> 表示乘积最 开始 的 5 个数位，<suf> 表示删除后缀 0 之后 结尾的 5 个数位。如果 d <= 10 ，我们不对它做修改。
比方说，我们将 1234567654321 表示为 12345...54321 ，但是 1234567 仍然表示为 1234567 。
最后，将乘积表示为 字符串 "<pre>...<suf>eC" 。
比方说，12345678987600000 被表示为 "12345...89876e5" 。
请你返回一个字符串，表示 闭区间 [left, right] 中所有整数 乘积 的 缩写 。

```python
class Solution:
    def abbreviateProduct(self, left: int, right: int) -> str:
        # py纯暴力算一下
        k = 1
        for i in range(left,right+1):
            k *= i 
        
        k = str(k)
        p = len(k)-1
        cntZero = 0
        while p >= 0:
            if k[p] == "0":
                p -= 1
                cntZero += 1
            else:
                p += 1
                break 
        if p <= 10:
            origin = k[:p] + 'e' + str(cntZero)
            return origin
        else:
            pre = k[:5]
            suf = k[p-5:p]
            return pre + '...' + suf + 'e' + str(cntZero)
```

## [489. 扫地机器人](https://leetcode-cn.com/problems/robot-room-cleaner/)【API理解】+ DFS，参照了题解

房间（用格栅表示）中有一个扫地机器人。格栅中的每一个格子有空和障碍物两种可能。

扫地机器人提供4个API，可以向前进，向左转或者向右转。每次转弯90度。

当扫地机器人试图进入障碍物格子时，它的碰撞传感器会探测出障碍物，使它停留在原地。

请利用提供的4个API编写让机器人清理整个房间的算法。

```python
# """
# This is the robot's control interface.
# You should not implement it, or speculate about its implementation
# """
#class Robot:
#    def move(self):
#        """
#        Returns true if the cell in front is open and robot moves into the cell.
#        Returns false if the cell in front is blocked and robot stays in the current cell.
#        :rtype bool
#        """
#
#    def turnLeft(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def turnRight(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def clean(self):
#        """
#        Clean the current cell.
#        :rtype void
#        """

class Solution:
    def __init__(self):
        self.visited = set()
        self.direc = [(-1,0),(0,1),(1,0),(0,-1)] # 默认向上
    
    def cleanRoom(self, robot):
        """
        :type robot: Robot
        :rtype: None
        """
        def dfs(now_i,now_j,di): # di表示direc,嵌套在里面无需再传递robot
            self.visited.add((now_i,now_j))
            robot.clean()
            # print(now_i,now_j)
            for i in range(4):
                new_i = now_i + self.direc[(di+i)%4][0]
                new_j = now_j + self.direc[(di+i)%4][1]
                if (new_i,new_j) not in self.visited and robot.move():
                    dfs(new_i,new_j,(di+i)%4)
                    self.goBack(robot)
                robot.turnRight()
        
        dfs(0,0,0)
    
    def goBack(self,robot): # 返回  
        # 先转180
        robot.turnRight()
        robot.turnRight()
        # 回撤一步
        robot.move()
        # 再转180
        robot.turnRight()
        robot.turnRight()

        
```

```go
/**
 * // This is the robot's control interface.
 * // You should not implement it, or speculate about its implementation
 * type Robot struct {
 * }
 * 
 * // Returns true if the cell in front is open and robot moves into the cell.
 * // Returns false if the cell in front is blocked and robot stays in the current cell.
 * func (robot *Robot) Move() bool {}
 *
 * // Robot will stay in the same cell after calling TurnLeft/TurnRight.
 * // Each turn will be 90 degrees.
 * func (robot *Robot) TurnLeft() {}
 * func (robot *Robot) TurnRight() {}
 *
 * // Clean the current cell.
 * func (robot *Robot) Clean() {}
 */

// 清扫算法
func cleanRoom(robot *Robot) {
    direc := make([][]int,4,4)
    
    direc[0] = []int{-1,0}
    direc[1] = []int{0,1}
    direc[2] = []int{1,0}
    direc[3] = []int{0,-1}
    memo := make(map[string]int)
    start_i,start_j := 0,0
    originDirec := 0
    dfs(robot,start_i,start_j,originDirec,memo,direc)   
}

// 注意参数传入，需要memo记忆已经去过的地方，direc方向数组
func dfs(robot *Robot,now_i,now_j,di int,memo map[string]int,direc [][]int) {
    memo[toKey(now_i,now_j)] = 1
    robot.Clean()
    for i:=0;i<4;i++ {
        new_i := now_i + direc[(di+i)%4][0]
        new_j := now_j + direc[(di+i)%4][1]
        _,ok := memo[toKey(new_i,new_j)]
        if !ok && robot.Move() == true {
            dfs(robot,new_i,new_j,(di+i)%4,memo,direc)
            goBack(robot)
        }
        robot.TurnRight()
        
    }
}
// 将坐标转化成key给map使用
func toKey(x,y int) string {
    return strconv.Itoa(x)+","+strconv.Itoa(y)
}
// 回退函数
func goBack(robot *Robot) {
    robot.TurnRight()
    robot.TurnRight()
    robot.Move()
    robot.TurnRight()
    robot.TurnRight()
}

```

## [1269. 停在原地的方案数](https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/) dp + 边界限制

有一个长度为 arrLen 的数组，开始有一个指针在索引 0 处。

每一步操作中，你可以将指针向左或向右移动 1 步，或者停在原地（指针不能被移动到数组范围外）。

给你两个整数 steps 和 arrLen ，请你计算并返回：在恰好执行 steps 次操作以后，指针仍然指向索引 0 处的方案数。

由于答案可能会很大，请返回方案数 模 10^9 + 7 后的结果。

```python
class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        # dp[i][j]表示运行i步时，到j的方案数
        # dp[0][0] = 1 其他为0
        
        # 状态转移 dp[i][j] = (dp[i-1][j-1]+dp[i-1][j+1]+dp[i-1][j]) % mod
        # 初始化为
        arrLen = min(arrLen,500) # 只需要长度500以内即可, min(arrLen,steps)也可以
        dp = [[0 for j in range(arrLen)] for i in range(steps+1)]
        dp[0][0] = 1 
        mod = 10**9 + 7
        for i in range(1,steps+1):
            for j in range(arrLen):
                state1 = dp[i-1][j-1] if j >= 1 else 0
                state2 = dp[i-1][j]
                state3 = dp[i-1][j+1] if j < arrLen-1 else 0 
                dp[i][j] = (state1+state2+state3)%mod
        
        # print(dp)
        return dp[steps][0]
```

## [5979. 全部开花的最早一天](https://leetcode-cn.com/problems/earliest-possible-day-of-full-bloom/)【单周赛，greedy】

你有 n 枚花的种子。每枚种子必须先种下，才能开始生长、开花。播种需要时间，种子的生长也是如此。给你两个下标从 0 开始的整数数组 plantTime 和 growTime ，每个数组的长度都是 n ：

plantTime[i] 是 播种 第 i 枚种子所需的 完整天数 。每天，你只能为播种某一枚种子而劳作。无须 连续几天都在种同一枚种子，但是种子播种必须在你工作的天数达到 plantTime[i] 之后才算完成。
growTime[i] 是第 i 枚种子完全种下后生长所需的 完整天数 。在它生长的最后一天 之后 ，将会开花并且永远 绽放 。
从第 0 开始，你可以按 任意 顺序播种种子。

返回所有种子都开花的 最早 一天是第几天。

```python
class Solution:
    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        n = len(plantTime)
        # 播种需要消耗一天，之后不用管
        # 一盆花开花的那一天是,序号从0开始 plantTime[i] + growTime[i] + start
        # 每天只能做一次灰花盆，灰花盆需要做满要求，之后自然生长
        # 最晚开花的最先种
        pair = [(plantTime[i],growTime[i]) for i in range(n)]
        pair.sort(key = lambda x:-x[1]) # 按照进盆子开花排序
        # print(pair)
        # 排序整理花盆，记录到目前为止的end时间
        end = 0
        now = 0 # 记录种灰色花盆所花的时间
        for i in range(n):
            now += pair[i][0]
            end = max(end,now+pair[i][1])
        return end

    # 证明，由于灰色花盆是可以随机调整的，那么只要使得花盆长好的越慢的越先被护理好，则一定更短。
```

# 第七周：01.10 - 01.16 【七题】

## [1579. 保证图可完全遍历](https://leetcode-cn.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/) 【需要使用优化版本并查集 + greedy】

Alice 和 Bob 共有一个无向图，其中包含 n 个节点和 3  种类型的边：

类型 1：只能由 Alice 遍历。
类型 2：只能由 Bob 遍历。
类型 3：Alice 和 Bob 都可以遍历。
给你一个数组 edges ，其中 edges[i] = [typei, ui, vi] 表示节点 ui 和 vi 之间存在类型为 typei 的双向边。请你在保证图仍能够被 Alice和 Bob 完全遍历的前提下，找出可以删除的最大边数。如果从任何节点开始，Alice 和 Bob 都可以到达所有其他节点，则认为图是可以完全遍历的。

返回可以删除的最大边数，如果 Alice 和 Bob 无法完全遍历图，则返回 -1 。

```python
class UF: # 
  # 优化版本并查集，优化并查集
    def __init__(self, n: int):
        self.n = n
        self.rank = [1] * n
        self.f = list(range(n))
    
    def find(self, x: int) -> int:
        if self.f[x] == x:
            return x
        self.f[x] = self.find(self.f[x]) # 递归
        return self.f[x]
    
    def union(self, x: int, y: int):
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return
        if self.rank[fx] < self.rank[fy]:
            fx, fy = fy, fx
        self.rank[fx] += self.rank[fy]
        self.f[fy] = fx
    
    def isConnected(self,x,y):
    	return self.find(x) == self.find(y)

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # 并查集 + 贪心 ，优先考虑各自可以删的，再考虑公共可以删的
        # 需要用优化并查集

        commonEdge = []
        aliceEdge = []
        bobEdge = []
        for ty,u,v in edges:
            if ty == 3:
                commonEdge.append((u-1,v-1))
            elif ty == 1:
                aliceEdge.append((u-1,v-1))
            elif ty == 2:
                bobEdge.append((u-1,v-1))

        ufset1 = UF(n)
        ufset2 = UF(n)

        cnt = 0 # 记录可以删除的边数

        # 先链接公共edge

        for u,v in commonEdge:
            if ufset1.isConnected(u,v) == False:
                ufset1.union(u,v)
                ufset2.union(u,v)
            else:
                cnt += 1

        for u,v in aliceEdge:
            if ufset1.isConnected(u,v) == False:
                ufset1.union(u,v)
            else:
                cnt += 1
        for u,v in bobEdge:
            if ufset2.isConnected(u,v) == False:
                ufset2.union(u,v)
            else:
                cnt += 1
        
        root1 = set()
        for i in range(n):
            root1.add(ufset1.find(i))
        root2 = set()
        for i in range(n):
            root2.add(ufset2.find(i))
        
        # print(root1,root2)
        if len(root1) == 1 and len(root2) == 1:
            return cnt 
        else:
            return -1

```

## [642. 设计搜索自动补全系统](https://leetcode-cn.com/problems/design-search-autocomplete-system/) 【细节Trie】

为搜索引擎设计一个搜索自动补全系统。用户会输入一条语句（最少包含一个字母，以特殊字符 '#' 结尾）。除 '#' 以外用户输入的每个字符，返回历史中热度前三并以当前输入部分为前缀的句子。下面是详细规则：

一条句子的热度定义为历史上用户输入这个句子的总次数。
返回前三的句子需要按照热度从高到低排序（第一个是最热门的）。如果有多条热度相同的句子，请按照 ASCII 码的顺序输出（ASCII 码越小排名越前）。
如果满足条件的句子个数少于 3，将它们全部输出。
如果输入了特殊字符，意味着句子结束了，请返回一个空集合。

```python
# 带权重的字典树
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(27)]
        self.val = 0 
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,word,weight): # 注意空格ascii码要小，这里和常规Trie的索引有区别
        node = self.root 
        for ch in word:
            if ch != " ":
                index = ord(ch)-ord('a')+1
            else:
                index = 0 
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.val += weight  #注意这里是+=
    
    def search(self,prefix):
        # 由于需要获取前3个字典序的prefix，所以需要用dfs比较好
        # 先移动前缀,注意前缀是累加前缀
        node = self.root 
        for ch in prefix:
            if ch != " ":
                index = ord(ch)-ord('a')+1
            else:
                index = 0 
            if node.children[index] == None:
                return [] # 没有这样的前缀
            node = node.children[index]
        # 移动到了前缀节点,由于需要频率排序，所以需要搜完全部的
        ansList = []
        def dfs(node,path): # dfs+回溯
            if node.val != 0:
                ansList.append((-node.val,prefix + "".join(path))) # 这里排序用到了负权                   
            for i in range(27):
                if node.children[i] != None:
                    if i == 0:
                        path.append(" ")                    
                    else:
                        path.append(chr(ord('a')+i-1))
                    dfs(node.children[i],path)
                    path.pop()

        dfs(node,[])
        # 整理
        ansList.sort()
        ansList = ansList[:3]
        final = [e[1] for e in ansList]
        return final


class AutocompleteSystem:

    def __init__(self, sentences: List[str], times: List[int]):
        self.tree = Trie()
        n = len(sentences)
        self.buf = [] # 缓存性质的存前缀
        for i in range(n):
            self.tree.insert(sentences[i],times[i])
        
    def input(self, c: str) -> List[str]: # 输入数据也会对整体缓存造成影响
        state = False
        if c[-1] == "#":
            c = c[:-1]
            state = True
        self.buf.append(c)
        pre = "".join(self.buf)
        if not state:            
            ans = self.tree.search(pre)
        if state == True:
            self.buf = [] # 清空
            self.tree.insert(pre,1)
            ans = []
        return ans
```

## [1036. 逃离大迷宫](https://leetcode-cn.com/problems/escape-a-large-maze/) 【限制扩散次数的BFS】【另外：其他人有离散化、限制扩散面积的BFS方法】

在一个 106 x 106 的网格中，每个网格上方格的坐标为 (x, y) 。

现在从源方格 source = [sx, sy] 开始出发，意图赶往目标方格 target = [tx, ty] 。数组 blocked 是封锁的方格列表，其中每个 blocked[i] = [xi, yi] 表示坐标为 (xi, yi) 的方格是禁止通行的。

每次移动，都可以走到网格中在四个方向上相邻的方格，只要该方格 不 在给出的封锁列表 blocked 上。同时，不允许走出网格。

只有在可以通过一系列的移动从源方格 source 到达目标方格 target 时才返回 true。否则，返回 false。

```python
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        # 这一题的难度在找到突破口
        # 由于blocked的长度有限，那么转换成集合快速查询
        # 每次从起点扩散，终点扩散，同时扩散200次，如果直到最后一次，其扩散区域仍旧可以变大，不被限定，那么最终可以联通
        blocked = set(tuple(e) for e in blocked)

        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        limit = 10**6
        # 注意点的坐标只能去到999999。所以limit用的符号是严格小于

        # 对起点
        t = 0 
        queue = [tuple(source)]
        while t <= 200 and len(queue) != 0:
            new_queue = []
            for x,y in queue:
                for di in direc:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if 0<=new_x<limit and 0<=new_y<limit and (new_x,new_y) not in blocked:
                        blocked.add((new_x,new_y))
                        new_queue.append((new_x,new_y))
            t += 1
            queue = new_queue
        # 注意有可能不到200次就退出了，这时候如果终点在范围内，是可以的
        if tuple(target) in blocked:
            return True 
        if len(queue) == 0:
            return False 
        
        # print(tuple(target) in blocked)
        t = 0 
        queue = [tuple(target)]
        while t <= 200 and len(queue) != 0:
            new_queue = []
            for x,y in queue:
                for di in direc:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if 0<=new_x<limit and 0<=new_y<limit and (new_x,new_y) not in blocked:
                        blocked.add((new_x,new_y))
                        new_queue.append((new_x,new_y))
            t += 1
            queue = new_queue
        
        if len(queue) != 0:
            return True 
        else:
            return False
```

## [1312. 让字符串成为回文串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/) 【经典回文串方法：掐头去尾dp】【倒着思考，找最长回文子序列】

给你一个字符串 s ，每一次操作你都可以在字符串的任意位置插入任意字符。

请你返回让 s 成为回文串的 最少操作次数 。

「回文串」是正读和反读都相同的字符串。

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        # 闭区间dp, dp[i][j] 表示闭区间内的成为回文串的最少插入次数
        n = len(s)
        dp = [[0xffffffff for j in range(n)] for i in range(n)] # 初始化为极大值
       	for i in range(n):
            dp[i][i] = 0
       
        for i in range(n-1): # 初始状态转移
            dp[i][i+1] = 1 if s[i] != s[i+1] else 0
        
        # 考虑掐头去尾
        # if s[i] == s[j]: dp[i][j] = dp[i+1][j-1]
        # elif s[i] != s[j]: dp[i][j] = min(dp[i][j-1],dp[i+1][j]) + 1
        # 
        for j in range(2,n): # 画图得到扫描顺序
            for i in range(j-2,-1,-1):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = min(dp[i][j-1],dp[i+1][j]) + 1
        
        return dp[0][n-1]
```

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        # dp,先找到最长回文子序列
        n = len(s)
        dp = [[0 for j in range(n)] for i in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n-1):
            dp[i][i+1] = 2 if s[i] == s[i+1] else 1 
        
        # 状态转移
        # if s[i] == s[j]: dp[i][j] = dp[i+1][j-1] + 2
        # elif s[i] != s[j]: dp[i][j] = max(dp[i+1][j],dp[i][j-1])
        for j in range(2,n):
            for i in range(j-2,-1,-1):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j],dp[i][j-1])
        
        return n - dp[0][n-1]
```

## [2081. k 镜像数字的和](https://leetcode-cn.com/problems/sum-of-k-mirror-numbers/) 【进制转化，暴力枚举，在超时的边缘】

一个 k 镜像数字 指的是一个在十进制和 k 进制下从前往后读和从后往前读都一样的 没有前导 0 的 正 整数。

比方说，9 是一个 2 镜像数字。9 在十进制下为 9 ，二进制下为 1001 ，两者从前往后读和从后往前读都一样。
相反地，4 不是一个 2 镜像数字。4 在二进制下为 100 ，从前往后和从后往前读不相同。
给你进制 k 和一个数字 n ，请你返回 k 镜像数字中 最小 的 n 个数 之和 。

```python
class Solution:
    def kMirror(self, k: int, n: int) -> int:
        # 枚举十进制的数的前半边，转成k进制，检测k进制是否回文
        def toK(number:str,k:int):
            lst = []
            number = int(number)
            while number:
                remain = number%k 
                number //= k 
                lst.append(remain)
            return lst # 返回的是数组形式表示的，左低位，右高位，不必转换回数值数字
        
        def check(lst):
            left = 0
            right = len(lst)-1
            while left < right:
                if lst[left] != lst[right]:
                    return False 
                else:
                    left += 1
                    right -= 1
            return True 

        lst = []
        limit = min(30,n)
        gapLst = ["0","1","2",'3','4','5','6','7','8','9']

        for i in range(1,10):
            number = str(i)
            v = toK(number,k)
            if v == v[::-1]:
                lst.append(int(number))

        for length in range(1,10):
            if len(lst) >= limit: break  
            for i in range(10**(length-1),10**length): # 构造全长为偶数的
                number = str(i) + str(i)[::-1]
                # print(number)
                v = toK(number,k)
                if check(v):
                    lst.append(int(number))
                    if len(lst) >= limit: break  
            for i in range(10**(length-1),10**length):
                for gap in gapLst:
                    number = str(i) + gap + str(i)[::-1] # 构造全长为奇数的
                    # print(number)
                    v = toK(number,k)
                    if check(v):
                        lst.append(int(number))
                        if len(lst) >= limit: break  
                
        return sum(lst[:n])
```

## [906. 超级回文数](https://leetcode-cn.com/problems/super-palindromes/) 【折半枚举，注意分析复杂度】

如果一个正整数自身是回文数，而且它也是一个回文数的平方，那么我们称这个数为超级回文数。

现在，给定两个正整数 L 和 R （以字符串形式表示），返回包含在范围 [L, R] 中的超级回文数的数目。

```python
class Solution:
    def superpalindromesInRange(self, left: str, right: str) -> int:
        # 数据范围在其他语言的longlong
        # 枚举回文数，然后截断，找长度
        # 枚举短的，判断长的，
        # 一个数的平方根的长度下限在 len(本身)//2+1 ,
        # 


        # print("len(left) = ",len(left),"len(right) = ",len(right))
        # 构造长度为k的回文数，最多需要k//2 + (1)位数
        # 枚举回文数到 k//2+(1) , 用它的平方去判断超级回文数
        

        gap = '0123456789'
        lst = [1,2,3,4,5,6,7,8,9]
        for t in range(1,10):
            for i in range(ceil(10**(t-1)),10**t):
                number = int(str(i)+str(i)[::-1])
                lst.append(number)
            for i in range(ceil(10**(t-1)),10**t):
                for g in gap:
                    number = int(str(i)+g+str(i)[::-1])
                    lst.append(number)
            if t*4+1 > len(right):
                break
        
        start = bisect_left(lst,int(sqrt(int(left))))
        end = bisect_right(lst,int(sqrt(int(right))))  
        
        cnt = 0
        for i in range(start,end):
            temp = lst[i]**2
            if str(temp) == str(temp)[::-1]:
                cnt += 1
        return cnt

```

## [778. 水位上升的泳池中游泳](https://leetcode-cn.com/problems/swim-in-rising-water/) 【并查集做法不难】

在一个 N x N 的坐标方格 grid 中，每一个方格的值 grid[i][j] 表示在位置 (i,j) 的平台高度。

现在开始下雨了。当时间为 t 时，此时雨水导致水池中任意位置的水位为 t 。你可以从一个平台游向四周相邻的任意一个平台，但是前提是此时水位必须同时淹没这两个平台。假定你可以瞬间移动无限距离，也就是默认在方格内部游动是不耗时的。当然，在你游泳的时候你必须待在坐标方格里面。

你从坐标方格的左上平台 (0，0) 出发。最少耗时多久你才能到达坐标方格的右下平台 (N-1, N-1)？

```python
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
        
    def isConnected(self,x,y):
        return self.find(x) == self.find(y)

class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        # 并查集，记录左上角和右下角是否联通
        # 记录每个点的上下左右，邻居的序号
        graph = collections.defaultdict(list)
        n = len(grid)
        for i in range(n):
            for j in range(n):
                now = grid[i][j]
                up =  grid[i-1][j] if i > 0 else None 
                down = grid[i+1][j] if i+1<n else None 
                left = grid[i][j-1] if j > 0 else None 
                right = grid[i][j+1] if j+1<n else None 
                graph[now] = [up,down,left,right]

        ufset = UF(n*n)
        start = grid[0][0]
        end = grid[-1][-1]


        for i in range(n**2):
            scanNode = i # 只需要扫描当前节点即可
            for neigh in graph[scanNode]:
                if neigh != None:
                    if neigh <= scanNode: # 小于等于才能连接
                        ufset.union(scanNode,neigh)
            if ufset.isConnected(start,end):
                return i 
```

# 第八周：01.17 - 01.23 【八题】+ 【四个学的题】

## [759. 员工空闲时间](https://leetcode-cn.com/problems/employee-free-time/) 【上下车算法】

给定员工的 schedule 列表，表示每个员工的工作时间。

每个员工都有一个非重叠的时间段  Intervals 列表，这些时间段已经排好序。

返回表示 所有 员工的 共同，正数长度的空闲时间 的有限时间段的列表，同样需要排好序。

```python
"""
# Definition for an Interval.
class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end
"""

class Solution:
    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        # 注意元素是个类，用属性访问而不是下标访问
        # 上下车算法
        up = collections.defaultdict(int)     
        for every in schedule:
            for s in every:
                up[s.start] += 1
                up[s.end] -= 1

        upList = [[0xffffffff,1]] # 初始化好边界
        for key in up:
            upList.append([key,up[key]])
        upList.sort()

        # print(upList)

        ans = []
        now = 0
        pre = None
        for i in range(len(upList)):
            if pre != None:
                ans.append([pre,upList[i][0]])
                pre = None
            now += upList[i][1]
            if now == 0 and pre == None:
                pre = upList[i][0]

        ans.pop()

        final = []
        for a,b in ans:
            final.append(Interval(a,b))
        return final
```

## [336. 回文对](https://leetcode-cn.com/problems/palindrome-pairs/) 【字典查询:通过】【字典树查询：我写的丐版字典树超时】

给定一组 **互不相同** 的单词， 找出所有 **不同** 的索引对 `(i, j)`，使得列表中的两个单词， `words[i] + words[j]` ，可拼接成回文串。

```python
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.isEnd = None
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,word,index):
        node = self.root
        for ch in word:
            if node.children.get(ch) == None:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.isEnd = index 
    
    def find(self,prefix):
        node = self.root 
        for ch in prefix:
            if node.children.get(ch) == None:
                return None
            node = node.children[ch]
        return node.isEnd

class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        # 分析，需要对于两个字符串
        # s1,s2, 假设s1长度大于等于s2，那么切割s1的前缀长度,然后判断切割点之后的是不是回文串
        # 注意可以有空串
        # 用字典树，先全插入，再查询
        tree = Trie()
        for i,w in enumerate(words):
            tree.insert(w,i)
        # 然后查询,查询每个单词时逆序，枚举它的每一种切割，当切割在字典树中可以被找到，且剩下的部分回文的时候，则加入两者索引
        ans = []
        def judge(s):
            left = 0
            right = len(s)-1
            while left < right:
                if s[left] != s[right]:
                    return False 
                left += 1
                right -= 1
            return True 
        
        hasSpace = -1
        for i,w in enumerate(words):
            w = w[::-1] # 逆序查找，并且枚举每一种切割，包括空串
            if w == "": hasSpace = i
            for length in range(len(w)):
                # 它的索引是i，查找到的索引是index， 索引对顺序为[index,i]
                searchWord = w[:length]
                remain = w[length:]
                index = tree.find(searchWord)
                isP = judge(remain)
                if index != None and isP and index != i:
                    ans.append([index,i])
                # 它的索引是i，查找到的索引是index， 索引对顺序为[i,index]
                searchWord = w[length:]
                remain = w[:length]
                index = tree.find(searchWord)
                isP = judge(remain)
                if index != None and isP and index != i:
                    ans.append([i,index])                
        
        # 补丁，所有回文串都可以和空串组上,空串序号在头
        
        if hasSpace != -1:
            for i,w in enumerate(words):
                if w != "" and judge(w):
                    ans.append([i,hasSpace])

        return ans
```

```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        def getPreSuf(word):
            pre,suf = [],[]
            for i in range(len(word)+1):
                if word[:i] == word[:i][::-1]:  # 前缀是回文
                    pre.append(word[i:][::-1]) # 需要查找除了前缀的部分
                if word[i:] == word[i:][::-1]: # 后缀是回文
                    suf.append(word[:i][::-1]) # 需要查找出了后缀的部分
            return pre,suf 

        dataset = {w: i for i, w in enumerate(words)}

        ans = []
        for index,word in enumerate(words):
            pre,suf = getPreSuf(word)
            for p in pre:
                #p[::-1] != word 前缀判断中过滤掉单词本身与其他单词形成回文的情况，避免在后缀判断中重复
                if p in dataset and index != dataset[p] and p[::-1] != word:
                    ans.append([dataset[p],index])

            for s in suf:
                if s in dataset and index != dataset[s]:
                    ans.append([index,dataset[s]])
        return ans
```

## [214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/) 【RK算法】

给定一个字符串 ***s***，你可以通过在字符串前面添加字符将其转换为回文串。找到并返回可以用这种方式转换的最短回文串。

```python
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        # r-k算法
        # 哈希做法，找到最短的字符串s',使得s'+s是回文串，其中s'是原串s的后缀逆序
        # 把s分割成 s1 + s2 ， 那么显然 目标回文串 -》 s2[逆] + s1 + s2
        # 所以s1要求是回文串，所以等价于找到最长的s1，使得s1是回文串
        # 这一题就变成了找最长回文前缀
        n = len(s)
        base, mod = 131, 10**9 + 7
        left,right = 0,0
        mul = 1
        best = -1
        for i in range(n):
            left = (left * base + ord(s[i])) % mod 
            right = (right + mul*ord(s[i])) % mod 
            if left == right: # 哈希碰撞的时候再次检验
                if s[:i+1] == s[:i+1][::-1]:
                    best = i
            mul = (mul * base) % mod 
        
        add = ("" if best == n-1 else s[best+1:])
        return add[::-1] + s

```

## [1606. 找到处理最多请求的服务器](https://leetcode-cn.com/problems/find-servers-that-handled-most-number-of-requests/) 【不会】

你有 k 个服务器，编号为 0 到 k-1 ，它们可以同时处理多个请求组。每个服务器有无穷的计算能力但是 不能同时处理超过一个请求 。请求分配到服务器的规则如下：

第 i （序号从 0 开始）个请求到达。
如果所有服务器都已被占据，那么该请求被舍弃（完全不处理）。
如果第 (i % k) 个服务器空闲，那么对应服务器会处理该请求。
否则，将请求安排给下一个空闲的服务器（服务器构成一个环，必要的话可能从第 0 个服务器开始继续找下一个空闲的服务器）。比方说，如果第 i 个服务器在忙，那么会查看第 (i+1) 个服务器，第 (i+2) 个服务器等等。
给你一个 严格递增 的正整数数组 arrival ，表示第 i 个任务的到达时间，和另一个数组 load ，其中 load[i] 表示第 i 个请求的工作量（也就是服务器完成它所需要的时间）。你的任务是找到 最繁忙的服务器 。最繁忙定义为一个服务器处理的请求数是所有服务器里最多的。

请你返回包含所有 最繁忙服务器 序号的列表，你可以以任意顺序返回这个列表。

```python
class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        busy = [] # 是一个heap
        free = [i for i in range(k)]
        ans = [0 for i in range(k)]

        for i,start in enumerate(arrival):
            # 1.busy执行完入队free
            while busy and busy[0][0] <= start:
                _,index = heapq.heappop(busy)
                heappush(free,i+(index-i)%k)
            # 2.从free中取出一个server执行当前任务
            if len(free) != 0:
                index = heapq.heappop(free) % k 
                heapq.heappush(busy,(start+load[i],index))
                ans[index] += 1
        
        m = max(ans)
        final = []
        for i,v in enumerate(ans):
            if v == m:
                final.append(i)
        return final

```

## [218. 天际线问题](https://leetcode-cn.com/problems/the-skyline-problem/) 【不会】

城市的天际线是从远处观看该城市中所有建筑物形成的轮廓的外部轮廓。给你所有建筑物的位置和高度，请返回由这些建筑物形成的 天际线 。

每个建筑物的几何信息由数组 buildings 表示，其中三元组 buildings[i] = [lefti, righti, heighti] 表示：

lefti 是第 i 座建筑物左边缘的 x 坐标。
righti 是第 i 座建筑物右边缘的 x 坐标。
heighti 是第 i 座建筑物的高度。
天际线 应该表示为由 “关键点” 组成的列表，格式 [[x1,y1],[x2,y2],...] ，并按 x 坐标 进行 排序 。关键点是水平线段的左端点。列表中最后一个点是最右侧建筑物的终点，y 坐标始终为 0 ，仅用于标记天际线的终点。此外，任何两个相邻建筑物之间的地面都应被视为天际线轮廓的一部分。

注意：输出天际线中不得有连续的相同高度的水平线。例如 [...[2 3], [4 5], [7 5], [11 5], [12 7]...] 是不正确的答案；三条高度为 5 的线应该在最终输出中合并为一个：[...[2 3], [4 5], [12 7], ...]

```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # 优先级队列
        from sortedcontainers import SortedList
        ans = []
        # 预处理所有的点，为了方便排序，对于左端点，令高度为负，右端点令高度为正
        ps = []
        for l,r,h in buildings:
            ps.append([l,-h])
            ps.append([r,h])
        # 按照横坐标排序，如果横坐标相同，按照左端点排序，如果相同的左右端点，则按照高度进行排序
        ps.sort()

        pre = 0
        q = SortedList([pre]) # 有序列表充当大根堆
        for point,height in ps:
            if height < 0:
                # 如果是左端点，说明存在一条往右延伸可记录的边，将高度存入优先队列
                q.add(-height) # 转化为正数
            else:
                # 如果是右端点，说明这条边结束了，将高度从优先级队列移除,这里的remove优化成了lgn
                q.remove(height)
            cur = q[-1]
            if cur != pre:
                ans.append([point,cur])
                pre = cur 
        return ans   
```

## [862. 和至少为 K 的最短子数组](https://leetcode-cn.com/problems/shortest-subarray-with-sum-at-least-k/) 【不会】

给你一个整数数组 nums 和一个整数 k ，找出 nums 中和至少为 k 的 最短非空子数组 ，并返回该子数组的长度。如果不存在这样的 子数组 ，返回 -1 。

子数组 是数组中 连续 的一部分。

```python
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        preList = []
        pre = 0
        for i in range(n):
            preList.append(pre)
            pre += nums[i]
        preList.append(pre)
        
        ans = n+1 # 
        monoq = collections.deque()
        for y,preY in enumerate(preList):
            while monoq and preY <= preList[monoq[-1]]:
                monoq.pop()
            while monoq and preY - preList[monoq[0]] >= k:
                ans = min(ans,y-monoq.popleft())
            
            monoq.append(y)

        return ans if ans < n+1 else -1
```

## [358. K 距离间隔重排字符串](https://leetcode-cn.com/problems/rearrange-string-k-distance-apart/) 【排队问题】

### 子问题：[767. 重构字符串](https://leetcode-cn.com/problems/reorganize-string/)

给定一个字符串`S`，检查是否能重新排布其中的字母，使得两相邻的字符不同。

若可行，输出任意可行的结果。若不可行，返回空字符串。

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        # 计数
        ct = collections.Counter(s)
        # 大顶堆弹出
        ans = [] # 栈存储
        maxHeap = [[-ct[key],key] for key in ct]
        heapq.heapify(maxHeap)


        queue = [] # 里面存的是一系列last
        while maxHeap:
            times,element = heapq.heappop(maxHeap)
            ans.append(element)
            # 然后把这个元素丢到queue里面排队

            queue.append([times+1,element])
            if len(queue) == 2: # 这里是2
                times,element = queue.pop(0)
                if times != 0:
                    heapq.heappush(maxHeap,[times,element])
        
        ans = "".join(ans)
        return ans if len(ans) == len(s) else ""

```

给你一个非空的字符串 s 和一个整数 k，你要将这个字符串中的字母进行重新排列，使得重排后的字符串中相同字母的位置间隔距离至少为 k。

所有输入的字符串都由小写字母组成，如果找不到距离至少为 k 的重排结果，请返回一个空字符串 ""。

```python
class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        if k == 0:
            return s
        # 计数
        ct = collections.Counter(s)
        # 大顶堆弹出
        ans = [] # 栈存储
        maxHeap = [[-ct[key],key] for key in ct]
        heapq.heapify(maxHeap)


        queue = collections.deque() # 里面存的是一系列last
        while maxHeap:
            times,element = heapq.heappop(maxHeap)
            ans.append(element)
            # 然后把这个元素丢到queue里面排队

            queue.append([times+1,element])
            if len(queue) == k:
                times,element = queue.popleft()
                if times != 0:
                    heapq.heappush(maxHeap,[times,element])
        
        ans = "".join(ans)
        # print(ans)
        return ans if len(ans) == len(s) else ""
```

## [1392. 最长快乐前缀](https://leetcode-cn.com/problems/longest-happy-prefix/) 【RK字符串哈希算法】

「快乐前缀」是在原字符串中既是 非空 前缀也是后缀（不包括原字符串自身）的字符串。

给你一个字符串 s，请你返回它的 最长快乐前缀。

如果不存在满足题意的前缀，则返回一个空字符串。

```python
class Solution:
    def longestPrefix(self, s: str) -> str:
        ans = ""
        n = len(s)
        left = 0
        right = 0
        mod = 10**9 + 7
        base = 131
        for i in range(n-1):
            left = (left*base + ord(s[i]))%mod
            right = (pow(base,i,mod)*ord(s[n-i-1]) + right)%mod 
            # print(left,right,s[:i+1],s[n-i-1:])
            if left == right:
                if len(ans) < i+1:
                    ans = s[:i+1]
        return ans
```

## [924. 尽量减少恶意软件的传播](https://leetcode-cn.com/problems/minimize-malware-spread/) 【并查集，讨论毒点数目分类0，1，morethan2】

在节点网络中，只有当 graph[i][j] = 1 时，每个节点 i 能够直接连接到另一个节点 j。

一些节点 initial 最初被恶意软件感染。只要两个节点直接连接，且其中至少一个节点受到恶意软件的感染，那么两个节点都将被恶意软件感染。这种恶意软件的传播将继续，直到没有更多的节点可以被这种方式感染。

假设 M(initial) 是在恶意软件停止传播之后，整个网络中感染恶意软件的最终节点数。

如果从初始列表中移除某一节点能够最小化 M(initial)， 返回该节点。如果有多个节点满足条件，就返回索引最小的节点。

请注意，如果某个节点已从受感染节点的列表 initial 中删除，它以后可能仍然因恶意软件传播而受到感染。

```python
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
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        # 并查集方法
        # 最多有300个点，先计算有多少个联通分量
        # 获取到每个联通分量中具体的点，其中一个有毒则全区有毒，如果这个联通分量中有两个以及以上的毒点，则无论是否移除，不会改变毒点数目
        # 如果这个联通分量中只有一个毒点，那么尝试移除，可以救活这个集合
        # 如果这个联通分量没有毒点，无需考虑
        posion = set(initial)
        n = len(graph)
        ufset = UF(n)
        for i in range(n):
            for j in range(i+1,n):
                if graph[i][j] == 1:
                    ufset.union(i,j)
        
        memo = collections.defaultdict(list)
        for i in range(n):
            root = ufset.find(i)
            memo[root].append(i)
        
        # print(memo)

        moreThanTwo = [] # 存的是【代表元，对应元素】
        equalToOne = []
        other = [] # 其实不用考虑
        for key in memo:
            ts = memo[key]
            cnt = []
            for ele in ts:
                if ele in initial:
                    cnt.append(ele)
            if len(cnt) >= 2:
                moreThanTwo.append([key,cnt])
            elif len(cnt) == 1:
                equalToOne.append([key,cnt[0]])
        
        # print(equalToOne,moreThanTwo)
        # 检查每一个equalToOne的，找到其中的cnt
        maxSize = 0
        tempList = []
        for every in equalToOne:
            if maxSize < len(memo[every[0]]):
                maxSize = len(memo[every[0]])
                tempList = [every[1]]
            elif maxSize == len(memo[every[0]]):
                tempList.append(every[1])

        tempList = set(tempList)
        if len(tempList) == 0: # 返回索引最小的就行
            initial.sort()
            return initial[0]
        else:
            tt = []
            for every in initial:
                if every in tempList:
                    tt.append(every)
            tt.sort()
            return tt[0]
```

## [2029. 石子游戏 IX ](https://leetcode-cn.com/problems/stone-game-ix/)【博弈论，周赛不会，现在还是不会】

Alice 和 Bob 再次设计了一款新的石子游戏。现有一行 n 个石子，每个石子都有一个关联的数字表示它的价值。给你一个整数数组 stones ，其中 stones[i] 是第 i 个石子的价值。

Alice 和 Bob 轮流进行自己的回合，Alice 先手。每一回合，玩家需要从 stones 中移除任一石子。

如果玩家移除石子后，导致 所有已移除石子 的价值 总和 可以被 3 整除，那么该玩家就 输掉游戏 。
如果不满足上一条，且移除后没有任何剩余的石子，那么 Bob 将会直接获胜（即便是在 Alice 的回合）。
假设两位玩家均采用 最佳 决策。如果 Alice 获胜，返回 true ；如果 Bob 获胜，返回 false 。

```python
class Solution:
    def stoneGameIX(self, stones: List[int]) -> bool:
        # 博弈论，思考最佳决策和状态的关系
        # a先b后。
        # 记录模3之后的石头个数ct0,ct1,ct2, 和目前所移除的总和tsum
        cnt0 = 0
        cnt1 = 0
        cnt2 = 0
        for s in stones:
            if s%3 == 0: cnt0 += 1
            elif s%3 == 1: cnt1 += 1
            elif s%3 == 2: cnt2 += 1
        
        # A要赢则需要使得B移除后，所有已经移除的价值总和模3余0
        # B要赢可以是A移除后，所有已经移除的价值总和模3余0；或者是全空
        # 考虑0的个数的影响
        # 0为偶数，无影响 【这个是需要证明的】
        # 0为奇数的时候，总可以变成1个0的状态

        # 再考虑1，2的数量的影响
        # 移除序列不直接死亡时候需要是 
        # 
        # 1 1 2 1 2 1 2 ... 总序列
        # 2 2 1 2 1 2 1 ... 总序列
        # Alice 序列 1 2 2 2 2   或者 Alice 序列 2 1 1 1 1
        # Bob 序列   1 1 1 1 1   或者 Bob   序列 2 2 2 2 2 
        # 先分析序列type1:  只考虑Alice赢的情况
        # 1. 类型1恰好1个，类型2至少1个， Alice赢
        # 2. 类型1大于等于2个，且不能比类型2的石头多： Alice 赢
        #      如果1多1个，那么Alice在移除最后一个类型2的石头后，游戏结束。 Bob因为条件2获胜
        #      如果1多2个，那么Bob在移除最后一个1类型后，所有石头空。游戏结束，Bob因为条件2获胜
        #      star：如果多了超过2个，Alice会在某一步没有2可以移了，Bob由于条件1获胜
        # 3. 如果一样多或者2更多，Bob会没有1可以移除了，Alice获胜

        # 上述四条可以归纳为有类型1的石头，且不能比类型2的石头多
        # 反过来，可以归纳为有类型2的时候，且不能比类型1的石头多
        # 这两条又归纳为 有至少1个1和至少1个2. 则Alice赢

        # 总结：如果0的个数为偶数，那么Alice获胜仅当1和2至少有一个
        # 如果0的个数为奇数，Alice获胜当且仅当没有类型0的石头下，Bob获胜且不是因为所有石头被移除。对应到上面的分析即为star条件

        if cnt0%2 == 0:
            return cnt1 >= 1 and cnt2 >= 1
        else:
            return cnt1-cnt2 > 2 or cnt2 - cnt1 > 2
```

## [5974. 分隔长廊的方案数](https://leetcode-cn.com/problems/number-of-ways-to-divide-a-long-corridor/)【画图就能发现是切割点累乘了，注意边界条件】

在一个图书馆的长廊里，有一些座位和装饰植物排成一列。给你一个下标从 0 开始，长度为 n 的字符串 corridor ，它包含字母 'S' 和 'P' ，其中每个 'S' 表示一个座位，每个 'P' 表示一株植物。

在下标 0 的左边和下标 n - 1 的右边 已经 分别各放了一个屏风。你还需要额外放置一些屏风。每一个位置 i - 1 和 i 之间（1 <= i <= n - 1），至多能放一个屏风。

请你将走廊用屏风划分为若干段，且每一段内都 恰好有两个座位 ，而每一段内植物的数目没有要求。可能有多种划分方案，如果两个方案中有任何一个屏风的位置不同，那么它们被视为 不同 方案。

请你返回划分走廊的方案数。由于答案可能很大，请你返回它对 109 + 7 取余 的结果。如果没有任何方案，请返回 0 。

```python
class Solution:
    def numberOfWays(self, corridor: str) -> int:
        mod = 10**9 + 7
        n = len(corridor)
        c = corridor
        gap = []
        ct = 0
        # 当S数量被激活到两个后，开始计算直到下一个S前的P的个数
        p = 0 
        valid = False 
        tree = 0
        while p < n:
            if c[p] == 'S':
                if valid:
                    gap.append(tree)
                    tree = 0
                    valid = False 
                ct += 1
                p += 1
                if ct == 2:
                    valid = True 
                    ct = 0
            elif c[p] == 'P':
                if valid:
                    tree += 1
                p += 1
        
        # print(gap,ct)
        if ct not in {0,2}:
            return 0
        else:
            if len(gap) == 0 and ct == 2:
                return 1
            if len(gap) == 0 and 'S' not in c:
                return 0
            k = 1
            for i in range(len(gap)):
                k = (k*(gap[i]+1))%mod
            return k
        
```

## [5992. 基于陈述统计最多好人数](https://leetcode-cn.com/problems/maximum-good-people-based-on-statements/) 【先用一种纯暴力方法过了，之后学一下状态压缩】

游戏中存在两种角色：

好人：该角色只说真话。
坏人：该角色可能说真话，也可能说假话。
给你一个下标从 0 开始的二维整数数组 statements ，大小为 n x n ，表示 n 个玩家对彼此角色的陈述。具体来说，statements[i][j] 可以是下述值之一：

0 表示 i 的陈述认为 j 是 坏人 。
1 表示 i 的陈述认为 j 是 好人 。
2 表示 i 没有对 j 作出陈述。
另外，玩家不会对自己进行陈述。形式上，对所有 0 <= i < n ，都有 statements[i][i] = 2 。

根据这 n 个玩家的陈述，返回可以认为是 好人 的 最大 数目。

```python
class Solution:
    def maximumGood(self, statements: List[List[int]]) -> int:
        ans = 0
        n = len(statements)
        # 注意坏人不会说半真半假的话
        st = statements
        # 全长最多15个二进制数，上限是2**len
        # 当i是好人的时候，那么他的描述中不能有和情况矛盾
        # 二进制中0表示坏人，1表示好人
        
        for i in range(2**(n)):
            prop = [None for j in range(15)]
            group = []
            for p in range(15):
                if (i>>p) % 2 == 1:
                    prop[p] = True
                    group.append(statements[p])
                else:
                    prop[p] = False
            # 组里的都是真话，那么这些真话不能矛盾
            # 具体而言就是，说他是好人的话，它的描述和prop要对应上
            state = True
            # group里面的是[[1,2,2]]这样的，如果他描述的是1则一定是True，如果描述的是0则一定是False
            for every in group:
                if state:
                    for t in range(min(15,n)):
                        if every[t] == 1 and prop[t] == False:
                            state = False 
                            break 
                        if every[t] == 0 and prop[t] == True:
                            state = False 
                            break
            if state:
                #print(prop,group)
                ans = max(ans,bin(i).count('1'))        
        return ans
```

# 第九周：01.24 - 02.06 【两题】

## [2045. 到达目的地的第二短时间](https://leetcode-cn.com/problems/second-minimum-time-to-reach-destination/) 【BFS改编，暂存每个节点的最短路和次短路线，注意算最终的time的方法】

城市用一个 双向连通 图表示，图中有 n 个节点，从 1 到 n 编号（包含 1 和 n）。图中的边用一个二维整数数组 edges 表示，其中每个 edges[i] = [ui, vi] 表示一条节点 ui 和节点 vi 之间的双向连通边。每组节点对由 最多一条 边连通，顶点不存在连接到自身的边。穿过任意一条边的时间是 time 分钟。

每个节点都有一个交通信号灯，每 change 分钟改变一次，从绿色变成红色，再由红色变成绿色，循环往复。所有信号灯都 同时 改变。你可以在 任何时候 进入某个节点，但是 只能 在节点 信号灯是绿色时 才能离开。如果信号灯是  绿色 ，你 不能 在节点等待，必须离开。

第二小的值 是 严格大于 最小值的所有值中最小的值。

例如，[2, 3, 4] 中第二小的值是 3 ，而 [2, 2, 4] 中第二小的值是 4 。
给你 n、edges、time 和 change ，返回从节点 1 到节点 n 需要的 第二短时间 。

注意：

你可以 任意次 穿过任意顶点，包括 1 和 n 。
你可以假设在 启程时 ，所有信号灯刚刚变成 绿色 。

```python
class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        graph = collections.defaultdict(list)
        for a,b in edges:
            graph[a-1].append(b-1)
            graph[b-1].append(a-1)
        
        # 维护每个点最短路距离和次短路距离
        # dist[i] = [distance1,distance2]
        dist = [[inf,inf] for i in range(n)]
        dist[0] = [0,inf] # base 
        queue = [0]
        steps = 0
        while len(queue):
            new_queue = []
            for node in queue:
                for neigh in graph[node]:
                    if dist[neigh][0] > steps + 1:
                        dist[neigh][0] = steps + 1
                    elif dist[neigh][0] < steps + 1 < dist[neigh][1]:
                        dist[neigh][1] = steps + 1
                        if neigh == n-1: # 此时说明到了需要收集结果的时候了
                            # print(dist)
                            ans = 0
                            for i in range(dist[n-1][1]):
                                ans += time 
                                if i < dist[n-1][1]-1 and (ans//change)%2 == 1:
                                    ans = (ans + change)//change * change 
                            return ans
                    else: # 注意这里，其余情况不再更新queue，用continue开启下一轮
                        continue
                    new_queue.append(neigh)
            steps += 1
            queue = new_queue
```

## [968. 监控二叉树](https://leetcode-cn.com/problems/binary-tree-cameras/) 【树动态规划，注意状态确立和状态转移】

给定一个二叉树，我们在树的节点上安装摄像头。

节点上的每个摄影头都可以监视**其父对象、自身及其直接子对象。**

计算监控树的所有节点所需的最小摄像头数量。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        # 三个状态
        # dp[0]: 当前节点安装摄像头，一共用了多少个摄像头
        # dp[1]: 当前节点不安装摄像头，被覆盖, 一共用了多少个摄像头
        # dp[2]: 当前节点不安装摄像头，不被覆盖，一共用了多少个摄像头
        # 后续遍历

        def postOrder(node):
            if node == None:
                # 注意空节点的初始化条件
                return [inf,0,inf]
            dp = [0,0,0]
            # dp[0]自己安装了之后，左孩子节点和右孩子节点都可以安装或者不装，但是总相机数+1
            leftPart = postOrder(node.left)
            rightPart = postOrder(node.right)
            dp[0] = 1 + min(leftPart[0],min(leftPart[1],leftPart[2])) + \
                 min(rightPart[0],min(rightPart[1],rightPart[2]))
            # 不安装相机，但是能被覆盖到，说明其孩子节点至少有一个安装了相机，因为自己不安装相机，如果孩子节点也不安装，那个节点只能是已被覆盖到的
            dp[1] = min((leftPart[0]+min(rightPart[0],rightPart[1])) ,
                    (rightPart[0]+min(leftPart[0],leftPart[1])))
            # 不安装相机，也不能被覆盖到，说明其孩子节点都没有安装相机，因为自己没有安装相机，其孩子节点也必须是已被覆盖到的
            dp[2] = leftPart[1] + rightPart[1]
            return dp 
        
        a,b,c = postOrder(root)
        return min(a,b)

```

# 第十一周：02.07-

## [1001. 网格照明](https://leetcode-cn.com/problems/grid-illumination/)【哈希表模拟题，注意defaultdict的使用】

在大小为 n x n 的网格 grid 上，每个单元格都有一盏灯，最初灯都处于 关闭 状态。

给你一个由灯的位置组成的二维数组 lamps ，其中 lamps[i] = [rowi, coli] 表示 打开 位于 grid[rowi][coli] 的灯。即便同一盏灯可能在 lamps 中多次列出，不会影响这盏灯处于 打开 状态。

当一盏灯处于打开状态，它将会照亮 自身所在单元格 以及同一 行 、同一 列 和两条 对角线 上的 所有其他单元格 。

另给你一个二维数组 queries ，其中 queries[j] = [rowj, colj] 。对于第 j 个查询，如果单元格 [rowj, colj] 是被照亮的，则查询结果为 1 ，否则为 0 。在第 j 次查询之后 [按照查询的顺序] ，关闭 位于单元格 grid[rowj][colj] 上及相邻 8 个方向上（与单元格 grid[rowi][coli] 共享角或边）的任何灯。

返回一个整数数组 ans 作为答案， ans[j] 应等于第 j 次查询 queries[j] 的结果，1 表示照亮，0 表示未照亮。

```python
class Solution:
    def gridIllumination(self, n: int, lamps: List[List[int]], queries: List[List[int]]) -> List[int]:
        # 记录每个灯泡，利用哈希表记次数。
        record = collections.defaultdict(int) # 记录灯泡被点亮
        lightingDict1 = collections.defaultdict(int) # 记录哪些对角线被点亮，要分开记录
        lightingDict2 = collections.defaultdict(int) # 记录哪些对角线被点亮，要分开记录
        lightingDictI = collections.defaultdict(int) # 记录哪些横行被点亮
        lightingDictJ = collections.defaultdict(int) # 记录哪些纵行被点亮
        # 对角线点亮原则(i,j) 
        # y = (x-i)+j ,x-y == i-j被点亮
        # y = -(x-i)+j, x+y == i+j被点亮
        # 灯只能被打开一次
        for i,j in lamps:
            if (i,j) in record: continue
            record[(i,j)] = 1
            lightingDict1[i-j] += 1
            lightingDict2[i+j] += 1
            lightingDictI[i] += 1
            lightingDictJ[j] += 1
        ans = []
        direc = [
            (-1,-1),(-1,0),(-1,1),
            (0,-1),(0,0),(0,1),
            (1,-1),(1,0),(1,1),
        ]
        for x,y in queries:
            state = False  # 默认状态没有被照亮
            if lightingDict1[x-y] > 0:
                state = True
            if lightingDict2[x+y] > 0:
                state = True 
            if lightingDictI[x] > 0:
                state = True 
            if lightingDictJ[y] > 0:
                state = True 

            for di in direc: # 熄灯
                i,j = x+di[0],y+di[1]
                # print(record)
                if (i,j) in record:
                    del record[(i,j)] # 熄灯
                    lightingDict1[i-j] -= 1
                    lightingDict2[i+j] -= 1
                    lightingDictI[i] -= 1
                    lightingDictJ[j] -= 1
                    state = True
            if state:
                ans.append(1)
            else:
                ans.append(0)
        # print(record,lightingDict1,lightingDict2)
        return ans
```


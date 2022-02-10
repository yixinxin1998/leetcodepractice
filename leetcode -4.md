# 6. Z 字形变换

将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        # 改变一下方向，变成z形
        # 纯模拟尬模
        mat = [["" for j in range(numRows)]]
        p = 0
        cur = 0
        i = 0
        while p < len(s):
            while p < len(s) and cur < numRows:
                mat[i][cur] = s[p]
                cur += 1 
                p += 1
            cur -= 1
            while p < len(s) and cur-1 > 0:
                cur -= 1
                tempLine = ["" for j in range(numRows)]
                tempLine[cur] = s[p]
                mat.append(tempLine)
                i += 1
                p += 1           
            mat.append(["" for j in range(numRows)])
            i += 1
            cur = 0
        
        # 然后纵列
        stack = []
        m = len(mat)
        n = len(mat[0])
        for j in range(n):
            for i in range(m):
                stack.append(mat[i][j])
        return "".join(stack)
```

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        # 参考k神的解法
        if numRows == 1:
            return s
            
        ans = [[] for i in range(numRows)]
        i = 0
        flag = 0
        for ch in s:
            if i < numRows and flag == 0:
                ans[i].append(ch)
                i += 1
                if i == numRows:
                    i = numRows - 2
                    flag = 1                    
            elif i >= 0 and flag == 1:
                ans[i].append(ch)
                i -= 1
                if i == -1:
                    i = 1
                    flag = 0
        
        final = "".join("".join(e) for e in ans) # pythonic的写法
        return final
```

# 10. 正则表达式匹配

给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

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



# 29. 两数相除

给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。

返回被除数 dividend 除以除数 divisor 得到的商。

整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2

```
// 时间复杂度：O(1)
func divide(a int, b int) int {
    if a == math.MinInt32 && b == -1 {
        return math.MaxInt32
    }

    sign := 1 // 符号位
    if (a > 0 && b < 0) || (a < 0 && b > 0) {
        sign = -1
    }

    a = abs(a)
    b = abs(b)

    res := 0
    for i := 31; i >= 0; i-- {
        if (a >> i) - b >= 0 {
            a = a - (b << i)
            res += 1 << i
        }
    }
    return sign * res
}

func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}
```



# 42. 接雨水

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # 带查询的解法，
        n = len(height)
        if n <= 2:
            return 0

        leftMax = [0 for i in range(n)]
        rightMax = [0 for i in range(n)]
        # 注意这两行
        leftMax[0] = height[0]
        rightMax[n-1] = height[n-1]

        # 更新当前柱子左边最高的柱子，如果没有，则定为当前柱高
        for i in range(1,n):
            leftMax[i] = max(height[i],leftMax[i-1])
        for i in range(n-2,-1,-1):
            rightMax[i] = max(height[i],rightMax[i+1])
        # 然后计算1～n-2索引的木桶效应
        # print(leftMax,rightMax)
        ans = 0
        for i in range(1,n-1): # 
            ans += min(leftMax[i],rightMax[i])-height[i]
        return ans
        
```

```go
func trap(height []int) int {
    if len(height) <= 2 {
        return 0
    }
    n := len(height)
    leftMax := make([]int,n)
    rightMax := make([]int,n)
    leftMax[0] = height[0]
    rightMax[n-1] = height[n-1]
    
    for i:=1; i<n; i += 1{
        leftMax[i] = getInt(height[i],leftMax[i-1])
    }
    for i:=n-2; i>=0; i-= 1 {
        rightMax[i] = getInt(height[i],rightMax[i+1])
    }
    
    ans := 0
    for i:=1; i<=n-2; i += 1 {
        ans += getMin(leftMax[i],rightMax[i])-height[i]
    }
    return ans
}

func getInt(a,b int) int {
    if a < b {
        return b
    } else {
        return a
    }
}

func getMin(a,b int) int {
    if a < b {
        return a
    }else {
        return b
    }
}
```

# 44. 通配符匹配

给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。

'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
两个字符串完全匹配才算匹配成功。

说明:

s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m = len(s)
        n = len(p)
        dp = [[False for j in range(n+1)] for i in range(m+1)]
        dp[0][0] = True
        for j in range(1,n+1):
            dp[0][j] = dp[0][j-1] and (p[j-1] == "*")

        for i in range(1,m+1):
            for j in range(1,n+1):
                if p[j-1] != '*':
                    state1 = dp[i-1][j-1] and s[i-1] == p[j-1]
                    state2 = dp[i-1][j-1] and p[j-1] == '?'
                    dp[i][j] = state1 or state2 
                elif p[j-1] == '*':
                    state1 = dp[i][j-1] # *当作没有
                    state2 = dp[i-1][j] # *的链式反应。有记忆化递归的意思 
                    dp[i][j] = state1 or state2
                    # 实质上是要看dp[i-k][j]，k一直变化的时候是否能匹配。，但是由于i是递增的，那么每一个i继承了前一个i
        return dp[-1][-1]


```

```go
func isMatch(s string, p string) bool {
    m,n := len(s),len(p)
    dp := make([][]bool,m+1)
    for i := 0; i < m+1; i ++ {
        dp[i] = make([]bool,n+1)
    }
    dp[0][0] = true
    for j := 1; j < n+1; j++ {
        dp[0][j] = dp[0][j-1] && p[j-1] == '*'
    }

    for i := 1; i < m+1; i ++ {
        for j := 1; j < n+1; j ++ {
            if p[j-1] != '*' {
                state1 := dp[i-1][j-1] && s[i-1] == p[j-1]
                state2 := dp[i-1][j-1] && p[j-1] == '?'
                dp[i][j] = state1 || state2
            } else if p[j-1] == '*' {
                state1 := dp[i][j-1]
                state2 := dp[i-1][j]
                dp[i][j] = state1 || state2
            }
        }
    }
    return dp[m][n]
}
```

# 71. 简化路径

给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为更加简洁的规范路径。

在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。任意多个连续的斜杠（即，'//'）都被视为单个斜杠 '/' 。 对于此问题，任何其他格式的点（例如，'...'）均被视为文件/目录名称。

请注意，返回的 规范路径 必须遵循下述格式：

始终以斜杠 '/' 开头。
两个目录名之间必须只有一个斜杠 '/' 。
最后一个目录名（如果存在）不能 以 '/' 结尾。
此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 '.' 或 '..'）。
返回简化后得到的 规范路径 。

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        # 栈问题
        # 先以"/"split
        # python取巧了。。
        stack = collections.deque()
        temp = path.split("/")
        for ch in temp:
            if ch == "":
                pass
            elif ch == ".":
                pass
            elif ch == "..":
                if len(stack) > 0:
                    stack.pop()
            else:
                stack.append(ch)
            
        return "/"+"/".join(stack)
        
```

# 85. 最大矩形

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:

        def calcArea(lst):
            # 单调栈,从左到右单调递增
            n = len(lst)
            stack = [-1]
            maxArea = 0
            for i in range(n):
                while stack[-1] != -1 and lst[stack[-1]] >= lst[i]: # 遇到了相同的也出栈
                    h = lst[stack.pop()]
                    w = i-stack[-1]-1
                    maxArea = max(maxArea,h*w)
                stack.append(i)
            while stack[-1] != -1:
                h = lst[stack.pop()]
                w = n-stack[-1]-1
                maxArea = max(maxArea,h*w)
            return maxArea
        
        ans = 0
        m = len(matrix)
        if m == 0: return 0
        n = len(matrix[0])
        if n == 0: return 0
        lst = [0 for i in range(n)]
        for i in range(m):
            for j in range(n):
                if int(matrix[i][j]) == 1: # 注意这里 
                    lst[j] += int(matrix[i][j])
                else:
                    lst[j] = 0
            # print(lst)
            # print(calcArea(lst))
            ans = max(ans,calcArea(lst))
        return ans
```

# 91. 解码方法

一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：

'A' -> 1
'B' -> 2
...
'Z' -> 26
要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：

"AAJF" ，将消息分组为 (1 1 10 6)
"KJF" ，将消息分组为 (11 10 6)
注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。

给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。

题目数据保证答案肯定是一个 32 位 的整数。

```python
class Solution:
    def numDecodings(self, s: str) -> int: 
        # dp
        # dp[i]的意思是包括到i为止的编码总数，注意是1～26
        # 状态转移为dp[i] = dp[i-1] , 如果使用则为dp[i] = dp[i-1] + dp[i-2]
        if s[0] == "0":
            return 0
        dp = [0 for i in range(len(s))]
        dp = [1] + dp # 添加哨兵方便处理边界条件
        s = " " + s # 添加哨兵
        for i in range(1,len(s)):
            # 当前字母在1~9则继承: dp[i] += dp[i-1]
            # (当前字母+前一个字母)在 10～26，继承 dp[i] += dp[i-2]
            a = int(s[i])
            b = int(s[i-1:i+1])
            if 1 <= a <= 9:
                dp[i] += dp[i-1]
            if 10 <= b <= 26:
                dp[i] += dp[i-2]
        return dp[-1]
```

```go
func numDecodings(s string) int {
    if s[0] == '0' {
        return 0
    }
    dp := make([]int,len(s)+1)
    dp[0] = 1
    s = " " + s 
    for i := 1; i < len(s); i++ {
        a := s[i] - '0'
        b := 10 * (s[i-1]-'0') + s[i]-'0'
        if 1 <= a && a <= 9 {
            dp[i] += dp[i-1]
        }
        if 10 <= b && b <= 26 {
            dp[i] += dp[i-2]
        }
    }

    return dp[len(s)-1]
}
```

# 95. 不同的二叉搜索树 II

给你一个整数 `n` ，请你生成并返回所有由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的不同 **二叉搜索树** 。可以按 **任意顺序** 返回答案。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:

        def recur(left,right): # 闭区间递归搜索
            if left > right:
                return [None]
            
            nowTree = []
            for i in range(left,right+1): # 闭区间
                leftPart = recur(left,i-1)
                rightPart = recur(i+1,right)

                for l in leftPart:
                    for r in rightPart:
                        theRoot = TreeNode(i)
                        theRoot.left = l 
                        theRoot.right = r 
                        nowTree.append(theRoot)
            return nowTree
        
        ans = recur(1,n)
        return ans
```

# 97. 交错字符串

给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。

两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
提示：a + b 意味着字符串 a 和 b 连接。

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1)+len(s2) != len(s3):
            return False
        # 二维dp
        m,n = len(s1),len(s2)
        dp = [[False for j in range(n+1)] for i in range(m+1)]
        # 本题字符串序号标记采用从1开始 1～m 和 1 ～ n
        # dp[i][j]表示 1~i和1～j是否匹配
        # 基态
        dp[0][0] = True 
        for i in range(1,m+1):
            dp[i][0]=(dp[i-1][0] and s1[i-1]==s3[i-1])
        for i in range(1,n+1):
            dp[0][i]=(dp[0][i-1] and s2[i-1]==s3[i-1])

        for i in range(1,m+1):
            for j in range(1,n+1):
                # 需要匹配的字符要么来自于1去尾，然后1匹配,此时的字符python序号i-1
                # 要么来自于2去尾,然后2匹配
                state1 = (dp[i-1][j] and s1[i-1] == s3[i+j-1])
                state2 = (dp[i][j-1] and s2[j-1] == s3[i+j-1])
                dp[i][j] = (state1 or state2 )
        return dp[-1][-1]
```

# 127. 单词接龙

字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：

序列中第一个单词是 beginWord 。
序列中最后一个单词是 endWord 。
每次转换只能改变一个字母。
转换过程中的中间单词必须是字典 wordList 中的单词。
给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if len(word_set) == 0 or endWord not in word_set:
            return 0
        if beginWord in word_set:
            word_set.remove(beginWord)
        
        queue = []
        queue.append(beginWord)
        steps = 1
        n = len(beginWord)
        visited = set()
        visited.add(beginWord)

        while len(queue) != 0:
            new_queue = []
            for tempWord in queue:
                tempWordList = list(tempWord)
                for i in range(n):
                    originChar = tempWordList[i]
                    for t in range(26):
                        tempWordList[i] = chr(ord('a')+t) # 变更
                        newWord = "".join(tempWordList)
                        if newWord in word_set:
                            if newWord == endWord:
                                return steps + 1
                            if newWord not in visited:
                                new_queue.append(newWord)
                                visited.add(newWord)
                        tempWordList[i] = originChar # 复原
            steps += 1
            queue = new_queue
        return 0
```

# 139. 单词拆分

给你一个字符串 s 和一个字符串列表 wordDict 作为字典，判定 s 是否可以由空格拆分为一个或多个在字典中出现的单词。

说明：拆分时可以重复使用字典中的单词。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    # 记忆化递归
        memo = dict()
        memo[0] = True
        wordDict = set(wordDict)

        def recur(n):
            if n in memo:
                return memo[n]
            group = []
            for i in range(0,n):
                group.append(recur(i) and s[i:n] in wordDict)
            memo[n] = any(group) # 有一个成立即可
            return memo[n]
        
        return recur(len(s))
```

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False for j in range(n+1)]
        dp[0] = True
        wordDict = set(wordDict)
        
        # dp[i]是前i个字符[0~i-1]是否可以被表示
        for i in range(1,n+1):
            group = []
            for j in range(i):
                group.append(dp[j] and s[j:i] in wordDict)
            dp[i] = any(group) # 有一个成立即可
        
        return dp[-1]

```



# 152. 乘积最大子数组

给你一个整数数组 `nums` ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        # 特殊情况需要打补丁
        if len(nums) == 1:
            return nums[0]
        # 两列，以i结尾的最大正数积和最大负数积
        n = len(nums)
        dp = [[0 for i in range(n)] for t in range(2)]
        # dp[0]为最大正数积，dp[1]为最大负数积
        if nums[0] > 0:
            dp[0][0] = nums[0]
            dp[1][0] = 0
        elif nums[0] < 0:
            dp[0][0] = 0
            dp[1][0] = nums[0]
        
        # 这个动态规划需要写纸上
        for i in range(1,n):
            if nums[i] == 0:
                continue 
            elif nums[i] > 0:
                if dp[0][i-1] > 0:
                    dp[0][i] = nums[i]*dp[0][i-1]
                else:
                    dp[0][i] = nums[i]
                if dp[1][i-1] < 0:
                    dp[1][i] = nums[i]*dp[1][i-1]

            elif nums[i] < 0:
                if dp[1][i-1] < 0:
                    dp[0][i] = nums[i]*dp[1][i-1]
                if dp[0][i-1] > 0:
                    dp[1][i] = nums[i]*dp[0][i-1]
                else:
                    dp[1][i] = nums[i]
        
        return max(dp[0])
```



# 163. 缺失的区间

给定一个排序的整数数组 ***nums*** ，其中元素的范围在 **闭区间** **[\*lower, upper\*]** 当中，返回不包含在数组中的缺失区间。

```python
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
        # 注意卡边界条件,lower可以为负数,结果可以为空
        nums = [lower-1] + nums
        nums = nums + [upper+1]
        # 注意处理
        if len(nums) == 1:
            return [str(nums[0])]
        
        # print(nums)
        p = 1
        ans = []        
        while p < len(nums):
            if nums[p] - nums[p-1] == 1:
                pass
            elif nums[p] - nums[p-1] == 2:
                ans.append(str(nums[p]-1))
            else:
                ans.append(str(nums[p-1]+1)+"->"+str(nums[p]-1))
            p += 1

        return ans
```

# 201. 数字范围按位与

给你两个整数 left 和 right ，表示区间 [left, right] ，返回此区间内所有数字 按位与 的结果（包含 left 、right 端点）。

```python
# 很巧妙的位运算思路
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        #  n 的后缀 0abcd... 累加到 m 的后缀 1hijk... 这个过程中，
        # 不管abcd...，hijk... 取值如何，必然要经过 10000...
        # 0abcd... 和 10000... 使得答案中长度为 31-x 的后缀必然都为 0。
		# 参考官解
        # 找到left,和right的最长公共前缀，
        shift = 0
        while left != right:
            left = left >> 1
            right = right >> 1
            shift += 1
        right = right << shift
        return right
```



# 221. 最大正方形

在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        # dp[i][j]表示以i,j为右下角的最大正方形边长
        # 技巧型很强的dp,和1277类似
        m,n = len(matrix),len(matrix[0])
        maxLength = 0
        dp = [[0 for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[i][j] = int(matrix[i][j])
                elif matrix[i][j] == "1": # 注意这里别写成dp[i][j]
                    dp[i][j] = min(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1]
                        )+1
                maxLength = max(maxLength,dp[i][j])
        return maxLength**2
```

# 229. 求众数 II

给定一个大小为 *n* 的整数数组，找出其中所有出现超过 `⌊ n/3 ⌋` 次的元素。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        # 普通法,这一题要严格大于
        limit = math.floor(len(nums)/3)
        ct = collections.Counter(nums)
        ans = []
        for key in ct:
            if ct[key] > limit:
                ans.append(key)
        return ans

```

```

```

# 265. 粉刷房子 II

假如有一排房子，共 n 个，每个房子可以被粉刷成 k 种颜色中的一种，你需要粉刷所有的房子并且使其相邻的两个房子颜色不能相同。

当然，因为市场上不同颜色油漆的价格不同，所以房子粉刷成不同颜色的花费成本也是不同的。每个房子粉刷成不同颜色的花费是以一个 n x k 的矩阵来表示的。

例如，costs[0][0] 表示第 0 号房子粉刷成 0 号颜色的成本花费；costs[1][2] 表示第 1 号房子粉刷成 2 号颜色的成本花费，以此类推。请你计算出粉刷完所有房子最少的花费成本。

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

# 273. 整数转换英文表示

将非负整数 `num` 转换为其对应的英文表示。

```python
singles = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
thousands = ["", "Thousand", "Million", "Billion"]

class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return "Zero"

        ans = []
        # 三位三位的划分，用迭代划分
        def helper(n):
            s = ""
            if n == 0:
                return ''
            if 0 < n < 10:
                return singles[n] + " "
            elif 10 <= n < 20:
                return teens[n-10] + " "
            elif 20 <= n < 100:
                return tens[n//10] + " " + helper(n%10)
            else:
                return singles[n//100] + " Hundred " + helper(n%100)
        
        s = ""
        # 然后需要用三位三位划分
        the_format = 10**9
        cur = 3# 初始化指向Billion
        for i in range(4):             
            need = num // the_format
            if need > 0:
                s += helper(need) + thousands[cur] + " "
            cur -= 1
            num = num % the_format
            the_format //= 1000

        return s.strip()
```

```go
var singles = []string {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"}
var teens = []string {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"}
var tens = []string {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"}
var thousands = []string{"", "Thousand", "Million", "Billion"}

func numberToWords(num int) string {
    if num == 0 {
        return "Zero"
    }
    s := ""
    theFormat := 1000000000
    cur := 3
    for i := 0; i < 4; i++ {
        need := num/theFormat
        if need > 0 {
            temp := help(need)
            s += temp + thousands[cur] + " "
        }
        cur -= 1
        num = num % theFormat
        theFormat = theFormat / 1000
    }
    return strings.TrimSpace(s)
}

func help(n int) string {
    if n == 0 {
        return ""
    }
    if 0 < n && n< 10 {
        return singles[n] + " "
    } else if 10 <= n && n < 20 {
        return teens[n-10] + " "
    } else if 20 <= n && n < 100 {
        return tens[n/10] + " " + help(n%10)
    } else {
        return singles[n/100] + " Hundred " + help(n%100)
    }
}
```

# 276. 栅栏涂色

有 k 种颜色的涂料和一个包含 n 个栅栏柱的栅栏，请你按下述规则为栅栏设计涂色方案：

每个栅栏柱可以用其中 一种 颜色进行上色。
相邻的栅栏柱 最多连续两个 颜色相同。
给你两个整数 k 和 n ，返回所有有效的涂色 方案数 。

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        if n == 1:
            return k 
        elif n == 2:
            return k**2
        # 动态规划
        # 状态转移方程
        # 1.当前颜色和前一根不相同，方案数为 dp[i] += dp[i-1]*(k-1)
        # 2.当前颜色和前一根相同，且前一根和前前一根不相同，那么绑定这两根：
        # 2的方案数为 dp[i] += dp[i-2]*(k-1)
        dp = [0 for i in range(n)]
        dp[0] = k
        dp[1] = k**2
        for i in range(2,n):
            dp[i] += dp[i-1]*(k-1)
            dp[i] += dp[i-2]*(k-1)
        return dp[-1]
```

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        if n == 1:
            return k
        if n == 2:
            return k**2
        # 二维dp
        dp = [[0 for j in range(n)] for i in range(2)]
        # 第一行表示与前者颜色相同的方案数目:dp[0][j] = dp[1][j-1]
        # 第二行表示与前者颜色不同的方案数目:dp[1][j] = dp[0][j-1]*(k-1) + dp[1][j-1]*(k-1)
        dp[0][0] = 0
        dp[1][0] = k
        for j in range(1,n):
            dp[0][j] = dp[1][j-1]
            dp[1][j] = dp[0][j-1]*(k-1) + dp[1][j-1]*(k-1)
        return dp[0][-1] + dp[1][-1]
```

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        # 记忆化递归版本
        memo = dict()
        def recur(n):
            if n == 1:
                return k
            if n == 2:
                return k**2
            if n in memo:
                return memo[n]
            state1 = recur(n-1)*(k-1) # 与前一个方案颜色不同的方案
            state2 = recur(n-2)*(k-1) # 与前一个方案颜色相同的方案，即把这最后两根绑定，返回recur(n-2)*(k-1)
            memo[n] = state1 + state2
            return memo[n]        
        return recur(n)
```

# 282. 给表达式添加运算符

给定一个仅包含数字 0-9 的字符串 num 和一个目标值整数 target ，在 num 的数字之间添加 二元 运算符（不是一元）+、- 或 * ，返回所有能够得到目标值的表达式。

```python
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        ans = []
        num = list(num)

        def parse(lst): # 传入一个列表，返回解析函数值
            lst.append("+") # 加个尾巴
            num = 0
            stack = []
            symbol = "+"
            for ch in lst:
                if ch.isdigit():
                    num = num*10 + int(ch)
                elif ch.isdigit() == False and symbol == "+":
                    stack.append(num)
                    symbol = ch 
                    num = 0
                elif ch.isdigit() == False and symbol == "*":
                    stack[-1] = num*stack[-1]
                    symbol = ch 
                    num = 0
                elif ch.isdigit() == False and symbol == "-":
                    stack.append(-num)
                    symbol = ch 
                    num = 0  
            lst.pop() # 去掉原来的小尾巴 
            return sum(stack)

        # print(parse(list("1*2*3*4*5")))
        # 还需要处理前导0的问题

        def check(lst): # 前导0检查
            temp = lst.copy()
            for i in range(len(temp)):
                if temp[i].isdigit() == False:
                    temp[i] = ","
            temp = "".join(temp)
            temp = temp.split(",")
            state = True 
            for e in temp:
                if len(str(int(e))) != len(e):
                    state = False
                    break 
            return state
            
        def backtracking(lst,index,path):
            if index >= len(num):
                if check(path) and parse(path) == target:
                    ans.append("".join(path))
                return 

            path.append(num[index])
            if len(path) != 0 and path[-1].isdigit() and index != len(num)-1:
                path.append("+")
                backtracking(lst,index+1,path)
                path.pop()
                path.append("-")
                backtracking(lst,index+1,path)
                path.pop()
                path.append("*")
                backtracking(lst,index+1,path)
                path.pop()
            backtracking(lst,index+1,path)
            path.pop()

        backtracking(num,0,[])
        ans.sort()
        return ans
```



# 293. 翻转游戏

你和朋友玩一个叫做「翻转游戏」的游戏。游戏规则如下：

给你一个字符串 currentState ，其中只含 '+' 和 '-' 。你和朋友轮流将 连续 的两个 "++" 反转成 "--" 。当一方无法进行有效的翻转时便意味着游戏结束，则另一方获胜。

计算并返回 一次有效操作 后，字符串 currentState 所有的可能状态，返回结果可以按 任意顺序 排列。如果不存在可能的有效操作，请返回一个空列表 [] 。

```python
class Solution:
    def generatePossibleNextMoves(self, currentState: str) -> List[str]:
        # 一次操作
        if len(currentState) <= 1:
            return []
        p = 1
        ans = []
        while p < len(currentState):
            if currentState[p-1:p+1] == "++":
                cp = list(currentState)
                cp[p-1] = "-"
                cp[p] = "-"
                ans.append("".join(cp))
            p += 1
        return ans
```

# 294. 翻转游戏 II

你和朋友玩一个叫做「翻转游戏」的游戏。游戏规则如下：

给你一个字符串 currentState ，其中只含 '+' 和 '-' 。你和朋友轮流将 连续 的两个 "++" 反转成 "--" 。当一方无法进行有效的翻转时便意味着游戏结束，则另一方获胜。默认每个人都会采取最优策略。

请你写出一个函数来判定起始玩家 是否存在必胜的方案 ：如果存在，返回 true ；否则，返回 false 。

```python
class Solution:
    # 记忆化搜索
    # 这一题实际测试用例很小，没有达到60
    def __init__(self):
        self.memo = dict()

    def canWin(self, currentState: str) -> bool:
        if currentState in self.memo:
            return self.memo[currentState]
        s = list(currentState)
        n = len(s)
        for i in range(n - 1):
            if s[i] == '+' and  s[i+1] == '+':
                s[i] = '-'
                s[i+1] = '-'
                if self.canWin( ''.join(s) ) == False:
                    self.memo[currentState] = True
                    return True
                s[i] = '+'              #回溯，有借有还
                s[i+1] = '+'            #回溯，有借有还
        self.memo[currentState] = False
        return False

```

# 299. 猜数字游戏

你在和朋友一起玩 猜数字（Bulls and Cows）游戏，该游戏规则如下：

写出一个秘密数字，并请朋友猜这个数字是多少。朋友每猜测一次，你就会给他一个包含下述信息的提示：

猜测数字中有多少位属于数字和确切位置都猜对了（称为 "Bulls", 公牛），
有多少位属于数字猜对了但是位置不对（称为 "Cows", 奶牛）。也就是说，这次猜测中有多少位非公牛数字可以通过重新排列转换成公牛数字。
给你一个秘密数字 secret 和朋友猜测的数字 guess ，请你返回对朋友这次猜测的提示。

提示的格式为 "xAyB" ，x 是公牛个数， y 是奶牛个数，A 表示公牛，B 表示奶牛。

请注意秘密数字和朋友猜测的数字都可能含有重复数字。

```python
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        # 先筛掉公牛，再筛掉奶牛

        bulls = 0
        cows = 0
        n = len(secret)
        secret = list(secret)
        guess = list(guess)

        for i in range(n):
            if secret[i] == guess[i]:
                bulls += 1
                secret[i],guess[i] = " "," "
        
        ct1 = collections.Counter(secret)
        ct2 = collections.Counter(guess)

        for key in ct2:
            if key != " " and key in ct1:
                cows += min(ct2[key],ct1[key])
        
        return str(bulls)+"A"+str(cows)+"B"
```

# 301. 删除无效的括号

给你一个由若干括号和字母组成的字符串 `s` ，删除最小数量的无效括号，使得输入的字符串有效。

返回所有可能的结果。答案可以按 **任意顺序** 返回。

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        # 先统计一遍有多少个无效括号
        def judgeDelete(s):# 看需要删除多少个括号
            left,right = 0,0            
            stack = []
            for ch in s:
                if len(stack) == 0 and ch == ")":
                    right += 1
                elif len(stack) != 0 and ch == ")":
                    stack.pop()
                elif ch == "(":
                    stack.append(ch)
            left = len(stack)
            return left,right

        deleteLeft,deleteRight = judgeDelete(s)
        ans = set() # 回溯解决

        def backtracking(path,index,l,r):
            if l < 0 or r < 0:
                return 
            if len(s) - index  < l + r: # 剪枝条,全删了长度都不够的时候
                return 
            if index == len(s):
                temp = "".join(path)
                if l == 0 and r == 0:
                    if judgeDelete(temp) == (0,0):
                        ans.add(temp)
                return 
            path.append(s[index])
            if s[index] == "(":
                path.pop()
                backtracking(path,index+1,l-1,r)
                path.append(s[index])
            if s[index] == ")":
                path.pop()
                backtracking(path,index+1,l,r-1)
                path.append(s[index])
            backtracking(path,index+1,l,r)
            path.pop()
        
        backtracking([],0,deleteLeft,deleteRight)
        ans = list(ans)
        # ans.sort()
        return ans
```

# 302. 包含全部黑色像素的最小矩形

图片在计算机处理中往往是使用二维矩阵来表示的。

给你一个大小为 m x n 的二进制矩阵 image 表示一张黑白图片，0 代表白色像素，1 代表黑色像素。

黑色像素相互连接，也就是说，图片中只会有一片连在一块儿的黑色像素。像素点是水平或竖直方向连接的。

给你两个整数 x 和 y 表示某一个黑色像素的位置，请你找出包含全部黑色像素的最小矩形（与坐标轴对齐），并返回该矩形的面积。

你必须设计并实现一个时间复杂度低于 O(mn) 的算法来解决此问题。

```python
class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        # 方法1:尬做法。未利用到x,y
        m = len(image)
        n = len(image[0])
        for i in range(m):
            for j in range(n):
                image[i][j] = int(image[i][j])
        rowMark = [False for i in range(m)]
        colMark = [False for j in range(n)]
        for i in range(m):
            if sum(image[i]) != 0:
                rowMark[i] = True
        for j in range(n):
            tempSum = 0
            for i in range(m):
                tempSum += image[i][j]
            if tempSum != 0:
                colMark[j] = True 
        
        p1 = 0
        p2 = m - 1
        while p1 < m and rowMark[p1] != True:
            p1 += 1
        while p2 >= 0 and rowMark[p2] != True:
            p2 -= 1
        deltaM = abs(p1-p2)+1

        p3 = 0
        p4 = n - 1
        while p3 < n and colMark[p3] != True:
            p3 += 1
        while p4 >= 0 and colMark[p4] != True:
            p4 -= 1
        deltaN = abs(p3-p4)+1
        return deltaM * deltaN
```

```python
class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        # 方法2: bfs,图像小的时候稍微省了时间，利用到了x,y
        upLimit = x
        downLimit = x
        leftLimit = y
        rightLimit = y
        m = len(image)
        n = len(image[0])
        queue = [(x,y)]
        direc = [(0,1),(0,-1),(-1,0),(1,0)]
        visited = [[False for j in range(n)] for i in range(m)]
        visited[x][y] = True

        while len(queue) != 0:
            new_queue = []
            for i,j in queue:
                upLimit = min(upLimit,i)
                downLimit = max(downLimit,i)
                leftLimit = min(leftLimit,j)
                rightLimit = max(rightLimit,j)
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and image[new_i][new_j] == "1":
                        visited[new_i][new_j] = True 
                        new_queue.append([new_i,new_j])
            queue = new_queue
        
        return (abs(upLimit-downLimit)+1) * (abs(leftLimit-rightLimit)+1)

```

```python
class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        # 二分法
        # 基础是只在有必要的时候进行二分
        # 需要四次二分，二分初始值设定比较关键
        m = len(image)
        n = len(image[0])

        # 注意每个left，right的分配初始值
        # case1
        left = 0
        right = x
        while left <= right:
            mid = (left+right)//2
            state = False 
            for element in image[mid]:
                if element == "1": # 这一行存在1，
                    state = True
                    break 
            if state: # 这一行存在1，收缩right
                right = mid - 1 # 收缩right,可能收缩过头，但是返回的是left所有没有关系
            if not state: # 这一行不存在1，收缩left
                left = mid + 1        
        upLimit = left # 

        # case2
        left = x
        right = m-1
        while left <= right:
            mid = (left+right)//2
            state = False 
            for element in image[mid]:
                if element == "1": # 这一行存在1，收缩left  
                    state = True 
                    break 
            if state: # 这一行存在1，收缩left
                left = mid + 1
            if not state:
                right = mid - 1
        downLimit = right

        # case3
        left = 0
        right = y
        while left <= right:
            mid = (left+right)//2
            state = False 
            for i in range(m):
                if image[i][mid] == "1": # 这一列存在1，收缩right
                    state = True 
                    break 
            if state:
                right = mid - 1 # # 收缩right,可能收缩过头，但是返回的是left所有没有关系
            if not state:
                left = mid + 1
        leftLimit = left 

        # case4
        left = y
        right = n - 1
        while left <= right:
            mid = (left+right)//2
            state = False 
            for i in range(m):
                if image[i][mid] == "1": # 这一列存在1，收缩left 
                    state = True 
                    break 
            if state:
                left = mid + 1
            if not state:
                right = mid - 1
        rightLimit = right

        # print(upLimit,downLimit,leftLimit,rightLimit)
        return (abs(upLimit-downLimit)+1) * (abs(leftLimit-rightLimit)+1)
```

# 309. 最佳买卖股票时机含冷冻期

给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 三行dp数组
        # dp[0][i],dp[1][i],dp[2][i]
        # dp[0][i]是没有持有股票的最大收益【非冷冻期
        # dp[1][i]是处于冷冻期的最大收益【手上没有股票，今天卖出的】
        # dp[2][i]是持有股票的最大收益
        # 注意dp的含义是收益，而不是付出和收获的总和
        n = len(prices)
        dp = [[0 for i in range(n)] for j in range(3)]
        # 初始化
        dp[0][0],dp[1][0],dp[2][0] = 0,0,-prices[0]
        # dp[0][i] 要么前一天也没有股票，且非冷冻，收益不会有卖股票
        # dp[0][i] = max(dp[0][i-1],dp[1][i-1])
        # dp[1][i] 只能来源于dp[2][i-1]卖出股票
        # dp[1][i] = dp[2][i-1]+prices[i]
        # dp[2][i] 要么前一天有股票，要么前一天没有股票[不能来自于冷冻期]，买了 
        # dp[2][i] = max(dp[2][i-1],dp[0][i-1]-prices[i])
        for i in range(1,n):
            dp[0][i] = max(dp[0][i-1],dp[1][i-1])
            dp[1][i] = dp[2][i-1]+prices[i]
            dp[2][i] = max(dp[2][i-1],dp[0][i-1]-prices[i])  
        return max(dp[0][-1],dp[1][-1],dp[2][-1])
```

```go
func maxProfit(prices []int) int {
    dp := make([][]int,3,3)
    n := len(prices)
    for i:=0;i<3;i++ {
        dp[i] = make([]int,n,n)
    }
    dp[0][0],dp[1][0],dp[2][0] = 0,0,-prices[0]
    // dp[0][i] 没有股票
    // dp【1】【i】 今天卖出的，没有股票
    // dp【2】【i】 有股票
    for i:=1;i<n;i++ {
        dp[0][i] = max(dp[0][i-1],dp[1][i-1])
        dp[1][i] = dp[2][i-1] + prices[i]
        dp[2][i] = max(dp[2][i-1],dp[0][i-1]-prices[i])
    }
    return max(dp[0][n-1],dp[1][n-1])
}

func max(a,b int) int {
    if a > b {
        return a
    } else {
        return b
    }
}
```

# 316. 去除重复字母

给你一个字符串 `s` ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 **返回结果的字典序最小**（要求不能打乱其他字符的相对位置）。

```python
# 这一题挺难的
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # 先考虑一个问题，需对于一个字符串，要得到字典序最小，找到第一个s[i]，当s[i]>s[i+1]的时候，删除它
        # 原字符串s中的每个字符都需要出现在新字符串中，且只能出现一次。为了让新字符串满足该要求，之前讨论的算法需要进行以下两点的更改。在考虑字符 s[i]时，如果它已经存在于栈中，则不能加入字符 s[i]。为此，需要记录每个字符是否出现在栈中。在弹出栈顶字符时，如果字符串在后面的位置上再也没有这一字符，则不能弹出栈顶字符。为此，需要记录每个字符的剩余数量，当这个值为 0 时，就不能弹出栈顶字符了。
        stack = []
        remain = collections.Counter(s)  # 初始化为每个字符的计数，随着使用递减
        visited = set()

        for ch in s:
            if ch in visited:
                remain[ch] -= 1
                continue
            while len(stack) > 0 and stack[-1] > ch and remain[stack[-1]] != 0:
                e = stack.pop()
                visited.remove(e)
            stack.append(ch)
            visited.add(ch)
            remain[ch] -= 1

        return "".join(stack)
```



# 329. 矩阵中的最长递增路径

给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
      # 记忆化搜索
        direc = [(0,1),(0,-1),(-1,0),(1,0)]
        m = len(matrix)
        n = len(matrix[0])
        grid = [[0 for j in range(n)] for i in range(m)]

        def dfs(i,j): # 一种后序遍历的思想
            if grid[i][j] != 0: # 说明被更新过了，那么直接使用它的值，减少重复运算
                return grid[i][j]
            group = [1] # 基态为1，
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and matrix[new_i][new_j] > matrix[i][j]:
                    group.append(dfs(new_i,new_j)+1) # 所有邻居入组
            grid[i][j] = max(group) # 获得最大值
            return grid[i][j] 
        
        longest = 1
        for i in range(m):
            for j in range(n):
                grid[i][j] = dfs(i,j)
                longest = max(longest,grid[i][j])
        # print(grid)
        return longest
```

# 337. 打家劫舍 III

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        # 后续遍历
        memoDict = dict()  # key是node的内存地址,不加记忆化会超时

        def postOrder(node):
            if node == None:
                return 0
            if node in memoDict:
                return memoDict[node]
            leftPart = postOrder(node.left)
            rightPart = postOrder(node.right)
            # 偷了孩子的就不能偷现在的,偷现在的就不能偷孩子的，得偷孙子的
            leftAllson = 0
            if node.left != None:
                leftAllson += postOrder(node.left.left)+postOrder(node.left.right)
            rightAllson = 0
            if node.right != None:
                rightAllson = postOrder(node.right.left)+postOrder(node.right.right)
            now = max(leftPart+rightPart,node.val+leftAllson+rightAllson)
            memoDict[node] = now 
            return now
        
        ans = postOrder(root)
        return ans
```

# 348. 设计井字棋

请在 n × n 的棋盘上，实现一个判定井字棋（Tic-Tac-Toe）胜负的神器，判断每一次玩家落子后，是否有胜出的玩家。

在这个井字棋游戏中，会有 2 名玩家，他们将轮流在棋盘上放置自己的棋子。

在实现这个判定器的过程中，你可以假设以下这些规则一定成立：

      1. 每一步棋都是在棋盘内的，并且只能被放置在一个空的格子里；
    
      2. 一旦游戏中有一名玩家胜出的话，游戏将不能再继续；
    
      3. 一个玩家如果在同一行、同一列或者同一斜对角线上都放置了自己的棋子，那么他便获得胜利。

```python
class TicTacToe:
# 模拟
# 注意两条对角线可能被同时激活，不用elif

    def __init__(self, n: int):
        self.grid = [[" " for j in range(n)] for i in range(n)]
        self.n = n

    def check(self,x,y,now):
        n = self.n
        state1 = ""
        for p in range(n):
            state1 += self.grid[x][p]
        if state1 == now*n:
            return True
        state2 = ""
        for p in range(n):
            state2 += self.grid[p][y]
        if state2 == now*n:
            return True

        state3 = ""
        if x+y == n-1:
            temp = ""
            for tempX in range(n):
                temp += self.grid[tempX][n-tempX-1]
            if temp == now*n:
                return True 
        if x == y:
            temp = ""
            for tempX in range(n):
                temp += self.grid[tempX][tempX]
            if temp == now*n:
                return True
        return False

    def move(self, row: int, col: int, player: int) -> int:
        self.grid[row][col] = str(player)
        if self.check(row,col,str(player)) == True:
            return player
        else:
            return 0
```



# 352. 将数据流变为多个不相交区间

给你一个由非负整数 a1, a2, ..., an 组成的数据流输入，请你将到目前为止看到的数字总结为不相交的区间列表。

实现 SummaryRanges 类：

SummaryRanges() 使用一个空数据流初始化对象。
void addNum(int val) 向数据流中加入整数 val 。
int[][] getIntervals() 以不相交区间 [starti, endi] 的列表形式返回对数据流中整数的总结。

```python
class SummaryRanges:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # 不使用sortedcontainers
        self.sl = []
        # 加入哨兵
        self.sl.append([-2,-2]) # 注意这个哨兵要足够小，[-1,-1]不行
        self.sl.append([99999,99999])

    def addNum(self, val: int) -> None:
        # 获取插入的index
        index = bisect.bisect_left(self.sl,[val]) # 注意这个参数用[val]表示
        # 分开情况讨论
        if self.sl[index][0] <= val <= self.sl[index][-1]: # 已经存在，直接pass
            return 
        if self.sl[index-1][0] <= val <= self.sl[index-1][-1]: # 已经存在，直接pass,这一行很重要，
        # 例如已经是[6,8][inf,inf],插入7的时候会在[inf,inf]位置，那么此时需要检查的是前一个
            return 
        if val-1 == self.sl[index-1][-1] and val+1 == self.sl[index][0]: # 链接左右
            temp = [self.sl[index-1][0],self.sl[index][-1]]
            self.sl[index-1:index+1] = [temp]
        elif val-1 == self.sl[index-1][-1]: # 接在左边
            self.sl[index-1][-1] = val 
        elif val + 1 == self.sl[index][0]: # 接在右边
            self.sl[index][0] = val
        else:
            self.sl.insert(index,[val,val])

    def getIntervals(self) -> List[List[int]]:
        return self.sl[1:-1]
```

# 356. 直线镜像

在一个二维平面空间中，给你 n 个点的坐标。问，是否能找出一条平行于 y 轴的直线，让这些点关于这条直线成镜像排布？

**注意**：题目数据中可能有重复的点。

```python
class Solution:
    def isReflected(self, points: List[List[int]]) -> bool:
        # 平行于y轴的直线，那么需要根据所有的y坐标进行分组
        # 数学解法
        memoY = collections.defaultdict(set)
        for x,y in points:
            memoY[y].add(x)
        standard = sum(memoY[points[0][1]])/len(memoY[points[0][1]])
        for key in memoY:
            temp = sum(memoY[key])/len(memoY[key])
            if temp != standard:
                return False
            # temp是标准中轴，循环看是否所有元素都有对应
            for e in memoY[key]:
                mirror = 2*temp-e
                if mirror not in memoY[key]:
                    return False
        return True
```

# 368. 最大整除子集

给你一个由 无重复 正整数组成的集合 nums ，请你找出并返回其中最大的整除子集 answer ，子集中每一元素对 (answer[i], answer[j]) 都应当满足：
answer[i] % answer[j] == 0 ，或
answer[j] % answer[i] == 0
如果存在多个有效解子集，返回其中任何一个均可。

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        n = len(nums)
        nums.sort()
        # 所有数互不相同
        # dp[i]是以nums[i]结尾的最大长度
        dp = [1 for i in range(n)] # 初始化均为1
        for i in range(n):
            group = [1]
            for j in range(i):
                if nums[i]%nums[j] == 0: # 那么就可以把它加进去
                    group.append(dp[j]+1)
            dp[i] = max(group)
        # 由于最大整除子集不一定以最后一个元素结尾，那么先找到最大值,然后倒推
        ans = []
        longest = max(dp)
        index = dp.index(longest)
        ans.append(nums[index])
        while index >= 0:
            p = index-1
            while p >= 0 and (dp[p] != dp[index]-1 or nums[index]%nums[p] != 0):
                p -= 1
            if p >= 0 and (dp[p] == dp[index]-1 and nums[index]%nums[p] == 0):
                ans.append(nums[p])
            index = p
        return ans[::-1]
```

```go
func largestDivisibleSubset(nums []int) []int {
    sort.Ints(nums) // 排序不要忘了
    n := len(nums)
    dp := make([]int,n,n)
    dp[0] = 1
    for i:=0;i<n;i++ {
        group := []int{1}
        for j:=0;j<i;j++ {
            if nums[i]%nums[j] == 0 {
                group = append(group,dp[j]+1)
            }
            dp[i] = max(group)
        }
    }
    
    longest := max(dp)
    index := -1
    for i:=0;i<n;i++ {
        if dp[i] == longest {
            index = i
            break
        }
    }
    
    //fmt.Println(dp)
    ans := []int{nums[index]}
    
    for index >= 0 {
        p := index - 1
        for p >= 0 && (dp[p] != dp[index]-1 || nums[index]%nums[p] != 0) {
            p -= 1
        }
        if p >= 0 && (dp[p] == dp[index]-1 && nums[index]%nums[p] == 0) {
            ans = append(ans,nums[p])
        }
        index = p
    }
    
    mid := len(ans)/2
    k := len(ans)
    for i:=0;i<mid;i++ {
        ans[i],ans[k-i-1] = ans[k-i-1],ans[i]
    }
    return ans
}

func max (arr []int) int {
    temp := arr[0]
    for _,v := range(arr) {
        if temp < v {
            temp = v
        }
    }
    return temp
}
```



# 375. 猜数字大小 II

我们正在玩一个猜数游戏，游戏规则如下：

我从 1 到 n 之间选择一个数字。
你来猜我选了哪个数字。
如果你猜到正确的数字，就会 赢得游戏 。
如果你猜错了，那么我会告诉你，我选的数字比你的 更大或者更小 ，并且你需要继续猜数。
每当你猜了数字 x 并且猜错了的时候，你需要支付金额为 x 的现金。如果你花光了钱，就会 输掉游戏 。
给你一个特定的数字 n ，返回能够 确保你获胜 的最小现金数，不管我选择那个数字 。

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # 利用记忆化搜索,传入参数为二维参数tuple
        memo = dict()
        memo[(1,1)] = 0

        def recur(i,j): # 表示从i～j闭区间的搜索
            if (i,j) in memo:
                return memo[(i,j)]
            group = [0xffffffff]
            if j > i: # 注意这里
                for k in range(i,j+1):
                    group.append(k+max(recur(i,k-1),recur(k+1,j)))
            memo[(i,j)] = min(group)
            if memo[(i,j)] == 0xffffffff:
                memo[(i,j)] = 0
            return memo[(i,j)]
        
        return recur(1,n)
```

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        memo = dict()
		# 好理解一点的
        def recur(i,j):# 闭区间搜索
            if i >= j:
                return 0xffffffff
            if (i,j) in memo:
                return memo[(i,j)]
            group = []

            for k in range(i,j+1):
                left = recur(i,k-1) # 左半边
                right = recur(k+1,j) # 右半边
                temp = [0]
                if left != 0xffffffff:
                    temp.append(left)
                if right != 0xffffffff:
                    temp.append(right)
                val = k + max(temp)
                group.append(val)

            memo[(i,j)] = min(group)
            return memo[(i,j)]
        
        ans = recur(1,n)
        return ans if ans != 0xffffffff else 0
```

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0 for j in range(n+1)] for i in range(n+1)]
        # dp[i][j]是闭合区间
        # dp[i][j] = min(max(k+dp[i][k-1],k+dp[k+1][j])) # k的范围为 k>=i, k<=j
        for i in range(n,0,-1):
            for j in range(i,n+1):
                group = []
                for k in range(i,j): # 范围限制不来源于坐标有意义，而是来源于选择意义，k一定是遍历闭区间【i,j】获得
                    # 但是由于[k+1]不能越界所以右边界为j-1的闭区间
                    group.append(max(dp[i][k-1],dp[k+1][j])+k)
                dp[i][j] = min(group) if len(group)!=0 else 0
        # print(dp)
        return dp[1][n]

```

# 391. 完美矩形

给你一个数组 rectangles ，其中 rectangles[i] = [xi, yi, ai, bi] 表示一个坐标轴平行的矩形。这个矩形的左下顶点是 (xi, yi) ，右上顶点是 (ai, bi) 。

如果所有矩形一起精确覆盖了某个矩形区域，则返回 true ；否则，返回 false 。

```python
class Solution:
    def isRectangleCover(self, rectangles: List[List[int]]) -> bool:
        countPoints = collections.defaultdict(int) # 存坐标
        # 这一题的陷阱是不能用左下角和右下角求完美面积，然后判断所有小矩形是否和其总面积相等
        # 因为可能缺一块并且重叠一块，这样面积也有可能相等
        # 采取扫描做法
        # 完美矩形存在四个点止出现一次，而其他点要么出现四次【例如十字】，要么出现两次【非十字】
        
        def toPoints(a,b,c,d):
            return ((a,b),(c,d),(a,d),(c,b))
        
        def calcArea(a,b,c,d):
            return abs(a-c)*abs(b-d)
        
        allArea = 0
        for a,b,c,d in rectangles:
            for eachPoint in toPoints(a,b,c,d):
                countPoints[eachPoint] += 1
            allArea += calcArea(a,b,c,d)

        allP = 4*len(rectangles)
        one,two,four = 0 
        oneGroup = []
        for key in countPoints:
            if countPoints[key] == 1:
                oneGroup.append(key)
                one += 1
            elif countPoints[key] == 2:
                two += 2
            elif countPoints[key] == 4:
                four += 4
            else:
                return False

        if one != 4 :
            return False 
        # 还需要判断这四个点是否是矩形
        t = 0
        for a,b in oneGroup: # 矩形的每个坐标出现两次，用异或
            t ^= a 
            t ^= b
        if t != 0:
            return False
        oneGroup.sort()
        (a,b),(c,d) = oneGroup[0],oneGroup[-1]
        if calcArea(a,b,c,d) != allArea:
            return False
        if one+two+four != allP:
            return False 
        return True
```

# 396. 旋转函数

给定一个长度为 n 的整数数组 A 。

假设 Bk 是数组 A 顺时针旋转 k 个位置后的数组，我们定义 A 的“旋转函数” F 为：

F(k) = 0 * Bk[0] + 1 * Bk[1] + ... + (n-1) * Bk[n-1]。

计算F(0), F(1), ..., F(n-1)中的最大值。

注意:
可以认为 n 的值小于 10^5。

```python
class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        # 利用前缀和
        # 相邻的差为 sig(B0...Bn-1) + n*Bk 
        first = sum(i*nums[i] for i in range(len(nums)))
        ans = first 
        pre = sum(nums)
        n = len(nums)

        for i in range(n-1,-1,-1):
            first += pre - n*nums[i]
            ans = max(ans,first)
        return ans
```

# 407. 接雨水 II

给你一个 `m x n` 的矩阵，其中的值均为非负整数，代表二维高度图每个单元的高度，请计算图中形状最多能接多少体积的雨水。

```python
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        m,n = len(heightMap),len(heightMap[0])
        if m <= 2 or n <= 2:
            return 0
        
        # 小根堆做法
        # 把外面一圈先加入
        # 堆始终是一个闭合环路！
        # 类似于dj算法
        heap = []
        direc = [(0,1),(0,-1),(-1,0),(1,0)]
        visited = [[False for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 or i == m-1 or j == 0 or j == n-1:
                    heapq.heappush(heap,(heightMap[i][j],[i,j]))
                    visited[i][j] = True 
        
        ans = 0

        while len(heap) != 0:
            element = heapq.heappop(heap)
            theH = element[0]
            i,j = element[1]
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False:
                    if heightMap[new_i][new_j] < theH:
                        ans += theH - heightMap[new_i][new_j]
                    visited[new_i][new_j] = True 
                    heapq.heappush(heap,(max(theH,heightMap[new_i][new_j]),[new_i,new_j]))
        
        return ans
```

# 419. 甲板上的战舰

给定一个二维的甲板， 请计算其中有多少艘战舰。 战舰用 'X'表示，空位用 '.'表示。 你需要遵守以下规则：

给你一个有效的甲板，仅由战舰或者空位组成。
战舰只能水平或者垂直放置。换句话说,战舰只能由 1xN (1 行, N 列)组成，或者 Nx1 (N 行, 1 列)组成，其中N可以是任意大小。
两艘战舰之间至少有一个水平或垂直的空位分隔 - 即没有相邻的战舰。

```python
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        # 方法1，朴素变更流dfs
        m,n = len(board),len(board[0])
        count = 0
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        def dfs(i,j):
            board[i][j] = "."
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and board[new_i][new_j] == "X":
                    dfs(new_i,new_j)
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == "X":
                    count += 1
                    dfs(i,j)
        return count
```

```python
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        # 方法2: 找舰头， 只有这一位是X且左边和上边都是。的时候才计数,注意处理越界
        count = 0
        m,n = len(board),len(board[0])
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'X':
                    s1 = True
                    if i>=1 and board[i-1][j] != ".": s1 = False
                    s2 = True 
                    if j>=1 and board[i][j-1] != ".": s2 = False 
                    if s1 and s2:
                        count += 1
        return count
```



# 433. 最小基因变化

一条基因序列由一个带有8个字符的字符串表示，其中每个字符都属于 "A", "C", "G", "T"中的任意一个。

假设我们要调查一个基因序列的变化。一次基因变化意味着这个基因序列中的一个字符发生了变化。

例如，基因序列由"AACCGGTT" 变化至 "AACCGGTA" 即发生了一次基因变化。

与此同时，每一次基因变化的结果，都需要是一个合法的基因串，即该结果属于一个基因库。

现在给定3个参数 — start, end, bank，分别代表起始基因序列，目标基因序列及基因库，请找出能够使起始基因序列变化为目标基因序列所需的最少变化次数。如果无法实现目标变化，请返回 -1。

注意：

起始基因序列默认是合法的，但是它并不一定会出现在基因库中。
如果一个起始基因序列需要多次变化，那么它每一次变化之后的基因序列都必须是合法的。
假定起始基因序列与目标基因序列是不一样的。

```python
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        # BFS,这一题数据量小
        if end not in bank:
            return -1
        if start not in bank:
            bank.append(start)
        # 建立关系
        n = len(bank)
        graph = collections.defaultdict(list)
        for i in range(n):
            for j in range(i+1,n):
                w1 = bank[i]
                w2 = bank[j]
                diff = 0
                for t in range(8):
                    if w1[t] != w2[t]:
                        diff += 1
                    if diff >= 2:
                        break 
                if diff <= 1:
                    graph[w1].append(w2)
                    graph[w2].append(w1)
        
        visited = set()
        queue = [start]
        visited.add(start)
        steps = 0
        while len(queue) != 0:
            new_queue = []
            for w in queue:
                if w == end:
                    return steps 
                for neigh in graph[w]:
                    if neigh not in visited:
                        new_queue.append(neigh)
                        visited.add(neigh)
            steps += 1
            queue = new_queue 
        return -1
```



# 436. 寻找右区间

给你一个区间数组 intervals ，其中 intervals[i] = [starti, endi] ，且每个 starti 都 不同 。

区间 i 的 右侧区间 可以记作区间 j ，并满足 startj >= endi ，且 startj 最小化 。

返回一个由每个区间 i 的 右侧区间 的最小起始位置组成的数组。如果某个区间 i 不存在对应的 右侧区间 ，则下标 i 处的值设为 -1 。

```python
class Solution:
    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        # 利用元组当key,值是原来的index
        cp = intervals.copy()
        memoDict = dict()
        for k,v in enumerate(intervals):
            start,end = v
            memoDict[(start,end)] = k 
        intervals.sort()
        # 
        mirror = dict()
        for start,end in intervals:
            index = bisect.bisect_left(intervals,[end]) # 找到插入位置
            if index == len(intervals):
                mirror[(start,end)] = -1
            else:
                # 得到它现在的位置之后，找到这个元祖，
                # 还原成原来位置，然后记录进入镜
                tp = tuple(intervals[index]) #
                mirror[(start,end)] = memoDict[tp]
        
        ans = []
        for pair in cp: # 扫描原始数据和镜
            ans.append(mirror[tuple(pair)])
        return ans

```

# 458. 可怜的小猪

有 buckets 桶液体，其中 正好 有一桶含有毒药，其余装的都是水。它们从外观看起来都一样。为了弄清楚哪只水桶含有毒药，你可以喂一些猪喝，通过观察猪是否会死进行判断。不幸的是，你只有 minutesToTest 分钟时间来确定哪桶液体是有毒的。

喂猪的规则如下：

选择若干活猪进行喂养
可以允许小猪同时饮用任意数量的桶中的水，并且该过程不需要时间。
小猪喝完水后，必须有 minutesToDie 分钟的冷却时间。在这段时间里，你只能观察，而不允许继续喂猪。
过了 minutesToDie 分钟后，所有喝到毒药的猪都会死去，其他所有猪都会活下来。
重复这一过程，直到时间用完。
给你桶的数目 buckets ，minutesToDie 和 minutesToTest ，返回在规定时间内判断哪个桶有毒所需的 最小 猪数。

```python
class Solution:
    def poorPigs(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
        # 超级厉害的数学解法
        # 先确定能测几轮
        circles = int(minutesToTest//minutesToDie)
        # n+1个桶子，其中一个有毒，那么用一只猪去测试的时候只需要测试n次，就可以完全覆盖所有可能性
        # 那么一只猪可以携带n+1的信息量，那么 (n+1)**(pigs) >= buckets的最小pig
        # 而n来源于对轮数，如果有5轮，可以鉴别六个桶子
        # 求(circles+1)**(pigs) >= buckets ,取对数
        pigs = math.ceil(math.log(buckets)/math.log(circles+1))
        return pigs
```



# 467. 环绕字符串中唯一的子字符串

把字符串 s 看作是“abcdefghijklmnopqrstuvwxyz”的无限环绕字符串，所以 s 看起来是这样的："...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd....". 

现在我们有了另一个字符串 p 。你需要的是找出 s 中有多少个唯一的 p 的非空子串，尤其是当你的输入是字符串 p ，你需要输出字符串 s 中 p 的不同的非空子串的数目。 

注意: p 仅由小写的英文字母组成，p 的大小可能超过 10000。

```python
class Solution:
    def findSubstringInWraproundString(self, p: str) -> int:
        # 技巧型前缀和
        # 计算的是唯一的非空子串,去重是要点
        pre = 1
        n = len(p)
        lengthMap = collections.defaultdict(int)
        lengthMap[p[0]] = 1
        for i in range(1,n):
            if ord(p[i])-ord(p[i-1]) == 1 or ord(p[i])-ord(p[i-1]) == -25:
                pre += 1
            else:
                pre = 1
            # 点睛之笔
            # 取以某个字母结尾的最长长度，那么它有多长，就是有多少个不同字符串
            lengthMap[p[i]] = max(lengthMap[p[i]],pre)
        
        return sum(lengthMap.values())

```

```python
func findSubstringInWraproundString(p string) int {
    p = "^" + p // 通用做法
    visitedMap := make(map[byte]int) // 码点
    
    n := len(p)
    pre := 0
    for i:=1;i<n;i++ {
        if p[i]-p[i-1] == 1 || p[i-1]-p[i] == 25 {
            pre += 1   
        } else {
            pre = 1
        }
        visitedMap[p[i]] = max(pre,visitedMap[p[i]])
    }
    ans := 0
    
    for _,v := range(visitedMap) {
        ans += v
    }
    return ans
    
}

func max (a,b int) int {
    if a > b {
        return a
    } else {
        return b
    }
}
```



# 482. 密钥格式化

有一个密钥字符串 S ，只包含字母，数字以及 '-'（破折号）。其中， N 个 '-' 将字符串分成了 N+1 组。

给你一个数字 K，请你重新格式化字符串，使每个分组恰好包含 K 个字符。特别地，第一个分组包含的字符个数必须小于等于 K，但至少要包含 1 个字符。两个分组之间需要用 '-'（破折号）隔开，并且将所有的小写字母转换为大写字母。

给定非空字符串 S 和数字 K，按照上面描述的规则进行格式化。

```python
class Solution:
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        # 数据范围为1～10w,不考虑效率。。。
        stack = collections.deque()
        for ch in s:
            if ch != "-":
                if ch.isalpha():
                    stack.append(ch.upper())
                else:
                    stack.append(ch)
        # 分成k组
        ans = [] # 使用list接收
        n = len(stack)
        if n == 0:
            return ""
        countGaps = math.ceil(n/k)-1 # 计算有多少个横杠
        ans = ""
        while countGaps != 0:
            temp = []
            for i in range(k):
                temp.append(stack.pop())
            temp = temp[::-1]
            ans = "-"+"".join(temp) + ans
            countGaps -= 1

        temp = []
        while len(stack) != 0:
            temp.append(stack.pop())
        temp = temp[::-1]
        ans = "".join(temp) + ans
        return ans


```

# 486. 预测赢家

给你一个整数数组 nums 。玩家 1 和玩家 2 基于这个数组设计了一个游戏。

玩家 1 和玩家 2 轮流进行自己的回合，玩家 1 先手。开始时，两个玩家的初始分值都是 0 。每一回合，玩家从数组的任意一端取一个数字（即，nums[0] 或 nums[nums.length - 1]），取到的数字将会从数组中移除（数组长度减 1 ）。玩家选中的数字将会加到他的得分上。当数组中没有剩余数字可取时，游戏结束。

如果玩家 1 能成为赢家，返回 true 。如果两个玩家得分相等，同样认为玩家 1 是游戏的赢家，也返回 true 。你可以假设每个玩家的玩法都会使他的分数最大化。

```python
# 记忆化递归python
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        # 记忆化递归
        memoDict = dict()
        def getScore(nums,leftBound,rightBound):
            if leftBound == rightBound:
                return nums[leftBound]
            if (leftBound,rightBound) in memoDict:
                return memoDict[(leftBound,rightBound)]
            # 记忆化思路：选择这个数的时候，对手选择另一个数的最大分差
            state1 = nums[leftBound]-getScore(nums,leftBound+1,rightBound)
            state2 = nums[rightBound]-getScore(nums,leftBound,rightBound-1)
            score = max(state1,state2) # 我需要取分差大的
            memoDict[(leftBound,rightBound)] = score
            return score 
        
        return getScore(nums,0,len(nums)-1) >= 0
```

```go
// 记忆化递归 go
func PredictTheWinner(nums []int) bool {
    memo := make(map[string]int)
    return getScore(nums,0,len(nums)-1,memo) >= 0
}

func getScore(nums []int,leftBound int,rightBound int,memo map[string]int) int {
        if leftBound == rightBound {
            return nums[leftBound]
        }
        now := strconv.Itoa(leftBound) + "#" + strconv.Itoa(rightBound)
        v,ok := memo[now]
        if ok {
            return v
        }
        state1 := nums[leftBound] - getScore(nums,leftBound+1,rightBound,memo)
        state2 := nums[rightBound] - getScore(nums,leftBound,rightBound-1,memo)
        state := max(state1,state2)
        memo[now] = state
        return memo[now]
    }

func max(a,b int) int {
    if a > b {
        return a
    } else {
        return b
    }
}

```

```python
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        # 动态规划，dp[i][j]的意思是i~j闭区间内做选择能拿到的最大分差
        n = len(nums)
        # 基态填充
        dp = [[0 for i in range(n)] for i in range(n)]
        for i in range(n):
            dp[i][i] = nums[i]
        # 状态转移方程 dp[i][j]
        # 选择left的时候，相对分数差为 nums[i] - dp[i+1][j]
        # 选择right的时候，相对分数差为 nums[j] - dp[i][j-1]
        # 画图可知填表方向，由左和下得到本位置
        for i in range(n-2,-1,-1):
            for j in range(i+1,n):
                dp[i][j] = max(nums[i]-dp[i+1][j],nums[j]-dp[i][j-1])
        # 返回第一行的最后一个值
        return dp[0][-1] >= 0
```

```go
func PredictTheWinner(nums []int) bool {
    n := len(nums)
    dp := make([][]int,n)
    for i := 0; i < n; i++ {
        dp[i] = make([]int,n)
        dp[i][i] = nums[i]
    }

    for i := n-2; i > -1; i-- {
        for j := i+1; j < n; j ++ {
            state1 := nums[i] - dp[i+1][j]
            state2 := nums[j] - dp[i][j-1]
            dp[i][j] = max(state1,state2)
        }
    }
    
    return dp[0][n-1] >= 0
}

func max(a,b int) int {
    if a > b {
        return a
    } else {
        return b
    }
}
```

# 488. 祖玛游戏

你正在参与祖玛游戏的一个变种。

在这个祖玛游戏变体中，桌面上有 一排 彩球，每个球的颜色可能是：红色 'R'、黄色 'Y'、蓝色 'B'、绿色 'G' 或白色 'W' 。你的手中也有一些彩球。

你的目标是 清空 桌面上所有的球。每一回合：

从你手上的彩球中选出 任意一颗 ，然后将其插入桌面上那一排球中：两球之间或这一排球的任一端。
接着，如果有出现 三个或者三个以上 且 颜色相同 的球相连的话，就把它们移除掉。
如果这种移除操作同样导致出现三个或者三个以上且颜色相同的球相连，则可以继续移除这些球，直到不再满足移除条件。
如果桌面上所有球都被移除，则认为你赢得本场游戏。
重复这个过程，直到你赢了游戏或者手中没有更多的球。
给你一个字符串 board ，表示桌面上最开始的那排球。另给你一个字符串 hand ，表示手里的彩球。请你按上述操作步骤移除掉桌上所有球，计算并返回所需的 最少 球数。如果不能移除桌上所有的球，返回 -1 。

```python
COLORS = ["R","Y","B","G","W"]

class Solution:
    def findMinStep(self, board: str, hand: str) -> int:
        # 检测是否所有颜色即存在，又不足三个
        ct_b = Counter(board)
        ct_h = Counter(hand)
        # Counter自带defaultdict效果
        for key in ct_b:
            if ct_b[key] + ct_h[key] < 3:
                return -1
        
        originLength = len(hand)

        @lru_cache(None)
        def dfs(bd,hd): #bd传入的是字符串，hd传入的是元组数组【因为要使用lru，传入必须不可变类型】
            if len(bd) <= 0:
                return originLength - sum(hd)
            n = len(bd)
            ans = 0xffffffff 
            for i,v in enumerate(hd):
                if v != 0:
                    cp = list(hd) # 转成list进行操作
                    cp[i] -= 1
                    new_hd = tuple(cp) # 转成tuple进行dfs
                    for insert in range(n+1):
                        ans = min(ans,dfs(eliminate(bd[:insert]+COLORS[i]+bd[insert:]),new_hd))
            return ans

        
        @lru_cache(None)
        def eliminate(bd):
            l,r = 0,0
            while l < len(bd):
                while r < len(bd) and bd[l] == bd[r]:
                    r += 1
                if r - l > 2:
                    return eliminate(bd[:l]+bd[r:])
                l = r 
            return bd 
        
        start = [ct_h[c] for c in COLORS]
        ans = dfs(board,tuple(start))
        return ans if ans != 0xffffffff else -1
```

```python
class Solution:
    def findMinStep(self, board: str, hand: str) -> int:
        # 用例太弱，其实很容易超时
        ct_b = collections.Counter(board)
        ct_h = collections.Counter(hand)
        for key in ct_b:
            if ct_b[key] + ct_h[key] < 3:
                return -1
        
        originLength = len(hand)

        colors = ["R","Y","B","G","W"]
        memo_dfs = collections.defaultdict()


        # 非lru_cache
        memo_bd = collections.defaultdict()
        def eliminate(bd):
            if bd in memo_bd:
                return memo_bd[bd]
            l,r = 0,0
            while l < len(bd):
                state = False 
                while r < len(bd) and bd[l] == bd[r]:
                    r += 1
                if r - l > 2:
                    return eliminate(bd[:l]+bd[r:])
                l = r 
            memo_bd[bd] = bd
            return memo_bd[bd]
		
		# 非lru_cache
        def dfs(boardState,handState):
            # print(boardState,handState)
            mark = boardState + str(handState)
            if mark in memo_dfs:
                return memo_dfs[mark]
            if len(boardState) == 0:
                return originLength - sum(handState)
            
            n = len(boardState) #插入用 
            ans = 0xffffffff 

            for i,v in enumerate(handState):
                if v > 0:
                    handState[i] -= 1
                    for insert in range(n+1):
                        ans = min(ans,dfs(eliminate(boardState[:insert]+colors[i]+boardState[insert:]),handState))
                    handState[i] += 1
            memo_dfs[mark] = ans
            return ans
        
        start = [ct_h[c] for c in colors]
        ans = dfs(board,start)
        return ans if ans != 0xffffffff else -1
```



# 492. 构造矩形

作为一位web开发者， 懂得怎样去规划一个页面的尺寸是很重要的。 现给定一个具体的矩形页面面积，你的任务是设计一个长度为 L 和宽度为 W 且满足以下要求的矩形的页面。要求：

1. 你设计的矩形页面必须等于给定的目标面积。

2. 宽度 W 不应大于长度 L，换言之，要求 L >= W 。

3. 长度 L 和宽度 W 之间的差距应当尽可能小。
你需要按顺序输出你设计的页面的长度 L 和宽度 W。

```python
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        # 因式分解,正向搜索
        start = 1
        left = []
        right = []
        while start <= int(math.sqrt(area)):
            if area%start == 0:
                left.append(start)
                right.append(area//start)
            start += 1
        
        return [right[-1],left[-1]]
```

```python
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        # 因式分解,从根号附近逆向搜索
        start = int(math.sqrt(area)) + 1
        left = []
        right = []
        while start > 0:
            if area%start == 0:
                left.append(start)
                right.append(area//start)
                break 
            start -= 1
        
        ans = [right[-1],left[-1]]
        ans.sort(reverse = True)
        return ans
```

```python
func constructRectangle(area int) []int {
    // 菜狗版本go
    start := int(math.Sqrt(float64(area)))
    left := -1
    right := -1
    for start > 0 {
        if area%start == 0 {
            left = start 
            right = area/start
            break 
        }
        start -= 1        
    }
    ans := make([]int,0,0)
    ans = append(ans,left)
    ans = append(ans,right)
    ans = sort(ans)
    return ans
}

func sort(arr []int) []int {
    if arr[0] < arr[1] {
        arr[0],arr[1] = arr[1],arr[0]
    }
    final := make([]int,0,0)
    final = append(final,arr[0])
    final = append(final,arr[1])
    return final
}
```

# 519. 随机翻转矩阵
给你一个 m x n 的二元矩阵 matrix ，且所有值被初始化为 0 。请你设计一个算法，随机选取一个满足 matrix[i][j] == 0 的下标 (i, j) ，并将它的值变为 1 。所有满足 matrix[i][j] == 0 的下标 (i, j) 被选取的概率应当均等。

尽量最少调用内置的随机函数，并且优化时间和空间复杂度。

实现 Solution 类：

Solution(int m, int n) 使用二元矩阵的大小 m 和 n 初始化该对象
int[] flip() 返回一个满足 matrix[i][j] == 0 的随机下标 [i, j] ，并将其对应格子中的值变为 1
void reset() 将矩阵中所有的值重置为 0

```python
# 拒绝采样法
class Solution:

    def __init__(self, m: int, n: int):
        # 拒绝采样法
        self.refuse = set()
        self.index = m*n - 1 # python的random是双闭合区间
        self.m = m
        self.n = n


    def flip(self) -> List[int]:
        get = False
        m = self.m
        n = self.n
        while not get:
            index = random.randint(0,self.index)
            if index not in self.refuse:
                self.refuse.add(index)
                get = True 
            i = index//n
            j = index - i*n
        return [i,j]


    def reset(self) -> None:
        self.refuse = set()

```

# 562. 矩阵中最长的连续1线段

给定一个01矩阵 **M**，找到矩阵中最长的连续1线段。这条线段可以是水平的、垂直的、对角线的或者反对角线的。

```python
class Solution:
    def longestLine(self, mat: List[List[int]]) -> int:
        # 实质上只需要考虑两次dp
        # 一次考虑，左边，上边，左上
        # 一次考虑，右边，上边，右上
        m,n = len(mat),len(mat[0])
        longest = 0
        dp1 = [[[0,0,0] for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    dp1[i][j][0] = dp1[i][j-1][0]+1 if j>=1 else 1 # 左边
                    dp1[i][j][1] = dp1[i-1][j][1]+1 if i>=1 else 1 # 上边
                    dp1[i][j][2] = dp1[i-1][j-1][2]+1 if i>=1 and j>=1 else 1 # 左上
                longest = max(longest,max(dp1[i][j]))

        
        dp2 = [[[0,0,0] for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n-1,-1,-1):
                if mat[i][j] == 1:
                    dp2[i][j][0] = dp2[i][j+1][0]+1 if j+1<n else 1 # 右边
                    dp2[i][j][1] = dp2[i-1][j][1]+1 if i>=1 else 1 # 上边
                    dp2[i][j][2] = dp2[i-1][j+1][2]+1 if i>=1 and j+1<n else 1 # 右上
                longest = max(longest,max(dp2[i][j]))
        
        return longest
```

```python
class Solution:
    def longestLine(self, mat: List[List[int]]) -> int:
        # 实质上只需要考虑两次dp
        # 一次考虑，左边，上边，左上
        # 一次考虑，右边，上边，右上
        m,n = len(mat),len(mat[0])
        longest = 0
        dp1 = [[[0,0,0] for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    dp1[i][j][0] = dp1[i][j-1][0]+1 if j>=1 else 1 # 左边
                    dp1[i][j][1] = dp1[i-1][j][1]+1 if i>=1 else 1 # 上边
                    dp1[i][j][2] = dp1[i-1][j-1][2]+1 if i>=1 and j>=1 else 1 # 左上
                longest = max(longest,max(dp1[i][j]))

        # 省略右边和上边
        dp2 = [[0 for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n-1,-1,-1):
                if mat[i][j] == 1:
                    dp2[i][j] = dp2[i-1][j+1]+1 if i>=1 and j+1<n else 1 # 右上
                longest = max(longest,dp2[i][j])
        
        return longest
```



# 594. 最长和谐子序列

和谐数组是指一个数组里元素的最大值和最小值之间的差别 正好是 1 。

现在，给你一个整数数组 nums ，请你在所有可能的子序列中找到最长的和谐子序列的长度。

数组的子序列是一个由数组派生出来的序列，它可以通过删除一些元素或不删除元素、且不改变其余元素的顺序而得到。

```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        # 每个数做个Counter
        longgest = 0
        ct = collections.Counter(nums)
        # 要有差别且正好是1
        for key in ct:
            if key - 1 in ct:
                longgest = max(longgest,ct[key]+ct[key-1])
            if key + 1 in ct:
                longgest = max(longgest,ct[key]+ct[key+1])
        return longgest
```

# 621. 任务调度器

给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。

然而，两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。

你需要计算完成所有任务所需要的 最短时间 。

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # 参照热评第一题解
        # 先获取最大频次，构建这么多个桶， 然后每个桶容量为n+1
        # 计算可能性1： [(频次-1)*(n+1)]+[具有最大频次的任务数目]
        ct = collections.Counter(tasks)
        # 
        maxTimes = 1
        for key in ct:
            if ct[key] > maxTimes:
                maxTimes = ct[key]
        taskNum = 0
        for key in ct:
            if ct[key] == maxTimes:
                taskNum += 1
        state1 = ((maxTimes-1)*(n+1)) + taskNum
        # 可能性2，任务足够多，安排时可以直接考虑大小
        state2 = len(tasks)
        return max(state1,state2)
        
```

# 628. 三个数的最大乘积

给你一个整型数组 `nums` ，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        # 考虑数字的组成。取
        # 0，3；1，2；2，1；3；0四个状态中最大的一个
        nums.sort()
        state1 = nums[-1]*nums[-2]*nums[-3]
        state2 = nums[0]*nums[-1]*nums[-2]
        state3 = nums[0]*nums[1]*nums[-1]
        state4 = nums[0]*nums[1]*nums[2]

        return max(state1,state2,state3,state4)
```

# 629. K个逆序对数组

给出两个整数 n 和 k，找出所有包含从 1 到 n 的数字，且恰好拥有 k 个逆序对的不同的数组的个数。

逆序对的定义如下：对于数组的第i个和第 j个元素，如果满i < j且 a[i] > a[j]，则其为一个逆序对；否则不是。

由于答案可能很大，只需要返回 答案 mod 109 + 7 的值。

```python
# 版本1: 未优化的直接思路版本
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        mod = 10**9 + 7
        # dp, dp[n][k]是使用1~n，恰好有k. 
        dp = [[0 for j in range(k+1)] for i in range(n+1)]
        # dp[i][j]是，使用了1～i-1.枚举插入i【可以插入i次】额外增添的逆序
        # 头插入增添了i-1对，次位差增添了i-2对，。。。尾插增添了0对
        # 那么还需要j-(i-1);j-(i-2);...;j-(0)
        # dp[i][j] = dp[i-1][j-(i-1)] + dp[i-1][j-(i-2)] + ... + dp[i-1][j-0] 共计i项目,需要上一行
        # 则遍历顺序从左到右，从上到下

        # 无优化先跑一遍
        # 基态，dp[1][0] = 1
        dp[1][0] = 1

        for i in range(2,n+1):
            for j in range(0,k+1):
                temp = 0
                for t in range(j,j-(i-1)-1,-1):
                    if t >= 0:
                        temp += dp[i-1][t]
                    else:
                        break
                dp[i][j] = temp 
        return dp[n][k]%mod
```

```python
# 版本2: 略微优化dp序列,思想来源于消元法
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        mod = 10**9 + 7
        # dp, dp[n][k]是使用1~n，恰好有k. 
        dp = [[0 for j in range(k+1)] for i in range(n+1)]
        # dp[i][j]是，使用了1～i-1.枚举插入i【可以插入i次】额外增添的逆序
        # 头插入增添了i-1对，次位差增添了i-2对，。。。尾插增添了0对
        # 那么还需要j-(i-1);j-(i-2);...;j-(0)
        # dp[i][j] = dp[i-1][j-(i-1)] + dp[i-1][j-(i-2)] + ... + dp[i-1][j-0] 共计i项目,需要上一行
        # 则遍历顺序从左到右，从上到下

        # 基态，dp[1][0] = 1
        dp[1][0] = 1
        # 优化dp[i][j] =                        dp[i-1][j-(i-1)] + dp[i-1][j-(i-2)] + ... + dp[i-1][j-0] 共计i项目
        #    dp[i][j-1] = dp[i-1][j-1-(i-1)] + dp[i-1][j-1-(i-2)] + ... + dp[i-1][j-1-0] 共计i项目
        # 错位减得到： dp[i][j] = dp[i][j-1] - dp[i-1][j-1-(i-1)] + dp[i-1][j] 注意符号

        for i in range(2,n+1): # 注意从2开始
            for j in range(0,k+1): # 注意符号
                s1 = dp[i][j-1] if j-1 >=0 else 0
                s2 = -dp[i-1][j-1-(i-1)] if j-1-(i-1)>=0 else 0
                s3 = dp[i-1][j] if j>=0 else 0
                dp[i][j] = s1+s2+s3

        return dp[n][k]%mod

```

```python
# 进一步优化和精简，在途中就算好了mod
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        dp = [[0 for j in range(k+1)] for i in range(n+1)]
        dp[1][0] = 1
        # dp[i][j] = dp[i-1][j-0]+dp[i-1][j-1]+...+dp[i-1][j-(i-1)]
        # dp[i][j-1] =            dp[i-1][j-1-0] + dp[i-1][j-1-1] + ... + dp[i-1][j-1-(i-1)]
        # dp[i][j]-dp[i][j-1] = dp[i-1][j-0]-dp[i-1][j-i]
        # dp[i][j] = dp[i][j-1] + dp[i-1][j] - dp[i-1][j-i] 
        mod = 10**9+7
        for i in range(2,n+1): # 注意从2开始
            for j in range(k+1):
                s1 = dp[i][j-1] if j >= 1 else 0
                s2 = dp[i-1][j]
                s3 = -dp[i-1][j-i] if j >= i else 0
                dp[i][j] = (s1 + s2 + s3)%mod
        return dp[n][k]

```

```go
func kInversePairs(n int, k int) int {
    dp := make([][]int,n+1,n+1)
    for i:=0;i<n+1;i++{
        dp[i] = make([]int,k+1,k+1)
    }
    dp[1][0] = 1
    mod := 1000000007
    // dp[i][j] = dp[i-1][j-(i-1)] + dp[i-1][j-(i-2)] +...+dp[i-1][j-0]
    // dp[i][j-1] = dp[i-1][j-1-(i-1)] + ... + dp[i-1][j-1 - (0)]
    // dp[i][j]-dp[i][j-1] = -dp[i-1][j-i] + dp[i-1][j]
    // dp[i][j] = dp[i][j-1]+dp[i-1][j]-dp[i-1][j-i]
    
    for i:=2;i<n+1;i++ {
        for j:=0;j<k+1;j++ {
            s1,s2,s3 := 0,0,0
            if j-1 >= 0{
                s1 = dp[i][j-1]
            }
            s2 = dp[i-1][j]
            if j >= i {
                s3 = -dp[i-1][j-i]
            }
            dp[i][j] = (s1+s2+s3)%mod
            if dp[i][j] < 0 {
                dp[i][j] += mod
            }
        }
    }
    return dp[n][k]
}
```



# 638. 大礼包

在 LeetCode 商店中， 有 n 件在售的物品。每件物品都有对应的价格。然而，也有一些大礼包，每个大礼包以优惠的价格捆绑销售一组物品。

给你一个整数数组 price 表示物品价格，其中 price[i] 是第 i 件物品的价格。另有一个整数数组 needs 表示购物清单，其中 needs[i] 是需要购买第 i 件物品的数量。

还有一个数组 special 表示大礼包，special[i] 的长度为 n + 1 ，其中 special[i][j] 表示第 i 个大礼包中内含第 j 件物品的数量，且 special[i][n] （也就是数组中的最后一个整数）为第 i 个大礼包的价格。

返回 确切 满足购物清单所需花费的最低价格，你可以充分利用大礼包的优惠活动。你不能购买超出购物清单指定数量的物品，即使那样会降低整体价格。任意大礼包可无限次购买。

```python
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        # 方法1: dfs暴搜
        n = len(price)
        valid = []
        ans = sum(price[i]*needs[i] for i in range(n)) # 初始化为全买单件

        start = [0 for i in range(n)]

        def backtracking(now,consume): # consume是大礼包花的钱
            nonlocal ans
            for i in range(n): # 有超额则返回
                if now[i] > needs[i]:
                    return 
            
            temp = consume 
            for i in range(n): # 补全
                temp += (needs[i]-now[i])*price[i]
            ans = min(ans,temp)

            for choice in special:
                consume += choice[-1] # 选择
                for i in range(n):
                    now[i] += choice[i]

                backtracking(now,consume)

                for i in range(n): # 回溯
                    now[i] -= choice[i]
                consume -= choice[-1]
        
        backtracking(start,0)
        return ans
```

```python
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        # 方法2: dfs,预处理special
        n = len(price)
        valid = []
        ans = sum(price[i]*needs[i] for i in range(n)) # 初始化为全买单件

        start = [0 for i in range(n)]

        # 预处理部分
        valid = []
        for choice in special:
            state = True
            for i in range(n): # 超量部分
                if choice[i] > needs[i]:
                    state = False
                    break 
            nondiscount = 0
            for i in range(n):
                nondiscount += choice[i]*price[i]
            if nondiscount < choice[-1]: # 没打折还便宜
                state = False 
            if state:
                valid.append(choice)

        special = valid 
				# 预处理结束
				
        def backtracking(now,consume): # consume是大礼包花的钱
            nonlocal ans
            for i in range(n): # 有超额则返回
                if now[i] > needs[i]:
                    return 
            
            temp = consume 
            for i in range(n): # 补全
                temp += (needs[i]-now[i])*price[i]
            ans = min(ans,temp)

            for choice in special:
                consume += choice[-1] # 选择
                for i in range(n):
                    now[i] += choice[i]

                backtracking(now,consume)

                for i in range(n): # 回溯
                    now[i] -= choice[i]
                consume -= choice[-1]
        
        backtracking(start,0)
        return ans
```

# 652. 寻找重复的子树

给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其中任意**一棵**的根结点即可。

两棵树重复是指它们具有相同的结构以及相同的结点值。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        # 返回的是节点列表，序列化二叉树
        memoDict = collections.defaultdict(list) # k-v是序列-节点

        def postOrder(node): # 后序遍历提升性能
            mark = ""
            if node == None:
                return ""
            leftPart = postOrder(node.left)
            rightPart = postOrder(node.right)
            mark += str(node.val) + "#"
            if node.left == None:
                mark += "#"
            else:
                mark += leftPart
            if node.right == None:
                mark += "#"
            else:
                mark += rightPart
            if memoDict.get(mark) == None:
                memoDict[mark] = [node,0]
            else:
                memoDict[mark][1] += 1
            return mark
        
        postOrder(root)
        ans = []
        for key in memoDict:
            if memoDict[key][1] > 0:
                ans.append(memoDict[key][0])
        return ans
```

# 663. 均匀树划分

给定一棵有 `n` 个结点的二叉树，你的任务是检查是否可以通过去掉树上的一条边将树分成两棵，且这两棵树结点之和相等。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def checkEqualTree(self, root: TreeNode) -> bool:
        # 先序遍历计算总和，后序遍历算汇总
        # 注意特殊节点

        allSum = 0

        def preOrder(node):
            nonlocal allSum
            if node == None:
                return
            allSum += node.val
            preOrder(node.left)
            preOrder(node.right)
        
        preOrder(root)
        
        if allSum % 2 != 0:
            return False 

        state = False 
        def postOrder(node):
            nonlocal state
            nonlocal allSum
            if node == None:
                return 0
            leftSum = postOrder(node.left)
            rightSum = postOrder(node.right)
            now = node.val + leftSum + rightSum
            if now == allSum//2 and node != root: # 注意这一条
                state = True 
            return now

        postOrder(root)
        return state

```

# 666. 路径总和 IV

对于一棵深度小于 5 的树，可以用一组三位十进制整数来表示。

对于每个整数：

百位上的数字表示这个节点的深度 D，1 <= D <= 4。
十位上的数字表示这个节点在当前层所在的位置 P， 1 <= P <= 8。位置编号与一棵满二叉树的位置编号相同。
个位上的数字表示这个节点的权值 V，0 <= V <= 9。
给定一个包含三位整数的升序数组，表示一棵深度小于 5 的二叉树，请你返回从根到所有叶子结点的路径之和。

```python
class Solution:
    def pathSum(self, nums: List[int]) -> int:
        # 用数组模拟树
        temp = str(nums[-1])[0]
        size = 2**(int(temp))-1
        tree = [None for i in range(size)]

        for n in nums:
            n = str(n)
            index = 2**(int(n[0])-1)-2+int(n[1])
            val = int(n[2])
            tree[index] = val 
        
        path = []
        ans = []
        def dfs(index):
            if index >= len(tree):
                return 
            if tree[index] == None:
                return 
            path.append(tree[index]) # 先选择
            if 2*index+1 >= len(tree) and 2*index+2 >= len(tree): # 底层叶子
                ans.append(path[:])
            elif tree[2*index+1] == None and tree[2*index+2] == None: # 普通叶子
                ans.append(path[:])
            dfs(2*index+1)
            dfs(2*index+2)
            path.pop()
        
        dfs(0)
        res = 0
        for line in ans:
            for element in line:
                if element != None:
                    res += element
        return res

```

# 681. 最近时刻

给定一个形如 “HH:MM” 表示的时刻，利用当前出现过的数字构造下一个距离当前时间最近的时刻。每个出现数字都可以被无限次使用。

你可以认为给定的字符串一定是合法的。例如，“01:34” 和 “12:09” 是合法的，“1:34” 和 “12:9” 是不合法的。

```python
class Solution:
    def nextClosestTime(self, time: str) -> str:
        # 暴力枚举加判断
        # 未剪枝
        def judge(s):
            hh = s[:2]
            mm = s[3:]
            return 0<=int(hh)<24 and 0<=int(mm)<60
        
        def calcGap(s1,s2):# s1是现在的时间，s2是要比对的时间
            hh1 = s1[:2]
            mm1 = s1[3:]
            t1 = int(hh1)*60+int(mm1)
            hh2 = s2[:2]
            mm2 = s2[3:]
            t2 = int(hh2)*60+int(mm2)
            return (t2-t1)%1440
        
        tempList = []
        timetemp = [time[0],time[1],time[3],time[4]]
        choice = set(timetemp)
        if len(choice) == 1: # 他如果只有一种数字那么可以是相同的。。。
            return time

        def backtracking(path):
            if len(path) == 4:
                tempList.append("".join(path[:2])+":"+"".join(path[2:]))
                return 
            for e in choice:
                path.append(e)
                backtracking(path)
                path.pop()
        
        backtracking([])
        validTime = []
        for e in tempList:
            if judge(e):
                validTime.append(e)
        
        ans = None
        mingap = 99999
        for e in validTime:
            theGap = calcGap(time,e)
            if theGap == 0:
                continue
            if theGap < mingap:
                ans = e
                mingap = theGap
        return ans
```

# 694. 不同岛屿的数量

给定一个非空 01 二维数组表示的网格，一个岛屿由四连通（上、下、左、右四个方向）的 1 组成，你可以认为网格的四周被海水包围。

请你计算这个网格中共有多少个形状不同的岛屿。两个岛屿被认为是相同的，当且仅当一个岛屿可以通过平移变换（不可以旋转、翻转）和另一个岛屿重合。

```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        # 需要设计合适的哈希函数
        pathdict = collections.defaultdict(int)
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        m,n = len(grid),len(grid[0])
        visited = [[False for j in range(n)] for i in range(m)]

        # 这个路径coord有可能巨长
        def dfs(i,j,start_i,start_j,path):
            coord = "["+str(i-start_i)+","+str(j-start_j)+"]"
            path.append(coord)
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0 <= new_i < m and 0 <= new_j < n and grid[new_i][new_j] == 1 and visited[new_i][new_j] == False:
                    visited[new_i][new_j] = True 
                    dfs(new_i,new_j,start_i,start_j,path)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and visited[i][j] == False:
                    visited[i][j] = True
                    path = []
                    dfs(i,j,i,j,path)
                    path = "".join(path)
                    pathdict[path] += 1 
        
        return len(pathdict)
```

# 696. 计数二进制子串

给定一个字符串 s，计算具有相同数量 0 和 1 的非空（连续）子字符串的数量，并且这些子字符串中的所有 0 和所有 1 都是连续的。

重复出现的子串要计算它们出现的次数。

```python
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        # 分组统计，group数组两两为不同的0/1的统计数量
        # 对于 000111: 它的合法子串为000111,0011,01
        # 对于 00011：它的合法子串为0011,01 # 于是有相邻的取min
        group = []
        temp = 0
        mark = s[0]
        for ch in s:
            if ch == mark:
                temp += 1
            elif ch != mark:
                group.append(temp)
                temp = 1
                mark = ch 
        if temp != 0:
            group.append(temp)
        
        if len(group) == 1:
            return 0
        
        count = 0
        p = 1
        while p < len(group):
            count += min(group[p],group[p-1])
            p += 1
        return count
```



# 697. 数组的度

给定一个非空且只包含非负数的整数数组 nums，数组的度的定义是指数组里任一元素出现频数的最大值。

你的任务是在 nums 中找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。

```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        # 先找到数组的度，然后双指针
        ct = collections.Counter(nums)
        degree = 0
        for key in ct:
            if ct[key] > degree:
                degree = ct[key]
        
        left = 0
        right = 0
        n = len(nums)
        ans = n
        windowLength = 0
        window = collections.defaultdict(int)

        while right < n:
            add = nums[right]
            right += 1
            windowLength += 1
            window[add] += 1
            if window[add] == degree:
                ans = min(ans,windowLength)
            while left < right and window[add] >= degree:
                delete = nums[left]
                left += 1
                windowLength -= 1
                window[delete] -= 1
                if window[add] == degree: # 注意这个用的还是add
                    ans = min(ans,windowLength)
        return ans
```

# 712. 两个字符串的最小ASCII删除和

给定两个字符串`s1, s2`，找到使两个字符串相等所需删除字符的ASCII值的最小和

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        # dp双序列问题，先求需要删除的步骤数目
        dp = [[0xffffffff for j in range(len(s2)+1)] for i in range(len(s1)+1)]
        # 初始化
        dp[0][0] = 0
        for i in range(1,len(s1)+1):
            dp[i][0] = dp[i-1][0] + ord(s1[i-1]) 
        for j in range(1,len(s2)+1):
            dp[0][j] = dp[0][j-1] + ord(s2[j-1])
        # 如果新加入的字符相等，则不用删，如果不相等，则各自去尾巴，挑选着删
        for i in range(1,len(s1)+1):
            for j in range(1,len(s2)+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + ord(s1[i-1]),
                        dp[i][j-1] + ord(s2[j-1])
                    )
        return dp[-1][-1]
```



# 714. 买卖股票的最佳时机含手续费

给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        # dp,两行dp
        n = len(prices)
        dp = [[0 for i in range(n)] for t in range(2)]
        # dp[0][i]为当前没有股票的收益
        # dp[1][i]为当前持有股票的收益
        dp[0][0],dp[1][0] = 0,-prices[0]
        for i in range(1,n):
            dp[0][i] = max(dp[0][i-1],dp[1][i-1]+prices[i]-fee)
            dp[1][i] = max(dp[1][i-1],dp[0][i-1]-prices[i])
        return max(dp[0][-1],dp[1][-1])
```

```python
func maxProfit(prices []int, fee int) int {
    n := len(prices)
    dp := make([][]int,2,2)
    dp[0] = make([]int,n,n)
    dp[1] = make([]int,n,n)
    dp[1][0] = -prices[0]
    
    for i:=1;i<n;i++ {
        dp[0][i] = max(dp[0][i-1],dp[1][i-1]+prices[i]-fee)
        dp[1][i] = max(dp[1][i-1],dp[0][i-1]-prices[i])
    }
    return max(dp[0][n-1],dp[1][n-1])
}

func max(a,b int) int {
    if a > b {
        return a
    } else {
        return b
    }
}
```



# 717. 1比特与2比特字符

有两种特殊字符。第一种字符可以用一比特0来表示。第二种字符可以用两比特(10 或 11)来表示。

现给一个由若干比特组成的字符串。问最后一个字符是否必定为一个一比特字符。给定的字符串总是由0结束。

```python
class Solution:
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        # 只要是1开头跨两位，只要是0开头跨一位
        n = len(bits)
        p = 0
        while p < n-1:
            if bits[p] == 1:
                p += 2
            elif bits[p] == 0:
                p += 1
        
        return p == n-1
```

# 718. 最长重复子数组

给两个整数数组 `A` 和 `B` ，返回两个数组中公共的、长度最长的子数组的长度。

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        a = len(nums1)
        b = len(nums2)
        dp = [[0 for j in range(b+1)] for i in range(a+1)]
        # a纵b横
        # 加了一圈外圈，数字个数从1开始算起，dp[i][j]代表 nums1的前i个数和nums2的前j个数的最长公共子序列长度
        # 状态转移方程dp[i][j],
        # 两个新加入的相等，那么从dp[i-1][j-1]+1
        # 两个新加入的不相等，重置
        longest = 0
        for j in range(1,b+1):
            for i in range(1,a+1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                if dp[i][j] > longest: # 不用max函数，可以降低复杂度。。
                    longest = dp[i][j]
        # print(dp)
        return longest
```

```go
func findLength(nums1 []int, nums2 []int) int {
    m := len(nums1)
    n := len(nums2)
    dp := make([][]int,m+1)
    for i:=0;i< m+1;i += 1{
        dp[i] = make([]int,n+1)
    }
    ans := 0
    for i:=1;i<m+1;i++ {
        for j:=1;j<n+1;j++ {
            if nums1[i-1] == nums2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            }
            if ans < dp[i][j] {
                ans = dp[i][j]
            }
        }
    }
    return ans
}
```

# 743. 网络延迟时间

有 n 个网络节点，标记为 1 到 n。

给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。

现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # dj算法，狄杰斯特拉算法
        # 邻接表存图法
        graph = collections.defaultdict(list)
        for a,b,c in times: # 在这里都已经预先减去1
            graph[a-1].append((b-1,c))
        
        distance = [0xffffffff for i in range(n)]
        queue = collections.deque()
        queue.append((k-1,0))
        distance[k-1] = 0 # 到这里是0

        while len(queue) != 0:
            cur,nowTime = queue.popleft()
            if distance[cur] < nowTime:
                continue # 开启下一轮
            for neigh,neighT in graph[cur]:
                if distance[neigh] > nowTime + neighT:
                    distance[neigh] = nowTime + neighT
                    queue.append((neigh,nowTime+neighT))
        
        ans = max(distance)
        return ans if ans != 0xffffffff else -1
      
```

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        graph = collections.defaultdict(list)
        for a,b,c in times:
            graph[a-1].append((b-1,c))
        
        queue = collections.deque()
        queue.append((k-1,0))
        distance = [0xffffffff for i in range(n)]
				
        # 另一种判断逻辑
        while len(queue) != 0:
            cur,nowTime = queue.popleft()
            if distance[cur] > nowTime:
                distance[cur] = nowTime
            for neigh,addTime in graph[cur]:
                if distance[neigh] > nowTime + addTime:
                    queue.append((neigh,nowTime+addTime))
        
        ans = max(distance)
        if ans == 0xffffffff:
            return -1
        else:
            return ans
```

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        graph = collections.defaultdict(list)
        for a,b,c in times:
            graph[a-1].append((b-1,c))
        
        # 堆优化版
        queue = [] # 做heap
        queue.append((0,k-1))
        distance = [0xffffffff for i in range(n)]

        while len(queue) != 0:
            nowTime,cur = heapq.heappop(queue)
            if distance[cur] > nowTime:
                distance[cur] = nowTime
            for neigh,addTime in graph[cur]:
                if distance[neigh] > nowTime + addTime:
                    heapq.heappush(queue,(nowTime+addTime,neigh))

        ans = max(distance)
        if ans == 0xffffffff:
            return -1
        else:
            return ans
```

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # dj算法，用邻接矩阵
        graph = [[0xffffffff for j in range(n)] for i in range(n)]
        for a,b,c in times:
            graph[a-1][b-1] = c 

        queue = collections.deque()
        queue.append((k-1,0))
        distance = [0xffffffff for i in range(n)]
        distance[k-1] = 0

        while len(queue) != 0:
            cur,nowTime = queue.popleft()
            if distance[cur] < nowTime:
                continue
            for neigh,addTime in enumerate(graph[cur]):
                if distance[neigh] > nowTime + addTime:
                    distance[neigh] = nowTime + addTime
                    queue.append((neigh,nowTime+addTime))
        
        ans = max(distance)
        if ans == 0xffffffff:
            return -1 
        else:
            return ans
```

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # dj算法，用邻接矩阵
        graph = [[0xffffffff for j in range(n)] for i in range(n)]
        for a,b,c in times:
            graph[a-1][b-1] = c 

        queue = collections.deque()
        queue.append((k-1,0))
        distance = [0xffffffff for i in range(n)]
				
				
				# 另一种逻辑
        while len(queue) != 0:
            cur,nowTime = queue.popleft()
            if distance[cur] > nowTime:
                distance[cur] = nowTime
            for neigh,addTime in enumerate(graph[cur]):
                if distance[neigh] > nowTime + addTime:
                    queue.append((neigh,nowTime+addTime))
        
        ans = max(distance)
        if ans == 0xffffffff:
            return -1 
        else:
            return ans
```

# 763. 划分字母区间

字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        n = len(s)
        lst = [-1 for i in range(n)]
        alphadict = [-1 for i in range(26)]

        for i in range(n):
            index = ord(s[i])-ord('a')
            alphadict[index] = i 

        for i in range(n):
            lst[i] = alphadict[ord(s[i])-ord('a')]
        
        # 遍历的过程中更新目前的边界
        ans = []
        cut = -1
        bound = lst[0]
        for i in range(n):
            if lst[i] > bound:
                bound = lst[i]
            elif i == bound:
                ans.append(bound-cut) # 添加长度,并且把cut更新
                cut = bound
                bound = lst[i+1] if i+1 < n else None
        return ans
```



# 779. 第K个语法符号

在第一行我们写上一个 `0`。接下来的每一行，将前一行中的`0`替换为`01`，`1`替换为`10`。

给定行数 `N` 和序数 `K`，返回第 `N` 行中第 `K`个字符。（`K`从1开始）

```python
#第一行: 0
#第二行: 01
#第三行: 0110
#第四行: 01101001
class Solution:
    def kthGrammar(self, n: int, k: int) -> int:
        # 递归运算，找准基态
        # 还需要找到规律，关于中间反对称
        length = 2**(n-1)
        half = length//2
        if n == 1:
            return 0
        if k > half:
            temp = self.kthGrammar(n,k-half) # 偏移量为half
            if temp == 1:
                return 0
            else:
                return 1
        elif k <= half:
            return self.kthGrammar(n-1,k)
```

# 794. 有效的井字游戏

用字符串数组作为井字游戏的游戏板 board。当且仅当在井字游戏过程中，玩家有可能将字符放置成游戏板所显示的状态时，才返回 true。

该游戏板是一个 3 x 3 数组，由字符 " "，"X" 和 "O" 组成。字符 " " 代表一个空位。

以下是井字游戏的规则：

玩家轮流将字符放入空位（" "）中。
第一个玩家总是放字符 “X”，且第二个玩家总是放字符 “O”。
“X” 和 “O” 只允许放置在空位中，不允许对已放有字符的位置进行填充。
当有 3 个相同（且非空）的字符填充任何行、列或对角线时，游戏结束。
当所有位置非空时，也算为游戏结束。
如果游戏结束，玩家不允许再放置字符。

```python
class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        # 判断是否可以为当前放置的状态
        # 1. X-O == 0 or 1
        # 2. 不能同时有双方的胜局
        def judge(board,symbol):
            for line in board:
                if line == symbol*3:
                    return True 
            for i in range(3):
                t = board[0][i]+board[1][i]+board[2][i]
                if t == symbol*3:
                    return True 
            t = board[0][0]+board[1][1]+board[2][2]
            if t == symbol*3:
                return True 
            t = board[0][2]+board[1][1]+board[2][0]
            if t == symbol*3:
                return True 
            return False 
        
        ctx = 0
        cto = 0
        for line in board:
            for ch in line:
                if ch == "X":
                    ctx += 1
                if ch == "O":
                    cto += 1
        if ctx-cto != 0 and ctx-cto != 1:
            return False 
        s1 = judge(board,"X")
        s2 = judge(board,"O")
        if s1 == True and s2 == True:
            return False 
        
        # 两者数目相等的时候，X不能是TrueTrue
        if ctx == cto:
            if judge(board,"X"):
                return False 
        # x更多的时候，O不能是True
        if ctx == cto + 1:
            if judge(board,"O"):
                return False 
        return True
```



# 795. 区间子数组个数

给定一个元素都是正整数的数组`A` ，正整数 `L` 以及 `R` (`L <= R`)。

求连续、非空且其中最大元素满足大于等于`L` 小于等于`R`的子数组个数。

```python
class Solution:
    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        # 技巧型超级强的前缀和
        # 计数方式为，以当前位置结尾，如果当前元素小于等于limit，则技术
        # 抽象出函数，使用两次的值相减
        def lowerThanLimit(nums,limit):
            count = 0
            pre = 0
            # 当这个数符合限制时，那么以它结尾的数都可以,有一种动态规划的思想
            for n in nums:
                if n <= limit:
                    pre += 1
                else: # 否则重置
                    pre = 0
                count += pre
            return count 
        
        # 对right使用和对left-1使用，两者值减
        ans = lowerThanLimit(nums,right)-lowerThanLimit(nums,left-1)
        return ans
```

```go
func numSubarrayBoundedMax(nums []int, left int, right int) int {
    // 技巧型前缀和
    lowerThanLimit := func(nums *([]int),limit int) int {
        pre := 0
        count := 0
        
        for _,v := range(*nums) {
            if v <= limit {
                pre += 1
            } else {
                pre = 0
            }
            count += pre
        }
        return count
    }
    
    ans := lowerThanLimit(&nums,right) - lowerThanLimit(&nums,left-1)
    return ans
}
```

# 811. 子域名访问计数

一个网站域名，如"discuss.leetcode.com"，包含了多个子域名。作为顶级域名，常用的有"com"，下一级则有"leetcode.com"，最低的一级为"discuss.leetcode.com"。当我们访问域名"discuss.leetcode.com"时，也同时访问了其父域名"leetcode.com"以及顶级域名 "com"。

给定一个带访问次数和域名的组合，要求分别计算每个域名被访问的次数。其格式为访问次数+空格+地址，例如："9001 discuss.leetcode.com"。

接下来会给出一组访问次数和域名组合的列表cpdomains 。要求解析出所有域名的访问次数，输出格式和输入格式相同，不限定先后顺序。

```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        # 先格式化
        memo = collections.defaultdict(int)
        for infomation in cpdomains:
            p = 0
            while infomation[p] != " ":
                p += 1
            times = int(infomation[:p])
            while infomation[p] == " ":
                p += 1
            part = infomation[p:]
            part = part.split(".")
            if len(part) == 2:
                memo[part[-1]] += times
                memo[part[0]+"."+part[1]] += times
            if len(part) == 3:
                memo[part[-1]] += times 
                memo[part[1]+"."+part[2]] += times
                memo[part[0]+"."+part[1]+"."+part[2]] += times
        
        ans = []
        for key in memo:
            temp = str(memo[key])+" "+key
            ans.append(temp)
        return ans
```

# 819. 最常见的单词

给定一个段落 (paragraph) 和一个禁用单词列表 (banned)。返回出现次数最多，同时不在禁用列表中的单词。

题目保证至少有一个词不在禁用列表中，而且答案唯一。

禁用列表中的单词用小写字母表示，不含标点符号。段落中的单词不区分大小写。答案都是小写字母。

```python
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        paragraph = paragraph.lower()
        paragraph = list(paragraph)
        theSet = set("!?',;.")
        for i in range(len(paragraph)):
            if paragraph[i] in theSet:
                paragraph[i] = ' '
        paragraph = "".join(paragraph)
        temp = paragraph.split(" ")
        ct = collections.Counter(temp)
        banned = set(banned)
        theList = []
        maxTimes = 0
        ans = None
        for key in ct:
            if key == "": continue 
            if key not in banned:
                if ct[key] > maxTimes:
                    maxTimes = ct[key]
                    ans = key 
        return ans

# 纯模拟
```

# 830. 较大分组的位置

在一个由小写字母构成的字符串 s 中，包含由一些连续的相同字符所构成的分组。

例如，在字符串 s = "abbxxxxzyy" 中，就含有 "a", "bb", "xxxx", "z" 和 "yy" 这样的一些分组。

分组可以用区间 [start, end] 表示，其中 start 和 end 分别表示该分组的起始和终止位置的下标。上例中的 "xxxx" 分组用区间表示为 [3,6] 。

我们称所有包含大于或等于三个连续字符的分组为 较大分组 。

找到每一个 较大分组 的区间，按起始位置下标递增顺序排序后，返回结果。

```python
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        # 计数
        times = 0
        pivot = s[0]
        p = 0
        ans = []
        while p < len(s):
            if s[p] == pivot:
                times += 1
            elif s[p] != pivot:
                if times >= 3:
                    ans.append([p-times,p-1])
                pivot = s[p]
                times = 1
            p += 1
        if times >= 3:
            ans.append([p-times,p-1])
        return ans
```

# 831. 隐藏个人信息

给你一条个人信息字符串 `S`，它可能是一个 **邮箱地址** ，也可能是一串 **电话号码**。

我们将隐藏它的隐私信息，通过如下规则: 略

```python
class Solution:
    def changeEmail(self,s):
        index1 = None
        index2 = None
        s = list(s)
        for i,ch in enumerate(s):
            if ch == "@":
                index1 = i
                continue
            elif ch == ".":
                index2 = i
                continue 
            else:
                s[i] = s[i].lower()
        
        n1 = s[:index1]
        mark1 = n1[0]
        mark2 = n1[-1]
        n = len(n1)
        n1 = ['*' for i in range(5+2)]
        n1[0],n1[-1] = mark1,mark2
        n1 = ''.join(n1)
        n2 = ''.join(s[index1+1:index2])
        n3 = ''.join(s[index2+1:])
        return n1+"@"+n2+"."+n3

    def changeTel(self,s):
        stack = []
        set1 = set("01234567890")
        for ch in s:
            if ch in set1:
                stack.append(ch)
        
        n = len(stack)
        s1,s2 = stack[:n-10],stack[n-10:n]

        for i in range(6):
            s2[i] = "*"
        s2 = ''.join(s2[:3])+"-"+''.join(s2[3:6])+"-"+"".join(s2[6:])
        if len(s1) != 0:
            s2 = "+"+len(s1)*"*"+"-"+s2
        return s2
        
    def maskPII(self, s: str) -> str:
        state = 1 # 0表示电子邮箱，1表示电话
        if "@" in s:
            state = 0
        if state == 0:
            return self.changeEmail(s)
        else:
            return self.changeTel(s)
```

# 835. 图像重叠

给你两个图像 img1 和 img2 ，两个图像的大小都是 n x n ，用大小相同的二维正方形矩阵表示。（并且为二进制矩阵，只包含若干 0 和若干 1 ）

转换其中一个图像，向左，右，上，或下滑动任何数量的单位，并把它放在另一个图像的上面。之后，该转换的 重叠 是指两个图像都具有 1 的位置的数目。

（请注意，转换 不包括 向任何方向旋转。）

最大可能的重叠是多少？

```python
class Solution:
    def largestOverlap(self, img1: List[List[int]], img2: List[List[int]]) -> int:
        # 注意，不要求联通
        # 直接暴力判断
        # 固定1，偏移2
        # 从四个角都要判断
        # 8次判断，可以写一个函数封装。。
        # 需要稍微剪枝
        n = len(img1)
        maxArea = 0
        for x_P in range(n):
            for y_P in range(n):
                tempArea = 0
                for i in range(n):
                    if 0<=i+x_P<n:
                        for j in range(n):
                            if 0<=j+y_P<n:
                                tempArea += img1[i][j]&img2[i+x_P][j+y_P]
                            else:
                                break
                maxArea = max(maxArea,tempArea)
        
        for x_P in range(-n,1):
            for y_P in range(n):
                tempArea = 0
                for i in range(n):
                    if 0<=i+x_P<n:
                        for j in range(n):
                            if 0<=j+y_P<n:
                                tempArea += img1[i][j]&img2[i+x_P][j+y_P]
                            else:
                                break
                maxArea = max(maxArea,tempArea)

        for x_P in range(n):
            for y_P in range(-n,1):
                tempArea = 0
                for i in range(n):
                    if 0<=i+x_P<n:
                        for j in range(n):
                            if 0<=j+y_P<n:
                                tempArea += img1[i][j]&img2[i+x_P][j+y_P]
                            else:
                                break
                maxArea = max(maxArea,tempArea)

        for x_P in range(-n,1):
            for y_P in range(-n,1):
                tempArea = 0
                for i in range(n):
                    if 0<=i+x_P<n:
                        for j in range(n):
                            if 0<=j+y_P<n:
                                tempArea += img1[i][j]&img2[i+x_P][j+y_P]
                            else:
                                break
                maxArea = max(maxArea,tempArea)

        # 交换一轮,不交换会漏
        img1,img2 = img2,img1
        for x_P in range(n):
            for y_P in range(n):
                tempArea = 0
                for i in range(n):
                    if 0<=i+x_P<n:
                        for j in range(n):
                            if 0<=j+y_P<n:
                                tempArea += img1[i][j]&img2[i+x_P][j+y_P]
                            else:
                                break
                maxArea = max(maxArea,tempArea)
        
        for x_P in range(-n,1):
            for y_P in range(n):
                tempArea = 0
                for i in range(n):
                    if 0<=i+x_P<n:
                        for j in range(n):
                            if 0<=j+y_P<n:
                                tempArea += img1[i][j]&img2[i+x_P][j+y_P]
                            else:
                                break
                maxArea = max(maxArea,tempArea)

        for x_P in range(n):
            for y_P in range(-n,1):
                tempArea = 0
                for i in range(n):
                    if 0<=i+x_P<n:
                        for j in range(n):
                            if 0<=j+y_P<n:
                                tempArea += img1[i][j]&img2[i+x_P][j+y_P]
                            else:
                                break
                maxArea = max(maxArea,tempArea)

        for x_P in range(-n,1):
            for y_P in range(-n,1):
                tempArea = 0
                for i in range(n):
                    if 0<=i+x_P<n:
                        for j in range(n):
                            if 0<=j+y_P<n:
                                tempArea += img1[i][j]&img2[i+x_P][j+y_P]
                            else:
                                break
                maxArea = max(maxArea,tempArea)

        return maxArea
```

# 839. 相似字符串组

如果交换字符串 X 中的两个不同位置的字母，使得它和字符串 Y 相等，那么称 X 和 Y 两个字符串相似。如果这两个字符串本身是相等的，那它们也是相似的。

例如，"tars" 和 "rats" 是相似的 (交换 0 与 2 的位置)； "rats" 和 "arts" 也是相似的，但是 "star" 不与 "tars"，"rats"，或 "arts" 相似。

总之，它们通过相似性形成了两个关联组：{"tars", "rats", "arts"} 和 {"star"}。注意，"tars" 和 "arts" 是在同一组中，即使它们并不相似。形式上，对每个组而言，要确定一个单词在组中，只需要这个词和该组中至少一个单词相似。

给你一个字符串列表 strs。列表中的每个字符串都是 strs 中其它所有字符串的一个字母异位词。请问 strs 中有多少个相似字符串组？

```python
class UF: # 并查集
    def __init__(self,size):
        self.root = [i for i in range(size)]
    
    def union(self,x,y): # 并
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootX] = rootY
    
    def find(self,x): # 查
        while x != self.root[x]:
            x = self.root[x]
        return x
  
    def is_connect(self,x,y): # 判断是否连接
        return self.find(x) == self.find(y)

class Solution:
    def numSimilarGroups(self, strs: List[str]) -> int:
        # 已知所有词是异位词
        def judge(w1,w2):
            diff = 0
            for i in range(len(w1)):
                if w1[i] != w2[i]:
                    diff += 1
                if diff >= 3:
                    return False
            return True
        
        n = len(strs)
        ufSet = UF(n)
        countSet = set()
        for i in range(n):
            for j in range(i+1,n):
                if judge(strs[i],strs[j]):
                    ufSet.union(i,j)

        for i in range(n):
            countSet.add(ufSet.find(i))
        return len(countSet)
```

# 848. 字母移位
有一个由小写字母组成的字符串 S，和一个整数数组 shifts。

我们将字母表中的下一个字母称为原字母的 移位（由于字母表是环绕的， 'z' 将会变成 'a'）。

例如·，shift('a') = 'b'， shift('t') = 'u',， 以及 shift('z') = 'a'。

对于每个 shifts[i] = x ， 我们会将 S 中的前 i+1 个字母移位 x 次。

返回将所有这些移位都应用到 S 后最终得到的字符串。

```python
class Solution:
    def shiftingLetters(self, s: str, shifts: List[int]) -> str:
        # 反着的前缀和
        shifts = shifts[::-1]
        pre = 0
        preSum = []
        for n in shifts:
            pre += n 
            preSum.append(pre)
        preSum = preSum[::-1] # 还原
        for i in range(len(preSum)):
            preSum[i] = preSum[i]%26

        # 下面这个解析式是化简版本        
        tempList = [chr((ord(s[i])-ord("a")+preSum[i])%26 + ord("a")) for i in range(len(s))]
        return "".join(tempList)
```

# 866. 回文素数

求出大于或等于 N 的最小回文素数。

回顾一下，如果一个数大于 1，且其因数只有 1 和它自身，那么这个数是素数。

例如，2，3，5，7，11 以及 13 是素数。

回顾一下，如果一个数从左往右读与从右往左读是一样的，那么这个数是回文数。

例如，12321 是回文数。

```python
class Solution:
    def primePalindrome(self, n: int) -> int:
        # 打表大师法。
        target = [2, 3, 5, 7, 11, 101, 131, 151, 181, 191, 313, 353, 373, 383, 727, 757, 787, 797, 919, 929, 10301, 10501, 10601, 11311, 11411, 12421, 12721, 12821, 13331, 13831, 13931, 14341, 14741, 15451, 15551, 16061, 16361, 16561, 16661, 17471, 17971, 18181, 18481, 19391, 19891, 19991, 30103, 30203, 30403, 30703, 30803, 31013, 31513, 32323, 32423, 33533, 34543, 34843, 35053, 35153, 35353, 35753, 36263, 36563, 37273, 37573, 38083, 38183, 38783, 39293, 70207, 70507, 70607, 71317, 71917, 72227, 72727, 73037, 73237, 73637, 74047, 74747, 75557, 76367, 76667, 77377, 77477, 77977, 78487, 78787, 78887, 79397, 79697, 79997, 90709, 91019, 93139, 93239, 93739, 94049, 94349, 94649, 94849, 94949, 95959, 96269, 96469, 96769, 97379, 97579, 97879, 98389, 98689, 1003001, 1008001, 1022201, 1028201, 1035301, 1043401, 1055501, 1062601, 1065601, 1074701, 1082801, 1085801, 1092901, 1093901, 1114111, 1117111, 1120211, 1123211, 1126211, 1129211, 1134311, 1145411, 1150511, 1153511, 1160611, 1163611, 1175711, 1177711, 1178711, 1180811, 1183811, 1186811, 1190911, 1193911, 1196911, 1201021, 1208021, 1212121, 1215121, 1218121, 1221221, 1235321, 1242421, 1243421, 1245421, 1250521, 1253521, 1257521, 1262621, 1268621, 1273721, 1276721, 1278721, 1280821, 1281821, 1286821, 1287821, 1300031, 1303031, 1311131, 1317131, 1327231, 1328231, 1333331, 1335331, 1338331, 1343431, 1360631, 1362631, 1363631, 1371731, 1374731, 1390931, 1407041, 1409041, 1411141, 1412141, 1422241, 1437341, 1444441, 1447441, 1452541, 1456541, 1461641, 1463641, 1464641, 1469641, 1486841, 1489841, 1490941, 1496941, 1508051, 1513151, 1520251, 1532351, 1535351, 1542451, 1548451, 1550551, 1551551, 1556551, 1557551, 1565651, 1572751, 1579751, 1580851, 1583851, 1589851, 1594951, 1597951, 1598951, 1600061, 1609061, 1611161, 1616161, 1628261, 1630361, 1633361, 1640461, 1643461, 1646461, 1654561, 1657561, 1658561, 1660661, 1670761, 1684861, 1685861, 1688861, 1695961, 1703071, 1707071, 1712171, 1714171, 1730371, 1734371, 1737371, 1748471, 1755571, 1761671, 1764671, 1777771, 1793971, 1802081, 1805081, 1820281, 1823281, 1824281, 1826281, 1829281, 1831381, 1832381, 1842481, 1851581, 1853581, 1856581, 1865681, 1876781, 1878781, 1879781, 1880881, 1881881, 1883881, 1884881, 1895981, 1903091, 1908091, 1909091, 1917191, 1924291, 1930391, 1936391, 1941491, 1951591, 1952591, 1957591, 1958591, 1963691, 1968691, 1969691, 1970791, 1976791, 1981891, 1982891, 1984891, 1987891, 1988891, 1993991, 1995991, 1998991, 3001003, 3002003, 3007003, 3016103, 3026203, 3064603, 3065603, 3072703, 3073703, 3075703, 3083803, 3089803, 3091903, 3095903, 3103013, 3106013, 3127213, 3135313, 3140413, 3155513, 3158513, 3160613, 3166613, 3181813, 3187813, 3193913, 3196913, 3198913, 3211123, 3212123, 3218123, 3222223, 3223223, 3228223, 3233323, 3236323, 3241423, 3245423, 3252523, 3256523, 3258523, 3260623, 3267623, 3272723, 3283823, 3285823, 3286823, 3288823, 3291923, 3293923, 3304033, 3305033, 3307033, 3310133, 3315133, 3319133, 3321233, 3329233, 3331333, 3337333, 3343433, 3353533, 3362633, 3364633, 3365633, 3368633, 3380833, 3391933, 3392933, 3400043, 3411143, 3417143, 3424243, 3425243, 3427243, 3439343, 3441443, 3443443, 3444443, 3447443, 3449443, 3452543, 3460643, 3466643, 3470743, 3479743, 3485843, 3487843, 3503053, 3515153, 3517153, 3528253, 3541453, 3553553, 3558553, 3563653, 3569653, 3586853, 3589853, 3590953, 3591953, 3594953, 3601063, 3607063, 3618163, 3621263, 3627263, 3635363, 3643463, 3646463, 3670763, 3673763, 3680863, 3689863, 3698963, 3708073, 3709073, 3716173, 3717173, 3721273, 3722273, 3728273, 3732373, 3743473, 3746473, 3762673, 3763673, 3765673, 3768673, 3769673, 3773773, 3774773, 3781873, 3784873, 3792973, 3793973, 3799973, 3804083, 3806083, 3812183, 3814183, 3826283, 3829283, 3836383, 3842483, 3853583, 3858583, 3863683, 3864683, 3867683, 3869683, 3871783, 3878783, 3893983, 3899983, 3913193, 3916193, 3918193, 3924293, 3927293, 3931393, 3938393, 3942493, 3946493, 3948493, 3964693, 3970793, 3983893, 3991993, 3994993, 3997993, 3998993, 7014107, 7035307, 7036307, 7041407, 7046407, 7057507, 7065607, 7069607, 7073707, 7079707, 7082807, 7084807, 7087807, 7093907, 7096907, 7100017, 7114117, 7115117, 7118117, 7129217, 7134317, 7136317, 7141417, 7145417, 7155517, 7156517, 7158517, 7159517, 7177717, 7190917, 7194917, 7215127, 7226227, 7246427, 7249427, 7250527, 7256527, 7257527, 7261627, 7267627, 7276727, 7278727, 7291927, 7300037, 7302037, 7310137, 7314137, 7324237, 7327237, 7347437, 7352537, 7354537, 7362637, 7365637, 7381837, 7388837, 7392937, 7401047, 7403047, 7409047, 7415147, 7434347, 7436347, 7439347, 7452547, 7461647, 7466647, 7472747, 7475747, 7485847, 7486847, 7489847, 7493947, 7507057, 7508057, 7518157, 7519157, 7521257, 7527257, 7540457, 7562657, 7564657, 7576757, 7586857, 7592957, 7594957, 7600067, 7611167, 7619167, 7622267, 7630367, 7632367, 7644467, 7654567, 7662667, 7665667, 7666667, 7668667, 7669667, 7674767, 7681867, 7690967, 7693967, 7696967, 7715177, 7718177, 7722277, 7729277, 7733377, 7742477, 7747477, 7750577, 7758577, 7764677, 7772777, 7774777, 7778777, 7782877, 7783877, 7791977, 7794977, 7807087, 7819187, 7820287, 7821287, 7831387, 7832387, 7838387, 7843487, 7850587, 7856587, 7865687, 7867687, 7868687, 7873787, 7884887, 7891987, 7897987, 7913197, 7916197, 7930397, 7933397, 7935397, 7938397, 7941497, 7943497, 7949497, 7957597, 7958597, 7960697, 7977797, 7984897, 7985897, 7987897, 7996997, 9002009, 9015109, 9024209, 9037309, 9042409, 9043409, 9045409, 9046409, 9049409, 9067609, 9073709, 9076709, 9078709, 9091909, 9095909, 9103019, 9109019, 9110119, 9127219, 9128219, 9136319, 9149419, 9169619, 9173719, 9174719, 9179719, 9185819, 9196919, 9199919, 9200029, 9209029, 9212129, 9217129, 9222229, 9223229, 9230329, 9231329, 9255529, 9269629, 9271729, 9277729, 9280829, 9286829, 9289829, 9318139, 9320239, 9324239, 9329239, 9332339, 9338339, 9351539, 9357539, 9375739, 9384839, 9397939, 9400049, 9414149, 9419149, 9433349, 9439349, 9440449, 9446449, 9451549, 9470749, 9477749, 9492949, 9493949, 9495949, 9504059, 9514159, 9526259, 9529259, 9547459, 9556559, 9558559, 9561659, 9577759, 9583859, 9585859, 9586859, 9601069, 9602069, 9604069, 9610169, 9620269, 9624269, 9626269, 9632369, 9634369, 9645469, 9650569, 9657569, 9670769, 9686869, 9700079, 9709079, 9711179, 9714179, 9724279, 9727279, 9732379, 9733379, 9743479, 9749479, 9752579, 9754579, 9758579, 9762679, 9770779, 9776779, 9779779, 9781879, 9782879, 9787879, 9788879, 9795979, 9801089, 9807089, 9809089, 9817189, 9818189, 9820289, 9822289, 9836389, 9837389, 9845489, 9852589, 9871789, 9888889, 9889889, 9896989, 9902099, 9907099, 9908099, 9916199, 9918199, 9919199, 9921299, 9923299, 9926299, 9927299, 9931399, 9932399, 9935399, 9938399, 9957599, 9965699, 9978799, 9980899, 9981899, 9989899]

        for e in target:
            if e >= n:
                return e
        return 100030001
```

# 870. 优势洗牌

给定两个大小相等的数组 A 和 B，A 相对于 B 的优势可以用满足 A[i] > B[i] 的索引 i 的数目来描述。

返回 A 的任意排列，使其相对于 B 的优势最大化。

```python
class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 田忌赛马最优解
        # 注意可能有重复
        nums1.sort()
        origin = nums2[:]
        helper = [[nums2[i],i] for i in range(len(nums2))]
        nums2.sort()
        # 然后用k-v映射求解
        # A要找到刚刚大于B的马

        cur1 = 0
        cur2 = 0
        n = len(nums1)
        used = [False for i in range(n)]
        tempans = []

        while cur1 < n and cur2 < n:
            if nums1[cur1] <= nums2[cur2]: # 如果不大于
                cur1 += 1 # 换马
            elif nums1[cur1] > nums2[cur2]: # 如果大于
                tempans.append(nums1[cur1])
                used[cur1] = True
                cur1 += 1
                cur2 += 1
        
        # 找完之后，肯定是cur1有可能先被用完,没有匹配上的cur1被cur2之后的随便匹配都行
        p = 0
        while cur2 < n and p < n:
            while p < n and used[p] == True:
                p += 1
            tempans.append(nums1[p])
            p += 1
            cur2 += 1
        
        # 此时的tempans和排序后的B对应,使用原来的helper

        helper.sort()
        
        for i in range(len(helper)):
            helper[i][0] = tempans[i]
        
        helper.sort(key = lambda x:x[1]) # 利用其还原
        ans = [i[0] for i in helper]
        return ans
```

# 890. 查找和替换模式

你有一个单词列表 words 和一个模式  pattern，你想知道 words 中的哪些单词与模式匹配。

如果存在字母的排列 p ，使得将模式中的每个字母 x 替换为 p(x) 之后，我们就得到了所需的单词，那么单词与模式是匹配的。

（回想一下，字母的排列是从字母到字母的双射：每个字母映射到另一个字母，没有两个字母映射到同一个字母。）

返回 words 中与给定模式匹配的单词列表。

你可以按任何顺序返回答案。

```python
class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        # 将所有字母都转换成小写字母
        # 这一题条件相当强
        def getP(word):
            distinct = dict()
            start = 97
            ans = ""
            for ch in word:
                if ch not in distinct:
                    distinct[ch] = chr(start)
                    start += 1
                ans += distinct[ch]
            return ans 
        
        p = getP(pattern)
        ans = []
        for w in words:
            if p == getP(w):
                ans.append(w)
        return ans
```

# 904. 水果成篮

在一排树中，第 i 棵树产生 tree[i] 型的水果。
你可以从你选择的任何树开始，然后重复执行以下步骤：

把这棵树上的水果放进你的篮子里。如果你做不到，就停下来。
移动到当前树右侧的下一棵树。如果右边没有树，就停下来。
请注意，在选择一颗树后，你没有任何选择：你必须执行步骤 1，然后执行步骤 2，然后返回步骤 1，然后执行步骤 2，依此类推，直至停止。

你有两个篮子，每个篮子可以携带任何数量的水果，但你希望每个篮子只携带一种类型的水果。

用这个程序你能收集的水果树的最大总量是多少？

```python
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        # 滑动窗口
        ans = 0
        left = 0
        right = 0
        n = len(fruits)
        window = collections.defaultdict(int)
        windowLength =0
        while right < n:
            add = fruits[right]
            right += 1
            window[add] += 1
            windowLength += 1
            if len(window) <= 2:
                ans = max(ans,windowLength)
            while len(window) > 2 and left < right: # 收缩
                delete = fruits[left]
                window[delete] -= 1
                if window[delete] == 0: del window[delete]
                left += 1
                windowLength -= 1
        return ans
```

```go
func totalFruit(fruits []int) int {
    n := len(fruits)
    left := 0
    right := 0
    ans := 0
    window := make(map[int]int)

    for right < n {
        addN := fruits[right]
        window[addN] += 1
        right += 1
        if len(window) <= 2 {
            if right-left > ans { // 这里的right已经加过1了
                ans = right-left
            }
        }
        for left < right && len(window) > 2 { // 收缩
            deleteN := fruits[left]
            window[deleteN] -= 1
            if window[deleteN] == 0 {
                delete(window,deleteN)
            }
            left += 1
        }
    }
    return ans
}
```

# 911. 在线选举

给你两个整数数组 persons 和 times 。在选举中，第 i 张票是在时刻为 times[i] 时投给候选人 persons[i] 的。

对于发生在时刻 t 的每个查询，需要找出在 t 时刻在选举中领先的候选人的编号。

在 t 时刻投出的选票也将被计入我们的查询之中。在平局的情况下，最近获得投票的候选人将会获胜。

实现 TopVotedCandidate 类：

TopVotedCandidate(int[] persons, int[] times) 使用 persons 和 times 数组初始化对象。
int q(int t) 根据前面描述的规则，返回在时刻 t 在选举中领先的候选人的编号。

```python
class TopVotedCandidate:
# 注意查询的qt，并不是单调递增的
    def __init__(self, persons: List[int], times: List[int]):
        # 每个候选人带一队list
        self.p = collections.defaultdict(list)
        n = len(persons)
        for i in range(n):
            self.p[persons[i]].append(times[i]) # k-v 是人，时间

    def q(self, t: int) -> int:
        # 对每个人进行二分查询，获取index
        # 平局的情况下，最近得票的候选人获胜
        ans = -1
        pivot = None
        near = []
        for key in self.p:
            index = bisect.bisect_right(self.p[key],t)
            if index > ans:
                ans = index
                pivot = key
                near = []
                near.append([self.p[key][index-1],key])
            elif index == ans:
                near.append([self.p[key][index-1],key])
        # print(near)
        near.sort()

        return near[-1][1]
```

```python
class TopVotedCandidate:

    def __init__(self, persons: List[int], times: List[int]):
        n = len(persons)
        self.top = [-1 for i in range(n)]
        self.times = times
        record = collections.defaultdict(int) # k-v是人，票数
        nowMax = 0
        pivot = None
        for i in range(n):
            record[persons[i]] += 1
            if record[persons[i]] >= nowMax: # 注意平局时候是最近的候选人胜利
                nowMax = record[persons[i]]
                pivot = persons[i]
            self.top[i] = pivot
        # print(self.top) 此时是每个时刻的获胜者

    def q(self, t: int) -> int:
        index = bisect.bisect(self.times,t)
        if index < len(self.times) and self.times[index] == t:
            return self.top[index]
        else:
            return self.top[index-1]
        # 下面这种写法也可以
        # index = bisect.bisect_right(self.times,t)-1
        # return self.top[index]
```

# 918. 环形子数组的最大和

给定一个由整数数组 A 表示的环形数组 C，求 C 的非空子数组的最大可能和。

在此处，环形数组意味着数组的末端将会与开头相连呈环状。（形式上，当0 <= i < A.length 时 C[i] = A[i]，且当 i >= 0 时 C[i+A.length] = C[i]）

此外，子数组最多只能包含固定缓冲区 A 中的每个元素一次。（形式上，对于子数组 C[i], C[i+1], ..., C[j]，不存在 i <= k1, k2 <= j 其中 k1 % A.length = k2 % A.length）

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        # 最大值可能来源于两种情况
        # ------[   ]------ 来自于中间
        # [     ]----[      ] 来自于两头
        # 来自于中间的情况为情况1,来自于两头的为情况2.情况1即为leetcode53:最大子序和。情况2需要使得中间部分最小

        pre = 0
        n = len(nums)
        dp = [nums[i] for i in range(n)]
        # 算当前位置的前缀和
        for i in range(n):
            pre += nums[i]
            if pre < 0:
                pre = 0
                continue
            elif pre >= 0:
                dp[i] = pre 
        state1 = max(dp)

        pre = 0
        dp = [nums[i] for i in range(n)]
        for i in range(n):
            pre += nums[i]
            if pre > 0:
                pre = 0
                continue 
            else:
                dp[i] = pre 
        state2 = sum(nums)-min(dp)

        # 如果所有数都是负数的情况下,那么直接返回state1
        if state1 < 0:
            return state1
        else:
            return max(state1,state2)
```

```go
func maxSubarraySumCircular(nums []int) int {
    n := len(nums)
    pre := 0
    dp := make([]int,n,n)
    for i,v := range(nums) {
        pre += v 
        if pre > 0 {
            dp[i] = pre
        } else {
            dp[i] = nums[i]
            pre = 0
        }
    }
    state1 := max(dp)
    if state1 < 0 {
        return state1
    }
    pre = 0
    dp = make([]int,n,n)
    for i,v := range(nums) {
        pre += v 
        if pre > 0 {
            dp[i] = nums[i]
            pre = 0
        } else {
            dp[i] = pre
        }
    }
    state2 := sum(nums) - min(dp)
    
    if state1 > state2 {
        return state1 
    } else {
        return state2
    }
}

func max(arr []int) int {
    ans := arr[0]
    for _,v := range(arr) {
        if ans < v {
            ans = v
        }
    }
    return ans
}

func min (arr []int) int {
    ans := arr[0]
    for _,v := range(arr) {
        if ans > v {
            ans = v
        }
    }
    return ans 
}

func sum (arr []int) int {
    ans := 0
    for _,v := range(arr) {
        ans += v
    }
    return ans 
}
```

# 926. 将字符串翻转到单调递增

如果一个由 '0' 和 '1' 组成的字符串，是以一些 '0'（可能没有 '0'）后面跟着一些 '1'（也可能没有 '1'）的形式组成的，那么该字符串是单调递增的。

我们给出一个由字符 '0' 和 '1' 组成的字符串 S，我们可以将任何 '0' 翻转为 '1' 或者将 '1' 翻转为 '0'。

返回使 S 单调递增的最小翻转次数。

```python
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        # 动态规划开两行数组
        n = len(s)
        dp = [[0xffffffff for j in range(n+1)] for i in range(2)]
        dp[0][0],dp[0][1] = 0,0 # 初始化
        # 第一行是一0结尾的最小翻转次数，第二行是一1结尾的最少翻转次数
        # 状态转移方程为：
        # 如果当前位置是0
        # dp[0][j] = dp[0][j-1]
        # dp[1][j] = min(dp[0][j-1],dp[1][j-1])+1
        # 如果当前位置是1
        # dp[0][j] = dp[0][j-1]+1
        # dp[1][j] = min(dp[0][j-1],dp[1][j-1])
        for j in range(1,n+1):
            if s[j-1] == "0":
                dp[0][j] = dp[0][j-1]
                dp[1][j] = min(dp[0][j-1],dp[1][j-1])+1
            elif s[j-1] == "1":
                dp[0][j] = dp[0][j-1]+1
                dp[1][j] = min(dp[0][j-1],dp[1][j-1])

        return min(dp[0][-1],dp[1][-1])

```

```go
func minFlipsMonoIncr(s string) int {
    // 开两行dp
    n := len(s)
    dp := make([][]int,2)
    inf := 999999
    dp[0] = make([]int,n+1)
    dp[1] = make([]int,n+1)
    for j:=1; j< n+1; j++ {
        dp[0][j] = inf
        dp[1][j] = inf
    }

    for j := 1; j < n+1; j ++ {
        if s[j-1] == '0' {
            dp[0][j] = dp[0][j-1]
            dp[1][j] = min(dp[0][j-1],dp[1][j-1]) + 1
        } else if s[j-1] == '1' {
            dp[0][j] = dp[0][j-1] + 1
            dp[1][j] = min(dp[0][j-1],dp[1][j-1])
        }
    }

    if dp[0][n] < dp[1][n] {
        return dp[0][n]
    } else {
        return dp[1][n]
    }
}

func min(a,b int) int {
    if a < b {
        return a
    } else {
        return b
    }
}
```

# 929. 独特的电子邮件地址

每封电子邮件都由一个本地名称和一个域名组成，以 @ 符号分隔。

例如，在 alice@leetcode.com中， alice 是本地名称，而 leetcode.com 是域名。

除了小写字母，这些电子邮件还可能包含 '.' 或 '+'。

如果在电子邮件地址的本地名称部分中的某些字符之间添加句点（'.'），则发往那里的邮件将会转发到本地名称中没有点的同一地址。例如，"alice.z@leetcode.com” 和 “alicez@leetcode.com” 会转发到同一电子邮件地址。 （请注意，此规则不适用于域名。）

如果在本地名称中添加加号（'+'），则会忽略第一个加号后面的所有内容。这允许过滤某些电子邮件，例如 m.y+name@email.com 将转发到 my@email.com。 （同样，此规则不适用于域名。）

可以同时使用这两个规则。

给定电子邮件列表 emails，我们会向列表中的每个地址发送一封电子邮件。实际收到邮件的不同地址有多少？

```python
class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        # 先筛一下
        def seive(email):
            stack = []
            lst = []
            temp = email.split("@")
            for ch in temp[0]:
                if ch == '.':
                    continue
                if ch == "+":
                    break 
                else:
                    stack.append(ch)
            return "".join(stack) + "@"+ temp[1] 
        
        ansSet = set() # 用集合去重复
        for e in emails:
            ansSet.add(seive(e))
        return len(ansSet)
```

# 934. 最短的桥

在给定的二维二进制数组 A 中，存在两座岛。（岛是由四面相连的 1 形成的一个最大组。）

现在，我们可以将 0 变为 1，以使两座岛连接起来，变成一座岛。

返回必须翻转的 0 的最小数目。（可以保证答案至少是 1 。）

```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        # 把两座岛的坐标分别加入坐标集，计算曼哈顿距离,桥长为曼哈顿-1
        # 需要用边缘点计算曼哈顿
        m = len(grid)
        n = len(grid[0])
        direc = [(0,1),(0,-1),(-1,0),(1,0)]
        visited = [[False for j in range(n)] for i in range(m)]

        def dfs(i,j,countset):
            state = False 
            for di in direc: # 只算边缘
                neigh_i = i + di[0]
                neigh_j = j + di[1]
                if 0<=neigh_i<m and 0<=neigh_j<n and grid[neigh_i][neigh_j] == 0:
                    state = True 
                    break 
            if state:
                countset.add((i,j))
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and grid[new_i][new_j] == 1 and visited[new_i][new_j] == False:
                    visited[new_i][new_j] = True
                    dfs(new_i,new_j,countset)
        
        setList = [set(),set()]
        t = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and visited[i][j] == False:
                    visited[i][j] = True 
                    dfs(i,j,setList[t])
                    t += 1
        
        shortest = 0xffffffff
        for coord1 in setList[0]:
            for coord2 in setList[1]:
                manhutum = abs(coord1[0]-coord2[0])+abs(coord1[1]-coord2[1])
                shortest = min(shortest,manhutum-1)
        return shortest
```

```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        # dfs染色+bfs多源最短路径判断
        m = len(grid)
        n = len(grid[0])
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        visited = [[False for j in range(m)] for i in range(n)]
        queue = []
        def dfs(i,j):
            visited[i][j] = True
            grid[i][j] = 2
            queue.append((i,j))
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and grid[new_i][new_j] == 1:
                    dfs(new_i,new_j)
        
        state = True
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and state:
                    dfs(i,j)
                    state = False 
        
        steps = -1
        while len(queue) != 0:
            new_queue = []
            for i,j in queue:
                if grid[i][j] == 1:
                    return steps 
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and grid[new_i][new_j] != 2 and visited[new_i][new_j] == False:
                        visited[new_i][new_j] = True 
                        new_queue.append((new_i,new_j))
            queue = new_queue
            steps += 1

```

# 937. 重新排列日志文件

给你一个日志数组 logs。每条日志都是以空格分隔的字串，其第一个字为字母与数字混合的 标识符 。

有两种不同类型的日志：

字母日志：除标识符之外，所有字均由小写字母组成
数字日志：除标识符之外，所有字均由数字组成
请按下述规则将日志重新排序：

所有 字母日志 都排在 数字日志 之前。
字母日志 在内容不同时，忽略标识符后，按内容字母顺序排序；在内容相同时，按标识符排序。
数字日志 应该保留原来的相对顺序。
返回日志的最终顺序。

```python
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        # 格式化，有标识符，
        digitLogs = []
        alphaLogs = []
        for log in logs:
            p = 0
            while log[p] != " ":
                p += 1
            p += 1
            if log[p].isdigit():
                digitLogs.append(log)
            else:
                alphaLogs.append((log[:p],log[p:]))
        alphaLogs.sort(key = lambda x:(x[1],x[0]))
        for i in range(len(alphaLogs)):
            temp = ""
            for w in alphaLogs[i]:
                temp += w
            alphaLogs[i] = temp
        return alphaLogs + digitLogs
```

# 941. 有效的山脉数组

给定一个整数数组 arr，如果它是有效的山脉数组就返回 true，否则返回 false。

让我们回顾一下，如果 A 满足下述条件，那么它是一个山脉数组：

arr.length >= 3
在 0 < i < arr.length - 1 条件下，存在 i 使得：
arr[0] < arr[1] < ... arr[i-1] < arr[i]
arr[i] > arr[i+1] > ... > arr[arr.length - 1]

```python
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        if len(arr) < 3:
            return False 
        # 找到最大值，gap不能有0
        theMax = max(arr)
        maxIndex = None
        for i in range(len(arr)):
            if arr[i] == theMax:
                maxIndex = i 
                break 
        if maxIndex == 0 or maxIndex == len(arr)-1:  #不能在开头或者结尾
            return False 

        sli1 = arr[:maxIndex]
        cpsli1 = sli1[:]
        cpsli1.sort()

        if not (sli1 == cpsli1 and len(set(sli1)) == len(sli1)):
            return False 

        sli2 = arr[maxIndex:]
        cpsli2 = sli2[:]
        cpsli2.sort(key = lambda x:-x)

        if not (sli2 == cpsli2 and len(set(sli2)) == len(sli2)):
            return False 

        return True
```

# 949. 给定数字能组成的最大时间

给定一个由 4 位数字组成的数组，返回可以设置的符合 24 小时制的最大时间。

24 小时格式为 "HH:MM" ，其中 HH 在 00 到 23 之间，MM 在 00 到 59 之间。最小的 24 小时制时间是 00:00 ，而最大的是 23:59 。从 00:00 （午夜）开始算起，过得越久，时间越大。

以长度为 5 的字符串，按 "HH:MM" 格式返回答案。如果不能确定有效时间，则返回空字符串。

```python
class Solution:
    def largestTimeFromDigits(self, arr: List[int]) -> str:
        # 一个合法函数，一个回溯
        def judgeValid(s):
            hh = int(s[:2])
            mm = int(s[3:])
            t = hh*60 + mm
            return 0<=hh<24 and 0<=mm<60

        def toMinute(s):
            hh = int(s[:2])
            mm = int(s[3:])
            t = hh*60 + mm
            return t

        tempList = []
        
        # 回溯
        def backtracking(path,choice):
            if len(path) == 4:
                tempList.append("".join(path[:2])+":"+"".join(path[2:]))
                return 
            for e in choice:
                cp = choice.copy()
                cp.remove(e)
                path.append(str(e))
                backtracking(path,cp)
                path.pop()
        
        backtracking([],arr)
        
        validList = []
        for time in tempList:
            if judgeValid(time):
                validList.append(time)
        
        ans = ""
        now = 0
        for time in validList:
            if toMinute(time) >= now:
                now = toMinute(time)
                ans = time 
        return ans
```

# 954. 二倍数对数组

给定一个长度为偶数的整数数组 arr，只有对 arr 进行重组后可以满足 “对于每个 0 <= i < len(arr) / 2，都有 arr[2 * i + 1] = 2 * arr[2 * i]” 时，返回 true；否则，返回 false。

```python
class Solution:
    def canReorderDoubled(self, arr: List[int]) -> bool:
        # 只讲究是否成立，不讲究构造方法
        # 对0特殊处理
        arr.sort() # 预先排序
        ct = collections.Counter(arr)
        if ct[0] % 2 != 0:
            return False 
        
        neg = collections.defaultdict(int)
        posi = collections.defaultdict(int)
        # 分为正负数考虑
        for key in ct:
            if key > 0:
                posi[key] = ct[key]
            elif key < 0:
                neg[key] = ct[key]
        
        # print(posi,neg)
        # 正数要消除完毕，从小消到大
        for key in posi:
            if posi[key] > 0:
                if posi.get(key*2) == None:
                    return False
                posi[key*2] -= posi[key]
                posi[key] = 0
                if posi[key*2] < 0:
                    return False 

        # 负数用更负的消除
        for key in neg:
            if neg[key] > 0:
                if neg.get(key//2) == None:
                    return False
                if key/2*2 == key//2 * 2: # 对奇数需要处理
                    neg[key//2] -= neg[key]
                    neg[key] = 0
                    if neg[key//2] < 0:
                        return False 

        # print(posi,neg)
        s1 = sum(v1 for v1 in posi.values())
        s2 = sum(v2 for v2 in neg.values())
        if s1 == 0 and s2 == 0:
            return True
        else:
            return False 
```



# 970. 强整数

给定两个正整数 x 和 y，如果某一整数等于 x^i + y^j，其中整数 i >= 0 且 j >= 0，那么我们认为该整数是一个强整数。

返回值小于或等于 bound 的所有强整数组成的列表。

你可以按任何顺序返回答案。在你的回答中，每个值最多出现一次。

```python
class Solution:
    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        # 找到搜索上限
        # math.log(a,b) == log a  b
        if bound == 0:
            return []
        if x == 1:
            limit_i = 1
        else:
            limit_i = math.ceil(math.log(bound,x))
        if y == 1:
            limit_j = 1
        else:
            limit_j = math.ceil(math.log(bound,y))

        ans = set()
        for i in range(limit_i+1):
            for j in range(limit_j+1):
                temp = x**i + y**j
                if temp <= bound:
                    ans.add(x**i + y**j)
        ans = list(ans)
        return ans
```

# 986. 区间列表的交集

给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而 secondList[j] = [startj, endj] 。每个区间列表都是成对 不相交 的，并且 已经排序 。

返回这 两个区间列表的交集 。

形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。

两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        # 用一下递归解法
        ans = []

        def submethod(lst1,lst2):
            if lst1 == []:
                submethod(lst2,[])
                return 
            if lst2 == []:
                return 
            # 否则找两者的头，看是否有重合
            if lst1[0] >= lst2[0]:
                lst1,lst2 = lst2,lst1
            # 调整使得lst1小于lst2
            a,b = lst1[0]
            c,d = lst2[0]

            # 注意下面这一段分析
            if c <= b: # 有交集，则处理
                ans.append([c,min(b,d)])
            
            # 根据交集情况递归处理
            if b >= d:
                submethod(lst1,lst2[1:])
            else:
                submethod(lst1[1:],lst2)
        
        submethod(firstList,secondList)

        return ans
```

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        # 双指针思路
        p1,p2 = 0,0
        ans = []
        while p1 < len(firstList) and p2 < len(secondList):
            low = max(firstList[p1][0],secondList[p2][0])
            high = min(firstList[p1][1],secondList[p2][1])

            if low <= high:
                ans.append([low,high])
            
            # 移除区间时，移除端点更靠后的那一个
            if firstList[p1][1] < secondList[p2][1]:
                p1 += 1
            else:
                p2 += 1
        return ans
```

# 988. 从叶结点开始的最小字符串

给定一颗根结点为 root 的二叉树，树中的每一个结点都有一个从 0 到 25 的值，分别代表字母 'a' 到 'z'：值 0 代表 'a'，值 1 代表 'b'，依此类推。

找出按字典序最小的字符串，该字符串从这棵树的一个叶结点开始，到根结点结束。

（小贴士：字符串中任何较短的前缀在字典序上都是较小的：例如，在字典序上 "ab" 比 "aba" 要小。叶结点是指没有子结点的结点。）

```python
class Solution:
    def smallestFromLeaf(self, root: TreeNode) -> str:
        # 收集所有字符串
        temp = []
        def backtracking(node,path):
            if node == None:
                return 
            if node.left == None and node.right == None:
                path.append(chr(node.val+ord("a")))
                temp.append("".join(path)[::-1])
                path.pop()
                return 
            path.append(chr(node.val+ord("a")))
            backtracking(node.left,path)
            backtracking(node.right,path)
            path.pop()
        
        backtracking(root,[])
        temp.sort()
        return temp[0]
```



# 1003. 检查替换后的词是否有效

给你一个字符串 s ，请你判断它是否 有效 。
字符串 s 有效 需要满足：假设开始有一个空字符串 t = "" ，你可以执行 任意次 下述操作将 t 转换为 s ：

将字符串 "abc" 插入到 t 中的任意位置。形式上，t 变为 tleft + "abc" + tright，其中 t == tleft + tright 。注意，tleft 和 tright 可能为 空 。
如果字符串 s 有效，则返回 true；否则，返回 false。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        # 栈匹配问题
        stack = []
        for ch in s:
            stack.append(ch)
            while len(stack) >= 3 and stack[-3]+stack[-2]+stack[-1] == "abc":
                for t in range(3):
                    stack.pop()
        return len(stack) == 0
```

# 1010. 总持续时间可被 60 整除的歌曲

在歌曲列表中，第 i 首歌曲的持续时间为 time[i] 秒。

返回其总持续时间（以秒为单位）可被 60 整除的歌曲对的数量。形式上，我们希望索引的数字 i 和 j 满足  i < j 且有 (time[i] + time[j]) % 60 == 0。

```python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        # 前缀统计，找到前面的目标
        memo = collections.defaultdict(int)
        count = 0
        for t in time:
            t %= 60
            target = 60 - t
            if target != 60:
                count += memo[target]
            elif target == 60:
                count += memo[0]
            memo[t] += 1
        return count
```



# 1020. 飞地的数量

给出一个二维数组 A，每个单元格为 0（代表海）或 1（代表陆地）。

移动是指在陆地上从一个地方走到另一个地方（朝四个方向之一）或离开网格的边界。

返回网格中无法在任意次数的移动中离开网格边界的陆地单元格的数量。

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        # 从边界上的所有1进行bfs扩散并且标记为0，最后遍历一次数1
        m = len(grid)
        n = len(grid[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        queue = []
        for i in range(m):
            if grid[i][0] == 1:
                visited[i][0] = True
                queue.append((i,0))
            if n-1 != 0 and grid[i][n-1] == 1:
                visited[i][n-1] = True 
                queue.append((i,n-1))
        for j in range(n):
            if grid[0][j] == 1:
                visited[0][j] = True 
                queue.append((0,j))
            if m-1 != 0 and grid[m-1][j] == 1:
                visited[m-1][j] = True 
                queue.append((m-1,j))
        
        while len(queue) != 0:
            new_queue = []
            for i,j in queue:
                grid[i][j] = 0 # 置0
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and grid[new_i][new_j] == 1:
                        new_queue.append((new_i,new_j))
                        visited[new_i][new_j] = True
            queue = new_queue
        
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    count += 1
        
        return count     
```

# 1026. 节点与其祖先之间的最大差值

给定二叉树的根节点 root，找出存在于 不同 节点 A 和 B 之间的最大值 V，其中 V = |A.val - B.val|，且 A 是 B 的祖先。

（如果 A 的任何子节点之一为 B，或者 A 的任何子节点是 B 的祖先，那么我们认为 A 是 B 的祖先）

```python
class Solution:
    def maxAncestorDiff(self, root: TreeNode) -> int:
        # 后序遍历
        maxGap = -1

        def postOrder(node):
            nonlocal maxGap
            if node == None:
                return 0xffffffff,-1 #(childMin,childMax)
            if node.left == None and node.right == None:
                return node.val,node.val

            left_c_min,left_c_max = postOrder(node.left)
            right_c_min,right_c_max = postOrder(node.right)
            # print('node.val = ',node.val,left_c_min,left_c_max,right_c_min,right_c_max)
            group = [maxGap]
            if left_c_max != -1:
                group.append(abs(node.val-left_c_max))
            if left_c_min != 0xffffffff:
                group.append(abs(node.val-left_c_min))
            if right_c_max != -1:
                group.append(abs(node.val-right_c_max))
            if right_c_min != 0xffffffff:
                group.append(abs(node.val-right_c_min))
            maxGap = max(group)

            return min(node.val,left_c_min,right_c_min),max(node.val,left_c_max,right_c_max)
        
        postOrder(root)
        return maxGap
            
```

# 1034. 边框着色

给出一个二维整数网格 grid，网格中的每个值表示该位置处的网格块的颜色。

只有当两个网格块的颜色相同，而且在四个方向中任意一个方向上相邻时，它们属于同一连通分量。

连通分量的边界是指连通分量中的所有与不在分量中的正方形相邻（四个方向上）的所有正方形，或者在网格的边界上（第一行/列或最后一行/列）的所有正方形。

给出位于 (r0, c0) 的网格块和颜色 color，使用指定颜色 color 为所给网格块的连通分量的边界进行着色，并返回最终的网格 grid 。

```python
class Solution:
    def colorBorder(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        needChange = []

        m,n = len(grid),len(grid[0])
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        visited = [[False for j in range(n)] for i in range(m)]

        def dfs(i,j,origin):
            neiList = []
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n:
                    neiList.append([new_i,new_j])
            bound = False
            if len(neiList) != 4: # 如果邻居不足四个，一定是边界
                bound = True
            if not bound: # 如果邻居有四个，进一步检查
                for ni,nj in neiList:
                    if grid[ni][nj] != origin:
                        bound = True
                        break 
            if bound:
                needChange.append([i,j])
            
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and grid[new_i][new_j] == origin :
                    visited[new_i][new_j] = True
                    dfs(new_i,new_j,origin)

        visited[row][col] = True
        dfs(row,col,grid[row][col])
        
        for x,y in needChange:
            grid[x][y] = color
        return grid

```

# 1035. 不相交的线

在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。

现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足满足：

 nums1[i] == nums2[j]
且绘制的直线不与任何其他连线（非水平线）相交。
请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。

以这种方法绘制线条，并返回可以绘制的最大连线数。

```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        #
        m,n = len(nums1),len(nums2)
        # 考虑状态方程，如果存在nums1[i] == nums2[j],那么当前为前一个的递归解+这个解
        memo = dict() # key是元组
        def recur(i,j):
            if (i,j) in memo:
                return memo[(i,j)]
            if i < 0 or j < 0:
                return 0
            if nums1[i] == nums2[j]:
                memo[(i,j)] = recur(i-1,j-1)+1
            elif nums1[i] != nums2[j]:
                memo[(i,j)] = max(recur(i-1,j),recur(i,j-1))
            return memo[(i,j)]
        
        return recur(m-1,n-1)
```

```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        # dp版本
        m,n = len(nums1),len(nums2)
        # 考虑状态方程，如果存在nums1[i] == nums2[j],那么当前为前一个的递归解+这个解

        dp = [[0 for j in range(n+1)] for i in range(m+1)]

        for i in range(1,m+1):
            for j in range(1,n+1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i][j-1],dp[i-1][j])
        
        return dp[-1][-1]
```



# 1042. 不邻接植花

有 n 个花园，按从 1 到 n 标记。另有数组 paths ，其中 paths[i] = [xi, yi] 描述了花园 xi 到花园 yi 的双向路径。在每个花园中，你打算种下四种花之一。

另外，所有花园 最多 有 3 条路径可以进入或离开.

你需要为每个花园选择一种花，使得通过路径相连的任何两个花园中的花的种类互不相同。

以数组形式返回 任一 可行的方案作为答案 answer，其中 answer[i] 为在第 (i+1) 个花园中种植的花的种类。花的种类用  1、2、3、4 表示。保证存在答案。

```python
class Solution:
    def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
        graph = collections.defaultdict(list)
        for x,y in paths:
            graph[x-1].append(y-1)
            graph[y-1].append(x-1)
        # 四种花[1,2,3,4]
        flowers = [False,False,False,False]
        ans = [0 for i in range(n)]

        def dfs(n,flowers):
            if ans[n] != 0:
                return 

            for neigh in graph[n]: # 看周围的花
                if ans[neigh] != 0:
                    flowers[ans[neigh]-1] = True

            for i in range(4): # 选择一个可行的即可
                if flowers[i] == False:
                    ans[n] = i+1
                    flowers[i] = True
                    break

            # for neigh in graph[n]: # dfs
            #     if ans[neigh] == 0:
            #         dfs(neigh,flowers)
            
            for neigh in graph[n]: # 取消选择
                if ans[neigh] != 0:
                    flowers[ans[neigh]-1] = False 

            for i in range(4): # 取消选择一个可行的即可
                if flowers[i] == True:
                    flowers[i] = False
                    break
                         
        for i in range(n):
            if ans[i] == 0:
                dfs(i,flowers)
        return ans
```

```python
class Solution:
    def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
        graph = collections.defaultdict(list)
        for x,y in paths:
            graph[x-1].append(y-1)
            graph[y-1].append(x-1)
        flowers = [False for i in range(4)]
        ans = [0 for i in range(n)]

        def dfs(n,flowers):
            if ans[n] != 0:
                return
            # 找邻居
            for neigh in graph[n]:
                if ans[neigh] != 0:
                    flowers[ans[neigh]-1] = True 
            # 染色
            for i in range(4):
                if flowers[i] == False:
                    ans[n] = i+1
                    flowers[i] = True
                    break 
            # 取消选择
            for neigh in graph[n]:
                if ans[neigh] != 0:
                    flowers[ans[neigh]-1] = False
            # 取消
            for i in range(4):
                if flowers[i] == True:
                    flowers[i] = False
                    break 
        
        for i in range(n):
            if ans[i] == 0:
                dfs(i,flowers)
        return ans
```

# 1048. 最长字符串链

给出一个单词列表，其中每个单词都由小写英文字母组成。

如果我们可以在 word1 的任何地方添加一个字母使其变成 word2，那么我们认为 word1 是 word2 的前身。例如，"abc" 是 "abac" 的前身。

词链是单词 [word_1, word_2, ..., word_k] 组成的序列，k >= 1，其中 word_1 是 word_2 的前身，word_2 是 word_3 的前身，依此类推。

从给定单词列表 words 中选择单词组成词链，返回词链的最长可能长度。

```python
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        
        def isSub(patt,s): # 判断s是否是patt的子序列,注意：只能增加一个字母
            p1 = 0
            p2 = 0
            diff = 0
            while p1 < len(patt) and p2 < len(s):
                if patt[p1] == s[p2]:
                    p1 += 1
                    p2 += 1
                else:
                    diff += 1
                    p1 += 1
            if p2 == len(s) and (diff <= 1):
                return True 
            else:
                return False 

        words.sort(key = len) # 排序
        n = len(words)
        dp = [1 for i in range(n)]
        for i in range(n):
            group = [1]
            for j in range(i):
                if len(words[i]) != len(words[j]) + 1: continue
                if isSub(words[i],words[j]):
                    group.append(dp[j]+1)
            dp[i] = max(group)
        # print(dp)
        return max(dp)
```

# 1055. 形成字符串的最短路径

对于任何字符串，我们可以通过删除其中一些字符（也可能不删除）来构造该字符串的子序列。

给定源字符串 source 和目标字符串 target，找出源字符串中能通过串联形成目标字符串的子序列的最小数量。如果无法通过串联源字符串中的子序列来构造目标字符串，则返回 -1。

```python
class Solution:
    def shortestWay(self, source: str, target: str) -> int:
        # 如果target集合中有source中不含有的字母,返回-1
        setTarget = set(target)
        setSource = set(source)
        for e in setTarget:
            if e not in setSource:
                return -1
        # 否则用双指针更新
        k = len(source)
        n = len(target)
        count = 1
        ps = 0
        p = 0
        while p < n:
            if ps == k:
                count += 1
                ps = 0
            if target[p] == source[ps]:
                ps += 1
                p += 1
            elif target[p] != source[ps]:
                ps += 1
        return count
            
```



# 1065. 字符串的索引对

给出 字符串 text 和 字符串列表 words, 返回所有的索引对 [i, j] 使得在索引对范围内的子字符串 text[i]...text[j]（包括 i 和 j）属于字符串列表 words。

```python
class Solution:
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        # 简单题暴力做
        ans = []
        for w in words:
            length = len(w)
            for i in range(len(text)):
                if text[i:i+length] == w:
                    ans.append([i,i+length-1])
        ans.sort()
        return ans
```

# 1081. 不同字符的最小子序列

返回 `s` 字典序最小的子序列，该子序列包含 `s` 的所有不同字符，且只包含一次。

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # 先考虑一个问题，需对于一个字符串，要得到字典序最小，找到第一个s[i]，当s[i]>s[i+1]的时候，删除它
        # 原字符串s中的每个字符都需要出现在新字符串中，且只能出现一次。为了让新字符串满足该要求，之前讨论的算法需要进行以下两点的更改。在考虑字符 s[i]时，如果它已经存在于栈中，则不能加入字符 s[i]。为此，需要记录每个字符是否出现在栈中。在弹出栈顶字符时，如果字符串在后面的位置上再也没有这一字符，则不能弹出栈顶字符。为此，需要记录每个字符的剩余数量，当这个值为 0 时，就不能弹出栈顶字符了。
        stack = []
        remain = collections.Counter(s)  # 初始化为每个字符的计数，随着使用递减
        visited = set()

        for ch in s:
            if ch in visited:
                remain[ch] -= 1
                continue
            while len(stack) > 0 and stack[-1] > ch and remain[stack[-1]] != 0:
                e = stack.pop()
                visited.remove(e)
            stack.append(ch)
            visited.add(ch)
            remain[ch] -= 1

        return "".join(stack)
                    
                    
```

```go
func removeDuplicateLetters(s string) string {
    stack := make([]rune,0,0)
    visited := make(map[rune]bool)
    remain := Counter(s)
    
    for _,ch := range(s) {
        if visited[ch] == true {
            remain[ch] -= 1
            continue
        }
        for len(stack) > 0 && stack[len(stack)-1] > ch && remain[stack[len(stack)-1]] != 0 {
            e := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            visited[e] = false
        }
        stack = append(stack,ch)
        remain[ch] -= 1
        visited[ch] = true
    }
    
    ans := string(stack)
    return ans
}

func Counter(s string) map[rune]int {
    ct := make(map[rune]int)
    for _,ch := range(s) {
        ct[ch] += 1
    }
    return ct
}
```

# 1182. 与目标颜色间的最短距离

给你一个数组 colors，里面有  1、2、 3 三种颜色。

我们需要在 colors 上进行一些查询操作 queries，其中每个待查项都由两个整数 i 和 c 组成。

现在请你帮忙设计一个算法，查找从索引 i 到具有目标颜色 c 的元素之间的最短距离。

如果不存在解决方案，请返回 -1。

```python
class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        # 建立三个颜色队列，进行二分查找
        c1 = []
        c2 = []
        c3 = []
        for i,c in enumerate(colors):
            if c == 1:
                c1.append(i)
            elif c == 2:
                c2.append(i)
            else:
                c3.append(i)
        cband = [[],c1,c2,c3]
        ans = []
        # 根据查询到的结果，找当前位置，前/后一个位置，注意检查错误
        for i,c in queries:
            if len(cband[c]) == 0: # 如果颜色带为空，填充-1
                ans.append(-1)
                continue 
            # 非空，
            theIndex = bisect.bisect_left(cband[c],i) # 查找到的是cbandc的数组索引，需要转换成colors索引
            if 0<=theIndex<len(cband[c]):
                gap1 = abs(cband[c][theIndex]-i) # 注意索引转换
            else:
                gap1 = 0xffffffff
            if 0<=theIndex+1<len(cband[c]):
                gap2 = abs(cband[c][theIndex+1]-i)
            else:
                gap2 = 0xffffffff
            if 0<=theIndex-1<len(cband[c]):
                gap3 = abs(cband[c][theIndex-1]-i)
            else:
                gap3 = 0xffffffff
            ans.append(min(gap1,gap2,gap3))
        return ans
```



# 1186. 删除一次得到子数组最大和

给你一个整数数组，返回它的某个 非空 子数组（连续元素）在执行一次可选的删除操作后，所能得到的最大元素总和。

换句话说，你可以从原数组中选出一个子数组，并可以决定要不要从中删除一个元素（只能删一次哦），（删除后）子数组中至少应当有一个元素，然后该子数组（剩下）的元素总和是所有子数组之中最大的。

注意，删除一个元素后，子数组 不能为空。

```python
class Solution:
    def maximumSum(self, arr: List[int]) -> int:
        # 不算当前元素的前后缀和
        n = len(arr)
        pre = 0
        preList = [0 for i in range(n)]
        for i in range(n):
            preList[i] = pre 
            pre += arr[i]
            if pre < 0:
                pre = 0
        
        post = 0
        postList = [0 for i in range(n)]
        for i in range(n-1,-1,-1):
            postList[i] = post 
            post += arr[i]
            if post < 0:
                post = 0
        
        # 注意子数组不能为空

        ans = -0xffffffff
        for i in range(n):
            ans = max(ans,arr[i]+preList[i]+postList[i],preList[i]+postList[i])
        
        # # 注意子数组不能为空
        if ans == 0 and max(arr) < 0:
            return max(arr)
        return ans
```



# 1092. 最短公共超序列

给出两个字符串 str1 和 str2，返回同时以 str1 和 str2 作为子序列的最短字符串。如果答案不止一个，则可以返回满足条件的任意一个答案。

（如果从字符串 T 中删除一些字符（也可能不删除，并且选出的这些字符可以位于 T 中的 任意位置），可以得到字符串 S，那么 S 就是 T 的子序列）

```python
class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        # 反着求的动态规划？
        m,n = len(str1),len(str2)
        dp = [[None for j in range(n+1)] for i in range(m+1)]
        # dp[i][j],先求最长公共子序列
        dp[0][0] = ""
        #
        for i in range(m+1):
            dp[i][0] = ""
        for j in range(n+1):
            dp[0][j] = ""

        for i in range(1,m+1):
            for j in range(1,n+1):
                # 如果新加入的字符相等：
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + str1[i-1]
                else:
                    if len(dp[i][j-1]) > len(dp[i-1][j]):
                        dp[i][j] = dp[i][j-1]
                    else:
                        dp[i][j] = dp[i-1][j]
        LCS = dp[-1][-1]
        # 此时右下角为最长公共子序列. 扫str1
        # 
        ans = [] # 栈接收
        p1 = 0
        p2 = 0
        for ch in LCS:
            while str1[p1] != ch:
                ans.append(str1[p1])
                p1 += 1
            while str2[p2] != ch:
                ans.append(str2[p2])
                p2 += 1
            # 此时两者都指向了公共字符，加入公共字符，两者继续往后搜素
            ans.append(ch) # 
            p1 += 1
            p2 += 1
        # 如果两者没有走完，把接下来的也加了
        ans.append(str1[p1:])
        ans.append(str2[p2:])
        # 最后拼接
        final = "".join(ans)
        return final
```



# 1103. 分糖果 II

排排坐，分糖果。

我们买了一些糖果 candies，打算把它们分给排好队的 n = num_people 个小朋友。

给第一个小朋友 1 颗糖果，第二个小朋友 2 颗，依此类推，直到给最后一个小朋友 n 颗糖果。

然后，我们再回到队伍的起点，给第一个小朋友 n + 1 颗糖果，第二个小朋友 n + 2 颗，依此类推，直到给最后一个小朋友 2 * n 颗糖果。

重复上述过程（每次都比上一次多给出一颗糖果，当到达队伍终点后再次从队伍起点开始），直到我们分完所有的糖果。注意，就算我们手中的剩下糖果数不够（不比前一次发出的糖果多），这些糖果也会全部发给当前的小朋友。

返回一个长度为 num_people、元素之和为 candies 的数组，以表示糖果的最终分发情况（即 ans[i] 表示第 i 个小朋友分到的糖果数）。

```python
class Solution:
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        # 纯暴力模拟
        ans = [0 for j in range(num_people)]
        p = 0
        n = num_people
        while candies > 0:
            index = p%n
            ans[index] += p+1
            candies -= p+1
            p += 1
        if candies < 0: # 需要补偿
            p -= 1
            index = p%n
            ans[index] += candies
        return ans
```

# 1170. 比较字符串最小字母出现频次

定义一个函数 f(s)，统计 s  中（按字典序比较）最小字母的出现频次 ，其中 s 是一个非空字符串。

例如，若 s = "dcce"，那么 f(s) = 2，因为字典序最小字母是 "c"，它出现了 2 次。

现在，给你两个字符串数组待查表 queries 和词汇表 words 。对于每次查询 queries[i] ，需统计 words 中满足 f(queries[i]) < f(W) 的 词的数目 ，W 表示词汇表 words 中的每个词。

请你返回一个整数数组 answer 作为答案，其中每个 answer[i] 是第 i 次查询的结果。

```python
class Solution:
    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:

        def theFunc(s):
            s = list(s)
            s.sort()
            cts = Counter(s)
            times = None
            for key in cts:
                times = cts[key]
                break 
            return times
        
        lst1 = [theFunc(a) for a in queries]
        lst2 = [theFunc(b) for b in words]

        ans = []
        for n1 in lst1:
            temp = 0
            for n2 in lst2:
                if n1 < n2:
                    temp += 1
            ans.append(temp)
        return ans
```



# 1220. 统计元音字母序列的数目

给你一个整数 n，请你帮忙统计一下我们可以按下述规则形成多少个长度为 n 的字符串：

字符串中的每个字符都应当是小写元音字母（'a', 'e', 'i', 'o', 'u'）
每个元音 'a' 后面都只能跟着 'e'
每个元音 'e' 后面只能跟着 'a' 或者是 'i'
每个元音 'i' 后面 不能 再跟着另一个 'i'
每个元音 'o' 后面只能跟着 'i' 或者是 'u'
每个元音 'u' 后面只能跟着 'a'
由于答案可能会很大，所以请你返回 模 10^9 + 7 之后的结果。

```python
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        # 暴力动态规划
        dp = [[0 for j in range(n)] for i in range(5)]
        for i in range(5):
            dp[i][0] = 1
        # 0,1,2,3,4分别对应a,e,i,o,u
        # 需要翻译题干条件
        # 这个条件第一遍是从"ae", "ea", "ei", "ia", "ie", "io", "iu", "oi", "ou" 和 "ua"。推导出来的。。。

        # 实质为:原始条件
        # a -> e
        # e -> a,i
        # i -> a,e,o,u
        # o -> i,u
        # u -> a

        # 那么 a后面可以是e,i,u
        # e 后面可以是,a,i
        # i后面可以是e,o
        # o后面可以是i
        # u后面可以是i,o
        for i in range(1,n):
            dp[0][i] = dp[1][i-1] + dp[2][i-1] + dp[4][i-1]
            dp[1][i] = dp[0][i-1] + dp[2][i-1]
            dp[2][i] = dp[1][i-1] + dp[3][i-1]
            dp[3][i] = dp[2][i-1]
            dp[4][i] = dp[2][i-1] + dp[3][i-1]
        
        ans = 0
        for i in range(5):
            ans += dp[i][-1]
        return ans % (10**9+7)
```

# 1230. 抛掷硬币

有一些不规则的硬币。在这些硬币中，prob[i] 表示第 i 枚硬币正面朝上的概率。

请对每一枚硬币抛掷 一次，然后返回正面朝上的硬币数等于 target 的概率。

```
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        # 背包问题，开数组范围为宽度为prob，高度为target

        dp = [[0 for j in range(len(prob)+1)] for i in range(target+1)]
        # dp[i][j]表示用前j个硬币，凑出target为i的概率数目
        # 状态转移为 dp[i][j] = 【前j-1个硬币凑到i-1】或者【前j-1个硬币凑到i】
        # dp[i][j] = dp[i-1][j-1]*prob[j] + dp[i][j-1]*(1-prob[j])
        
        # 初始化第一行,正面朝上为0的概率
        pre = 1
        for j in range(1,len(prob)+1):
            pre *= (1-prob[j-1])
            dp[0][j] = pre 

        dp[0][0] = 1 # 注意这个初始化
        for i in range(1,target+1):
            for j in range(1,len(prob)+1):
                state1 = dp[i-1][j-1]*prob[j-1] 
                state2 = dp[i][j-1]*(1-prob[j-1])
                dp[i][j] = state1 + state2 
        # print(dp)
        return dp[-1][-1]
        
```



# 1237. 找出给定方程的正整数解

给你一个函数  f(x, y) 和一个目标结果 z，函数公式未知，请你计算方程 f(x,y) == z 所有可能的正整数 数对 x 和 y。满足条件的结果数对可以按任意顺序返回。

尽管函数的具体式子未知，但它是单调递增函数，也就是说：

f(x, y) < f(x + 1, y)
f(x, y) < f(x, y + 1)

1 <= function_id <= 9
1 <= z <= 100
题目保证 f(x, y) == z 的解处于 1 <= x, y <= 1000 的范围内。
在 1 <= x, y <= 1000 的前提下，题目保证 f(x, y) 是一个 32 位有符号整数。

```python
class Solution:
    def findSolution(self, customfunction: 'CustomFunction', z: int) -> List[List[int]]:
        # 找到x的上界，y的上界
        xLeft = 1
        xRight = 1000
        while xLeft <= xRight:
            xMid = (xLeft+xRight)//2
            if customfunction.f(xMid,1) == z: # 尝试收缩
                xRight = xMid - 1
            elif customfunction.f(xMid,1) > z: # 数值偏大，需要减小
                xRight = xMid - 1
            elif customfunction.f(xMid,1) < z: # 数值偏小，需要增大
                xLeft = xMid + 1
        yLeft = 1
        yRight = 1000
        while yLeft <= yRight:
            yMid = (yLeft+yRight)//2
            if customfunction.f(1,yMid) == z: # 尝试收缩
                yRight = yMid - 1
            if customfunction.f(1,yMid) > z: # 数值偏大，需要减小
                yRight = yMid - 1
            if customfunction.f(1,yMid) < z: # 数值偏小，需要增大
                yLeft = yMid + 1
        
        ans = []
        # x,y >= 1
        for x in range(1,xLeft+1):
            for y in range(1,yLeft+1):
                if customfunction.f(x,y) == z:
                    ans.append([x,y])
        return ans
```

# 1244. 力扣排行榜

新一轮的「力扣杯」编程大赛即将启动，为了动态显示参赛者的得分数据，需要设计一个排行榜 Leaderboard。

请你帮忙来设计这个 Leaderboard 类，使得它有如下 3 个函数：

addScore(playerId, score)：
假如参赛者已经在排行榜上，就给他的当前得分增加 score 点分值并更新排行。
假如该参赛者不在排行榜上，就把他添加到榜单上，并且将分数设置为 score。
top(K)：返回前 K 名参赛者的 得分总和。
reset(playerId)：将指定参赛者的成绩清零（换句话说，将其从排行榜中删除）。题目保证在调用此函数前，该参赛者已有成绩，并且在榜单上。
请注意，在初始状态下，排行榜是空的。

```python
class Leaderboard:
# 不知道这一题的可修改的topk有啥意义
    def __init__(self):
        import sortedcontainers
        self.Board = collections.defaultdict(int)
        self.scoreList = sortedcontainers.SortedList()
        self.scoreList.add(-1)
        self.scoreList.add(0xffffffff)


    def addScore(self, playerId: int, score: int) -> None:
        old = self.Board[playerId]
        self.Board[playerId] += score
        if old != 0:
            index = bisect.bisect_left(self.scoreList,old)
            self.scoreList.pop(index)
            self.scoreList.add(self.Board[playerId])
        if old == 0:
            self.scoreList.add(self.Board[playerId])
        
    def top(self, K: int) -> int:
        temp = 0
        for i in range(K):
            temp += self.scoreList[-2-i]
        return temp
        
    def reset(self, playerId: int) -> None:
        old = self.Board[playerId]
        self.Board[playerId] = 0
        index = bisect.bisect_left(self.scoreList,old)
        self.scoreList.pop(index)
```

# 1260. 二维网格迁移

给你一个 m 行 n 列的二维网格 grid 和一个整数 k。你需要将 grid 迁移 k 次。

每次「迁移」操作将会引发下述活动：

位于 grid[i][j] 的元素将会移动到 grid[i][j + 1]。
位于 grid[i][n - 1] 的元素将会移动到 grid[i + 1][0]。
位于 grid[m - 1][n - 1] 的元素将会移动到 grid[0][0]。
请你返回 k 次迁移操作后最终得到的 二维网格。

```python
class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
    # 读图后模拟，转一维rotate
        if k == 0:
            return grid
        m = len(grid)
        n = len(grid[0])
        origin = []
        for i in range(m):
            origin += grid[i]
        k = k % len(origin)
        k = len(origin) - k
        final = origin[k:] + origin[:k] 
        ans = []

        i = 0
        for t in range(m):
            ans.append(final[i:i+n])
            i += n 
        return ans
```

# 1275. 找出井字棋的获胜者

```python
class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        # 下完之后，如果不是角位，只需要检查同行或者同列是否获胜
        # 中心位需要检查四个方向
        # A先行动
        # 纯模拟
        def check(x,y,now):
            state1 = grid[x][0]+grid[x][1]+grid[x][2]
            state2 = grid[0][y]+grid[1][y]+grid[2][y]
            state3 = ""
            if (x,y) in ((0,0),(0,2),(2,0),(2,2)):
                if (x,y) == (0,0) or (x,y) == (2,2):
                    state3 = grid[0][0]+grid[1][1]+grid[2][2]
                elif (x,y) == (0,2) or (x,y) == (2,0):
                    state3 = grid[0][2]+grid[1][1]+grid[2][0]
            elif (x,y) == (1,1):
                state3 = grid[0][0]+grid[1][1]+grid[2][2]
                state4 = grid[0][2]+grid[1][1]+grid[2][0]
                if state1 == now*3 or state2 == now*3 or state3 == now*3 or state4 == now*3:
                    return True
            if state1 == now*3 or state2 == now*3 or state3 == now*3:
                return True 
            return False

        grid = [[" " for j in range(3)] for i in range(3)]
        count = 0
        for i in range(len(moves)):
            if i%2 == 0:
                x,y = moves[i]
                grid[x][y] = "A"
                now = "A"
            elif i%2 == 1:
                x,y = moves[i]
                grid[x][y] = "B"
                now = "B"
            if check(x,y,now):
                return now
            count += 1
        
        if count == 9:
            return "Draw"
        else:
            return "Pending"
```



# 1276. 不浪费原料的汉堡制作方案

圣诞活动预热开始啦，汉堡店推出了全新的汉堡套餐。为了避免浪费原料，请你帮他们制定合适的制作计划。

给你两个整数 tomatoSlices 和 cheeseSlices，分别表示番茄片和奶酪片的数目。不同汉堡的原料搭配如下：

巨无霸汉堡：4 片番茄和 1 片奶酪
小皇堡：2 片番茄和 1 片奶酪
请你以 [total_jumbo, total_small]（[巨无霸汉堡总数，小皇堡总数]）的格式返回恰当的制作方案，使得剩下的番茄片 tomatoSlices 和奶酪片 cheeseSlices 的数量都是 0。

如果无法使剩下的番茄片 tomatoSlices 和奶酪片 cheeseSlices 的数量为 0，就请返回 []。

```python
class Solution:
    def numOfBurgers(self, tomatoSlices: int, cheeseSlices: int) -> List[int]:
        # 设返回数组为[m,n]
        # 4*m + 2*n == tomatoSlices
        # m + n == cheeseSlices
        # m == (tomatoSlices-2*cheeseSlices)/2 # 需要判断它是不是大于零0的偶数
        # n = cheeseSlices - m
        if (tomatoSlices-2*cheeseSlices) % 2 != 0:
            return []
        if (tomatoSlices-2*cheeseSlices) < 0 :
            return []
        m = (tomatoSlices-2*cheeseSlices)//2
        n = cheeseSlices - m
        if n < 0:
            return []
        return [m,n]
```

# 1277. 统计全为 1 的正方形子矩阵

给你一个 `m * n` 的矩阵，矩阵中的元素不是 `0` 就是 `1`，请你统计并返回其中完全由 `1` 组成的 **正方形** 子矩阵的个数。

```python
class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        # 技巧型较强。一种特殊的dp
        # linked to 221
        m,n = len(matrix),len(matrix[0])
        dp = [[0 for j in range(n)] for i in range(m)]
        # dp[i][j]表示以i,j为右下角，的最大边长正方形，在数字上还等于以它为右下角的正方形的数目【对角线长度固定了】
        ans = 0
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[i][j] = matrix[i][j]
                elif matrix[i][j] == 1:
                    dp[i][j] = min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])+1
                ans += dp[i][j]
        return ans
```

# 1297. 子串的最大出现次数

给你一个字符串 s ，请你返回满足以下条件且出现次数最大的 任意 子串的出现次数：

子串中不同字母的数目必须小于等于 maxLetters 。
子串的长度必须大于等于 minSize 且小于等于 maxSize 。

```python
class Solution:
    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:
        # 用个window记录长度
        # maxSize没有意义,长的可以，短的更可以
        tempDict = collections.defaultdict(int)
        window = collections.defaultdict(int)
        left = 0
        right = 0
        n = len(s)
        windowLength = 0
        while right < n:
            add_char = s[right]
            right += 1
            window[add_char] += 1
            windowLength += 1
            if len(window) <= maxLetters and minSize<=windowLength<=maxSize:
                temp = s[right-windowLength:right]
                tempDict[temp] += 1
            while left < right and (len(window) > maxLetters or windowLength>minSize): # 注意这个收缩逻辑是到minSize
                delete_char = s[left]
                left += 1
                window[delete_char] -= 1
                if window[delete_char] == 0:
                    del window[delete_char]
                windowLength -= 1
                if len(window) <= maxLetters and minSize<=windowLength<=maxSize:
                    temp = s[right-windowLength:right]
                    tempDict[temp] += 1
        
        ans = 0
        for key in tempDict:
            if tempDict[key] > ans:
                ans = tempDict[key]
        return ans
```

```go
func maxFreq(s string, maxLetters int, minSize int, maxSize int) int {
    left := 0
    right := 0
    tempMap := make(map[string]int)
    window := make(map[string]int)
    windowLength := 0
    n := len(s)
    for right < n {
        addChar := string(s[right])
        right += 1
        window[addChar] += 1
        windowLength += 1
        if len(window) <= maxLetters && (minSize <= windowLength && windowLength <= maxSize) {
            temp := string(s[right-windowLength:right])
            tempMap[temp] += 1
        }  
        
        for left < right && (len(window) > maxLetters || windowLength > minSize) {
            deleteChar := string(s[left])
            left += 1
            window[deleteChar] -= 1
            if window[deleteChar] == 0 {
                delete(window,deleteChar)
            }
            windowLength -= 1
            if len(window) <= maxLetters && (minSize <= windowLength && windowLength <= maxSize) {
                temp := s[right-windowLength:right]
                tempMap[temp] += 1
            } 
        }
    }
    
    //fmt.Println(tempMap)
    ans := 0
    for _,v := range(tempMap) {
        if v > ans {
            ans = v
        }
    }
    return ans
}
```



# 1310. 子数组异或查询

有一个正整数数组 arr，现给你一个对应的查询数组 queries，其中 queries[i] = [Li, Ri]。

对于每个查询 i，请你计算从 Li 到 Ri 的 XOR 值（即 arr[Li] xor arr[Li+1] xor ... xor arr[Ri]）作为本次查询的结果。

并返回一个包含给定查询 queries 所有结果的数组。

```python
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        # 运用到了XOR关于4的性质，要把XOR运算缩到O(1)级别
        # 计算包含当前位置的前缀
        preSum = 0
        n = len(arr)
        preList = [0 for i in range(n)]
        for i in range(n):
            preSum ^= arr[i]
            preList[i] = preSum
        
        ans = [0 for i in range(len(queries))]
        for i in range(len(queries)):
            a = queries[i][0]
            b = queries[i][1]
            k = 0
            if a-1 >= 0: # 注意这里的处理
                k = preList[a-1]
            ans[i] = k^preList[b]
        return ans
```

# 1314. 矩阵区域和

给你一个 m x n 的矩阵 mat 和一个整数 k ，请你返回一个矩阵 answer ，其中每个 answer[i][j] 是所有满足下述条件的元素 mat[r][c] 的和： 

i - k <= r <= i + k,
j - k <= c <= j + k 且
(r, c) 在矩阵内。

```python
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        # 先做一个预处理矩阵
        m,n = len(mat),len(mat[0])
        preMat = [[0 for j in range(n)] for i in range(m)]
        # 效率稍低的前缀和预处理，但是方便
        for i in range(m):
            pre = 0
            for j in range(n):
                pre += mat[i][j] # 注意这里是对mat前缀和
                preMat[i][j] = pre

        for j in range(n):
            pre = 0
            for i in range(m):
                pre += preMat[i][j] # 注意这里是对preMat前缀和
                preMat[i][j] = pre 
        
        # 对每一个结果矩阵填充
        ans = [[0 for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                # [row1,col1]左上角，[row2,col2]右下角
                row1,col1 = max(0,i-k),max(0,j-k)
                row2,col2 = min(m-1,i+k),min(n-1,j+k)
                # 然后四个矩阵进行加减,画图即可
                s1 = preMat[row2][col2]
                s2 = preMat[row1-1][col1-1] if row1>=1 and col1>=1 else 0
                s3 = preMat[row1-1][col2] if row1>=1 else 0
                s4 = preMat[row2][col1-1] if col1>=1 else 0
                ans[i][j] = s1 + s2 - s3 - s4 
        return ans
```

# 1339. 分裂二叉树的最大乘积

给你一棵二叉树，它的根为 root 。请你删除 1 条边，使二叉树分裂成两棵子树，且它们子树和的乘积尽可能大。

由于答案可能会很大，请你将结果对 10^9 + 7 取模后再返回。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxProduct(self, root: TreeNode) -> int:
        # 后续遍历，一个是带到节点，一个是从root减去到该节点
        allSum = 0
        ans = 0

        def getAll(node):
            nonlocal allSum
            if node == None:
                return 
            allSum += node.val
            getAll(node.left)
            getAll(node.right)
        getAll(root)
        
        def postOrder(node):
            if node == None:
                return 0
            nonlocal ans 
            leftPart = postOrder(node.left)
            rightPart = postOrder(node.right)
            now = node.val + leftPart + rightPart
            minus = allSum - now
            if ans < minus * now:
                ans = minus*now 
            return now 
        
        postOrder(root)
        return ans%(10**9+7)
```

# 1362. 最接近的因数

给你一个整数 num，请你找出同时满足下面全部要求的两个整数：

两数乘积等于  num + 1 或 num + 2
以绝对差进行度量，两数大小最接近
你可以按任意顺序返回这两个整数。

```python
class Solution:
    def closestDivisors(self, num: int) -> List[int]:
        # 找到平方根
        n1 = num+1
        n2 = num+2

        start1 = int(sqrt(n1))+1
        temp1 = []
        for i in range(start1,0,-1):
            if n1%i == 0:
                temp1.append(i)
                temp1.append(n1//i)
                break 

        start2 = int(sqrt(n2))+1
        temp2 = []
        for i in range(start2,0,-1):
            if n2%i == 0:
                temp2.append(i)
                temp2.append(n2//i)
                break 
        
        abs1 = abs(temp1[0]-temp1[1])
        abs2 = abs(temp2[0]-temp2[1])
        if abs1 < abs2:
            return temp1
        else:
            return temp2

```

# 1370. 上升下降字符串

给你一个字符串 s ，请你根据下面的算法重新构造字符串：

从 s 中选出 最小 的字符，将它 接在 结果字符串的后面。
从 s 剩余字符中选出 最小 的字符，且该字符比上一个添加的字符大，将它 接在 结果字符串后面。
重复步骤 2 ，直到你没法从 s 中选择字符。
从 s 中选出 最大 的字符，将它 接在 结果字符串的后面。
从 s 剩余字符中选出 最大 的字符，且该字符比上一个添加的字符小，将它 接在 结果字符串后面。
重复步骤 5 ，直到你没法从 s 中选择字符。
重复步骤 1 到 6 ，直到 s 中所有字符都已经被选过。
在任何一步中，如果最小或者最大字符不止一个 ，你可以选择其中任意一个，并将其添加到结果字符串。

请你返回将 s 中字符重新排序后的 结果字符串 。

```python
class Solution:
    def sortString(self, s: str) -> str:
        ct = [0 for i in range(26)]
        for ch in s:
            index = ord(ch)-ord('a')
            ct[index] += 1
        
        bulider = []

        p = 0
        n = len(s)
        # 来回扫
        while n != 0:
            while p < 26:
                if ct[p] > 0:
                    bulider.append(chr(97+p))
                    ct[p] -= 1
                    n -= 1
                p += 1
            p -= 1
            while p >= 0:
                if ct[p] > 0:
                    bulider.append(chr(97+p))
                    ct[p] -= 1
                    n -= 1
                p -= 1
            p = 0
        
        return "".join(bulider)
```

# 1371. 每个元音包含偶数次的最长子字符串

给你一个字符串 `s` ，请你返回满足以下条件的最长子字符串的长度：每个元音字母，即 'a'，'e'，'i'，'o'，'u' ，在子字符串中都恰好出现了偶数次。

```python
class Solution:
    def findTheLongestSubstring(self, s: str) -> int:
        longest = 0
        n = len(s)
        pre = [0,0,0,0,0] # 压缩成字符串？
        # 可以用位运算
        preDict = collections.defaultdict(int) # 记录的是索引
        preDict["00000"] = -1
        reflec = {"a":0,"e":1,"i":2,"o":3,"u":4}
        for i in range(n):
            if s[i] in reflec:
                index = reflec[s[i]]
                pre[index] = (pre[index] + 1)%2
            key = "".join(str(count) for count in pre) # 这一步是真的慢
            if key in preDict:
                temgLength = i-preDict[key]
                longest = max(longest,temgLength)
            elif key not in preDict:
                preDict[key] = i 
            
        return longest
```

# 1372. 二叉树中的最长交错路径

给你一棵以 root 为根的二叉树，二叉树中的交错路径定义如下：

选择二叉树中 任意 节点和一个方向（左或者右）。
如果前进方向为右，那么移动到当前节点的的右子节点，否则移动到它的左子节点。
改变前进方向：左变右或者右变左。
重复第二步和第三步，直到你在树中无法继续移动。
交错路径的长度定义为：访问过的节点数目 - 1（单个节点的路径长度为 0 ）。

请你返回给定树中最长 交错路径 的长度。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: TreeNode) -> int:
        longest = 1

        # 后续遍历
        # 每个节点需要记录，当它的右子树是zig的最长路径，当它的左子树是zig的最长路径
        # 更新的时候
        def postOrder(node):
            nonlocal longest
            if node == None:
                return [0,0]
            
            leftPart = postOrder(node.left)
            rightPart = postOrder(node.right)
            nowLeft = 1
            nowRight = 1
            # 它如果有左孩子，那么加上左孩子的右zig，它如果有右孩子，那么加上右孩子的左zig
            if node.left != None:
                nowLeft += leftPart[1]
            if node.right != None:
                nowRight += rightPart[0]

            longest = max(longest,nowLeft,nowRight)

            return nowLeft,nowRight
        
        postOrder(root)
        return longest-1
```

# 1390. 四因数

给你一个整数数组 `nums`，请你返回该数组中恰有四个因数的这些整数的各因数之和。

如果数组中不存在满足题意的整数，则返回 `0` 。

```python
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # 除了1和本身之外，只能变成两个质数乘积
        # 预先计算所有质数
        up = max(nums)
        isPrime = [True for i in range(up+1)]
        primeList = []
        for i in range(2,up+1):
            if isPrime[i] == True:
                primeList.append(i)
                for j in range(i,up+1,i):
                    isPrime[j] = False
        
        
        ans = 0
        valid = dict() # k为该数，v为拆分
        # 双重for素数组合
        for i in range(len(primeList)):
            for j in range(len(primeList)):
                if i == j:
                    continue 
                key = primeList[i]*primeList[j]
                if key > up + 1:
                    break 
                valid[key] = 1+key+primeList[i]+primeList[j]
        
        # 不要遗漏了纯素数立方数:例如8
        for e in primeList:
            if e**3 > up+1:
                break
            key = e**3 
            valid[key] = 1+e+e**2+e**3
        for e in nums:
            if valid.get(e) != None:
                ans += valid[e]
        return ans
```



# 1415. 长度为 n 的开心字符串中字典序第 k 小的字符串

一个 「开心字符串」定义为：

仅包含小写字母 ['a', 'b', 'c'].
对所有在 1 到 s.length - 1 之间的 i ，满足 s[i] != s[i + 1] （字符串的下标从 1 开始）。
比方说，字符串 "abc"，"ac"，"b" 和 "abcbabcbcb" 都是开心字符串，但是 "aa"，"baa" 和 "ababbc" 都不是开心字符串。

给你两个整数 n 和 k ，你需要将长度为 n 的所有开心字符串按字典序排序。

请你返回排序后的第 k 个开心字符串，如果长度为 n 的开心字符串少于 k 个，那么请你返回 空字符串 。

```python
class Solution:
    def getHappyString(self, n: int, k: int) -> str:
        # 回溯暴力搜
        ans = []
        def backtracking(path):
            if len(path) == n:
                ans.append(path[:])
                return 
            if len(path) == 0 or path[-1] != 'a':
                path.append('a')
                backtracking(path)
                path.pop()
            if len(path) == 0 or path[-1] != 'b':
                path.append('b')
                backtracking(path)
                path.pop()
            if len(path) == 0 or path[-1] != 'c':
                path.append('c')
                backtracking(path)
                path.pop()
        backtracking([])
        if k > len(ans):
            return ""
        else:
            return "".join(ans[k-1])

```

# 1451. 重新排列句子中的单词

「句子」是一个用空格分隔单词的字符串。给你一个满足下述格式的句子 text :

句子的首字母大写
text 中的每个单词都用单个空格分隔。
请你重新排列 text 中的单词，使所有单词按其长度的升序排列。如果两个单词的长度相同，则保留其在原句子中的相对顺序。

请同样按上述格式返回新的句子。

```python
class Solution:
    def arrangeWords(self, text: str) -> str:
        lst = text.split(" ")
        first = lst[0][0].lower()+lst[0][1:]
        lst[0] = first
        lst.sort(key = len) # python偷懒了。。
        # 如果不用这个，可以用dict+长度列表append进行桶排序
        first = lst[0][0].upper()+lst[0][1:]
        lst[0] = first
        ans = ' '.join(lst)
        return ans

```

# 1452. 收藏清单

给你一个数组 favoriteCompanies ，其中 favoriteCompanies[i] 是第 i 名用户收藏的公司清单（下标从 0 开始）。

请找出不是其他任何人收藏的公司清单的子集的收藏清单，并返回该清单下标。下标需要按升序排列。

```python
class Solution:
    def peopleIndexes(self, favoriteCompanies: List[List[str]]) -> List[int]:
        theId = 0
        l = []
        for line in favoriteCompanies:
            l += line
        wordDict = dict()
        for w in l:
            if wordDict.get(w) == None:
                wordDict[w] = theId
                theId += 1
        
        # 转成唯一ID,且集合化
        for i in range(len(favoriteCompanies)):
            for j in range(len(favoriteCompanies[i])):
                favoriteCompanies[i][j] = wordDict[favoriteCompanies[i][j]]
            favoriteCompanies[i] = set(favoriteCompanies[i])
        
        ans = []
        # python自带了集合
        for i in range(len(favoriteCompanies)):
            state = True
            for j in range(len(favoriteCompanies)):
                if i == j: continue 
                s1 = favoriteCompanies[i]
                s2 = favoriteCompanies[j]
                if s1.issubset(s2):
                    state = False
                    break
            if state:
                ans.append(i)
        return ans
```

# 1455. 检查单词是否为句中其他单词的前缀

给你一个字符串 sentence 作为句子并指定检索词为 searchWord ，其中句子由若干用 单个空格 分隔的单词组成。

请你检查检索词 searchWord 是否为句子 sentence 中任意单词的前缀。

如果 searchWord 是某一个单词的前缀，则返回句子 sentence 中该单词所对应的下标（下标从 1 开始）。
如果 searchWord 是多个单词的前缀，则返回匹配的第一个单词的下标（最小下标）。
如果 searchWord 不是任何单词的前缀，则返回 -1 。
字符串 S 的 前缀 是 S 的任何前导连续子字符串。

```python
class Solution:
    def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
        sentence = sentence.split(" ")
        for index,w in enumerate(sentence):
            if len(w) >= len(searchWord):
                n = len(searchWord)
                if w[:n] == searchWord:
                    return index+1
        return -1
```

# 1472. 设计浏览器历史记录

你有一个只支持单个标签页的 浏览器 ，最开始你浏览的网页是 homepage ，你可以访问其他的网站 url ，也可以在浏览历史中后退 steps 步或前进 steps 步。

请你实现 BrowserHistory 类：

BrowserHistory(string homepage) ，用 homepage 初始化浏览器类。
void visit(string url) 从当前页跳转访问 url 对应的页面  。执行此操作会把浏览历史前进的记录全部删除。
string back(int steps) 在浏览历史中后退 steps 步。如果你只能在浏览历史中后退至多 x 步且 steps > x ，那么你只后退 x 步。请返回后退 至多 steps 步以后的 url 。
string forward(int steps) 在浏览历史中前进 steps 步。如果你只能在浏览历史中前进至多 x 步且 steps > x ，那么你只前进 x 步。请返回前进 至多 steps步以后的 url 。

```python
class BrowserHistory:
# 双栈法
    def __init__(self, homepage: str):
        self.home = homepage
        self.stack1 = [] # 主栈
        self.helper = [] # 辅助栈

    def visit(self, url: str) -> None:
        self.stack1.append(url)
        self.helper = [] # 清空辅助栈

    def back(self, steps: int) -> str:
        while len(self.stack1) != 0 and steps != 0:
            e = self.stack1.pop()
            self.helper.append(e)
            steps -= 1
        # 注意这个返回逻辑
        if len(self.stack1) == 0:
            return self.home
        else:
            return self.stack1[-1]


    def forward(self, steps: int) -> str:
        while len(self.helper) != 0 and steps != 0:
            e = self.helper.pop()
            self.stack1.append(e)
            steps -= 1
        # 注意这个返回逻辑
        if len(self.stack1) == 0:
            return self.home
        else:
            return self.stack1[-1]
```

# 1477. 找两个和为目标值且不重叠的子数组

给你一个整数数组 arr 和一个整数值 target 。

请你在 arr 中找 两个互不重叠的子数组 且它们的和都等于 target 。可能会有多种方案，请你返回满足要求的两个子数组长度和的 最小值 。

请返回满足要求的最小长度和，如果无法找到这样的两个子数组，请返回 -1 。

```python
class Solution:
    def minSumOfLengths(self, arr: List[int], target: int) -> int:
        tempList = []
        left = 0
        right = 0
        windowLength = 0
        window = 0
        n = len(arr)

        while right < n :
            add = arr[right]
            right += 1
            window += add 
            windowLength += 1
            if window == target:
                tempList.append([right-windowLength,right]) # 左闭右开
            while left < right and window > target:
                delete = arr[left]
                left += 1
                window -= delete
                windowLength -= 1
                if window == target:
                    tempList.append([right-windowLength,right])
        
        # 倒序得到长度最小
        minLength = [0 for i in range(len(tempList))]
        if len(minLength) == 0: # 无法找到
            return -1
        tempMin = tempList[-1][1]-tempList[-1][0]

        for i in range(len(tempList)-1,-1,-1):
            if tempList[i][1]-tempList[i][0] < tempMin:
                tempMin = tempList[i][1]-tempList[i][0]
            minLength[i] = tempMin
        
        # 二分查找
        # 查找比[a,b]中b大的最左端的下标
        ans = 0xffffffff
        for i in range(len(tempList)):
            a,b = tempList[i]
            index = bisect.bisect_left(tempList,[b])
            if index < len(tempList):
                ans = min(ans,b-a+minLength[index])
        
        return ans if ans != 0xffffffff else -1
```



# 1509. 三次操作后最大值与最小值的最小差

给你一个数组 nums ，每次操作你可以选择 nums 中的任意一个元素并将它改成任意值。

请你返回三次操作后， nums 中最大值与最小值的差的最小值。

```python
class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # 数学思想，只要数组长度小于等于4，一定可以变成全都一样
        if len(nums) <= 4:
            return 0
        # 否则先排序,暴力收拢？
        nums.sort()
        # 左3，右0 ： gap = nums[3] - nums[-1]  abs
        # 左2，右1 ： gap = nums[2] - nums[-2]  abs
        # 左1，右2 ： gap = nums[1] - nums[-3]  abs
        # 左0，右3 ： gap = nums[0] - nums[-4]  abs

        ans = 0xffffffff
        for i in range(4):
            a = nums[i]
            b = nums[-(4-i)]
            ans = min(ans,abs(a-b))
        return ans
      
      # 实际上可以无需排序，只需要维护八个数【大4，小4即可】
```

# 1513. 仅含 1 的子串数

给你一个二进制字符串 s（仅由 '0' 和 '1' 组成的字符串）。

返回所有字符都为 1 的子字符串的数目。

由于答案可能很大，请你将它对 10^9 + 7 取模后返回。

```python
class Solution:
    def numSub(self, s: str) -> int:
        # 利用排列组合简化运算
        mod = 10**9 + 7
        # 计算每次连续的1的数量
        s += '0' # 加个尾巴封端
        countList = []
        now = 0
        for ch in s:
            if ch == "0":
                if now != 0:
                    countList.append(now)
                now = 0
            elif ch == "1":
                now += 1
        ans = 0
        for n in countList:
            ans += (n+1)*n//2
        return ans % mod
            
```

# 1544. 整理字符串

给你一个由大小写英文字母组成的字符串 s 。

一个整理好的字符串中，两个相邻字符 s[i] 和 s[i+1]，其中 0<= i <= s.length-2 ，要满足如下条件:

若 s[i] 是小写字符，则 s[i+1] 不可以是相同的大写字符。
若 s[i] 是大写字符，则 s[i+1] 不可以是相同的小写字符。
请你将字符串整理好，每次你都可以从字符串中选出满足上述条件的 两个相邻 字符并删除，直到字符串整理好为止。

请返回整理好的 字符串 。题目保证在给出的约束条件下，测试样例对应的答案是唯一的。

注意：空字符串也属于整理好的字符串，尽管其中没有任何字符。

```python
class Solution:
    def makeGood(self, s: str) -> str:
        # 栈匹配问题
        stack = []
        for ch in s:
            stack.append(ch)
            if len(stack) >= 2:
                if stack[-1].upper() == stack[-2].upper():
                    if 65<=ord(stack[-1])<=90 and 97<=ord(stack[-2])<=122:
                        stack.pop()
                        stack.pop()
                    elif 65<=ord(stack[-2])<=90 and 97<=ord(stack[-1])<=122:
                        stack.pop()
                        stack.pop()
        return ''.join(stack)
```

# 1545. 找出第 N 个二进制字符串中的第 K 位

给你两个正整数 n 和 k，二进制字符串  Sn 的形成规则如下：

S1 = "0"
当 i > 1 时，Si = Si-1 + "1" + reverse(invert(Si-1))
其中 + 表示串联操作，reverse(x) 返回反转 x 后得到的字符串，而 invert(x) 则会翻转 x 中的每一位（0 变为 1，而 1 变为 0）。

例如，符合上述描述的序列的前 4 个字符串依次是：

S1 = "0"
S2 = "011"
S3 = "0111001"
S4 = "011100110110001"
请你返回  Sn 的 第 k 位字符 ，题目数据保证 k 一定在 Sn 长度范围以内。

```python
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        # 递归做法
        # 找准基态
        e = 2**(n-1) # 暂存加速运算
        if n == 1:
            return "0"
        if k == e:
            return "1"
        if  k > e:
            if self.findKthBit(n,2*e-k) == "1":
                return '0'
            else:
                return '1'
        if k < e:
            return self.findKthBit(n-1,k)
```

# 1551. 使数组中所有元素相等的最小操作数

存在一个长度为 n 的数组 arr ，其中 arr[i] = (2 * i) + 1 （ 0 <= i < n ）。

一次操作中，你可以选出两个下标，记作 x 和 y （ 0 <= x, y < n ）并使 arr[x] 减去 1 、arr[y] 加上 1 （即 arr[x] -=1 且 arr[y] += 1 ）。最终的目标是使数组中的所有元素都 相等 。题目测试用例将会 保证 ：在执行若干步操作后，数组中的所有元素最终可以全部相等。

给你一个整数 n，即数组的长度。请你返回使数组 arr 中所有元素相等所需的 最小操作数 。

```python
class Solution:
    def minOperations(self, n: int) -> int:
        # 跷跷板+贪心
        # 奇数直接找中位数,偶数直接找两个中间数//2
        # 实质上数值上都是n
        # 懒得操作了
        arr = [2*i+1 for i in range(n)]
        ans = 0
        midNum = n 
        for a in arr:
            ans += abs(a-midNum)
        return ans//2
```

# 1553. 吃掉 N 个橘子的最少天数

厨房里总共有 n 个橘子，你决定每一天选择如下方式之一吃这些橘子：

吃掉一个橘子。
如果剩余橘子数 n 能被 2 整除，那么你可以吃掉 n/2 个橘子。
如果剩余橘子数 n 能被 3 整除，那么你可以吃掉 2*(n/3) 个橘子。
每天你只能从以上 3 种方案中选择一种方案。

请你返回吃掉所有 n 个橘子的最少天数。

```python
class Solution:
    # 方法1: 超时，复杂度为O(n)，但是由于n巨大，所以gg
    # 支持10w以内
    def minDays(self, n: int) -> int:
        memo = dict() # k-v表示 k个橘子所对应的最小天数
        def recur(k):
            if k == 1:
                return 1
            if k in memo:
                return memo[k]

            state1 = recur(k-1)+1
            state2 = 0xffffffff
            state3 = 0xffffffff            
            if k%2 == 0:
                state2 = recur(k//2) + 1
            if k%3 == 0:
                state3 = recur(k//3) + 1

            memo[k] = min(state1,state2,state3)
            
            return memo[k]
        memo[1] = 1
        recur(n)
        return memo[n]
        
```

```

```

# 1554. 只有一个不同字符的字符串

给定一个字符串列表 `dict` ，其中所有字符串的长度都相同。

当存在两个字符串在相同索引处只有一个字符不同时，返回 `True` ，否则返回 `False` 。

```python
class Solution:
    def differByOne(self, dict: List[str]) -> bool:
        # 每个字符变成*尝试
        # 例: abcd --> *bcd  a*cd  ab*d  abc*
        # 已知所有字符都不同
        def change(s):
            group = []
            for i in range(len(s)):
                t = s[:i]+"*"+s[i+1:]
                group.append(t)
            return group
        
        ct = collections.defaultdict(int)
        for s in dict:
            for every in change(s):
                ct[every] += 1
                if ct[every] == 2:
                    return True
                    
        return False
```



# 1556. 千位分隔数

给你一个整数 `n`，请你每隔三位添加点（即 "." 符号）作为千位分隔符，并将结果以字符串格式返回。

```python
class Solution:
    def thousandSeparator(self, n: int) -> str:
        n = list(str(n))
        ans = []
        n = n[::-1]
        for i in range(0,len(n),3):
            ans += n[i:i+3]
            ans += ["."]
        ans.pop()
        ans = ans[::-1]
        return "".join(ans)
        
```

# 1559. 二维网格图中探测环

给你一个二维字符网格数组 grid ，大小为 m x n ，你需要检查 grid 中是否存在 相同值 形成的环。

一个环是一条开始和结束于同一个格子的长度 大于等于 4 的路径。对于一个给定的格子，你可以移动到它上、下、左、右四个方向相邻的格子之一，可以移动的前提是这两个格子有 相同的值 。

同时，你也不能回到上一次移动时所在的格子。比方说，环  (1, 1) -> (1, 2) -> (1, 1) 是不合法的，因为从 (1, 2) 移动到 (1, 1) 回到了上一次移动时的格子。

如果 grid 中有相同值形成的环，请你返回 true ，否则返回 false 。

```python
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        # 没有想到用合适的并查集思路，直接肝
        m = len(grid)
        n = len(grid[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        
        # 这个用来找是否存在环
        def dfs(i,j,aimChar,start_i,start_j,pre_i,pre_j,pathSet):
            nonlocal standard
            nonlocal state
            if i == start_i and j == start_j:
                state = True 
                return 
            if (i,j) in pathSet:
                state = True 
                return 
            if visited[i][j] == True:
                return 
            visited[i][j] = True 
            pathSet.add((i,j))               
            if start_i == None and start_j == None:
                start_i,start_j = standard            
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and (new_i,new_j) != (pre_i,pre_j) and grid[new_i][new_j] == aimChar: 
                    dfs(new_i,new_j,aimChar,start_i,start_j,i,j,pathSet)
        
        # 这个每次扫完之后，把连在一起的同字符清除，提升效率
        def clean(i,j,target): # 清除
            grid[i][j] = ""
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and grid[new_i][new_j] == target:
                    clean(new_i,new_j,target)

        for i in range(m):
            for j in range(n):
                if grid[i][j] != "":
                    state = False 
                    standard = (i,j)
                    aimChar = grid[i][j] # 目标元素
                    pathSet = set()
                    dfs(i,j,aimChar,None,None,None,None,pathSet)
                    # print(i,j,grid,visited,state,pathSet)
                    if state == True:
                        return True 
                    clean(i,j,aimChar) # 调用清除
        return False
```

# 1567. 乘积为正数的最长子数组长度

给你一个整数数组 nums ，请你求出乘积为正数的最长子数组的长度。

一个数组的子数组是由原数组中零个或者更多个连续数字组成的数组。

请你返回乘积为正数的最长子数组长度。

```python
class Solution:
    def getMaxLen(self, nums: List[int]) -> int:
        # 先格式化，
        if len(nums) == 1:
            return 1 if nums[0] > 0 else 0

        n = len(nums)
        for i in range(n):
            if nums[i] > 0:
                nums[i] = 1
            elif nums[i] < 0:
                nums[i] = -1
            else:
                nums[i] = 0

        # 开两行dp,第一行是大于0的，第二行是小于0的
        dp = [[0 for j in range(n)] for t in range(2)]
        if nums[0] > 0:
            dp[0][0] = 1
            dp[1][0] = 0
        elif nums[0] < 0:
            dp[0][0] = 0
            dp[1][0] = 1

        for i in range(1,n):
            if nums[i] == 0:
                continue 
            # 这个dp也得靠纸和思路想，核心思路继承
            if nums[i] > 0:
                dp[0][i] = dp[0][i-1]+1 if dp[0][i-1] > 0 else 1
                dp[1][i] = dp[1][i-1]+1 if dp[1][i-1] > 0 else 0
            elif nums[i] < 0:
                dp[0][i] = dp[1][i-1]+1 if dp[1][i-1] > 0 else 0
                dp[1][i] = dp[0][i-1]+1 if dp[0][i-1] > 0 else 1 
        
        return max(dp[0])
```

# 1577. 数的平方等于两数乘积的方法数

给你两个整数数组 nums1 和 nums2 ，请你返回根据以下规则形成的三元组的数目（类型 1 和类型 2 ）：

类型 1：三元组 (i, j, k) ，如果 nums1[i]2 == nums2[j] * nums2[k] 其中 0 <= i < nums1.length 且 0 <= j < k < nums2.length
类型 2：三元组 (i, j, k) ，如果 nums2[i]2 == nums1[j] * nums1[k] 其中 0 <= i < nums2.length 且 0 <= j < k < nums1.length

```python
class Solution:
    def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        ans = 0
        nums1.sort()
        nums2.sort()
        # 双指针原理，两个数字越接近，乘积越大
        def calc(nums1,nums2):
            ans = 0
            for e in nums1:
                target = e*e
                left = 0
                right = len(nums2)-1
                while left < right:
                    if nums2[left]*nums2[right] > target:
                        right -= 1
                    elif nums2[left]*nums2[right] < target:
                        left += 1
                    elif nums2[left]*nums2[right] == target:
                        ct1 = 1
                        ct2 = 1
                        if nums2[left] != nums2[right]:
                            while left + 1< right and nums2[left] == nums2[left+1]:
                                left += 1
                                ct1 += 1
                            while left < right - 1 and nums2[right] == nums2[right-1]:
                                right -= 1
                                ct2 += 1
                            ans += ct1 * ct2 
                        elif nums2[left] == nums2[right]:
                            t = (right-left+1)
                            ans += (t*(t-1))//2   
                            left = right                    
                        left += 1
                        right -= 1
            return ans
        ans += calc(nums1,nums2)
        ans += calc(nums2,nums1)
        return ans
```

# 1586. 二叉搜索树迭代器 II

实现二叉搜索树（BST）的中序遍历迭代器 BSTIterator 类：

BSTIterator(TreeNode root) 初始化 BSTIterator 类的实例。二叉搜索树的根节点 root 作为构造函数的参数传入。内部指针使用一个不存在于树中且小于树中任意值的数值来初始化。
boolean hasNext() 如果当前指针在中序遍历序列中，存在右侧数值，返回 true ，否则返回 false 。
int next() 将指针在中序遍历序列中向右移动，然后返回移动后指针所指数值。
boolean hasPrev() 如果当前指针在中序遍历序列中，存在左侧数值，返回 true ，否则返回 false 。
int prev() 将指针在中序遍历序列中向左移动，然后返回移动后指针所指数值。
注意，虽然我们使用树中不存在的最小值来初始化内部指针，第一次调用 next() 需要返回二叉搜索树中最小的元素。

你可以假设 next() 和 prev() 的调用总是有效的。即，当 next()/prev() 被调用的时候，在中序遍历序列中一定存在下一个/上一个元素。

进阶：你可以不提前遍历树中的值来解决问题吗？

```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        # 非迭代器思想
        self.lst = []
        def inorder(node):
            if node == None:
                return
            inorder(node.left)
            self.lst.append(node.val)
            inorder(node.right)
        self.p = -1
        inorder(root)

    def hasNext(self) -> bool:
        return self.p+1 < len(self.lst)

    def next(self) -> int:
        self.p += 1
        val = self.lst[self.p]
        return val

    def hasPrev(self) -> bool:
        return self.p-1 >= 0

    def prev(self) -> int:  
        self.p -= 1      
        val = self.lst[self.p]
        return val
```

# 1589. 所有排列中的最大和

有一个整数数组 nums ，和一个查询数组 requests ，其中 requests[i] = [starti, endi] 。第 i 个查询求 nums[starti] + nums[starti + 1] + ... + nums[endi - 1] + nums[endi] 的结果 ，starti 和 endi 数组索引都是 从 0 开始 的。

你可以任意排列 nums 中的数字，请你返回所有查询结果之和的最大值。

由于答案可能会很大，请你将它对 109 + 7 取余 后返回。

```python
class Solution:
    def maxSumRangeQuery(self, nums: List[int], requests: List[List[int]]) -> int:
        # 把request的频次统计起来,频次高的配大数字，注意是闭区间reque
        mod = 10**9 + 7
        n = len(nums)
        # 上下车算法
        up = collections.defaultdict(int)
        down = collections.defaultdict(int)

        for u,d in requests:
            up[u] += 1
            down[d] += 1
        
        now = 0
        tempList = []
        for i in range(n):
            now += up[i]
            tempList.append(now)
            now -= down[i]
        
        tempList.sort() # 从小到大排序
        nums.sort()
        ans = sum(nums[i]*tempList[i] for i in range(n))
        return ans%mod
```

```go
func maxSumRangeQuery(nums []int, requests [][]int) int {
    // 上下车算法
    up := make(map[int]int)
    down := make(map[int]int)
    mod := 1000000007
    n := len(nums)
    now := 0
    
    for _,v := range(requests) {
        u := v[0]
        d := v[1]
        up[u] += 1
        down[d] += 1
    }
    
    tempList := make([]int,n,n)
    for i:=0;i<n;i++ {
        now += up[i]
        tempList[i] = now
        now -= down[i]
    }
    
    sort.Ints(nums)
    sort.Ints(tempList)
    
    ans := 0
    for i:=0;i<n;i++ {
        ans += nums[i] * tempList[i]
    }
        
    return ans%mod
    
}
```

# 1593. 拆分字符串使唯一子字符串的数目最大

给你一个字符串 s ，请你拆分该字符串，并返回拆分后唯一子字符串的最大数目。

字符串 s 拆分后可以得到若干 非空子字符串 ，这些子字符串连接后应当能够还原为原字符串。但是拆分出来的每个子字符串都必须是 唯一的 。

注意：子字符串 是字符串中的一个连续字符序列。

```python
class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        # 回溯暴力搜
        memoDict = collections.defaultdict(int)

        maxLength = 1

        def backtracking(index,temp):
            nonlocal maxLength
            if index >= len(s):
                if temp == "":
                    maxLength = max(maxLength,len(memoDict))
                    # print(memoDict)
                return 
            temp += s[index]

            if temp not in memoDict: # 尝试在这里中断
                memoDict[temp] += 1
                backtracking(index+1,"") # 清空temp
                del memoDict[temp]

            backtracking(index+1,temp) # 继续搜
        
        backtracking(0,"")
        return maxLength
```

# 1600. 皇位继承顺序

一个王国里住着国王、他的孩子们、他的孙子们等等。每一个时间点，这个家庭里有人出生也有人死亡。

这个王国有一个明确规定的皇位继承顺序，第一继承人总是国王自己。我们定义递归函数 Successor(x, curOrder) ，给定一个人 x 和当前的继承顺序，该函数返回 x 的下一继承人。

Successor(x, curOrder):
    如果 x 没有孩子或者所有 x 的孩子都在 curOrder 中：
        如果 x 是国王，那么返回 null
        否则，返回 Successor(x 的父亲, curOrder)
    否则，返回 x 不在 curOrder 中最年长的孩子

```python
class ThroneInheritance:

    def __init__(self, kingName: str):
        self.king = kingName
        self.deathDict = collections.defaultdict(bool)
        self.record = collections.defaultdict(list)
        self.record[kingName] = []

    def birth(self, parentName: str, childName: str) -> None:
        self.record[parentName].append(childName)

    def death(self, name: str) -> None:
        self.deathDict[name] = True

    def dfs(self,nowName): # 先序遍历
        temp = []
        temp.append(nowName)
        for e in self.record[nowName]:
            temp += self.dfs(e)
        
        final = []
        for element in temp:
            if element not in self.deathDict:
                final.append(element)
        return final

    def getInheritanceOrder(self) -> List[str]:
        return self.dfs(self.king)
```

```python
class ThroneInheritance:

    def __init__(self, kingName: str):
        self.king = kingName
        self.deathDict = collections.defaultdict(bool)
        self.record = collections.defaultdict(list)
        self.record[kingName] = []

    def birth(self, parentName: str, childName: str) -> None:
        self.record[parentName].append(childName)

    def death(self, name: str) -> None:
        self.deathDict[name] = True

    def dfs(self,nowName): # 先序遍历
        temp = []
        temp.append(nowName)
        for e in self.record[nowName]:
            temp += self.dfs(e)        
        return temp

    def getInheritanceOrder(self) -> List[str]:
        temp = self.dfs(self.king)
        final = []
        for element in temp:
            if element not in self.deathDict:
                final.append(element)
        return final
```

# 1631. 最小体力消耗路径

你准备参加一场远足活动。给你一个二维 rows x columns 的地图 heights ，其中 heights[row][col] 表示格子 (row, col) 的高度。一开始你在最左上角的格子 (0, 0) ，且你希望去最右下角的格子 (rows-1, columns-1) （注意下标从 0 开始编号）。你每次可以往 上，下，左，右 四个方向之一移动，你想要找到耗费 体力 最小的一条路径。

一条路径耗费的 体力值 是路径上相邻格子之间 高度差绝对值 的 最大值 决定的。

请你返回从左上角走到右下角的最小 体力消耗值 。

```python
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        # 二分穷举搜索,dfs
        left = 0
        right = 10**6
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        m = len(heights)
        n = len(heights[0])
        visited = [[False for j in range(n)] for i in range(m)]

        def dfs(i,j,limit):
            nonlocal state
            if i == m-1 and j == n-1:
                state = True 
                return 
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and abs(heights[new_i][new_j]-heights[i][j]) <= limit:
                    visited[new_i][new_j] = True 
                    dfs(new_i,new_j,limit)

        while left <= right:
            mid = (left+right)//2
            state = False 
            visited = [[False for j in range(n)] for i in range(m)] # 重置
            visited[0][0] = True
            dfs(0,0,mid)
            if state == True: # 长度可能偏长，可以收缩
                right = mid - 1
            elif state == False: # 长度不够，往右边搜
                left = mid + 1
        
        return left

```

```python
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        # 二分穷举搜索,bfs
        left = 0
        right = 10**6
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        m = len(heights)
        n = len(heights[0])
        visited = [[False for j in range(n)] for i in range(m)]

        def bfs(limit):
            nonlocal state 
            queue = [(0,0)]
            while len(queue) != 0:
                new_queue = []
                for i,j in queue:
                    if (i,j) == (m-1,n-1):
                        state = True
                        return 
                    for di in direc:
                        new_i = i + di[0]
                        new_j = j + di[1]
                        if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and abs(heights[new_i][new_j]-heights[i][j]) <= limit:
                            visited[new_i][new_j] = True 
                            new_queue.append((new_i,new_j))
                queue = new_queue
            return False 

        while left <= right:
            mid = (left+right)//2
            state = False 
            visited = [[False for j in range(n)] for i in range(m)] # 重置
            visited[0][0] = True
            bfs(mid)
            if state == True: # 长度可能偏长，可以收缩
                right = mid - 1
            elif state == False: # 长度不够，往右边搜
                left = mid + 1
        
        return left

```

```python
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        # dj算法
        m = len(heights)
        n = len(heights[0])
        direc = ((0,1),(0,-1),(1,0),(-1,0))
        # 注意这个distance的key要么设计成元组，要么设计成i*n+j
        # key 设计成元组比较好算邻居
        distance = collections.defaultdict(lambda:0xffffffff)

        distance[(0,0)] = 0 # 起始
        queue = [(0,(0,0))] # （d,坐标)

        while len(queue) != 0:
            nowDistance,coord = heapq.heappop(queue)
            if coord == (m-1,n-1):
                break 
            i,j = coord
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n:
                    # 注意这一行
                    new_distance = max(distance[(i,j)],abs(heights[i][j]-heights[new_i][new_j]))
                    if new_distance < distance[(new_i,new_j)]:
                        distance[(new_i,new_j)] = new_distance
                        heapq.heappush(queue,(new_distance,(new_i,new_j)))
        
        return nowDistance
               
```

# 1640. 能否连接形成数组

给你一个整数数组 arr ，数组中的每个整数 互不相同 。另有一个由整数数组构成的数组 pieces，其中的整数也 互不相同 。请你以 任意顺序 连接 pieces 中的数组以形成 arr 。但是，不允许 对每个数组 pieces[i] 中的整数重新排序。

如果可以连接 pieces 中的数组形成 arr ，返回 true ；否则，返回 false 。

```python
class Solution:
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        set1 = set(arr)
        set2 = set()
        for e in pieces:
            for t in e:
                set2.add(t)
        if set1 != set2: # 集合不相等肯定false
            return False 
        
        # 哈希
        tempDict = collections.defaultdict(lambda :-1)
        for e in pieces:
            tempDict[e[0]] = e

        p = 0
        n = len(arr)
        while p < n:
            element = tempDict[arr[p]]
            if element == -1: # 找不到也直接False
                return False 
            if element != arr[p:p+len(element)]: # 这里面不相等，直接False
                return False 
            p += len(element)

        return True # 能过筛选，返回True
```



# 1660. 纠正二叉树

你有一棵二叉树，这棵二叉树有个小问题，其中有且只有一个无效节点，它的右子节点错误地指向了与其在同一层且在其右侧的一个其他节点。

给定一棵这样的问题二叉树的根节点 root ，将该无效节点及其所有子节点移除（除被错误指向的节点外），然后返回新二叉树的根结点。

自定义测试用例：

测试用例的输入由三行组成：

TreeNode root
int fromNode （在 correctBinaryTree 中不可见）
int toNode （在 correctBinaryTree 中不可见）
当以 root 为根的二叉树被解析后，值为 fromNode 的节点 TreeNode 将其右子节点指向值为 toNode 的节点 TreeNode 。然后， root 传入 correctBinaryTree 的参数中。

```python
# 这个题绕的一笔，不需要管是否在右边。。。直接二次检查，复杂度常数*2
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def correctBinaryTree(self, root: TreeNode) -> TreeNode:
        parentDict = dict()

        def dfs(node,p):
            if node == None:
                return 
            parentDict[node] = p
            dfs(node.left,node)
            dfs(node.right,node)
        
        dfs(root,None)

        queue = [root]
        mark = None 
        while len(queue) != 0:
            new_queue = []
            tempSet = set()
            for node in queue:
                tempSet.add(node.val)
                if node.left != None:
                    new_queue.append(node.left)
                if node.right != None:
                    new_queue.append(node.right)
            for node in queue: # 二次检查
                if node.right != None and node.right.val in tempSet:
                    mark = node 
                    new_queue = [] # 清空
                    break 
            queue = new_queue 
        
        # mark对应的是需要被删除的，找到他的父节点,mark是节点
        p_mark = parentDict[mark]
        if p_mark.left != None and p_mark.left.val == mark.val:
            p_mark.left = None
        elif p_mark.right != None and p_mark.right.val == mark.val:
            p_mark.right = None 
        return root
```

# 1663. 具有给定数值的最小字符串

小写字符 的 数值 是它在字母表中的位置（从 1 开始），因此 a 的数值为 1 ，b 的数值为 2 ，c 的数值为 3 ，以此类推。

字符串由若干小写字符组成，字符串的数值 为各字符的数值之和。例如，字符串 "abe" 的数值等于 1 + 2 + 5 = 8 。

给你两个整数 n 和 k 。返回 长度 等于 n 且 数值 等于 k 的 字典序最小 的字符串。

注意，如果字符串 x 在字典排序中位于 y 之前，就认为 x 字典序比 y 小，有以下两种情况：

x 是 y 的一个前缀；
如果 i 是 x[i] != y[i] 的第一个位置，且 x[i] 在字母表中的位置比 y[i] 靠前。

```python
class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        # 已知k一定合法
        # 贪心生成,提示性能用列表接字符
        lst = []

        # 先算a
        while (n - 1) * 26 >= k:
            lst.append(chr(97))
            n -= 1
            k -= 1
        # 算中间
        now = k % 26
        if now != 0:
            lst.append(chr(now+96))
        # 算结尾
        times = k//26
        for t in range(times):
            lst.append("z")
        
        return ''.join(lst)
```

# 1664. 生成平衡数组的方案数

给你一个整数数组 nums 。你需要选择 恰好 一个下标（下标从 0 开始）并删除对应的元素。请注意剩下元素的下标可能会因为删除操作而发生改变。

比方说，如果 nums = [6,1,7,4,1] ，那么：

选择删除下标 1 ，剩下的数组为 nums = [6,7,4,1] 。
选择删除下标 2 ，剩下的数组为 nums = [6,1,4,1] 。
选择删除下标 4 ，剩下的数组为 nums = [6,1,7,4] 。
如果一个数组满足奇数下标元素的和与偶数下标元素的和相等，该数组就是一个 平衡数组 。

请你返回删除操作后，剩下的数组 nums 是 平衡数组 的 方案数 。

```python
class Solution:
    def waysToMakeFair(self, nums: List[int]) -> int:
        # 强化前缀和，每个位置都不计算自身的前缀和后缀
        # 常数贼大
        n = len(nums)
        preSumOdd = 0
        preSumEven = 0
        preSumList = [None for i in range(n)]

        for i in range(n):
            preSumList[i] = [preSumOdd,preSumEven]
            if i % 2 == 0:
                preSumEven += nums[i]
            elif i % 2 == 1:
                preSumOdd += nums[i]

        # 镜像做后缀和
        postSumOdd = 0
        postSumEven = 0
        postSumList = [None for i in range(n)]
        for i in range(n-1,-1,-1):
            postSumList[i] = [postSumOdd,postSumEven]
            if i % 2 == 0:
                postSumEven += nums[i]
            else:
                postSumOdd += nums[i]
        
        # 删除该元素之后，后面的元素奇偶顺序被打乱,
        count = 0
        for i in range(n):
            even = preSumList[i][1] + postSumList[i][0]
            odd = preSumList[i][0] + postSumList[i][1]
            if even == odd:
                count += 1
        return count
```

# 1685. 有序数组中差绝对值之和

给你一个 非递减 有序整数数组 nums 。

请你建立并返回一个整数数组 result，它跟 nums 长度相同，且result[i] 等于 nums[i] 与数组中所有其他元素差的绝对值之和。

换句话说， result[i] 等于 sum(|nums[i]-nums[j]|) ，其中 0 <= j < nums.length 且 j != i （下标从 0 开始）。

```python
class Solution:
    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        # 用前缀和
        # 包括当前数的前缀和
        # 这里的后缀和是伪后缀和，仅仅用这个称呼
        preSum = 0
        n = len(nums)
        preArr = [0 for i in range(n)]

        postSum = 0
        postArr = [0 for i in range(n)]
        ans = [0 for i in range(n)]
        for i in range(n):
            preSum += nums[i]
            preArr[i] = preSum
        for i in range(n-1,-1,-1):
            postSum += nums[i]
            postArr[i] = postSum

        # 当前值应该为此时的前缀和-此时的数*(索引+1)+后缀和-数（长度-索引），都需要abs
        for i in range(n):
            ans[i] = abs(preArr[i]-nums[i]*(i+1))+abs(postArr[i]-nums[i]*(n-i))
        return ans
```



# 1698. 字符串的不同子字符串个数

给定一个字符串 s，返回 s 的不同子字符串的个数。

字符串的 子字符串 是由原字符串删除开头若干个字符（可能是 0 个）并删除结尾若干个字符（可能是 0 个）形成的字符串。

```python
class Solution:
    def countDistinct(self, s: str) -> int:
        # 需要开后缀数组才能达到On
        if len(s) == 1:
            return 1
        ansSet = set()
        for size in range(1,len(s)):
            for left in range(len(s)+1):
                t = s[left:left+size]
                ansSet.add(t)
        ans = []
        for e in ansSet:
            ans.append(e)
        return len(ans)
```

# 1722. 执行交换操作后的最小汉明距离

给你两个整数数组 source 和 target ，长度都是 n 。还有一个数组 allowedSwaps ，其中每个 allowedSwaps[i] = [ai, bi] 表示你可以交换数组 source 中下标为 ai 和 bi（下标从 0 开始）的两个元素。注意，你可以按 任意 顺序 多次 交换一对特定下标指向的元素。

相同长度的两个数组 source 和 target 间的 汉明距离 是元素不同的下标数量。形式上，其值等于满足 source[i] != target[i] （下标从 0 开始）的下标 i（0 <= i <= n-1）的数量。

在对数组 source 执行 任意 数量的交换操作后，返回 source 和 target 间的 最小汉明距离 。

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
    
    def isConnect(self,x,y):
        return self.root(x) == self.root(y)


class Solution:
    def minimumHammingDistance(self, source: List[int], target: List[int], allowedSwaps: List[List[int]]) -> int:
        # 并查集
        n = len(source)
        ufset = UF(n)
        for x,y in allowedSwaps:
            ufset.union(x,y)

        originDict = collections.defaultdict(list)
        for i in range(n):
            origin = ufset.find(i)
            originDict[origin].append(i)

        count = 0

        for key in originDict:
            lst1 = [source[i] for i in originDict[key]]
            lst2 = [target[i] for i in originDict[key]]
            length = len(originDict[key])
            lst1.sort()
            lst2.sort()
            p1 = 0
            p2 = 0            
            while p1 < length and p2 < length:
                if lst1[p1] == lst2[p2]:
                    count += 1
                    p1 += 1
                    p2 += 1
                elif lst1[p1] > lst2[p2]:
                    p2 += 1
                elif lst1[p1] < lst2[p2]:
                    p1 += 1

        return n-count 
```



# 1726. 同积元组

给你一个由 不同 正整数组成的数组 nums ，请你返回满足 a * b = c * d 的元组 (a, b, c, d) 的数量。其中 a、b、c 和 d 都是 nums 中的元素，且 a != b != c != d 。

```python
class Solution:
    def tupleSameProduct(self, nums: List[int]) -> int:
        # nums已经是不同的了,且都为正数
        # 数学规律化简，由于所有数不同，那么结果一定是8的倍数
        if len(nums) < 4:
            return 0
        nums.sort()
        n = len(nums)
        ansDict = collections.defaultdict(int)
        for i in range(n):
            for j in range(i+1,n):
                mul = nums[i]*nums[j]
                ansDict[mul] += 1
                
        # 每一对搭配有8个,cn2
        count = 0
        for key in ansDict:
            e = ansDict[key]
            if e >= 2:
                count += ((e-1)*e//2) * 8
        return count
```

# 1734. 解码异或后的排列

给你一个整数数组 perm ，它是前 n 个正整数的排列，且 n 是个 奇数 。

它被加密成另一个长度为 n - 1 的整数数组 encoded ，满足 encoded[i] = perm[i] XOR perm[i + 1] 。比方说，如果 perm = [1,3,2] ，那么 encoded = [2,1] 。

给你 encoded 数组，请你返回原始数组 perm 。题目保证答案存在且唯一。

```python
class Solution:
    def decode(self, encoded: List[int]) -> List[int]:
        n = len(encoded) + 1
        # 隔一个
        pivot = 0
        for i in range(1,len(encoded),2):
            pivot ^= encoded[i]
        for i in range(1,n+1):
            pivot ^= i 
        
        # 此时pivot是首项
        ans = []
        ans.append(pivot)
        for tp in encoded:
            pivot ^= tp
            ans.append(pivot)
        return ans
```

# 1742. 盒子中小球的最大数量

你在一家生产小球的玩具厂工作，有 n 个小球，编号从 lowLimit 开始，到 highLimit 结束（包括 lowLimit 和 highLimit ，即 n == highLimit - lowLimit + 1）。另有无限数量的盒子，编号从 1 到 infinity 。

你的工作是将每个小球放入盒子中，其中盒子的编号应当等于小球编号上每位数字的和。例如，编号 321 的小球应当放入编号 3 + 2 + 1 = 6 的盒子，而编号 10 的小球应当放入编号 1 + 0 = 1 的盒子。

给你两个整数 lowLimit 和 highLimit ，返回放有最多小球的盒子中的小球数量。如果有多个盒子都满足放有最多小球，只需返回其中任一盒子的小球数量。

```
class Solution:
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
    		# 方法一		
        # 纯模拟试试看
        
        n = len(str(highLimit))
        lst = [0 for i in range(n*9+1)]
        for i in range(lowLimit,highLimit+1):
            index = sum(int(t) for t in str(i))
            lst[index] += 1
        return max(lst)
```

```
# 方法2:动态规划

```



# 1746. 经过一次操作后的最大子数组和

你有一个整数数组 `nums`。你只能将一个元素 `nums[i]` 替换为 `nums[i] * nums[i]`。

返回替换后的最大子数组和。

```python
class Solution:
    def maxSumAfterOperation(self, nums: List[int]) -> int:
        # 二维dp
        # dp[0][i]的含义为不包括当前元素，前缀的最大子数组和
        # dp[1][i]的含义为倒着的不包括当前元素，后缀的最大子数组和
        length = len(nums)
        dp = [[-0xffffffff for j in range(length)] for i in range(2)]
        # 初始化
        dp[0][0] = 0
        dp[1][-1] = 0

        MaxpreSum = 0
        for i in range(len(nums)):
            if MaxpreSum < 0:
                MaxpreSum = 0
            dp[0][i] = MaxpreSum
            MaxpreSum += nums[i]
        MaxpostSum = 0
        for i in range(len(nums)-1,-1,-1):
            if MaxpostSum < 0:
                MaxpostSum = 0
            dp[1][i] = MaxpostSum
            MaxpostSum += nums[i]
        
        # 然后是取前缀+当前平方+后缀
        ans = 0
        for i in range(len(nums)):
            temp = dp[0][i] + nums[i]**2 + dp[1][i]
            ans = max(ans,temp)
        return ans 
```

# 1749. 任意子数组和的绝对值的最大值

给你一个整数数组 nums 。一个子数组 [numsl, numsl+1, ..., numsr-1, numsr] 的 和的绝对值 为 abs(numsl + numsl+1 + ... + numsr-1 + numsr) 。

请你找出 nums 中 和的绝对值 最大的任意子数组（可能为空），并返回该 最大值 。

abs(x) 定义如下：

如果 x 是负整数，那么 abs(x) = -x 。
如果 x 是非负整数，那么 abs(x) = x 。

```python
class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        # 贪心最大+最小都加入比较
        maxAns = 0
        n = len(nums)
        # 先找最大值
        tempSum = 0 
        dp1 = [-0xffffffff for i in range(n)]
        for i in range(n):
            if tempSum >= 0: # 前缀大于0
                dp1[i] = nums[i] + tempSum
            elif tempSum < 0:
                dp1[i] = nums[i]
                tempSum = 0 # 重置
            tempSum += nums[i]
        tempSum = 0
        dp2 = [0xffffffff for i in range(n)]
        for i in range(n):
            if tempSum < 0: # 前缀小于0
                dp2[i] = nums[i] + tempSum
            elif tempSum >= 0:
                dp2[i] = nums[i]
                tempSum = 0 # 重置
            tempSum += nums[i]
        
        a = max(dp1) # 最大值
        b = min(dp2) # 最小值
        return max(abs(a),abs(b))
```



# 1762. 能看到海景的建筑物

有 n 座建筑物。给你一个大小为 n 的整数数组 heights 表示每一个建筑物的高度。

建筑物的右边是海洋。如果建筑物可以无障碍地看到海洋，则建筑物能看到海景。确切地说，如果一座建筑物右边的所有建筑都比它 矮 时，就认为它能看到海景。

返回能看到海景建筑物的下标列表（下标 从 0 开始 ），并按升序排列。

```python
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        # 右边所有建筑物都比它矮，单调栈
        # 单调递减栈，最终栈内的所有元素下标都是结果
        # 需要严格小于
        stack = [] # 里面存的是索引
        for i in range(len(heights)):
            if len(stack) == 0:
                stack.append(i)
                continue 
            if heights[stack[-1]] <= heights[i]:
                while len(stack) != 0 and heights[stack[-1]] <= heights[i]:
                    stack.pop()
                stack.append(i)
            else:
                stack.append(i)
        return stack
```

# 1763. 最长的美好子字符串

当一个字符串 s 包含的每一种字母的大写和小写形式 同时 出现在 s 中，就称这个字符串 s 是 美好 字符串。比方说，"abABB" 是美好字符串，因为 'A' 和 'a' 同时出现了，且 'B' 和 'b' 也同时出现了。然而，"abA" 不是美好字符串因为 'b' 出现了，而 'B' 没有出现。

给你一个字符串 s ，请你返回 s 最长的 美好子字符串 。如果有多个答案，请你返回 最早 出现的一个。如果不存在美好子字符串，请你返回一个空字符串。

```python
class Solution:
    def longestNiceSubstring(self, s: str) -> str:
        # 暴力搜的方法
        def judge(strings):
            memoDict = collections.defaultdict(set)
            for ch in strings:
                memoDict[ch.lower()].add(ch)
            for key in memoDict:
                if len(memoDict[key]) != 2:
                    return False
            return True

        n = len(s)
        temp = []
        
        for i in range(n):
            for j in range(i+1,n):
                if judge(s[i:j+1]):
                    temp.append(s[i:j+1])
        if len(temp) == 0:
            return ''
        temp.sort(key = lambda x: -len(x))
        return temp[0]
                
```

```python
# 或者固定窗口为，含有1～26种元素。滑动窗口26次.元素上限可以预先用set。
```

# 1781. 所有子字符串美丽值之和

一个字符串的 美丽值 定义为：出现频率最高字符与出现频率最低字符的出现次数之差。

比方说，"abaacc" 的美丽值为 3 - 1 = 2 。
给你一个字符串 s ，请你返回它所有子字符串的 美丽值 之和。

```python
class Solution:
    def beautySum(self, s: str) -> int:
        # 数据量一般，支持暴力,需要一个数据结构能够方便的找到最大最小值，用数组模拟哈希表。
        beau = 0
        # 找到非0最小值，
        def findMin(lst):
            themin = 0xffff
            for n in lst:
                if n != 0:
                    if n < themin:
                        themin = n
            return themin

        for size in range(2,len(s)+1):
            right = 0 # 初始化
            window = [0 for i in range(26)]
            for t in range(size):
                index = ord(s[right])-ord("a")
                window[index] += 1
                right += 1
            maxCount = max(window)
            minCount = findMin(window)
            beau += (maxCount - minCount)
            left = 0
            while right < len(s):
                index = ord(s[right])-ord("a")
                window[index] += 1
                right += 1
                index = ord(s[left])-ord("a")
                window[index] -= 1
                left += 1
                maxCount = max(window)
                minCount = findMin(window)
                beau += (maxCount - minCount)
        return beau
```

# 1814. 统计一个数组中好对子的数目

给你一个数组 nums ，数组中只包含非负整数。定义 rev(x) 的值为将整数 x 各个数字位反转得到的结果。比方说 rev(123) = 321 ， rev(120) = 21 。我们称满足下面条件的下标对 (i, j) 是 好的 ：

0 <= i < j < nums.length
nums[i] + rev(nums[j]) == nums[j] + rev(nums[i])
请你返回好下标对的数目。由于结果可能会很大，请将结果对 109 + 7 取余 后返回。

```python
class Solution:
    def countNicePairs(self, nums: List[int]) -> int:
        # 统计nums[i]-rev(nums[i])的分组
        # 然后用组合计数
        def rev(x):
            x = str(x)
            x = x[::-1]
            x = int(x)
            return x 
        ct = collections.defaultdict(int)
        for i in range(len(nums)):
            key = nums[i] - rev(nums[i])
            ct[key] += 1
        
        # 如果它的值大于等于2，则返回k*(k-1)//2,包括1也行
        ans = 0
        for key in ct:
            k = ct[key]
            ans += (k-1)*k//2
        return ans%(10**9+7)
```

# 1817. 查找用户活跃分钟数

给你用户在 LeetCode 的操作日志，和一个整数 k 。日志用一个二维整数数组 logs 表示，其中每个 logs[i] = [IDi, timei] 表示 ID 为 IDi 的用户在 timei 分钟时执行了某个操作。

多个用户 可以同时执行操作，单个用户可以在同一分钟内执行 多个操作 。

指定用户的 用户活跃分钟数（user active minutes，UAM） 定义为用户对 LeetCode 执行操作的 唯一分钟数 。 即使一分钟内执行多个操作，也只能按一分钟计数。

请你统计用户活跃分钟数的分布情况，统计结果是一个长度为 k 且 下标从 1 开始计数 的数组 answer ，对于每个 j（1 <= j <= k），answer[j] 表示 用户活跃分钟数 等于 j 的用户数。

返回上面描述的答案数组 answer 。

```python
class Solution:
    def findingUsersActiveMinutes(self, logs: List[List[int]], k: int) -> List[int]:
        # 对于python这种自带元组+集合的这种题有点耍赖。
        ans = [0 for i in range(k)]
        for i in range(len(logs)):
            logs[i] = tuple(logs[i])
        logs = set(logs)

        count = collections.defaultdict(int) # k-v是 id-分钟数
        for log in logs:
            count[log[0]] += 1
        
        mirrorDict = collections.defaultdict(int) # k-v是 分钟数-人数
        for key in count:
            new_key = count[key]
            mirrorDict[new_key] += 1

        # 注意下标是从1开始
        for key in mirrorDict:
            ans[key-1] = mirrorDict[key]
        return ans
```

# 1829. 每个查询的最大异或值

给你一个 有序 数组 nums ，它由 n 个非负整数组成，同时给你一个整数 maximumBit 。你需要执行以下查询 n 次：

找到一个非负整数 k < 2maximumBit ，使得 nums[0] XOR nums[1] XOR ... XOR nums[nums.length-1] XOR k 的结果 最大化 。k 是第 i 个查询的答案。
从当前数组 nums 删除 最后 一个元素。
请你返回一个数组 answer ，其中 answer[i]是第 i 个查询的结果。

```python
class Solution:
    def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
        # ans里面存的是k,k是按位取反
        ans = []
        start = 0
        n = len(nums)
        p = n - 1
        for i in range(n):
            start ^= nums[i]
        
        def getOp(n,theLength): # 传入一个数，按位置取反
            s = list(bin(n)[2:])
            s = ["0" for i in range(theLength-len(s))] + s
            for i in range(len(s)):
                if s[i] == "1":
                    s[i] = 0
                elif s[i] == "0":
                    s[i] = 1
            s = s[::-1]
            ans = 0
            for i in range(len(s)):
                ans += s[i]*(2**i)
            return ans

        while p >= 0:            
            ans.append(getOp(start,maximumBit))
            start ^= nums[p]
            p -= 1
        
        return ans
```

```python
class Solution:
    def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
        # ans里面存的是k,k是按位取反
        ans = []
        start = 0
        n = len(nums)
        p = n - 1
        for i in range(n):
            start ^= nums[i]
        
        ops = pow(2,maximumBit)-1

        while p >= 0:            
            ans.append(start^ops) # 优化按位取反
            start ^= nums[p]
            p -= 1
        
        return ans
```

# 1839. 所有元音按顺序排布的最长子字符串

当一个字符串满足如下条件时，我们称它是 美丽的 ：

所有 5 个英文元音字母（'a' ，'e' ，'i' ，'o' ，'u'）都必须 至少 出现一次。
这些元音字母的顺序都必须按照 字典序 升序排布（也就是说所有的 'a' 都在 'e' 前面，所有的 'e' 都在 'i' 前面，以此类推）
比方说，字符串 "aeiou" 和 "aaaaaaeiiiioou" 都是 美丽的 ，但是 "uaeio" ，"aeoiu" 和 "aaaeeeooo" 不是美丽的 。

给你一个只包含英文元音字母的字符串 word ，请你返回 word 中 最长美丽子字符串的长度 。如果不存在这样的子字符串，请返回 0 。

子字符串 是字符串中一个连续的字符序列。

```python
class Solution:
    def longestBeautifulSubstring(self, word: str) -> int:
        window = collections.defaultdict(int)
        windowLength = 0 # 窗口长度
        validMaxLength = 0 # 最大合法长度
        right = 0
        now = 97 # 默认为'a'的ascii码
        # 其实可以优化每次right从是a开始搜
        while right < len(word):
            add = word[right]
            right += 1
            windowLength += 1
            if ord(add) >= now:
                now = ord(add)
                window[add] += 1
                if len(window) == 5:
                    validMaxLength = max(validMaxLength,windowLength)
            elif ord(add) < now:
                now = ord(add)
                window = collections.defaultdict(int) # 重置
                window[add] += 1
                windowLength = 1
        return validMaxLength
```

# 1848. 到目标元素的最小距离

给你一个整数数组 nums （下标 从 0 开始 计数）以及两个整数 target 和 start ，请你找出一个下标 i ，满足 nums[i] == target 且 abs(i - start) 最小化 。注意：abs(x) 表示 x 的绝对值。

返回 abs(i - start) 。

题目数据保证 target 存在于 nums 中。

```python
class Solution:
    def getMinDistance(self, nums: List[int], target: int, start: int) -> int:
        # 数据量小，直接爆搜
        indexList = []
        for i in range(len(nums)):
            if nums[i] == target:
                indexList.append((abs(i-start)))
        indexList.sort()
        return indexList[0]
```

# 1849. 将字符串拆分为递减的连续值

给你一个仅由数字组成的字符串 s 。

请你判断能否将 s 拆分成两个或者多个 非空子字符串 ，使子字符串的 数值 按 降序 排列，且每两个 相邻子字符串 的数值之 差 等于 1 。

例如，字符串 s = "0090089" 可以拆分成 ["0090", "089"] ，数值为 [90,89] 。这些数值满足按降序排列，且相邻值相差 1 ，这种拆分方法可行。
另一个例子中，字符串 s = "001" 可以拆分成 ["0", "01"]、["00", "1"] 或 ["0", "0", "1"] 。然而，所有这些拆分方法都不可行，因为对应数值分别是 [0,1]、[0,1] 和 [0,0,1] ，都不满足按降序排列的要求。
如果可以按要求拆分 s ，返回 true ；否则，返回 false 。

子字符串 是字符串中的一个连续字符序列。

```python
class Solution:
    def splitString(self, s: str) -> bool:
        n = len(s)
        state = False 
        
        def check(stack):
            t = [int(e) for e in stack]
            if len(t) <= 1:
                return False 
            p = 1
            while p < len(t):
                if t[p]-t[p-1] != -1:
                    return False
                p += 1
            return True 
        
        def backtracking(path,index,temp):
            nonlocal state
            if index == n:
                # print(path,temp)
                if not state:
                    if len(temp) == 0 and check(path): 
                        state = True
                return 

            if len(path) < 2 or len(path) >= 2 and check(path): # 这个剪枝很重要
                          
                    temp.append(s[index])
                    path.append("".join(temp))
                    backtracking(path,index+1,[])
                    path.pop()   
                    temp.pop()

                    temp.append(s[index])
                    backtracking(path,index+1,temp)
                    temp.pop()
            
        backtracking([],0,[])
        return state
```



# 1870. 准时到达的列车最小时速

给你一个浮点数 hour ，表示你到达办公室可用的总通勤时间。要到达办公室，你必须按给定次序乘坐 n 趟列车。另给你一个长度为 n 的整数数组 dist ，其中 dist[i] 表示第 i 趟列车的行驶距离（单位是千米）。

每趟列车均只能在整点发车，所以你可能需要在两趟列车之间等待一段时间。

例如，第 1 趟列车需要 1.5 小时，那你必须再等待 0.5 小时，搭乘在第 2 小时发车的第 2 趟列车。
返回能满足你准时到达办公室所要求全部列车的 最小正整数 时速（单位：千米每小时），如果无法准时到达，则返回 -1 。

生成的测试用例保证答案不超过 107 ，且 hour 的 小数点后最多存在两位数字 。

```python
class Solution:
    def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
        # 速度必须是正整数,二分搜索+判断

        left = 1
        right = 2**32
        # 注意这个二分的right不一定是最大距离对应的单位速度
        # [1,1,100000]，2.01     返回   10000000
        # 如果均以最大速度都不能满足，则返回False,注意最后一个需要特殊处理
        state1 = 0
        for d in dist[:-1]:
            state1 += math.ceil(d/right)        
        state1 += (dist[-1]/right)
        if state1 > hour:
            return -1

        while left <= right:
            mid = (left + right)//2
            tempSum = 0
            for d in dist[:-1]: # 注意对最后一个的特殊处理
                tempSum += ceil(d/mid)
            tempSum += (dist[-1]/mid)

            if tempSum <= hour: # 速度过快，可以减速，即便减过头了也没有关系，变更的是right，但是返回的是left
                right = mid - 1
            elif tempSum > hour: # 速度过慢，需要加速
                left = mid + 1
        return left
```

# 1881. 插入后的最大值

给你一个非常大的整数 n 和一个整数数字 x ，大整数 n 用一个字符串表示。n 中每一位数字和数字 x 都处于闭区间 [1, 9] 中，且 n 可能表示一个 负数 。

你打算通过在 n 的十进制表示的任意位置插入 x 来 最大化 n 的 数值 。但 不能 在负号的左边插入 x 。

例如，如果 n = 73 且 x = 6 ，那么最佳方案是将 6 插入 7 和 3 之间，使 n = 763 。
如果 n = -55 且 x = 2 ，那么最佳方案是将 2 插在第一个 5 之前，使 n = -255 。
返回插入操作后，用字符串表示的 n 的最大值。

```python
class Solution:
    def maxValue(self, n: str, x: int) -> str:
        # 根据符号来判断
        n = list(n)
        if n[0] != "-": # 正数
            # 找到第一个比x小的位置，插入
            # 为了方便边界处理，尾巴加上一个0
            n.append("0")
            for i in range(len(n)):
                if int(n[i]) < x:  # 插在这个数的前面
                    n = n[:i]+[str(x)]+n[i:]
                    break 
            n.pop()
            return "".join(n)
        elif n[0] == "-": # 负数
            # 找到第一个比x大的位置，在它之前插入
            state = False
            for i in range(1,len(n)):
                if int(n[i]) > x:
                    n = n[:i]+[str(x)]+n[i:]
                    state = True
                    break
            if not state: # 扫完了都没有插入过
                n.append(str(x))
            return "".join(n)
```

# 1891. 割绳子

给定一个整数数组 ribbons 和一个整数 k，数组每项 ribbons[i] 表示第 i 条绳子的长度。对于每条绳子，你可以将任意切割成一系列长度为正整数的部分，或者选择不进行切割。

例如，如果给你一条长度为 4 的绳子，你可以：

保持绳子的长度为 4 不变；
切割成一条长度为 3 和一条长度为 1 的绳子；
切割成两条长度为 2 的绳子；
切割成一条长度为 2 和两条长度为 1 的绳子；
切割成四条长度为 1 的绳子。
你的任务是最终得到 k 条完全一样的绳子，他们的长度均为相同的正整数。如果绳子切割后有剩余，你可以直接舍弃掉多余的部分。

对于这 k 根绳子，返回你能得到的绳子最大长度；如果你无法得到 k 根相同长度的绳子，返回 0。

```python
class Solution:
    def maxLength(self, ribbons: List[int], k: int) -> int:
        # 二分法，left是1，right是数组最大值
        left = 1
        right = max(ribbons)
        ans = 0
        while left <= right:
            mid = (left+right)//2
            cuts = 0
            for r in ribbons:
                cuts += r//mid 
            if cuts >= k: # 次数满足要求，收集答案且尝试曾长
                ans = max(ans,mid)
                left = mid + 1
            else: # 次数不满足要求
                right = mid - 1
        return ans
```

# 1904. 你完成的完整对局数

一款新的在线电子游戏在近期发布，在该电子游戏中，以 刻钟 为周期规划若干时长为 15 分钟 的游戏对局。这意味着，在 HH:00、HH:15、HH:30 和 HH:45 ，将会开始一个新的对局，其中 HH 用一个从 00 到 23 的整数表示。游戏中使用 24 小时制的时钟 ，所以一天中最早的时间是 00:00 ，最晚的时间是 23:59 。

给你两个字符串 startTime 和 finishTime ，均符合 "HH:MM" 格式，分别表示你 进入 和 退出 游戏的确切时间，请计算在整个游戏会话期间，你完成的 完整对局的对局数 。

例如，如果 startTime = "05:20" 且 finishTime = "05:59" ，这意味着你仅仅完成从 05:30 到 05:45 这一个完整对局。而你没有完成从 05:15 到 05:30 的完整对局，因为你是在对局开始后进入的游戏；同时，你也没有完成从 05:45 到 06:00 的完整对局，因为你是在对局结束前退出的游戏。
如果 finishTime 早于 startTime ，这表示你玩了个通宵（也就是从 startTime 到午夜，再从午夜到 finishTime）。

假设你是从 startTime 进入游戏，并在 finishTime 退出游戏，请计算并返回你完成的 完整对局的对局数 。

```python
class Solution:
    def numberOfRounds(self, startTime: str, finishTime: str) -> int:
        
        def toMinute(s):
            mm = int(s[:2])
            hh = int(s[3:])
            return mm*60+hh 
        
        st = toMinute(startTime)
        ft = toMinute(finishTime)
        # 处理通宵
        if st > ft:
            ft += 1440
        # 获取时间段,最好不要采取相对偏移发
        nowoff = int(st//15*15)
        count = 0
        while nowoff <= ft:
            if st <= nowoff and nowoff+15 <= ft:
                count += 1
            nowoff += 15
        return count
```

# 1922. 统计好数字的数目

我们称一个数字字符串是 好数字 当它满足（下标从 0 开始）偶数 下标处的数字为 偶数 且 奇数 下标处的数字为 质数 （2，3，5 或 7）。

比方说，"2582" 是好数字，因为偶数下标处的数字（2 和 8）是偶数且奇数下标处的数字（5 和 2）为质数。但 "3245" 不是 好数字，因为 3 在偶数下标处但不是偶数。
给你一个整数 n ，请你返回长度为 n 且为好数字的数字字符串 总数 。由于答案可能会很大，请你将它对 109 + 7 取余后返回 。

一个 数字字符串 是每一位都由 0 到 9 组成的字符串，且可能包含前导 0 。

```python
class Solution:
    def countGoodNumbers(self, n: int) -> int:
        # 注意可以有0,调用快速幂
        if n == 1:
            return 5
        mod = 10**9 + 7
        # 第一位选择有5种，第二位选择为4种
        # 如果n是奇数，则有n//2+1次5,n//2次4
        # 如果n是偶数，则有n//2次5,n//2次4
        # 需要调用快速幂
    
        if n%2 == 1:
            k1 = pow(5,n//2+1,mod)
            k2 = pow(4,n//2,mod)
        elif n%2 == 0:
            k1 = pow(5,n//2,mod)
            k2 = pow(4,n//2,mod)
        return (k1*k2)%mod 

```

```go
const mod = 1000000007
func countGoodNumbers(n int64) int {
    // 写个快速幂
    return quickPow(5, (int(n)+1)/2,mod) * quickPow(4, int(n)/2,mod) % mod
}

func quickPow(a,b,mod int) int {
    ans := 1
    for b != 0 {
        if b % 2 == 1 {
            ans = ans*a%mod
        }
        b /= 2
        a = a*a%mod
    }
    return ans
}
```



# 1947. 最大兼容性评分和

有一份由 n 个问题组成的调查问卷，每个问题的答案要么是 0（no，否），要么是 1（yes，是）。

这份调查问卷被分发给 m 名学生和 m 名导师，学生和导师的编号都是从 0 到 m - 1 。学生的答案用一个二维整数数组 students 表示，其中 students[i] 是一个整数数组，包含第 i 名学生对调查问卷给出的答案（下标从 0 开始）。导师的答案用一个二维整数数组 mentors 表示，其中 mentors[j] 是一个整数数组，包含第 j 名导师对调查问卷给出的答案（下标从 0 开始）。

每个学生都会被分配给 一名 导师，而每位导师也会分配到 一名 学生。配对的学生与导师之间的兼容性评分等于学生和导师答案相同的次数。

例如，学生答案为[1, 0, 1] 而导师答案为 [0, 0, 1] ，那么他们的兼容性评分为 2 ，因为只有第二个和第三个答案相同。
请你找出最优的学生与导师的配对方案，以 最大程度上 提高 兼容性评分和 。

给你 students 和 mentors ，返回可以得到的 最大兼容性评分和 。

```python
class Solution:
    def maxCompatibilitySum(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        # 记忆化+全排列
        n = len(students)
        memo = dict() # key是(sId,mId),v是分数

        for sId,s in enumerate(students):
            for mId,m in enumerate(mentors):
                length = len(s)
                score = 0
                for t in range(length):
                    if s[t] == m[t]:
                        score += 1
                memo[(sId,mId)] = score
        
        lst = [i for i in range(n)]
        ans = []

        def backtracking(path,lst):
            if len(path) == n:
                ans.append(path[:])
                return 
            for num in lst:
                cp = lst.copy()
                cp.remove(num)
                path.append(num)
                backtracking(path,cp)
                path.pop()

        backtracking([],lst)
        maxScores = 0
        
        # 不需要学生和老师都做映射。固定学生动老师就行
        for every2 in ans:
            temp = 0
            for i in range(n):
                pair = (i,every2[i]) 
                temp += memo[pair]
            maxScores = max(maxScores,temp)

        return maxScores    
```

# 1954. 收集足够苹果的最小花园周长

给你一个用无限二维网格表示的花园，每一个 整数坐标处都有一棵苹果树。整数坐标 (i, j) 处的苹果树有 |i| + |j| 个苹果。

你将会买下正中心坐标是 (0, 0) 的一块 正方形土地 ，且每条边都与两条坐标轴之一平行。

给你一个整数 neededApples ，请你返回土地的 最小周长 ，使得 至少 有 neededApples 个苹果在土地 里面或者边缘上。

|x| 的值定义为：

如果 x >= 0 ，那么值为 x
如果 x < 0 ，那么值为 -x

```python
class Solution:
    def minimumPerimeter(self, neededApples: int) -> int:
        # 第i圈有(i+2i)*(i+1)/2*8-4i-8i
        # 累加有 sigma(12*i^2) , 返回为8i
        # 总苹果由平方和公式有 2n(n+1)(2n+1)
        def calc(n):
            return 2*n*(n+1)*(2*n+1)


        left = 1
        right = neededApples

        while left <= right:
            mid = (left+right)//2
            if calc(mid) >= neededApples:
                right = mid - 1
            elif calc(mid) < neededApples:
                left = mid + 1
        return left * 8
```

```python
class Solution:
    def minimumPerimeter(self, neededApples: int) -> int:
        # 第i圈有(i+2i)*(i+1)/2*8-4i-8i
        # 累加有 sigma(12*i^2) , 返回为8i
        # 总苹果由平方和公式有 2n(n+1)(2n+1)
        def calc(n):
            return 2*n*(n+1)*(2*n+1)

        for n in range(neededApples+1):
            if calc(n) >= neededApples:
                return 8*n
            
```



# 1963. 使字符串平衡的最小交换次数

给你一个字符串 s ，下标从 0 开始 ，且长度为偶数 n 。字符串 恰好 由 n / 2 个开括号 '[' 和 n / 2 个闭括号 ']' 组成。

只有能满足下述所有条件的字符串才能称为 平衡字符串 ：

字符串是一个空字符串，或者
字符串可以记作 AB ，其中 A 和 B 都是 平衡字符串 ，或者
字符串可以写成 [C] ，其中 C 是一个 平衡字符串 。
你可以交换 任意 两个下标所对应的括号 任意 次数。

返回使 s 变成 平衡字符串 所需要的 最小 交换次数。

```python
class Solution:
    def minSwaps(self, s: str) -> int:
        # 最终只需要统计]]]][[[[这种形式
        # (pair + 1)//2为结果
        stack = []

        for ch in s:
            if ch == "[":
                stack.append(ch)
            if ch == "]":
                if len(stack) > 0 and stack[-1] == "[":
                    stack.pop()
                else:
                    stack.append(ch)

        pair = len(stack)//2
        return (pair+1)//2
```

# 1971. Find if Path Exists in Graph

There is a bi-directional graph with n vertices, where each vertex is labeled from 0 to n - 1 (inclusive). The edges in the graph are represented as a 2D integer array edges, where each edges[i] = [ui, vi] denotes a bi-directional edge between vertex ui and vertex vi. Every vertex pair is connected by at most one edge, and no vertex has an edge to itself.

You want to determine if there is a valid path that exists from vertex start to vertex end.

Given edges and the integers n, start, and end, return true if there is a valid path from start to end, or false otherwise.

```python
class Solution:
    def validPath(self, n: int, edges: List[List[int]], start: int, end: int) -> bool:
        # bfs,只需要判断是否合法,无重复边，无自环
        graph = collections.defaultdict(list)
        for a,b in edges:
            graph[a].append(b)
            graph[b].append(a)
        visited = set()

        queue = [start]
        visited.add(start)

        while len(queue) != 0:
            new_queue = []
            for e in queue:
                if e == end:
                    return True 
                for neigh in graph[e]:
                    if neigh not in visited:
                        new_queue.append(neigh)
                        visited.add(neigh)
            queue = new_queue
        return False
                

```

# 1973. Count Nodes Equal to Sum of Descendants

Given the root of a binary tree, return the number of nodes where the value of the node is equal to the sum of the values of its descendants.

A descendant of a node x is any node that is on the path from node x to some leaf node. The sum is considered to be 0 if the node has no descendants.

```python
class Solution:
    def equalToDescendants(self, root: Optional[TreeNode]) -> int:
        # 后续遍历
        ans = 0
        def postOrder(node):
            nonlocal ans 
            if node == None:
                return 0
            leftPart = postOrder(node.left)
            rightPart = postOrder(node.right)
            if leftPart + rightPart == node.val:
                ans += 1
            return leftPart+rightPart+node.val
        
        postOrder(root)
        return ans
```

# 2017. 网格游戏

给你一个下标从 0 开始的二维数组 grid ，数组大小为 2 x n ，其中 grid[r][c] 表示矩阵中 (r, c) 位置上的点数。现在有两个机器人正在矩阵上参与一场游戏。

两个机器人初始位置都是 (0, 0) ，目标位置是 (1, n-1) 。每个机器人只会 向右 ((r, c) 到 (r, c + 1)) 或 向下 ((r, c) 到 (r + 1, c)) 。

游戏开始，第一个 机器人从 (0, 0) 移动到 (1, n-1) ，并收集路径上单元格的全部点数。对于路径上所有单元格 (r, c) ，途经后 grid[r][c] 会重置为 0 。然后，第二个 机器人从 (0, 0) 移动到 (1, n-1) ，同样收集路径上单元的全部点数。注意，它们的路径可能会存在相交的部分。

第一个 机器人想要打击竞争对手，使 第二个 机器人收集到的点数 最小化 。与此相对，第二个 机器人想要 最大化 自己收集到的点数。两个机器人都发挥出自己的 最佳水平 的前提下，返回 第二个 机器人收集到的 点数 。

```python
class Solution:
    def gridGame(self, grid: List[List[int]]) -> int:
        n = len(grid[0])
        if n == 1:
            return 0
        # 画图辅助理解，对红色机器人
        # 它有n次往下走的权利，留白为n-1个白格，计算二维前缀和
        # 留白为左下角和右上角。第一个机器人的任务是使得这两者的和最小化
        accum1 = [0 for i in range(n)]
        accum2 = [0 for i in range(n)]
        # 都是包含本位置的前缀和
        p1 = 0
        p2 = 0
        for i in range(n):
            p1 += grid[0][i]
            accum1[i] = p1 
            p2 += grid[1][i]
            accum2[i] = p2 
        # print(accum1,accum2)
        # 拐点时，右上角为accum1[n-1]-accum1[i]
        # 左下角为 accum2[i]-grid[1][i]
        ans = 0xffffffff
        # 注意这一行！
        for i in range(n):
            ans = min(ans,max((accum1[n-1]-accum1[i]), (accum2[i]-grid[1][i])))
        return ans
```

```python
func gridGame(grid [][]int) int64 {
    // 前缀和
    n := len(grid[0])
    accum1 := make([]int,n,n)
    accum2 := make([]int,n,n)
    pre1 := 0
    pre2 := 0
    for i:=0;i<n;i++ {
        pre1 += grid[0][i]
        accum1[i] = pre1 
        pre2 += grid[1][i]
        accum2[i] = pre2
    }

    var ans int64= math.MaxInt64
    for i:=0;i<n;i++ {
        var temp3 int64= max(int64(accum1[n-1]-accum1[i]),int64(accum2[i]-grid[1][i]))
        ans = min(ans,temp3)
    }
    return int64(ans)
}

func max(a,b int64) int64 {
    if a > b {
        return int64(a)
    } else {
        return int64(b)
    }
}

func min(a,b int64) int64 {
    if a < b {
        return int64(a)
    } else {
        return int64(b)
    }
}
```



# 2047. 句子中的有效单词数

句子仅由小写字母（'a' 到 'z'）、数字（'0' 到 '9'）、连字符（'-'）、标点符号（'!'、'.' 和 ','）以及空格（' '）组成。每个句子可以根据空格分解成 一个或者多个 token ，这些 token 之间由一个或者多个空格 ' ' 分隔。

如果一个 token 同时满足下述条件，则认为这个 token 是一个有效单词：

仅由小写字母、连字符和/或标点（不含数字）。
至多一个 连字符 '-' 。如果存在，连字符两侧应当都存在小写字母（"a-b" 是一个有效单词，但 "-ab" 和 "ab-" 不是有效单词）。
至多一个 标点符号。如果存在，标点符号应当位于 token 的 末尾 。
这里给出几个有效单词的例子："a-b."、"afad"、"ba-c"、"a!" 和 "!" 。

给你一个字符串 sentence ，请你找出并返回 sentence 中 有效单词的数目 。

```python
# 爆肝逻辑
class Solution:
    def countValidWords(self, sentence: str) -> int:
        count = 0
        temp = sentence.split(" ")
        temp2 = []
        for e in temp:
            if e != "":
                temp2.append(e)
        for e in temp2:
            state = True
            for ch in e:
                if ch.isdigit():
                    state = False
                    break
            co = 0
            stack = []
            for index,ch in enumerate(e):
                if ch == "-":
                    co += 1
                    stack.append(index)
            if co >= 2:
                state = False 
            if len(stack) > 1:
                state = False 
            
            if len(stack)== 1:
                index = stack[0]
                if index-1 >= 0 and index + 1 < len(e):
                    if e[index-1].isalpha() and e[index-1].lower() == e[index-1]:
                        pass
                    else:
                        state = False 
                    if e[index+1].isalpha() and e[index+1].lower() == e[index+1]:
                        pass
                    else:
                        state = False 
                else:
                    state = False 
            
            co2 = 0
            for index,ch in enumerate(e):
                if ch in "!.,":
                    co2 += 1
            if co2 >= 2:
                state = False
            if co2 == 1:
                if e[-1] not in "!.,":
                    state = False 
            if state:
                count += 1
        return count 
                    
```

# 2048. 下一个更大的数值平衡数

如果整数  x 满足：对于每个数位 d ，这个数位 恰好 在 x 中出现 d 次。那么整数 x 就是一个 数值平衡数 。

给你一个整数 n ，请你返回 严格大于 n 的 最小数值平衡数 。

```python
class Solution:
    def nextBeautifulNumber(self, n: int) -> int:
        # 严格大于n
        # 10^6有7位，这一题最多只支持到【1～6】,其余的返回1224444
        if n >= 666666:
            return 1224444
        # 回溯生成之后用二分或者不用二分应该都可以
        # 1对应1； 2 对应22， 3 对应333和221，212，122； 4 对应【1+3】和4； 5对应【5，1+4，2+3】； 6对应【6,1+5,2+4，1+2+3】
        # 打表过程在pycharm
        lst = [1,22,333,122,212,221,4444,1333,3133,3313,3331,55555,14444,41444,44144,44414,44441,23233, 32323, 23332, 33223, 32233, 33322, 32332, 33232, 23323, 22333,666666,155555, 551555, 515555, 555155, 555515, 555551,444224, 442244, 444422, 244424, 424424, 442442, 242444, 422444, 424442, 444242, 244244, 424244, 442424, 244442, 224444, 312323, 313223, 132233, 322313, 232331, 312332, 323213, 233231, 313232, 221333, 332312, 231323, 322331, 223133, 333212, 323231, 332321, 233123, 231332, 333221, 312233, 321323, 233132, 323123, 321332, 332213, 331322, 123323, 323132, 231233, 333122, 123332, 232133, 332231, 133322, 213323, 223313, 321233, 213332, 322133, 323321, 331223, 332123, 122333, 233312, 123233, 331232, 132323, 223331, 332132, 133223, 233321, 313322, 132332, 212333, 133232, 213233, 323312, 232313, 233213]
        lst.sort()
        ans = -1
        for e in lst:
            if e > n:
                ans = e
                break 
        return ans
            
```

# 2049. 统计最高分的节点数目

给你一棵根节点为 0 的 二叉树 ，它总共有 n 个节点，节点编号为 0 到 n - 1 。同时给你一个下标从 0 开始的整数数组 parents 表示这棵树，其中 parents[i] 是节点 i 的父节点。由于节点 0 是根，所以 parents[0] == -1 。

一个子树的 大小 为这个子树内节点的数目。每个节点都有一个与之关联的 分数 。求出某个节点分数的方法是，将这个节点和与它相连的边全部 删除 ，剩余部分是若干个 非空 子树，这个节点的 分数 为所有这些子树 大小的乘积 。

请你返回有 最高得分 节点的 数目 。

```python

        
class Solution:
    def countHighestScoreNodes(self, parents: List[int]) -> int:
        graph = collections.defaultdict(list) # 由父亲找到孩子
        parentDict = collections.defaultdict(int) # 由孩子找到父亲
        n = len(parents)
        
        for i,v in enumerate(parents):
            parentDict[i] = v
        for i,v in enumerate(parents):
            graph[v].append(i)
            
        sizeDict = collections.defaultdict(int)
        # 后续遍历得到size
        
        def postOrder(node):
            if graph[node] == []:
                sizeDict[node] = 1
                return 1
            leftPart = postOrder(graph[node][0])
            if len(graph[node]) == 2:
                rightPart = postOrder(graph[node][1])
            else:
                rightPart = 0
            sizeDict[node] = 1 + leftPart + rightPart
            return sizeDict[node]
        
        postOrder(-1)

        maxVal = 0
        for i in range(n):
            v1 = n - sizeDict[i] # 总树减去本身大小
            if v1 == 0:
                v1 = 1
            if len(graph[i]) == 0:
                c1,c2 = None,None
            if len(graph[i]) == 1:
                c1 = graph[i][0]
                c2 = None
            if len(graph[i]) == 2:
                c1 = graph[i][0]
                c2 = graph[i][1]
            v2 = sizeDict[c1] if c1 != None else 1
            v3 = sizeDict[c2] if c2 != None else 1
            theVal = v1 * v2 * v3
            maxVal = max(maxVal,theVal)
        
        # print("max",maxVal)
        # print("size",sizeDict)
        # print("parent",parentDict)
        
        count = 0
        for i in range(n):
            v1 = n - sizeDict[i] # 树大小减去本身大小
            if v1 == 0:
                v1 = 1 # 修正
            if len(graph[i]) == 0:
                c1,c2 = None,None
            if len(graph[i]) == 1:
                c1 = graph[i][0]
                c2 = None
            if len(graph[i]) == 2:
                c1 = graph[i][0]
                c2 = graph[i][1]
            v2 = sizeDict[c1] if c1 != None else 1
            v3 = sizeDict[c2] if c2 != None else 1
            theVal = v1 * v2 * v3
            if theVal == maxVal:
                # print(i)
                count += 1
        return count 
```

```python
class Solution:
    def countHighestScoreNodes(self, parents: List[int]) -> int:
        # 后续遍历,改进版
        graph = collections.defaultdict(list)
        parentDict = collections.defaultdict(int)
        sizeDict = collections.defaultdict(int)
        n = len(parents)
        for i,v in enumerate(parents):
            graph[v].append(i)
            parentDict[i] = v
        
        def postOrder(node):
            if graph[node] == []:
                sizeDict[node] = 1
                return 1
            leftPart = postOrder(graph[node][0])
            rightPart = postOrder(graph[node][1]) if len(graph[node]) == 2 else 0
            sizeDict[node] = 1 + leftPart + rightPart
            return sizeDict[node]
        
        postOrder(-1)

        maxVal = 0
        count = 0
        for i in range(n):
            v1 = n - sizeDict[i]
            if v1 == 0:
                v1 = 1
            if len(graph[i]) == 0:
                v2,v3 = 1,1
            elif len(graph[i]) == 1:
                v2 = sizeDict[graph[i][0]]
                v3 = 1
            elif len(graph[i]) == 2:
                v2 = sizeDict[graph[i][0]]
                v3 = sizeDict[graph[i][1]]
            theVal = v1 * v2 * v3 
            if theVal > maxVal:
                maxVal = theVal
                count = 1
            elif theVal == maxVal:
                count += 1
        return count 
```

# [2073. 买票需要的时间](https://leetcode-cn.com/problems/time-needed-to-buy-tickets/)

有 n 个人前来排队买票，其中第 0 人站在队伍 最前方 ，第 (n - 1) 人站在队伍 最后方 。

给你一个下标从 0 开始的整数数组 tickets ，数组长度为 n ，其中第 i 人想要购买的票数为 tickets[i] 。

每个人买票都需要用掉 恰好 1 秒 。一个人 一次只能买一张票 ，如果需要购买更多票，他必须走到  队尾 重新排队（瞬间 发生，不计时间）。如果一个人没有剩下需要买的票，那他将会 离开 队伍。

返回位于位置 k（下标从 0 开始）的人完成买票需要的时间（以秒为单位）。

```python
class Solution:
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        times = 0
        p = 0
        n = len(tickets)
        while tickets[k] != 0:
            if tickets[p] > 0:
                tickets[p] -= 1
                times += 1
            p = (p+1)%len(tickets)
        return times
```

# [2094. 找出 3 位偶数](https://leetcode-cn.com/problems/finding-3-digit-even-numbers/)

给你一个整数数组 digits ，其中每个元素是一个数字（0 - 9）。数组中可能存在重复元素。

你需要找出 所有 满足下述条件且 互不相同 的整数：

该整数由 digits 中的三个元素按 任意 顺序 依次连接 组成。
该整数不含 前导零
该整数是一个 偶数
例如，给定的 digits 是 [1, 2, 3] ，整数 132 和 312 满足上面列出的全部条件。

将找出的所有互不相同的整数按 递增顺序 排列，并以数组形式返回。

```python
class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        ans = []
        digits.sort()
            
        n = len(digits)
        for i in range(n):
            if digits[i] == 0: continue
            for j in range(n):
                for k in range(n):
                    if i == j or i == k or j == k: continue
                    if digits[k]%2 != 0: continue
                    temp = 100*(digits[i])+10*(digits[j])+(digits[k])
                    ans.append(temp)
        ans = list(set(ans))
        ans.sort()
        return ans
```

# [2095. 删除链表的中间节点](https://leetcode-cn.com/problems/delete-the-middle-node-of-a-linked-list/)

给你一个链表的头节点 head 。删除 链表的 中间节点 ，并返回修改后的链表的头节点 head 。

长度为 n 链表的中间节点是从头数起第 ⌊n / 2⌋ 个节点（下标从 0 开始），其中 ⌊x⌋ 表示小于或等于 x 的最大整数。

```python
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None or head.next == None:
            return None
        dummy2 = ListNode(-1)
        dummy = ListNode(-1)
        dummy.next = dummy2
        dummy2.next = head
        fast = dummy
        slow = dummy
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next 
            
        if slow != None and slow.next != None:
            slow.next = slow.next.next 
        return head
```

# [5186. 区间内查询数字的频率](https://leetcode-cn.com/problems/range-frequency-queries/)

请你设计一个数据结构，它能求出给定子数组内一个给定值的 频率 。

子数组中一个值的 频率 指的是这个子数组中这个值的出现次数。

请你实现 RangeFreqQuery 类：

RangeFreqQuery(int[] arr) 用下标从 0 开始的整数数组 arr 构造一个类的实例。
int query(int left, int right, int value) 返回子数组 arr[left...right] 中 value 的 频率 。
一个 子数组 指的是数组中一段连续的元素。arr[left...right] 指的是 nums 中包含下标 left 和 right 在内 的中间一段连续元素。

```python
class RangeFreqQuery:

    def __init__(self, arr: List[int]):
        self.cnt = collections.defaultdict(list)
        # 记录每个数字出现的索引
        for i,num in enumerate(arr):
            self.cnt[num].append(i)

    def query(self, left: int, right: int, value: int) -> int:
        # 查询这个数字出现的
        k1 = bisect.bisect_left(self.cnt[value],left) # 查询小于等于它的第一个数
        k2 = bisect.bisect_right(self.cnt[value],right) # 查询大于它的第一个数
        return k2 - k1
```

# [5201. 给植物浇水](https://leetcode-cn.com/problems/watering-plants/)

你打算用一个水罐给花园里的 n 株植物浇水。植物排成一行，从左到右进行标记，编号从 0 到 n - 1 。其中，第 i 株植物的位置是 x = i 。x = -1 处有一条河，你可以在那里重新灌满你的水罐。

每一株植物都需要浇特定量的水。你将会按下面描述的方式完成浇水：

按从左到右的顺序给植物浇水。
在给当前植物浇完水之后，如果你没有足够的水 完全 浇灌下一株植物，那么你就需要返回河边重新装满水罐。
你 不能 提前重新灌满水罐。
最初，你在河边（也就是，x = -1），在 x 轴上每移动 一个单位 都需要 一步 。

给你一个下标从 0 开始的整数数组 plants ，数组由 n 个整数组成。其中，plants[i] 为第 i 株植物需要的水量。另有一个整数 capacity 表示水罐的容量，返回浇灌所有植物需要的 步数 。

```python
class Solution:
    def wateringPlants(self, plants: List[int], capacity: int) -> int:
        n = len(plants)
        ans = 0
        remain = capacity
        p = 0
        while p < n:
            if plants[p] <= remain:
                remain -= plants[p]
                ans += 1
                p += 1
            elif plants[p] > remain:
                remain = capacity
                remain -= plants[p]
                ans += 2*p+1
                p += 1
        return ans
```



# [5885. 使每位学生都有座位的最少移动次数](https://leetcode-cn.com/problems/minimum-number-of-moves-to-seat-everyone/)

一个房间里有 n 个座位和 n 名学生，房间用一个数轴表示。给你一个长度为 n 的数组 seats ，其中 seats[i] 是第 i 个座位的位置。同时给你一个长度为 n 的数组 students ，其中 students[j] 是第 j 位学生的位置。

你可以执行以下操作任意次：

增加或者减少第 i 位学生的位置，每次变化量为 1 （也就是将第 i 位学生从位置 x 移动到 x + 1 或者 x - 1）
请你返回使所有学生都有座位坐的 最少移动次数 ，并确保没有两位学生的座位相同。

请注意，初始时有可能有多个座位或者多位学生在 同一 位置。

```python
class Solution:
    def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
        n = len(seats)
        seats.sort()
        students.sort()
        ans = 0
        for i in range(n):
            ans += abs(seats[i]-students[i])
        return ans 
```

# [5886. 如果相邻两个颜色均相同则删除当前颜色](https://leetcode-cn.com/problems/remove-colored-pieces-if-both-neighbors-are-the-same-color/)

总共有 n 个颜色片段排成一列，每个颜色片段要么是 'A' 要么是 'B' 。给你一个长度为 n 的字符串 colors ，其中 colors[i] 表示第 i 个颜色片段的颜色。

Alice 和 Bob 在玩一个游戏，他们 轮流 从这个字符串中删除颜色。Alice 先手 。

如果一个颜色片段为 'A' 且 相邻两个颜色 都是颜色 'A' ，那么 Alice 可以删除该颜色片段。Alice 不可以 删除任何颜色 'B' 片段。
如果一个颜色片段为 'B' 且 相邻两个颜色 都是颜色 'B' ，那么 Bob 可以删除该颜色片段。Bob 不可以 删除任何颜色 'A' 片段。
Alice 和 Bob 不能 从字符串两端删除颜色片段。
如果其中一人无法继续操作，则该玩家 输 掉游戏且另一玩家 获胜 。
假设 Alice 和 Bob 都采用最优策略，如果 Alice 获胜，请返回 true，否则 Bob 获胜，返回 false。

```python
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        # 计数到目前为止的A,B的数量
        if len(colors) <= 2:
            return False 
        pivot = colors[0]
        times = 0
        memo = collections.defaultdict(list)
        p = 0
        while p < len(colors):
            if colors[p] == pivot:
                times += 1
            elif colors[p] != pivot:
                if times >= 3:
                    memo[pivot].append(times-2)
                pivot = colors[p]
                times = 1
            p += 1
        
        # 擦屁股
        if times >= 3:
            memo[pivot].append(times-2)
        # 数其中大于等于3的数量，看谁能删得多
        a = 0
        b = 0
        for key in memo:
            if key == "A":
                a = sum(memo[key])
            elif key == "B":
                b = sum(memo[key])
        
        return a > b # 需要严格大于      
```

# [5888. 网络空闲的时刻](https://leetcode-cn.com/problems/the-time-when-the-network-becomes-idle/)

给你一个有 n 个服务器的计算机网络，服务器编号为 0 到 n - 1 。同时给你一个二维整数数组 edges ，其中 edges[i] = [ui, vi] 表示服务器 ui 和 vi 之间有一条信息线路，在 一秒 内它们之间可以传输 任意 数目的信息。再给你一个长度为 n 且下标从 0 开始的整数数组 patience 。

题目保证所有服务器都是 相通 的，也就是说一个信息从任意服务器出发，都可以通过这些信息线路直接或间接地到达任何其他服务器。

编号为 0 的服务器是 主 服务器，其他服务器为 数据 服务器。每个数据服务器都要向主服务器发送信息，并等待回复。信息在服务器之间按 最优 线路传输，也就是说每个信息都会以 最少时间 到达主服务器。主服务器会处理 所有 新到达的信息并 立即 按照每条信息来时的路线 反方向 发送回复信息。

在 0 秒的开始，所有数据服务器都会发送各自需要处理的信息。从第 1 秒开始，每 一秒最 开始 时，每个数据服务器都会检查它是否收到了主服务器的回复信息（包括新发出信息的回复信息）：

如果还没收到任何回复信息，那么该服务器会周期性 重发 信息。数据服务器 i 每 patience[i] 秒都会重发一条信息，也就是说，数据服务器 i 在上一次发送信息给主服务器后的 patience[i] 秒 后 会重发一条信息给主服务器。
否则，该数据服务器 不会重发 信息。
当没有任何信息在线路上传输或者到达某服务器时，该计算机网络变为 空闲 状态。

请返回计算机网络变为 空闲 状态的 最早秒数 。

```python
class Solution:
    def networkBecomesIdle(self, edges: List[List[int]], patience: List[int]) -> int:
        graph = collections.defaultdict(list)
        for a,b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        n = len(patience)
        
        length = [0xffffffff for i in range(n)]

        visited = [False for i in range(n)]
        visited[0] = True
        steps = 0
        # bfs
        queue = [0]
        while len(queue) != 0:
            new_queue = []
            for e in queue:
                length[e] = steps
                for neigh in graph[e]:
                    if visited[neigh] == False:
                        visited[neigh] = True 
                        new_queue.append(neigh)
            steps += 1
            queue = new_queue
        
        print(length)
        
        ans = -1
        # 需要看收到信息的时候发送了几条信息
        for i in range(1,n):
            sendnum = ceil(2*length[i]/patience[i])
            # 收到这一条信息的时候最后这一条信息走了多远了
            p1 = (sendnum-1)*patience[i] # 还有这么久到
            ans = max(ans,2*length[i]+p1)
        return ans+1        
```

# [5894. 至少在两个数组中出现的值](https://leetcode-cn.com/problems/two-out-of-three/)

给你三个整数数组 nums1、nums2 和 nums3 ，请你构造并返回一个 不同 数组，且由 至少 在 两个 数组中出现的所有值组成。数组中的元素可以按 任意 顺序排列。

```python
class Solution:
    def twoOutOfThree(self, nums1: List[int], nums2: List[int], nums3: List[int]) -> List[int]:
        ans = []
        
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                if nums1[i] == nums2[j]:
                    ans.append(nums1[i])
        for i in range(len(nums1)):
            for j in range(len(nums3)):
                if nums1[i] == nums3[j] and nums1[i] not in ans:
                    ans.append(nums1[i])
        for i in range(len(nums2)):
            for j in range(len(nums3)):
                if nums2[i] == nums3[j] and nums2[i] not in ans:
                    ans.append(nums2[i])
        ans = set(ans)
        final = []
        for e in ans:
            final.append(e)
        return final
                    
```

# [5895. 获取单值网格的最小操作数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-uni-value-grid/)

给你一个大小为 m x n 的二维整数网格 grid 和一个整数 x 。每一次操作，你可以对 grid 中的任一元素 加 x 或 减 x 。

单值网格 是全部元素都相等的网格。

返回使网格化为单值网格所需的 最小 操作数。如果不能，返回 -1 。

```python
class Solution:
    def minOperations(self, grid: List[List[int]], x: int) -> int:
        m = len(grid)
        n = len(grid[0])
        lst = []
        for i in range(m):
            for j in range(n):
                lst.append(grid[i][j])
        lst.sort()
        if len(lst) % 2 == 1:
            ops = 0
            mid = (len(lst))//2
            pivot = lst[mid]
            for n in lst:
                if (n-pivot) % x == 0:
                    ops += abs(n-pivot) // x
                else:
                    return -1
            return ops
        
        if len(lst)%2 == 0:
            ops1 = 0
            mid = (len(lst)-1)//2
            pivot = lst[mid]
            state1 = True
            for n in lst:
                if (n-pivot) % x == 0:
                    ops1 += abs(n-pivot) // x
                else:
                    state1 = False
                    break 
            ops2 = 0
            pivot = lst[mid+1]
            state2 = True
            for n in lst:
                if (n-pivot) % x == 0:
                    ops2 += abs(n-pivot) // x
                else:
                    state2 = False
                    break 
            if state1 and state2:
                return min(ops1,ops2)
            if state1:
                return ops1
            if state2:
                return ops2
            return -1
            
```

# [5896. 股票价格波动](https://leetcode-cn.com/problems/stock-price-fluctuation/)

给你一支股票价格的数据流。数据流中每一条记录包含一个 时间戳 和该时间点股票对应的 价格 。

不巧的是，由于股票市场内在的波动性，股票价格记录可能不是按时间顺序到来的。某些情况下，有的记录可能是错的。如果两个有相同时间戳的记录出现在数据流中，前一条记录视为错误记录，后出现的记录 更正 前一条错误的记录。

请你设计一个算法，实现：

更新 股票在某一时间戳的股票价格，如果有之前同一时间戳的价格，这一操作将 更正 之前的错误价格。
找到当前记录里 最新股票价格 。最新股票价格 定义为时间戳最晚的股票价格。
找到当前记录里股票的 最高价格 。
找到当前记录里股票的 最低价格 。
请你实现 StockPrice 类：

StockPrice() 初始化对象，当前无股票价格记录。
void update(int timestamp, int price) 在时间点 timestamp 更新股票价格为 price 。
int current() 返回股票 最新价格 。
int maximum() 返回股票 最高价格 。
int minimum() 返回股票 最低价格 。

```python
class StockPrice:

    def __init__(self):
        # 一个排序容器
        import sortedcontainers
        self.sl = sortedcontainers.SortedList()
        self.maxlist = sortedcontainers.SortedList()
        # 哨兵
        self.sl.add([-5,-5])
        self.sl.add([0xffffffff,0xffffffff])
        self.maxlist.add(-5)
        self.maxlist.add(0xffffffff)


    def update(self, timestamp: int, price: int) -> None:
        index = bisect.bisect_left(self.sl,[timestamp])
        if self.sl[index][0] == timestamp: # 矫正
            old = self.sl[index][1]
            self.sl[index][1] = price
            ind = bisect.bisect_left(self.maxlist,old)
            self.maxlist.pop(ind)
            indx = bisect.bisect_left(self.maxlist,price)
            self.maxlist.add(price)
        else: # 更新
            self.sl.add([timestamp,price])
            self.maxlist.add(price)


    def current(self) -> int:
        return self.sl[-2][-1]


    def maximum(self) -> int:
        return self.maxlist[-2]


    def minimum(self) -> int:
        return self.maxlist[1]

```

# [5898. 数组中第 K 个独一无二的字符串](https://leetcode-cn.com/problems/kth-distinct-string-in-an-array/)

独一无二的字符串 指的是在一个数组中只出现过 一次 的字符串。

给你一个字符串数组 arr 和一个整数 k ，请你返回 arr 中第 k 个 独一无二的字符串 。如果 少于 k 个独一无二的字符串，那么返回 空字符串 "" 。

注意，按照字符串在原数组中的 顺序 找到第 k 个独一无二字符串。

```python
class Solution:
    def kthDistinct(self, arr: List[str], k: int) -> str:
        tempDict = collections.defaultdict(int)
        for t in arr:
            tempDict[t] += 1
        lst = []
        for key in tempDict:
            if tempDict[key] == 1:
                lst.append(key)
        if len(lst) < k:
            return ""
        return lst[k-1]
```

# [5899. 两个最好的不重叠活动](https://leetcode-cn.com/problems/two-best-non-overlapping-events/)

给你一个下标从 0 开始的二维整数数组 events ，其中 events[i] = [startTimei, endTimei, valuei] 。第 i 个活动开始于 startTimei ，结束于 endTimei ，如果你参加这个活动，那么你可以得到价值 valuei 。你 最多 可以参加 两个时间不重叠 活动，使得它们的价值之和 最大 。

请你返回价值之和的 最大值 。

注意，活动的开始时间和结束时间是 包括 在活动时间内的，也就是说，你不能参加两个活动且它们之一的开始时间等于另一个活动的结束时间。更具体的，如果你参加一个活动，且结束时间为 t ，那么下一个活动必须在 t + 1 或之后的时间开始。

```python
class Solution:
    def maxTwoEvents(self, events: List[List[int]]) -> int:
        events.sort(key = lambda x:(x[0],x[1],-x[2]))
        # 可能只有一个活动
        ans = 0
        n = len(events)
        # 枚举每个活动bisect找到合法
        # 倒序最大值
        maxVal = [0 for i in range(n)]
        tempMax = events[-1][2]
        for i in range(n-1,-1,-1):
            if events[i][2] > tempMax:
                tempMax = events[i][2]
            maxVal[i] = tempMax
            
        for a,b,c in events:
            temp = c
            index = bisect.bisect_left(events,[b+1])
            if index < n:
                temp += maxVal[index]
            ans = max(ans,temp)
        return ans
```

# [5900. 蜡烛之间的盘子](https://leetcode-cn.com/problems/plates-between-candles/)

给你一个长桌子，桌子上盘子和蜡烛排成一列。给你一个下标从 0 开始的字符串 s ，它只包含字符 '*' 和 '|' ，其中 '*' 表示一个 盘子 ，'|' 表示一支 蜡烛 。

同时给你一个下标从 0 开始的二维整数数组 queries ，其中 queries[i] = [lefti, righti] 表示 子字符串 s[lefti...righti] （包含左右端点的字符）。对于每个查询，你需要找到 子字符串中 在 两支蜡烛之间 的盘子的 数目 。如果一个盘子在 子字符串中 左边和右边 都 至少有一支蜡烛，那么这个盘子满足在 两支蜡烛之间 。

比方说，s = "||**||**|*" ，查询 [3, 8] ，表示的是子字符串 "*||**|" 。子字符串中在两支蜡烛之间的盘子数目为 2 ，子字符串中右边两个盘子在它们左边和右边 都 至少有一支蜡烛。
请你返回一个整数数组 answer ，其中 answer[i] 是第 i 个查询的答案。

```python
class Solution:
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        # 前缀和
        # 如果是棍子，则加入棍子list
        gunList = []
        n = len(s)
        pre = 0
        preList = [0 for i in range(n)]
        for i in range(n):
            ch = s[i]
            if s[i] == "*":
                pre += 1
                preList[i] = pre
            elif s[i] == "|":
                preList[i] = pre
                gunList.append(i)
        
        ans = []
        # 找到比queries:start大于等于的第一个棍的index，end小于等于的最后棍的index
        # print(gunList)
        # print(preList)
        
        for i,v in enumerate(queries):
            a,b = v
            startIndex = bisect.bisect_left(gunList,a)
            endIndex = bisect.bisect_right(gunList,b)-1

            # print("s = ",startIndex,",,,","e = ",endIndex)
            # print("preList[startIndex]",preList[startIndex],"preList[endIndex]",preList[endIndex])
            # print("gunList[endIndex]",gunList[endIndex],"gunList[startIndex",gunList[startIndex])
            if startIndex <= endIndex:
                t = preList[gunList[endIndex]] - preList[gunList[startIndex]]
            else:
                t = 0
            ans.append(t)
        
            
        return (ans)
                
                    
            
```

# [5902. 检查句子中的数字是否递增](https://leetcode-cn.com/problems/check-if-numbers-are-ascending-in-a-sentence/)

句子是由若干 token 组成的一个列表，token 间用 单个 空格分隔，句子没有前导或尾随空格。每个 token 要么是一个由数字 0-9 组成的不含前导零的 正整数 ，要么是一个由小写英文字母组成的 单词 。

示例，"a puppy has 2 eyes 4 legs" 是一个由 7 个 token 组成的句子："2" 和 "4" 是数字，其他像 "puppy" 这样的 tokens 属于单词。
给你一个表示句子的字符串 s ，你需要检查 s 中的 全部 数字是否从左到右严格递增（即，除了最后一个数字，s 中的 每个 数字都严格小于它 右侧 的数字）。

如果满足题目要求，返回 true ，否则，返回 false 。

```python
class Solution:
    def areNumbersAscending(self, s: str) -> bool:
        lst = s.split(" ")
        numlist = []
        for e in lst:
            if e.isdigit():
                numlist.append(int(e))
        # 严格递增
        if len(numlist) == 1:
            return True
        p = 1
        while p < len(numlist):
            gap = numlist[p]-numlist[p-1]
            if gap <= 0:
                return False
            p += 1
        return True
```

# [5903. 简易银行系统](https://leetcode-cn.com/problems/simple-bank-system/)

你的任务是为一个很受欢迎的银行设计一款程序，以自动化执行所有传入的交易（转账，存款和取款）。银行共有 n 个账户，编号从 1 到 n 。每个账号的初始余额存储在一个下标从 0 开始的整数数组 balance 中，其中第 (i + 1) 个账户的初始余额是 balance[i] 。

请你执行所有 有效的 交易。如果满足下面全部条件，则交易 有效 ：

指定的账户数量在 1 和 n 之间，且
取款或者转账需要的钱的总数 小于或者等于 账户余额。
实现 Bank 类：

Bank(long[] balance) 使用下标从 0 开始的整数数组 balance 初始化该对象。
boolean transfer(int account1, int account2, long money) 从编号为 account1 的账户向编号为 account2 的账户转帐 money 美元。如果交易成功，返回 true ，否则，返回 false 。
boolean deposit(int account, long money) 向编号为 account 的账户存款 money 美元。如果交易成功，返回 true ；否则，返回 false 。
boolean withdraw(int account, long money) 从编号为 account 的账户取款 money 美元。如果交易成功，返回 true ；否则，返回 false 。

```python
class Bank:

    def __init__(self, balance: List[int]):
        self.n = len(balance)
        self.balance = balance

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        index1 = account1 - 1
        index2 = account2 - 1
        if index1 >= self.n or index2 >= self.n:
            return False
        if self.balance[index1] >= money:
            self.balance[index1] -= money
            self.balance[index2] += money
            return True
        return False

    def deposit(self, account: int, money: int) -> bool:
        index = account - 1
        if index >= self.n:
            return False
        self.balance[index] += money
        return True



    def withdraw(self, account: int, money: int) -> bool:
        index = account - 1
        if index >= self.n:
            return False 
        if self.balance[index] >= money:
            self.balance[index] -= money
            return True
        return False
```

# [5904. 统计按位或能得到最大值的子集数目](https://leetcode-cn.com/problems/count-number-of-maximum-bitwise-or-subsets/)

给你一个整数数组 nums ，请你找出 nums 子集 按位或 可能得到的 最大值 ，并返回按位或能得到最大值的 不同非空子集的数目 。

如果数组 a 可以由数组 b 删除一些元素（或不删除）得到，则认为数组 a 是数组 b 的一个 子集 。如果选中的元素下标位置不一样，则认为两个子集 不同 。

对数组 a 执行 按位或 ，结果等于 a[0] OR a[1] OR ... OR a[a.length - 1]（下标从 0 开始）。

```python
class Solution:
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        # 这个数据量直接回溯

        ans = [] # 存储最终结果
        stack = [] # 存取临时结果
        n = len(nums)
        def backtracking(nums,stack,startindex,maxlenth): # 选择列表，路径，开始为止，收集结果的条件
            if len(stack) == maxlenth:
                ans.append(stack[:]) # 传值而不是传引用
                return 
            p = startindex 
            while p < len(nums):
                stack.append(nums[p])
                backtracking(nums,stack,p+1,maxlenth) # p+1表示只在这个之后搜
                stack.pop()
                p += 1
        for i in range(n+1): # 
            backtracking(nums,stack,0,i)
            
        maxVal = 1
        count = 0
        for pair in ans:
            temp = 0
            for e in pair:
                temp |= e
            maxVal = max(maxVal,temp)
        
        for pair in ans:
            temp = 0
            for e in pair:
                temp |= e
            if temp == maxVal:
                count += 1
        return count
```

# [5910. 检查两个字符串是否几乎相等](https://leetcode-cn.com/problems/check-whether-two-strings-are-almost-equivalent/)

如果两个字符串 word1 和 word2 中从 'a' 到 'z' 每一个字母出现频率之差都 不超过 3 ，那么我们称这两个字符串 word1 和 word2 几乎相等 。

给你两个长度都为 n 的字符串 word1 和 word2 ，如果 word1 和 word2 几乎相等 ，请你返回 true ，否则返回 false 。

一个字母 x 的出现 频率 指的是它在字符串中出现的次数。

```python
class Solution:
    def checkAlmostEquivalent(self, word1: str, word2: str) -> bool:
        ct1 = [0 for i in range(26)]
        ct2 = [0 for i in range(26)]
        for ch in word1:
            index = ord(ch)-ord('a')
            ct1[index] += 1
        for ch in word2:
            index = ord(ch)-ord('a')
            ct2[index] += 1
        
        for i in range(26):
            if abs(ct1[i]-ct2[i]) >= 4:
                return False 
        return True
```

# [5911. 模拟行走机器人 II](https://leetcode-cn.com/problems/walking-robot-simulation-ii/)

给你一个在 XY 平面上的 width x height 的网格图，左下角 的格子为 (0, 0) ，右上角 的格子为 (width - 1, height - 1) 。网格图中相邻格子为四个基本方向之一（"North"，"East"，"South" 和 "West"）。一个机器人 初始 在格子 (0, 0) ，方向为 "East" 。

机器人可以根据指令移动指定的 步数 。每一步，它可以执行以下操作。

沿着当前方向尝试 往前一步 。
如果机器人下一步将到达的格子 超出了边界 ，机器人会 逆时针 转 90 度，然后再尝试往前一步。
如果机器人完成了指令要求的移动步数，它将停止移动并等待下一个指令。

请你实现 Robot 类：

Robot(int width, int height) 初始化一个 width x height 的网格图，机器人初始在 (0, 0) ，方向朝 "East" 。
void move(int num) 给机器人下达前进 num 步的指令。
int[] getPos() 返回机器人当前所处的格子位置，用一个长度为 2 的数组 [x, y] 表示。
String getDir() 返回当前机器人的朝向，为 "North" ，"East" ，"South" 或者 "West" 。

```python
class Robot:

    def __init__(self, width: int, height: int):
        # 容易发现，只会走外圈,判断区间
        # 作图找映射关系，取模算steps
        # 是(0,0)这个格子，在一开始没有移动过的时候，它的朝向是East。但之后如果停在(0,0)，朝向一定是South。因此需要特判一下当前是否移动过。

        self.length = 2*(width+height)-4
        self.now = 0
        self.getReady = False
        self.w = width
        self.h = height


    def move(self, num: int) -> None:
        self.getReady = True 
        self.now = (self.now+num)%self.length


    def getPos(self) -> List[int]:
        w = self.w
        h = self.h
        if 0<=self.now<=w-1:
            return [self.now,0]
        elif w<=self.now<=w+h-2:
            return [w-1,self.now-w+1]
        elif w+h-1<=self.now<=2*w+h-3:
            t = self.now - (w+h-2)
            a = w-1 - t
            return [a,h-1]
        else:
            t = self.now - (2*w+h-3)
            a = h-1 - t           
            return [0,a]

    def getDir(self) -> str:
        if self.getReady == False:
            return "East"
        w = self.w
        h = self.h
        if 1<=self.now<=w-1:
            return "East"
        elif w<=self.now<=w+h-2:
            return "North"
        elif w+h-1<=self.now<=2*w+h-3:
            return "West"
        else:
            return "South"
```

# [5912. 每一个查询的最大美丽值](https://leetcode-cn.com/problems/most-beautiful-item-for-each-query/)

给你一个二维整数数组 items ，其中 items[i] = [pricei, beautyi] 分别表示每一个物品的 价格 和 美丽值 。

同时给你一个下标从 0 开始的整数数组 queries 。对于每个查询 queries[j] ，你想求出价格小于等于 queries[j] 的物品中，最大的美丽值 是多少。如果不存在符合条件的物品，那么查询的结果为 0 。

请你返回一个长度与 queries 相同的数组 answer，其中 answer[j]是第 j 个查询的答案。

```python
class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        # 价格排序
        items.sort(key = lambda x:(x[0],x[1])) # 价格，美丽
        
        theD = dict()
        for i in items:
            theD[i[0]] = i[1]
        
        theList = []
        for key in theD:
            theList.append([key,theD[key]])
        
        tMax = theList[0][1]
        for i in range(len(theList)):
            if tMax < theList[i][1]:
                tMax = theList[i][1]
            theList[i][1] = tMax 
        
        ans = []
        for q in queries:
            index = bisect.bisect_right(theList,[q+1])-1 # 这里left和right都可以，因为没有重复值
            #             index = bisect.bisect_left(theList,[q+1])-1
            if index < 0:
                ans.append(0)
            else:
                ans.append(theList[index][1])
        return ans
```



# [5914. 值相等的最小索引](https://leetcode-cn.com/problems/smallest-index-with-equal-value/)

给你一个下标从 0 开始的整数数组 nums ，返回 nums 中满足 i mod 10 == nums[i] 的最小下标 i ；如果不存在这样的下标，返回 -1 。

x mod y 表示 x 除以 y 的 余数 。

```python
class Solution:
    def smallestEqual(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            if i%10 == nums[i]:
                return i
        return -1
```

# [5915. 找出临界点之间的最小和最大距离](https://leetcode-cn.com/problems/find-the-minimum-and-maximum-number-of-nodes-between-critical-points/)

链表中的 临界点 定义为一个 局部极大值点 或 局部极小值点 。

如果当前节点的值 严格大于 前一个节点和后一个节点，那么这个节点就是一个  局部极大值点 。

如果当前节点的值 严格小于 前一个节点和后一个节点，那么这个节点就是一个  局部极小值点 。

注意：节点只有在同时存在前一个节点和后一个节点的情况下，才能成为一个 局部极大值点 / 极小值点 。

给你一个链表 head ，返回一个长度为 2 的数组 [minDistance, maxDistance] ，其中 minDistance 是任意两个不同临界点之间的最小距离，maxDistance 是任意两个不同临界点之间的最大距离。如果临界点少于两个，则返回 [-1，-1] 。

```python
class Solution:
    def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
        # 转化成数组
        lst = []
        cur = head
        while cur != None:
            lst.append(cur.val)
            cur = cur.next 
        n = len(lst)
        tempList = []
        for i in range(1,n-1):
            if (lst[i-1] < lst[i] and lst[i] > lst[i+1]) or (lst[i-1] > lst[i] and lst[i] < lst[i+1]):
                tempList.append(i)
        
        if len(tempList) < 2:
            return [-1,-1]
        
        maxGap = tempList[-1]-tempList[0]
        minGap = tempList[1]-tempList[0]
        for i in range(1,len(tempList)):
            if tempList[i]-tempList[i-1] < minGap:
                minGap = tempList[i]-tempList[i-1]
        
        return [minGap,maxGap]

```

# [5916. 转化数字的最小运算数](https://leetcode-cn.com/problems/minimum-operations-to-convert-number/)

给你一个下标从 0 开始的整数数组 nums ，该数组由 互不相同 的数字组成。另给你两个整数 start 和 goal 。

整数 x 的值最开始设为 start ，你打算执行一些运算使 x 转化为 goal 。你可以对数字 x 重复执行下述运算：

如果 0 <= x <= 1000 ，那么，对于数组中的任一下标 i（0 <= i < nums.length），可以将 x 设为下述任一值：

x + nums[i]
x - nums[i]
x ^ nums[i]（按位异或 XOR）
注意，你可以按任意顺序使用每个 nums[i] 任意次。使 x 越过 0 <= x <= 1000 范围的运算同样可以生效，但该该运算执行后将不能执行其他运算。

返回将 x = start 转化为 goal 的最小操作数；如果无法完成转化，则返回 -1 。

```python
class Solution:
    def minimumOperations(self, nums: List[int], start: int, goal: int) -> int:
        # dp动态规划
        nums = set(nums)
        
        queue = [start]
        steps = 0
        visited = set()
        visited.add(start)
        
        while len(queue) != 0:
            new_queue = []
            for e in queue:
                if e == goal:
                    return steps
                if 0<=e<=1000:
                    for element in nums:
                        t1 = e + element
                        if t1 not in visited:
                            visited.add(t1)
                            new_queue.append(t1)
                        t2 = e - element
                        if t2 not in visited:
                            visited.add(t2)
                            new_queue.append(t2)
                        t3 = e ^ element
                        if t3 not in visited:
                            visited.add(t3)
                            new_queue.append(t3)
            steps += 1
            queue = new_queue
        return -1
```

# [5918. 统计字符串中的元音子字符串](https://leetcode-cn.com/problems/count-vowel-substrings-of-a-string/)

子字符串 是字符串中的一个连续（非空）的字符序列。

元音子字符串 是 仅 由元音（'a'、'e'、'i'、'o' 和 'u'）组成的一个子字符串，且必须包含 全部五种 元音。

给你一个字符串 word ，统计并返回 word 中 元音子字符串的数目 。

```python
class Solution:
    def countVowelSubstrings(self, word: str) -> int:
        # 包含五种元音的暴力
        ans = 0
        element = set("aeiou")
        p1 = 0
        n = len(word)
        for p1 in range(n): 
            for p2 in range(p1+5,n+1):
                if set(word[p1:p2]) == element:
                    ans += 1
        return ans
```

```

```

# [5919. 所有子字符串中的元音](https://leetcode-cn.com/problems/vowels-of-all-substrings/)

给你一个字符串 word ，返回 word 的所有子字符串中 元音的总数 ，元音是指 'a'、'e'、'i'、'o' 和 'u' 。

子字符串 是字符串中一个连续（非空）的字符序列。

注意：由于对 word 长度的限制比较宽松，答案可能超过有符号 32 位整数的范围。计算时需当心。

```
class Solution:
    def countVowels(self, word: str) -> int:
        # 对于每一个位置，向前被计算（index+1)次，向后被计算（n-index)次，如果它是元音，加入计算
        ans = 0
        for index,ch in enumerate(word):
            if ch in set('aeiou'):
                ans += (index+1)*(len(word)-index)
        return ans 
```

# [5920. 分配给商店的最多商品的最小值](https://leetcode-cn.com/problems/minimized-maximum-of-products-distributed-to-any-store/)

给你一个整数 n ，表示有 n 间零售商店。总共有 m 种产品，每种产品的数目用一个下标从 0 开始的整数数组 quantities 表示，其中 quantities[i] 表示第 i 种商品的数目。

你需要将 所有商品 分配到零售商店，并遵守这些规则：

一间商店 至多 只能有 一种商品 ，但一间商店拥有的商品数目可以为 任意 件。
分配后，每间商店都会被分配一定数目的商品（可能为 0 件）。用 x 表示所有商店中分配商品数目的最大值，你希望 x 越小越好。也就是说，你想 最小化 分配给任意商店商品数目的 最大值 。
请你返回最小的可能的 x 。

```python
class Solution:
    def minimizedMaximum(self, n: int, quantities: List[int]) -> int:
        # 二分查找
        left = 1
        right = max(quantities)
        
        while left <= right:
            mid = (left+right)//2
            temp = 0
            for element in quantities:
                temp += math.ceil(element/mid)
            if temp <= n:
                right = mid - 1
            elif temp > n:
                left  = mid + 1
                
        return left
```

# [5922. 统计出现过一次的公共字符串](https://leetcode-cn.com/problems/count-common-words-with-one-occurrence/)

给你两个字符串数组 `words1` 和 `words2` ，请你返回在两个字符串数组中 **都恰好出现一次** 的字符串的数目。

```python
class Solution:
    def countWords(self, words1: List[str], words2: List[str]) -> int:
        ct1 = collections.Counter(words1)
        ct2 = collections.Counter(words2)
        ans = []
        for e1 in ct1:
            if e1 in ct2:
                if ct1[e1] == 1 and ct2[e1] == 1:
                    ans.append(e1)
        return len(ans)
```

# [5923. 从房屋收集雨水需要的最少水桶数](https://leetcode-cn.com/problems/minimum-number-of-buckets-required-to-collect-rainwater-from-houses/)

给你一个下标从 0 开始的字符串 street 。street 中每个字符要么是表示房屋的 'H' ，要么是表示空位的 '.' 。

你可以在 空位 放置水桶，从相邻的房屋收集雨水。位置在 i - 1 或者 i + 1 的水桶可以收集位置为 i 处房屋的雨水。一个水桶如果相邻两个位置都有房屋，那么它可以收集 两个 房屋的雨水。

在确保 每个 房屋旁边都 至少 有一个水桶的前提下，请你返回需要的 最少 水桶数。如果无解请返回 -1 。

```python
class Solution:
    def minimumBuckets(self, street: str) -> int:
        if "HHH" in street:
            return -1
        
        lst = list(street)
        p = 0
        n = len(street)
        
        ans = 0
        while p < n:
            if street[p:p+3] == "H.H":
                ans += 1
                if p < n:
                    lst[p] = "X"
                if p+2 < n:
                    lst[p+2] = "X" 
                p += 3
            else:
                p += 1
        
        for i,e in enumerate(lst):
            if lst[i] == "H":
                if i-1 >= 0 and lst[i-1] == ".":
                    ans += 1
                    continue
                if i+1 < n and lst[i+1] == ".":
                    ans += 1
                    continue
                else:
                    return -1
        return ans
            
```

# [5924. 网格图中机器人回家的最小代价](https://leetcode-cn.com/problems/minimum-cost-homecoming-of-a-robot-in-a-grid/)

给你一个 m x n 的网格图，其中 (0, 0) 是最左上角的格子，(m - 1, n - 1) 是最右下角的格子。给你一个整数数组 startPos ，startPos = [startrow, startcol] 表示 初始 有一个 机器人 在格子 (startrow, startcol) 处。同时给你一个整数数组 homePos ，homePos = [homerow, homecol] 表示机器人的 家 在格子 (homerow, homecol) 处。

机器人需要回家。每一步它可以往四个方向移动：上，下，左，右，同时机器人不能移出边界。每一步移动都有一定代价。再给你两个下标从 0 开始的额整数数组：长度为 m 的数组 rowCosts  和长度为 n 的数组 colCosts 。

如果机器人往 上 或者往 下 移动到第 r 行 的格子，那么代价为 rowCosts[r] 。
如果机器人往 左 或者往 右 移动到第 c 列 的格子，那么代价为 colCosts[c] 。
请你返回机器人回家需要的 最小总代价 。

```python
class Solution:
    def minCost(self, startPos: List[int], homePos: List[int], rowCosts: List[int], colCosts: List[int]) -> int:
        # 这个数据量不能bfs
        tx_d = min(startPos[0],homePos[0])
        tx_u = max(startPos[0],homePos[0])
        
        ty_d = min(startPos[1],homePos[1])
        ty_u = max(startPos[1],homePos[1])
        
        # print(tx_d,tx_u,ty_d,ty_u)
        
        cost = 0
        for i in range(tx_d,tx_u+1):            
            cost += rowCosts[i]
            
        for i in range(ty_d,ty_u+1):
            cost += colCosts[i]
        
        cost -= rowCosts[startPos[0]]
        cost -= colCosts[startPos[1]]
        return cost
```

# [5927. 反转偶数长度组的节点](https://leetcode-cn.com/problems/reverse-nodes-in-even-length-groups/)

给你一个链表的头节点 head 。

链表中的节点 按顺序 划分成若干 非空 组，这些非空组的长度构成一个自然数序列（1, 2, 3, 4, ...）。一个组的 长度 就是组中分配到的节点数目。换句话说：

节点 1 分配给第一组
节点 2 和 3 分配给第二组
节点 4、5 和 6 分配给第三组，以此类推
注意，最后一组的长度可能小于或者等于 1 + 倒数第二组的长度 。

反转 每个 偶数 长度组中的节点，并返回修改后链表的头节点 head 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseEvenLengthGroups(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 两个坑，要剩余长度为偶数才反转
        lst = []
        cur = head
        while cur != None:
            lst.append(cur.val)
            cur = cur.next
        
        n = len(lst)

        p = 1 
        add = 2
        while p < n:
            # print(p)
            # print(add)
            # print(lst[p:p+add])
            e = lst[p:p+add]
            if len(e)%2 == 0: # 这里
                lst[p:p+add] = e[::-1]
            p += add
            add += 1
            p += add 
            add += 1
        
        add -= 1
        p -= add
        if p >= 0 and len(lst[p:]) % 2 == 0: # 这里
            e = lst[p:]
            lst[p:] = e[::-1]
        cur = head
        for n in lst:
            cur.val = n
            cur = cur.next 
        return head
            
```

# [5928. 解码斜向换位密码](https://leetcode-cn.com/problems/decode-the-slanted-ciphertext/)

字符串 originalText 使用 斜向换位密码 ，经由 行数固定 为 rows 的矩阵辅助，加密得到一个字符串 encodedText 。

originalText 先按从左上到右下的方式放置到矩阵中。

```python
class Solution:
    def decodeCiphertext(self, encodedText: str, rows: int) -> str:
        if len(encodedText) == 0:
            return ""
        def toMat(s,rows):
            mat = []
            line = len(s)//rows
            for i in range(0,len(s),line):
                mat.append(s[i:i+line])
            return mat
        mat = toMat(encodedText,rows)
        
        m,n = len(mat),len(mat[0])
        stack = []
        for start in range(n):
            for i in range(m):
                if start + i < n:
                    stack.append(mat[i][start+i])
        while len(stack) > 0 and stack[-1] == " ":
            stack.pop()
        ans = "".join(stack)
        return ans
            
```

# [5930. 两栋颜色不同且距离最远的房子](https://leetcode-cn.com/problems/two-furthest-houses-with-different-colors/)

街上有 n 栋房子整齐地排成一列，每栋房子都粉刷上了漂亮的颜色。给你一个下标从 0 开始且长度为 n 的整数数组 colors ，其中 colors[i] 表示第  i 栋房子的颜色。

返回 两栋 颜色 不同 房子之间的 最大 距离。

第 i 栋房子和第 j 栋房子之间的距离是 abs(i - j) ，其中 abs(x) 是 x 的绝对值。

```python
class Solution:
    def maxDistance(self, colors: List[int]) -> int:
        n = len(colors)
        maxGap = 1
        for i in range(n):
            for j in range(i+1,n):
                if colors[i] != colors[j]:
                    maxGap = max(maxGap,abs(i-j))
        return maxGap
                
```

```python
class Solution:
    def maxDistance(self, colors: List[int]) -> int:
        n = len(colors)
        left = 0
        right = n-1
        if colors[left] != colors[right]:
            return right-left
        
        p1 = 0
        p2 = n-1
        while colors[p2] == colors[0]:
            p2 -= 1
        state1 = p2-p1 

        p3 = 0
        p4 = n-1
        while colors[p3] == colors[-1]:
            p3 += 1
        state2 = p4-p3 
        return max(state1,state2)
```

# [5934. 找到和最大的长度为 K 的子序列](https://leetcode-cn.com/problems/find-subsequence-of-length-k-with-the-largest-sum/)

给你一个整数数组 nums 和一个整数 k 。你需要找到 nums 中长度为 k 的 子序列 ，且这个子序列的 和最大 。

请你返回 任意 一个长度为 k 的整数子序列。

子序列 定义为从一个数组里删除一些元素后，不改变剩下元素的顺序得到的数组。

```python
class Solution:
    def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
        cp = nums[:]
        nums.sort(reverse = True)
        ans = []
        t = nums[:k]
        
        ct = collections.Counter(t)
        for n in cp:
            if n in ct and ct[n] > 0:
                ct[n] -= 1
                ans.append(n)
        
        return ans
```

# [5935. 适合打劫银行的日子](https://leetcode-cn.com/problems/find-good-days-to-rob-the-bank/)

你和一群强盗准备打劫银行。给你一个下标从 0 开始的整数数组 security ，其中 security[i] 是第 i 天执勤警卫的数量。日子从 0 开始编号。同时给你一个整数 time 。

如果第 i 天满足以下所有条件，我们称它为一个适合打劫银行的日子：

第 i 天前和后都分别至少有 time 天。
第 i 天前连续 time 天警卫数目都是非递增的。
第 i 天后连续 time 天警卫数目都是非递减的。
更正式的，第 i 天是一个合适打劫银行的日子当且仅当：security[i - time] >= security[i - time + 1] >= ... >= security[i] <= ... <= security[i + time - 1] <= security[i + time].

请你返回一个数组，包含 所有 适合打劫银行的日子（下标从 0 开始）。返回的日子可以 任意 顺序排列。

```python
class Solution:
    def goodDaysToRobBank(self, security: List[int], time: int) -> List[int]:
        n = len(security)
        preI = [0 for i in range(n)]
        preD = [0 for i in range(n)]
        pre = -1
        l = 0
        for i in range(n):
            if security[i] <= pre:
                l += 1
            else:
                l = 1
            pre = security[i]
            preI[i] = l 
        
        now = -1
        l = 0
        post = 0xffffffff
        for i in range(n-1,-1,-1):
            if security[i] <= post:
                l += 1
            else:
                l = 1
            post = security[i]
            preD[i] = l 
        
        ans = []
        for i in range(n):
            if preI[i] >= time+1 and preD[i] >= time+1:
                ans.append(i)
        # print(preI,preD)
        return ans
```

# [5936. 引爆最多的炸弹](https://leetcode-cn.com/problems/detonate-the-maximum-bombs/)

给你一个炸弹列表。一个炸弹的 爆炸范围 定义为以炸弹为圆心的一个圆。

炸弹用一个下标从 0 开始的二维整数数组 bombs 表示，其中 bombs[i] = [xi, yi, ri] 。xi 和 yi 表示第 i 个炸弹的 X 和 Y 坐标，ri 表示爆炸范围的 半径 。

你需要选择引爆 一个 炸弹。当这个炸弹被引爆时，所有 在它爆炸范围内的炸弹都会被引爆，这些炸弹会进一步将它们爆炸范围内的其他炸弹引爆。

给你数组 bombs ，请你返回在引爆 一个 炸弹的前提下，最多 能引爆的炸弹数目。

```python
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        # 用平方简化
        # 引爆不是双向的

        n = len(bombs)

        graph = collections.defaultdict(list)
        for i in range(n):
            for j in range(n):
                if i == j: 
                    continue 
                x1,y1,r1 = bombs[i]
                x2,y2,r2 = bombs[j]
                di = (x1-x2)**2 + (y1-y2)**2 
                rr = r1**2
                if di <= rr: # 距离够小
                    graph[i].append(j)

        # print(graph)
        
        
        def dfs(g):
            nonlocal visited 
            nonlocal cnt 
            if visited[g] == True:
                return 
            cnt += 1
            visited[g] = True 
            for neigh in graph[g]:
                dfs(neigh)
        
        ans = 1
        for i in range(n):
            visited = [False for t in range(n)]
            cnt = 0
            dfs(i)
            ans = max(ans,cnt)
        return ans            
```

# [5937. 序列顺序查询](https://leetcode-cn.com/problems/sequentially-ordinal-rank-tracker/)

一个观光景点由它的名字 name 和景点评分 score 组成，其中 name 是所有观光景点中 唯一 的字符串，score 是一个整数。景点按照最好到最坏排序。景点评分 越高 ，这个景点越好。如果有两个景点的评分一样，那么 字典序较小 的景点更好。

你需要搭建一个系统，查询景点的排名。初始时系统里没有任何景点。这个系统支持：

添加 景点，每次添加 一个 景点。
查询 已经添加景点中第 i 好 的景点，其中 i 是系统目前位置查询的次数（包括当前这一次）。
比方说，如果系统正在进行第 4 次查询，那么需要返回所有已经添加景点中第 4 好的。
注意，测试数据保证 任意查询时刻 ，查询次数都 不超过 系统中景点的数目。

请你实现 SORTracker 类：

SORTracker() 初始化系统。
void add(string name, int score) 向系统中添加一个名为 name 评分为 score 的景点。
string get() 查询第 i 好的景点，其中 i 是目前系统查询的次数（包括当前这次查询）。

```python
class SORTracker:
# python取巧
    def __init__(self):
        import sortedcontainers
        self.sl = sortedcontainers.sortedlist.SortedList()
        self.g = 0


    def add(self, name: str, score: int) -> None:
        self.sl.add((-score,name))
        
    def get(self) -> str:
        k = self.g
        self.g += 1
        n = len(self.sl)
        score,name = self.sl[k]
        return name
```

# [5938. 找出数组排序后的目标下标](https://leetcode-cn.com/problems/find-target-indices-after-sorting-array/)

给你一个下标从 0 开始的整数数组 nums 以及一个目标元素 target 。

目标下标 是一个满足 nums[i] == target 的下标 i 。

将 nums 按 非递减 顺序排序后，返回由 nums 中目标下标组成的列表。如果不存在目标下标，返回一个 空 列表。返回的列表必须按 递增 顺序排列。

```python
class Solution:
    def targetIndices(self, nums: List[int], target: int) -> List[int]:
        ans = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] == target:
                ans.append(i)
        return ans
        
```

# [5939. 半径为 k 的子数组平均值](https://leetcode-cn.com/problems/k-radius-subarray-averages/)

给你一个下标从 0 开始的数组 nums ，数组中有 n 个整数，另给你一个整数 k 。

半径为 k 的子数组平均值 是指：nums 中一个以下标 i 为 中心 且 半径 为 k 的子数组中所有元素的平均值，即下标在 i - k 和 i + k 范围（含 i - k 和 i + k）内所有元素的平均值。如果在下标 i 前或后不足 k 个元素，那么 半径为 k 的子数组平均值 是 -1 。

构建并返回一个长度为 n 的数组 avgs ，其中 avgs[i] 是以下标 i 为中心的子数组的 半径为 k 的子数组平均值 。

x 个元素的 平均值 是 x 个元素相加之和除以 x ，此时使用截断式 整数除法 ，即需要去掉结果的小数部分。

例如，四个元素 2、3、1 和 5 的平均值是 (2 + 3 + 1 + 5) / 4 = 11 / 4 = 3.75，截断后得到 3 。

```python
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        preSum = [0 for i in range(n)]
        pre = 0

        for i in range(n):
            pre += nums[i]
            preSum[i] = pre 
        
        ans = [-1 for i in range(n)]
        for i in range(n):
            if i-k-1 >= -1 and i+k < n:
                t1 = preSum[i-k-1] if i-k-1 != -1 else 0
                t2 = preSum[i+k]
                ans[i] = (t2-t1)//(2*k+1)
        return ans
```

# [5940. 从数组中移除最大值和最小值](https://leetcode-cn.com/problems/removing-minimum-and-maximum-from-array/)

给你一个下标从 0 开始的数组 nums ，数组由若干 互不相同 的整数组成。

nums 中有一个值最小的元素和一个值最大的元素。分别称为 最小值 和 最大值 。你的目标是从数组中移除这两个元素。

一次 删除 操作定义为从数组的 前面 移除一个元素或从数组的 后面 移除一个元素。

返回将数组中最小值和最大值 都 移除需要的最小删除次数。

```python
class Solution:
    def minimumDeletions(self, nums: List[int]) -> int:
        left = nums.index(max(nums))
        right = nums.index(min(nums))
        t = sorted([left,right])
        left,right = t 
        n = len(nums)
        s1 = max(left+1,right+1)
        s2 = max(n-left,n-right)
        s3 = (left+1+n-right)

        return min(s1,s2,s3)
```



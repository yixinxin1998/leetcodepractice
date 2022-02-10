# [5194. 得到目标值的最少行动次数](https://leetcode-cn.com/problems/minimum-moves-to-reach-target-score/)

你正在玩一个整数游戏。从整数 1 开始，期望得到整数 target 。

在一次行动中，你可以做下述两种操作之一：

递增，将当前整数的值加 1（即， x = x + 1）。
加倍，使当前整数的值翻倍（即，x = 2 * x）。
在整个游戏过程中，你可以使用 递增 操作 任意 次数。但是只能使用 加倍 操作 至多 maxDoubles 次。

给你两个整数 target 和 maxDoubles ，返回从 1 开始得到 target 需要的最少行动次数。

```python
class Solution:
    def minMoves(self, target: int, maxDoubles: int) -> int:
        # 贪心，偶数除以2，奇数减去1
        cnt = 0
        while target != 1 and maxDoubles > 0:
            if target % 2 == 1:
                cnt += 1
                
                target -= 1
            else:
                cnt += 1
                target //= 2 
                maxDoubles -= 1
        return target - 1 + cnt
            
```



# [5946. 句子中的最多单词数](https://leetcode-cn.com/problems/maximum-number-of-words-found-in-sentences/)

一个 句子 由一些 单词 以及它们之间的单个空格组成，句子的开头和结尾不会有多余空格。

给你一个字符串数组 sentences ，其中 sentences[i] 表示单个 句子 。

请你返回单个句子里 单词的最多数目 。

```python
class Solution:
    def mostWordsFound(self, sentences: List[str]) -> int:
        ans = 0
        for e in sentences:
            ans = max(ans,len(e.split()))
        return ans
            
```

# [5947. 从给定原材料中找到所有可以做出的菜](https://leetcode-cn.com/problems/find-all-possible-recipes-from-given-supplies/)

你有 n 道不同菜的信息。给你一个字符串数组 recipes 和一个二维字符串数组 ingredients 。第 i 道菜的名字为 recipes[i] ，如果你有它 所有 的原材料 ingredients[i] ，那么你可以 做出 这道菜。一道菜的原材料可能是 另一道 菜，也就是说 ingredients[i] 可能包含 recipes 中另一个字符串。

同时给你一个字符串数组 supplies ，它包含你初始时拥有的所有原材料，每一种原材料你都有无限多。

请你返回你可以做出的所有菜。你可以以 任意顺序 返回它们。

注意两道菜在它们的原材料中可能互相包含。

```python
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        n = len(recipes)
        graph = collections.defaultdict(list)
        inDegree = collections.defaultdict(int) # k是名字
        for i in range(n):
            for each in ingredients[i]:
                graph[each].append(recipes[i])
                inDegree[recipes[i]] += 1
        
        visited = set()
        queue = supplies[::]
        ans = set()
        while len(queue) != 0:
            new_queue = []
            for node in queue:
                for every in graph[node]:
                    inDegree[every] -= 1
                    if inDegree[every] == 0:
                        ans.add(every)
                        new_queue.append(every)
            queue = new_queue 
        return list(ans)
                        
```

# [5948. 判断一个括号字符串是否有效](https://leetcode-cn.com/problems/check-if-a-parentheses-string-can-be-valid/)

一个括号字符串是只由 '(' 和 ')' 组成的 非空 字符串。如果一个字符串满足下面 任意 一个条件，那么它就是有效的：

字符串为 ().
它可以表示为 AB（A 与 B 连接），其中A 和 B 都是有效括号字符串。
它可以表示为 (A) ，其中 A 是一个有效括号字符串。
给你一个括号字符串 s 和一个字符串 locked ，两者长度都为 n 。locked 是一个二进制字符串，只包含 '0' 和 '1' 。对于 locked 中 每一个 下标 i ：

如果 locked[i] 是 '1' ，你 不能 改变 s[i] 。
如果 locked[i] 是 '0' ，你 可以 将 s[i] 变为 '(' 或者 ')' 。
如果你可以将 s 变为有效括号字符串，请你返回 true ，否则返回 false 。

```python
class Solution:
    def canBeValid(self, s: str, locked: str) -> bool:
        n = len(s)
        if n%2 == 1:
            return False 
        # 栈匹配
        stack = [] 
        stackAny = []
        for i in range(n):
            if locked[i] == "0":
                stackAny.append(i)
            elif locked[i] == "1":
                if s[i] == "(":
                    stack.append(i)
                elif s[i] == ")":
                    if len(stack) > 0:
                        stack.pop()
                    elif len(stackAny) > 0:
                        stackAny.pop()
                    else:
                        return False
        # print(stack,stackAny)
        # 对stack进行索引检查，每个残余的stack中的元素
        while len(stack) and len(stackAny):
            if stack[-1] < stackAny[-1]:
                stack.pop()
                stackAny.pop()
            else:
                break
        if len(stack) != 0:
            return False 
        if len(stackAny)%2 != 0:
            return False 
        return True
       
```

# [5952. 环和杆](https://leetcode-cn.com/problems/rings-and-rods/)

总计有 n 个环，环的颜色可以是红、绿、蓝中的一种。这些环分布穿在 10 根编号为 0 到 9 的杆上。

给你一个长度为 2n 的字符串 rings ，表示这 n 个环在杆上的分布。rings 中每两个字符形成一个 颜色位置对 ，用于描述每个环：

第 i 对中的 第一个 字符表示第 i 个环的 颜色（'R'、'G'、'B'）。
第 i 对中的 第二个 字符表示第 i 个环的 位置，也就是位于哪根杆上（'0' 到 '9'）。
例如，"R3G2B1" 表示：共有 n == 3 个环，红色的环在编号为 3 的杆上，绿色的环在编号为 2 的杆上，蓝色的环在编号为 1 的杆上。

找出所有集齐 全部三种颜色 环的杆，并返回这种杆的数量。

```python
class Solution:
    def countPoints(self, rings: str) -> int:
        n = len(rings)
        pair = []
        for i in range(0,n,2):
            pair.append(rings[i:i+2])
        
        ct = [[] for i in range(10)]
        for c,ind in pair:
            ind = int(ind)
            ct[ind].append(c)

        ans = 0
        for i in range(10):
            if len(set(ct[i])) == 3:
                ans += 1
        return ans
```

# [5953. 子数组范围和](https://leetcode-cn.com/problems/sum-of-subarray-ranges/)

给你一个整数数组 nums 。nums 中，子数组的 范围 是子数组中最大元素和最小元素的差值。

返回 nums 中 所有 子数组范围的 和 。

子数组是数组中一个连续 非空 的元素序列。

```python
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        # n2做法
        n = len(nums)
        # 固定左端点枚举
        ans = 0
        for i in range(n):
            tmax = nums[i]
            tmin = nums[i]
            for j in range(i,n):
                tmax = max(tmax,nums[j])
                tmin = min(tmin,nums[j])
                ans += tmax-tmin
        return ans
```

```

```

# [5954. 给植物浇水 II](https://leetcode-cn.com/problems/watering-plants-ii/)

Alice 和 Bob 打算给花园里的 n 株植物浇水。植物排成一行，从左到右进行标记，编号从 0 到 n - 1 。其中，第 i 株植物的位置是 x = i 。

每一株植物都需要浇特定量的水。Alice 和 Bob 每人有一个水罐，最初是满的 。他们按下面描述的方式完成浇水：

 Alice 按 从左到右 的顺序给植物浇水，从植物 0 开始。Bob 按 从右到左 的顺序给植物浇水，从植物 n - 1 开始。他们 同时 给植物浇水。
如果没有足够的水 完全 浇灌下一株植物，他 / 她会立即重新灌满浇水罐。
不管植物需要多少水，浇水所耗费的时间都是一样的。
不能 提前重新灌满水罐。
每株植物都可以由 Alice 或者 Bob 来浇水。
如果 Alice 和 Bob 到达同一株植物，那么当前水罐中水更多的人会给这株植物浇水。如果他俩水量相同，那么 Alice 会给这株植物浇水。
给你一个下标从 0 开始的整数数组 plants ，数组由 n 个整数组成。其中，plants[i] 为第 i 株植物需要的水量。另有两个整数 capacityA 和 capacityB 分别表示 Alice 和 Bob 水罐的容量。返回两人浇灌所有植物过程中重新灌满水罐的 次数 。

```python
class Solution:
    def minimumRefill(self, plants: List[int], capacityA: int, capacityB: int) -> int:
        n = len(plants)
        nowA = capacityA
        nowB = capacityB 
        left = 0
        right = n-1
        times = 0 
        while left < right:
            if plants[left] <= nowA:
                nowA -= plants[left]
                left += 1
            elif plants[left] > nowA:
                nowA = capacityA
                nowA -= plants[left]
                left += 1
                times += 1
            
            if plants[right] <= nowB:
                nowB -= plants[right]
                right -= 1
            elif plants[right] > nowB:
                nowB = capacityB
                nowB -= plants[right]
                times += 1
                right -= 1
        
        if left == right:
            if plants[left] <= nowA or plants[right] <= nowB:
                pass
            else:
                times += 1
        return times
```

# [5956. 找出数组中的第一个回文字符串](https://leetcode-cn.com/problems/find-first-palindromic-string-in-the-array/)

给你一个字符串数组 words ，找出并返回数组中的 第一个回文字符串 。如果不存在满足要求的字符串，返回一个 空字符串 "" 。

回文字符串 的定义为：如果一个字符串正着读和反着读一样，那么该字符串就是一个 回文字符串 。

```python
class Solution:
    def firstPalindrome(self, words: List[str]) -> str:
        for w in words:
            if w == w[::-1]:
                return w 
        return ""
```

# [5957. 向字符串添加空格](https://leetcode-cn.com/problems/adding-spaces-to-a-string/)

给你一个下标从 0 开始的字符串 s ，以及一个下标从 0 开始的整数数组 spaces 。

数组 spaces 描述原字符串中需要添加空格的下标。每个空格都应该插入到给定索引处的字符值 之前 。

例如，s = "EnjoyYourCoffee" 且 spaces = [5, 9] ，那么我们需要在 'Y' 和 'C' 之前添加空格，这两个字符分别位于下标 5 和下标 9 。因此，最终得到 "Enjoy Your Coffee" 。
请你添加空格，并返回修改后的字符串。

```python
class Solution:
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        stack = []
        ps = 0
        p = 0
        while p < len(s):
            if ps < len(spaces) and p == spaces[ps]:
                stack.append(" ")
                stack.append(s[p])
                p += 1
                ps += 1
            else:
                stack.append(s[p])
                p += 1
        return "".join(stack)
```

# [5958. 股票平滑下跌阶段的数目](https://leetcode-cn.com/problems/number-of-smooth-descent-periods-of-a-stock/)

给你一个整数数组 prices ，表示一支股票的历史每日股价，其中 prices[i] 是这支股票第 i 天的价格。

一个 平滑下降的阶段 定义为：对于 连续一天或者多天 ，每日股价都比 前一日股价恰好少 1 ，这个阶段第一天的股价没有限制。

请你返回 平滑下降阶段 的数目。

```python
class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [1 for i in range(n)]
        for i in range(1,n):
            if prices[i]+1 == prices[i-1]:
                dp[i] = dp[i-1]+1
        return sum(dp)
```



# [5960. 将标题首字母大写](https://leetcode-cn.com/problems/capitalize-the-title/)

给你一个字符串 title ，它由单个空格连接一个或多个单词组成，每个单词都只包含英文字母。请你按以下规则将每个单词的首字母 大写 ：

如果单词的长度为 1 或者 2 ，所有字母变成小写。
否则，将单词首字母大写，剩余字母变成小写。
请你返回 大写后 的 title 。

```python
class Solution:
    def capitalizeTitle(self, title: str) -> str:
        stack = []
        t = title.split(' ')
        for w in t:
            if len(w) <= 2:
                stack.append(w.lower())
            else:
                stack.append(w[0].upper()+w[1:].lower())
        return " ".join(stack)
```

# [5961. 链表最大孪生和](https://leetcode-cn.com/problems/maximum-twin-sum-of-a-linked-list/)

在一个大小为 n 且 n 为 偶数 的链表中，对于 0 <= i <= (n / 2) - 1 的 i ，第 i 个节点（下标从 0 开始）的孪生节点为第 (n-1-i) 个节点 。

比方说，n = 4 那么节点 0 是节点 3 的孪生节点，节点 1 是节点 2 的孪生节点。这是长度为 n = 4 的链表中所有的孪生节点。
孪生和 定义为一个节点和它孪生节点两者值之和。

给你一个长度为偶数的链表的头节点 head ，请你返回链表的 最大孪生和 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        lst = []
        maxAns = -1
        cur = head 
        while cur:
            lst.append(cur.val)
            cur = cur.next 
        
        left = 0
        right = len(lst)-1
        while left < right:
            maxAns = max(maxAns,lst[left]+lst[right])
            left += 1
            right -= 1
        return maxAns
```

# [5962. 连接两字母单词得到的最长回文串](https://leetcode-cn.com/problems/longest-palindrome-by-concatenating-two-letter-words/)

给你一个字符串数组 words 。words 中每个元素都是一个包含 两个 小写英文字母的单词。

请你从 words 中选择一些元素并按 任意顺序 连接它们，并得到一个 尽可能长的回文串 。每个元素 至多 只能使用一次。

请你返回你能得到的最长回文串的 长度 。如果没办法得到任何一个回文串，请你返回 0 。

回文串 指的是从前往后和从后往前读一样的字符串。

```python
class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        ct = collections.defaultdict(int)
        # 区分真回文，和具有对称的
        # 
        for w in words:
            ct[w] += 1
        ans = 0
        used = set()
        maxSingle = 0
        for w in ct:
            if w in used:
                continue
            if w[::-1] in ct and w != w[::-1]:
                ans += len(w) * min(ct[w],ct[w[::-1]]) * 2
                used.add(w)
                used.add(w[::-1])
            elif w[::-1] == w: # 真回文
                used.add(w)
                ans += len(w) * (ct[w]//2*2)
                if ct[w]%2 == 1:
                    maxSingle = max(maxSingle,len(w))
   
        return ans + maxSingle
            
```

# [5963. 反转两次的数字](https://leetcode-cn.com/problems/a-number-after-a-double-reversal/)

反转 一个整数意味着倒置它的所有位。

例如，反转 2021 得到 1202 。反转 12300 得到 321 ，不保留前导零 。
给你一个整数 num ，反转 num 得到 reversed1 ，接着反转 reversed1 得到 reversed2 。如果 reversed2 等于 num ，返回 true ；否则，返回 false 。

```python
class Solution:
    def isSameAfterReversals(self, num: int) -> bool:
        r1 = int(str(num)[::-1])
        r2 = int(str(r1)[::-1])
        return r2 == num
```

# [5964. 执行所有后缀指令](https://leetcode-cn.com/problems/execution-of-all-suffix-instructions-staying-in-a-grid/)

现有一个 n x n 大小的网格，左上角单元格坐标 (0, 0) ，右下角单元格坐标 (n - 1, n - 1) 。给你整数 n 和一个整数数组 startPos ，其中 startPos = [startrow, startcol] 表示机器人最开始在坐标为 (startrow, startcol) 的单元格上。

另给你一个长度为 m 、下标从 0 开始的字符串 s ，其中 s[i] 是对机器人的第 i 条指令：'L'（向左移动），'R'（向右移动），'U'（向上移动）和 'D'（向下移动）。

机器人可以从 s 中的任一第 i 条指令开始执行。它将会逐条执行指令直到 s 的末尾，但在满足下述条件之一时，机器人将会停止：

下一条指令将会导致机器人移动到网格外。
没有指令可以执行。
返回一个长度为 m 的数组 answer ，其中 answer[i] 是机器人从第 i 条指令 开始 ，可以执行的 指令数目 。

```python
class Solution:
    def executeInstructions(self, n: int, startPos: List[int], s: str) -> List[int]:
    # 暴力，硬编码
        m = len(s)
        direcDict = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}
        ans = [0 for i in range(m)]
        for i in range(m):
            pp = s[i:]
            now = [startPos[0],startPos[1]]
            tempcnt = 0
            for every in pp:
                now[0] += direcDict[every][0]
                now[1] += direcDict[every][1]
                if 0<=now[0]<n and 0<=now[1]<n:
                    tempcnt += 1
                else:
                    break 
            ans[i] = tempcnt
        return ans
```

# [5965. 相同元素的间隔之和](https://leetcode-cn.com/problems/intervals-between-identical-elements/)

给你一个下标从 0 开始、由 n 个整数组成的数组 arr 。

arr 中两个元素的 间隔 定义为它们下标之间的 绝对差 。更正式地，arr[i] 和 arr[j] 之间的间隔是 |i - j| 。

返回一个长度为 n 的数组 intervals ，其中 intervals[i] 是 arr[i] 和 arr 中每个相同元素（与 arr[i] 的值相同）的 间隔之和 。

注意：|x| 是 x 的绝对值。

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

    def getDistances(self, arr: List[int]) -> List[int]:
        # 每一个元素收集其索引
        
        indexDict = collections.defaultdict(list)
        for i,key in enumerate(arr):
            indexDict[key].append(i)
        
        lst = []
        aimDict = collections.defaultdict(int)
        
        # 对每个元素
        # print(indexDict)
        
        for key in indexDict: # key是数值  
            aimDict[key] = self.getSumAbsoluteDifferences(indexDict[key])
        # print(aimDict)
        
        k = len(arr)
        ans = [0 for i in range(k)]
        tt1 = []
        tt2 = []
        for key in indexDict:
            tt1 += indexDict[key]
            tt2 += aimDict[key]
        # print(tt1,tt2)
        mirror = dict(zip(tt1,tt2))
        for i in range(k):
            ans[i] = mirror[i]
        return ans
```

# [5967. 检查是否所有 A 都在 B 之前](https://leetcode-cn.com/problems/check-if-all-as-appears-before-all-bs/)

给你一个 仅 由字符 'a' 和 'b' 组成的字符串  s 。如果字符串中 每个 'a' 都出现在 每个 'b' 之前，返回 true ；否则，返回 false 。

```python
class Solution:
    def checkString(self, s: str) -> bool:
        return 'ba' not in s
```

# [5968. 银行中的激光束数量](https://leetcode-cn.com/problems/number-of-laser-beams-in-a-bank/)

银行内部的防盗安全装置已经激活。给你一个下标从 0 开始的二进制字符串数组 bank ，表示银行的平面图，这是一个大小为 m x n 的二维矩阵。 bank[i] 表示第 i 行的设备分布，由若干 '0' 和若干 '1' 组成。'0' 表示单元格是空的，而 '1' 表示单元格有一个安全设备。

对任意两个安全设备而言，如果同时 满足下面两个条件，则二者之间存在 一个 激光束：

两个设备位于两个 不同行 ：r1 和 r2 ，其中 r1 < r2 。
满足 r1 < i < r2 的 所有 行 i ，都 没有安全设备 。
激光束是独立的，也就是说，一个激光束既不会干扰另一个激光束，也不会与另一个激光束合并成一束。

返回银行中激光束的总数量。

```python
class Solution:
    def numberOfBeams(self, bank: List[str]) -> int:
        # 相邻行的计算1的量的乘积
        ans = 0
        lst = []
        for line in bank:
            t = line.count('1')
            if t != 0:
                lst.append(t)
        for i in range(1,len(lst)):
            ans += lst[i]*lst[i-1]
        return ans
```

# [5969. 摧毁小行星](https://leetcode-cn.com/problems/destroying-asteroids/)

给你一个整数 mass ，它表示一颗行星的初始质量。再给你一个整数数组 asteroids ，其中 asteroids[i] 是第 i 颗小行星的质量。

你可以按 任意顺序 重新安排小行星的顺序，然后让行星跟它们发生碰撞。如果行星碰撞时的质量 大于等于 小行星的质量，那么小行星被 摧毁 ，并且行星会 获得 这颗小行星的质量。否则，行星将被摧毁。

如果所有小行星 都 能被摧毁，请返回 true ，否则返回 false 。

```python
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        ast = asteroids
        # 贪心
        ast.sort()
        for i in range(len(ast)):
            if mass >= ast[i]:
                mass += ast[i]
            else:
                return False 
        return True
```

# [5971. 打折购买糖果的最小开销](https://leetcode-cn.com/problems/minimum-cost-of-buying-candies-with-discount/)

一家商店正在打折销售糖果。每购买 两个 糖果，商店会 免费 送一个糖果。

免费送的糖果唯一的限制是：它的价格需要小于等于购买的两个糖果价格的 较小值 。

比方说，总共有 4 个糖果，价格分别为 1 ，2 ，3 和 4 ，一位顾客买了价格为 2 和 3 的糖果，那么他可以免费获得价格为 1 的糖果，但不能获得价格为 4 的糖果。
给你一个下标从 0 开始的整数数组 cost ，其中 cost[i] 表示第 i 个糖果的价格，请你返回获得 所有 糖果的 最小 总开销。

```python
class Solution:
    def minimumCost(self, cost: List[int]) -> int:
        ans = 0
        cost.sort(reverse=True)
        p = 0
        free = False 
        ct = 0
        while p < len(cost):
            if not free:
                ct += 1
                ans += cost[p]
                p += 1
                if ct == 2:
                    free = True 
                    ct = 0
            else:
                p += 1
                free = False 
        return ans
```

# [5972. 统计隐藏数组数目](https://leetcode-cn.com/problems/count-the-hidden-sequences/)

给你一个下标从 0 开始且长度为 n 的整数数组 differences ，它表示一个长度为 n + 1 的 隐藏 数组 相邻 元素之间的 差值 。更正式的表述为：我们将隐藏数组记作 hidden ，那么 differences[i] = hidden[i + 1] - hidden[i] 。

同时给你两个整数 lower 和 upper ，它们表示隐藏数组中所有数字的值都在 闭 区间 [lower, upper] 之间。

比方说，differences = [1, -3, 4] ，lower = 1 ，upper = 6 ，那么隐藏数组是一个长度为 4 且所有值都在 1 和 6 （包含两者）之间的数组。
[3, 4, 1, 5] 和 [4, 5, 2, 6] 都是符合要求的隐藏数组。
[5, 6, 3, 7] 不符合要求，因为它包含大于 6 的元素。
[1, 2, 3, 4] 不符合要求，因为相邻元素的差值不符合给定数据。
请你返回 符合 要求的隐藏数组的数目。如果没有符合要求的隐藏数组，请返回 0 。

```python
class Solution:
    def numberOfArrays(self, differences: List[int], lower: int, upper: int) -> int:
        n = len(differences)
        tempMin = 0xffffffff
        tempMax = -0xffffffff
        now = 0
        for n in differences:
            now += n 
            tempMin = min(now,tempMin)
            tempMax = max(now,tempMax)
        
        tempMax = max(0,tempMax)
        tempMin = min(0,tempMin)
        
        g1 = upper-lower
        g2 = tempMax-tempMin 
        if g1 < g2:
            return 0
        else:
            return g1-g2+1
```

# [5973. 价格范围内最高排名的 K 样物品](https://leetcode-cn.com/problems/k-highest-ranked-items-within-a-price-range/)

给你一个下标从 0 开始的二维整数数组 grid ，它的大小为 m x n ，表示一个商店中物品的分布图。数组中的整数含义为：

0 表示无法穿越的一堵墙。
1 表示可以自由通过的一个空格子。
所有其他正整数表示该格子内的一样物品的价格。你可以自由经过这些格子。
从一个格子走到上下左右相邻格子花费 1 步。

同时给你一个整数数组 pricing 和 start ，其中 pricing = [low, high] 且 start = [row, col] ，表示你开始位置为 (row, col) ，同时你只对物品价格在 闭区间 [low, high] 之内的物品感兴趣。同时给你一个整数 k 。

你想知道给定范围 内 且 排名最高 的 k 件物品的 位置 。排名按照优先级从高到低的以下规则制定：

距离：定义为从 start 到一件物品的最短路径需要的步数（较近 距离的排名更高）。
价格：较低 价格的物品有更高优先级，但只考虑在给定范围之内的价格。
行坐标：较小 行坐标的有更高优先级。
列坐标：较小 列坐标的有更高优先级。
请你返回给定价格内排名最高的 k 件物品的坐标，将它们按照排名排序后返回。如果给定价格内少于 k 件物品，那么请将它们的坐标 全部 返回。

```python
class Solution:
    def highestRankedKItems(self, grid: List[List[int]], pricing: List[int], start: List[int], k: int) -> List[List[int]]:
        ans = []
        # 先有距离，再有价格
        m,n = len(grid),len(grid[0])
        
        left,right = pricing[0],pricing[1]
        
        queue = [start]
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        steps = 0
        ans = [] # [距离，价格，横，纵]
        visited = set()
        visited.add(tuple(start))
        
        while len(queue):
            new_queue = []
            for i,j in queue:
                if left<=grid[i][j]<=right:
                    ans.append([steps,grid[i][j],[i,j]])
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and grid[new_i][new_j] >= 1 and (new_i,new_j) not in visited:
                        new_queue.append([new_i,new_j])
                        visited.add((new_i,new_j))
            queue = new_queue 
            steps += 1
        
        # print(ans)
        ans.sort()
        return [e[2] for e in ans[:k]]
```

# [5974. 分隔长廊的方案数](https://leetcode-cn.com/problems/number-of-ways-to-divide-a-long-corridor/)

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

# [5976. 检查是否每一行每一列都包含全部整数](https://leetcode-cn.com/problems/check-if-every-row-and-column-contains-all-numbers/)

对一个大小为 n x n 的矩阵而言，如果其每一行和每一列都包含从 1 到 n 的 全部 整数（含 1 和 n），则认为该矩阵是一个 有效 矩阵。

给你一个大小为 n x n 的整数矩阵 matrix ，请你判断矩阵是否为一个有效矩阵：如果是，返回 true ；否则，返回 false 。

```python
class Solution:
    def checkValid(self, matrix: List[List[int]]) -> bool:
        n = len(matrix)
        template = [i+1 for i in range(n)]
        
        for line in matrix:
            e = sorted(line)
            if e != template:
                return False 
        
        for j in range(n):
            lst = []           
            for i in range(n):            
                lst.append(matrix[i][j])            
            lst.sort()
            if lst != template:
                return False 
        return True
```

# [5977. 最少交换次数来组合所有的 1 II](https://leetcode-cn.com/problems/minimum-swaps-to-group-all-1s-together-ii/)

交换 定义为选中一个数组中的两个 互不相同 的位置并交换二者的值。

环形 数组是一个数组，可以认为 第一个 元素和 最后一个 元素 相邻 。

给你一个 二进制环形 数组 nums ，返回在 任意位置 将数组中的所有 1 聚集在一起需要的最少交换次数。

```python
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        # 首位拼接，滑动窗口,固定窗口长度
        size = 0
        for k in nums:
            if k == 1:
                size += 1
        left = 0
        right = size 
        nums = nums + nums
        n = len(nums)
        window = sum(nums[:size])
        maxOne = window # 初始化
        while right < n:
            add = nums[right]
            delete = nums[left]
            left += 1
            right += 1
            window += add - delete
            maxOne = max(maxOne,window)
        return size-maxOne
            
```

# [5978. 统计追加字母可以获得的单词数](https://leetcode-cn.com/problems/count-words-obtained-after-adding-a-letter/)

给你两个下标从 0 开始的字符串数组 startWords 和 targetWords 。每个字符串都仅由 小写英文字母 组成。

对于 targetWords 中的每个字符串，检查是否能够从 startWords 中选出一个字符串，执行一次 转换操作 ，得到的结果与当前 targetWords 字符串相等。

转换操作 如下面两步所述：

追加 任何 不存在 于当前字符串的任一小写字母到当前字符串的末尾。
例如，如果字符串为 "abc" ，那么字母 'd'、'e' 或 'y' 都可以加到该字符串末尾，但 'a' 就不行。如果追加的是 'd' ，那么结果字符串为 "abcd" 。
重排 新字符串中的字母，可以按 任意 顺序重新排布字母。
例如，"abcd" 可以重排为 "acbd"、"bacd"、"cbda"，以此类推。注意，它也可以重排为 "abcd" 自身。
找出 targetWords 中有多少字符串能够由 startWords 中的 任一 字符串执行上述转换操作获得。返回 targetWords 中这类 字符串的数目 。

注意：你仅能验证 targetWords 中的字符串是否可以由 startWords 中的某个字符串经执行操作获得。startWords  中的字符串在这一过程中 不 发生实际变更。

```python
class Solution:
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        # 单词转int，利用位来计算
        def toInt(s):
            k = 0
            lst = [0 for i in range(26)]
            for ch in s:
                index = ord(ch)-ord('a')
                lst[index] += 1
            for i in range(len(lst)):
                k += lst[i]*(2**i)
            return k 
        t1 = set(toInt(w) for w in startWords)
        t2 = [toInt(w) for w in targetWords]        
        # 要求必须执行一次转换，必须先追加，再重排
        # 扫描t2,掩蔽一个，看是否存在在t1中
        cnt = 0
        for w in t2:
            temp = []
            for i in range(27):
                if (w>>i)&1 == 1:
                    k = w ^ (1<<(i))
                    temp.append(k) 
            for every in temp:
                if every in t1:
                    cnt += 1
                    break 
        
        return cnt
```

# [5980. 将字符串拆分为若干长度为 k 的组](https://leetcode-cn.com/problems/divide-a-string-into-groups-of-size-k/)

字符串 s 可以按下述步骤划分为若干长度为 k 的组：

第一组由字符串中的前 k 个字符组成，第二组由接下来的 k 个字符串组成，依此类推。每个字符都能够成为 某一个 组的一部分。
对于最后一组，如果字符串剩下的字符 不足 k 个，需使用字符 fill 来补全这一组字符。
注意，在去除最后一个组的填充字符 fill（如果存在的话）并按顺序连接所有的组后，所得到的字符串应该是 s 。

给你一个字符串 s ，以及每组的长度 k 和一个用于填充的字符 fill ，按上述步骤处理之后，返回一个字符串数组，该数组表示 s 分组后 每个组的组成情况 。

```python
class Solution:
    def divideString(self, s: str, k: int, fill: str) -> List[str]:
        remain = k - len(s) % k 
        if remain != k:
            s += remain * fill 
        ans = []
        for i in range(0,len(s),k):
            ans.append(s[i:i+k])
        return ans
```

# [5981. 分组得分最高的所有下标](https://leetcode-cn.com/problems/all-divisions-with-the-highest-score-of-a-binary-array/)

给你一个下标从 0 开始的二进制数组 nums ，数组长度为 n 。nums 可以按下标 i（ 0 <= i <= n ）拆分成两个数组（可能为空）：numsleft 和 numsright 。

numsleft 包含 nums 中从下标 0 到 i - 1 的所有元素（包括 0 和 i - 1 ），而 numsright 包含 nums 中从下标 i 到 n - 1 的所有元素（包括 i 和 n - 1 ）。
如果 i == 0 ，numsleft 为 空 ，而 numsright 将包含 nums 中的所有元素。
如果 i == n ，numsleft 将包含 nums 中的所有元素，而 numsright 为 空 。
下标 i 的 分组得分 为 numsleft 中 0 的个数和 numsright 中 1 的个数之 和 。

返回 分组得分 最高 的 所有不同下标 。你可以按 任意顺序 返回答案。

```python
class Solution:
    def maxScoreIndices(self, nums: List[int]) -> List[int]:
    # 注意边界条件处理即可
        ans = []
        n = len(nums)
        maxScore = 0
        ct0 = [0 for i in range(n)]
        ct1 = [0 for i in range(n)]
        pre = 0 
        for i in range(n):
            ct0[i] = pre 
            if nums[i] == 0:
                pre += 1
        ct0.append(pre)
        pre = 0
        for i in range(n-1,-1,-1):
            if nums[i] == 1:
                pre += 1
            ct1[i] = pre 
        
        maxScore = ct0[-1]
        for i in range(n):
            maxScore = max(maxScore,ct0[i]+ct1[i])
        for i in range(n):
            if ct0[i]+ct1[i] == maxScore:
                ans.append(i)
        if ct0[-1] == maxScore:
            ans.append(n)
        return ans

```

# [5982. 解决智力问题](https://leetcode-cn.com/problems/solving-questions-with-brainpower/)

给你一个下标从 0 开始的二维整数数组 questions ，其中 questions[i] = [pointsi, brainpoweri] 。

这个数组表示一场考试里的一系列题目，你需要 按顺序 （也就是从问题 0 开始依次解决），针对每个问题选择 解决 或者 跳过 操作。解决问题 i 将让你 获得  pointsi 的分数，但是你将 无法 解决接下来的 brainpoweri 个问题（即只能跳过接下来的 brainpoweri 个问题）。如果你跳过问题 i ，你可以对下一个问题决定使用哪种操作。

比方说，给你 questions = [[3, 2], [4, 3], [4, 4], [2, 5]] ：
如果问题 0 被解决了， 那么你可以获得 3 分，但你不能解决问题 1 和 2 。
如果你跳过问题 0 ，且解决问题 1 ，你将获得 4 分但是不能解决问题 2 和 3 。
请你返回这场考试里你能获得的 最高 分数。

```python
class Solution:
    def mostPoints(self, questions: List[List[int]]) -> int:
        # 倒序dp
        n = len(questions)
        dp = [0 for i in range(n)]      
        dp[-1] = questions[-1][0]
        
        for i in range(n-2,-1,-1):
            pi = questions[i][0]
            bpi = questions[i][1]
            # 选择它
            state1 = pi + dp[i+bpi+1] if i+bpi+1 < n else pi 
            # 不选择它
            state2 = dp[i+1] 
            dp[i] = max(state1,state2)
        return dp[0] 
```

# [5984. 拆分数位后四位数字的最小和](https://leetcode-cn.com/problems/minimum-sum-of-four-digit-number-after-splitting-digits/)

给你一个四位 正 整数 num 。请你使用 num 中的 数位 ，将 num 拆成两个新的整数 new1 和 new2 。new1 和 new2 中可以有 前导 0 ，且 num 中 所有 数位都必须使用。

比方说，给你 num = 2932 ，你拥有的数位包括：两个 2 ，一个 9 和一个 3 。一些可能的 [new1, new2] 数对为 [22, 93]，[23, 92]，[223, 9] 和 [2, 329] 。
请你返回可以得到的 new1 和 new2 的 最小 和。

```python
class Solution:
    def minimumSum(self, num: int) -> int:
    # 暴力
        num = str(num)
        ans = []
        temp = []
        used = [False for i in range(4)]

        def backtracking(index,path):
            if len(path) == 4:
                temp.append("".join(path[:]))
            for i in range(4):
                if used[i] == False:
                    path.append(num[i])
                    used[i] = True
                    backtracking(i,path)
                    used[i] = False 
                    path.pop()
        
        backtracking(0,[])
        # print(temp)
        for each in temp:
            for i in range(1,4):
                ans.append([int(each[:i]),int(each[i:])])
        
        return min(sum(key) for key in ans)
```

# [5985. 根据给定数字划分数组](https://leetcode-cn.com/problems/partition-array-according-to-given-pivot/)

给你一个下标从 0 开始的整数数组 nums 和一个整数 pivot 。请你将 nums 重新排列，使得以下条件均成立：

所有小于 pivot 的元素都出现在所有大于 pivot 的元素 之前 。
所有等于 pivot 的元素都出现在小于和大于 pivot 的元素 中间 。
小于 pivot 的元素之间和大于 pivot 的元素之间的 相对顺序 不发生改变。
更正式的，考虑每一对 pi，pj ，pi 是初始时位置 i 元素的新位置，pj 是初始时位置 j 元素的新位置。对于小于 pivot 的元素，如果 i < j 且 nums[i] < pivot 和 nums[j] < pivot 都成立，那么 pi < pj 也成立。类似的，对于大于 pivot 的元素，如果 i < j 且 nums[i] > pivot 和 nums[j] > pivot 都成立，那么 pi < pj 。
请你返回重新排列 nums 数组后的结果数组。

```python
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        less = []
        equel = []
        more = []
        for number in nums:
            if number < pivot:
                less.append(number)
            elif number == pivot:
                equel.append(number)
            else:
                more.append(number)
        return less + equel + more
```

# [5986. 设置时间的最少代价](https://leetcode-cn.com/problems/minimum-cost-to-set-cooking-time/)

常见的微波炉可以设置加热时间，且加热时间满足以下条件：

至少为 1 秒钟。
至多为 99 分 99 秒。
你可以 最多 输入 4 个数字 来设置加热时间。如果你输入的位数不足 4 位，微波炉会自动加 前缀 0 来补足 4 位。微波炉会将设置好的四位数中，前 两位当作分钟数，后 两位当作秒数。它们所表示的总时间就是加热时间。比方说：

你输入 9 5 4 （三个数字），被自动补足为 0954 ，并表示 9 分 54 秒。
你输入 0 0 0 8 （四个数字），表示 0 分 8 秒。
你输入 8 0 9 0 ，表示 80 分 90 秒。
你输入 8 1 3 0 ，表示 81 分 30 秒。
给你整数 startAt ，moveCost ，pushCost 和 targetSeconds 。一开始，你的手指在数字 startAt 处。将手指移到 任何其他数字 ，需要花费 moveCost 的单位代价。每 输入你手指所在位置的数字一次，需要花费 pushCost 的单位代价。

要设置 targetSeconds 秒的加热时间，可能会有多种设置方法。你想要知道这些方法中，总代价最小为多少。

请你能返回设置 targetSeconds 秒钟加热时间需要花费的最少代价。

请记住，虽然微波炉的秒数最多可以设置到 99 秒，但一分钟等于 60 秒。

```python
class Solution:
    def minCostSetTime(self, startAt: int, moveCost: int, pushCost: int, targetSeconds: int) -> int:
        # target拆分成两种
        # 前导0可以略去不打
        # 是从左到右打
        # 注意补丁，大于6000秒的只能通过第二种方法得到

        t1_mm = targetSeconds//60
        t1_ss = targetSeconds%60 

        if t1_ss < 40 and t1_mm >= 1:
            t2_mm = t1_mm - 1
            t2_ss = t1_ss + 60 
        else:
            t2_mm = None 
            t2_ss = None 
        

        # 然后格式化
        def fmt(s):
            if s != None:
                s = str(s)
                l = 2-len(s)
                return l*'0'+s 
        
        t1_mm,t1_ss = fmt(t1_mm),fmt(t1_ss)
        t2_mm,t2_ss = fmt(t2_mm),fmt(t2_ss)
        # 去除前导0
        t1 = str(int(t1_mm+t1_ss))
        #
        if int(t1) >= 10000:
            t1 = None
        
        if t2_mm:
            t2 = str(int(t2_mm+t2_ss))
        else:
            t2 = None
        
        startAt = str(startAt)
        
        def check(t):
            if t == None:
                return inf
            cost = 0
            pre = startAt
            for num in t:
                if num != pre:
                    cost += moveCost
                    cost += pushCost
                    pre = num 
                elif num == pre:
                    cost += pushCost
                # print('cost',cost)
            return cost 
        
        ans = []        
        ans.append(check(t1))
        ans.append(check(t2))
        return min(ans)
```



# [5989. 元素计数](https://leetcode-cn.com/problems/count-elements-with-strictly-smaller-and-greater-elements/)

给你一个整数数组 `nums` ，统计并返回在 `nums` 中同时具有一个严格较小元素和一个严格较大元素的元素数目。

```python
class Solution:
    def countElements(self, nums: List[int]) -> int:
        mmax = max(nums)
        mmin = min(nums)
        ct = 0
        for n in nums:
            if mmin<n<mmax:
                ct += 1
        return ct
```

# [5990. 找出数组中的所有孤独数字](https://leetcode-cn.com/problems/find-all-lonely-numbers-in-the-array/)

给你一个整数数组 nums 。如果数字 x 在数组中仅出现 一次 ，且没有 相邻 数字（即，x + 1 和 x - 1）出现在数组中，则认为数字 x 是 孤独数字 。

返回 nums 中的 所有 孤独数字。你可以按 任何顺序 返回答案。

```python
class Solution:
    def findLonely(self, nums: List[int]) -> List[int]:
        ct = collections.Counter(nums)
        ans = []
        for key in ct:
            if ct[key] == 1 and key-1 not in ct and key + 1 not in ct:
                ans.append(key)
        return ans
```

# [5991. 按符号重排数组](https://leetcode-cn.com/problems/rearrange-array-elements-by-sign/)

给你一个下标从 0 开始的整数数组 nums ，数组长度为 偶数 ，由数目相等的正整数和负整数组成。

你需要 重排 nums 中的元素，使修改后的数组满足下述条件：

任意 连续 的两个整数 符号相反
对于符号相同的所有整数，保留 它们在 nums 中的 顺序 。
重排后数组以正整数开头。
重排元素满足上述条件后，返回修改后的数组。

```python
class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        pos = deque()
        neg = deque()
        for n in nums:
            if n < 0:
                neg.append(n)
            else:
                pos.append(n)
        ans = []
        while pos and neg:
            ans.append(pos.popleft())
            ans.append(neg.popleft())
        return ans
```

# [5992. 基于陈述统计最多好人数](https://leetcode-cn.com/problems/maximum-good-people-based-on-statements/)

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
        n = len(statements)
        ans = 0
        for i in range(2**n):
            prop = [None for j in range(n)]
            group = []
            for p in range(n):
                if (i>>p)%2 == 1:
                    prop[p] = True 
                    group.append(statements[p])
                else:
                    prop[p] = False     
            state = True
            for each in group:
                if state:
                    for t in range(n):
                        if each[t] == 0 and prop[t] == True:
                            state = False 
                            break 
                        if each[t] == 1 and prop[t] == False:
                            state = False 
                            break 
                else:
                    break 
            if state:
                ans = max(ans,bin(i).count('1'))
        return ans
```

# [5993. 将找到的值乘以 2](https://leetcode-cn.com/problems/keep-multiplying-found-values-by-two/)

给你一个整数数组 nums ，另给你一个整数 original ，这是需要在 nums 中搜索的第一个数字。

接下来，你需要按下述步骤操作：

如果在 nums 中找到 original ，将 original 乘以 2 ，得到新 original（即，令 original = 2 * original）。
否则，停止这一过程。
只要能在数组中找到新 original ，就对新 original 继续 重复 这一过程。
返回 original 的 最终 值。

```python
class Solution:
    def findFinalValue(self, nums: List[int], original: int) -> int:
        nums = set(nums)
        while original in nums:
            original *= 2 
        return original
```

# [5994. 查找给定哈希值的子串](https://leetcode-cn.com/problems/find-substring-with-given-hash-value/)

给定整数 p 和 m ，一个长度为 k 且下标从 0 开始的字符串 s 的哈希值按照如下函数计算：

hash(s, p, m) = (val(s[0]) * p0 + val(s[1]) * p1 + ... + val(s[k-1]) * pk-1) mod m.
其中 val(s[i]) 表示 s[i] 在字母表中的下标，从 val('a') = 1 到 val('z') = 26 。

给你一个字符串 s 和整数 power，modulo，k 和 hashValue 。请你返回 s 中 第一个 长度为 k 的 子串 sub ，满足 hash(sub, power, modulo) == hashValue 。

测试数据保证一定 存在 至少一个这样的子串。

子串 定义为一个字符串中连续非空字符组成的序列。

```python
class Solution:
    def subStrHash(self, s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
        # 返回第一个满足的
        n = len(s)
        p,mod = power,modulo 
        ans = []

        def val(s): return ord(s)-ord('a') + 1

        # 由于取模除法没有同余性质，所以需要倒着算
        win = 0
        for i in range(n-k,n):
            win += val(s[i])*pow(p,i-(n-k),mod)
        win %= mod 
        ans = []
        # print(win)
        if win == hashValue:
            ans.append(n-k)
        # 开始倒序
        right = n-1
        left = n-k-1
        g = pow(p,k-1,mod)
        while left >= 0:
            # print(win,s[left],s[right],s[left:left+k])
            win = (win - val(s[right])*g)*p + val(s[left])
            win %= mod
            if win == hashValue:
                ans.append(left)
            left -= 1
            right -= 1
        temp = ans[-1]
        return s[temp:temp+k]
```

# [6000. 对奇偶下标分别排序](https://leetcode-cn.com/problems/sort-even-and-odd-indices-independently/)

给你一个下标从 0 开始的整数数组 nums 。根据下述规则重排 nums 中的值：

按 非递增 顺序排列 nums 奇数下标 上的所有值。
举个例子，如果排序前 nums = [4,1,2,3] ，对奇数下标的值排序后变为 [4,3,2,1] 。奇数下标 1 和 3 的值按照非递增顺序重排。
按 非递减 顺序排列 nums 偶数下标 上的所有值。
举个例子，如果排序前 nums = [4,1,2,3] ，对偶数下标的值排序后变为 [2,1,4,3] 。偶数下标 0 和 2 的值按照非递减顺序重排。
返回重排 nums 的值之后形成的数组。

```python
class Solution:
    def sortEvenOdd(self, nums: List[int]) -> List[int]:
        odd = []
        even = []
        p = 0
        while p < len(nums):
            if p%2 == 0:
                even.append(nums[p])
            else:
                odd.append(nums[p])
            p += 1
        odd.sort(reverse=True)
        even.sort()
        ans = []
        p1,p2 = 0,0
        n1,n2 = len(even),len(odd)
        while p1<n1 and p2<n2:
            ans.append(even[p1])
            ans.append(odd[p2])
            p1 += 1
            p2 += 1
        while p1 < n1:
            ans.append(even[p1])
            p1 += 1
        while p2 < n2:
            ans.append(odd[p2])
            p2 += 1
        return ans
```

# [6001. 重排数字的最小值](https://leetcode-cn.com/problems/smallest-value-of-the-rearranged-number/)

给你一个整数 num 。重排 num 中的各位数字，使其值 最小化 且不含 任何 前导零。

返回不含前导零且值最小的重排数字。

注意，重排各位数字后，num 的符号不会改变。

```python
class Solution:
    def smallestNumber(self, num: int) -> int:
        if num == 0:
            return 0
        symbol = 1
        if num < 0:
            symbol = -1 
        num = abs(num)
        # 由于数字比较大，不能用全排列穷举
        # 如果正数
        if symbol == 1:
            tt = list(map(int,list(str(num))))
            tt.sort()
            p = 0
            ans = []
            while p < len(tt) and tt[p] == 0:
                p += 1
            if p < len(tt) and tt[p] != 0:
                ans.append(tt[p])
                tt.pop(p)          
            ans += tt 
            return int("".join(map(str,ans)))
        # 如果负数
        else:
            tt = list(map(int,list(str(num))))
            tt.sort(reverse=True)
            ans = tt
            return -int("".join(map(str,ans)))
```

# [6002. 设计位集](https://leetcode-cn.com/problems/design-bitset/)

位集 Bitset 是一种能以紧凑形式存储位的数据结构。

请你实现 Bitset 类。

Bitset(int size) 用 size 个位初始化 Bitset ，所有位都是 0 。
void fix(int idx) 将下标为 idx 的位上的值更新为 1 。如果值已经是 1 ，则不会发生任何改变。
void unfix(int idx) 将下标为 idx 的位上的值更新为 0 。如果值已经是 0 ，则不会发生任何改变。
void flip() 翻转 Bitset 中每一位上的值。换句话说，所有值为 0 的位将会变成 1 ，反之亦然。
boolean all() 检查 Bitset 中 每一位 的值是否都是 1 。如果满足此条件，返回 true ；否则，返回 false 。
boolean one() 检查 Bitset 中 是否 至少一位 的值是 1 。如果满足此条件，返回 true ；否则，返回 false 。
int count() 返回 Bitset 中值为 1 的位的 总数 。
String toString() 返回 Bitset 的当前组成情况。注意，在结果字符串中，第 i 个下标处的字符应该与 Bitset 中的第 i 位一致。

```python
class Bitset:

    def __init__(self, size: int):
        # 懒惰更新
        self.rec = [0 for i in range(size)]
        self.f = 0
        self.ones = 0


    def fix(self, idx: int) -> None:
        if self.f == 0:
            if self.rec[idx] == 0:
                self.ones += 1
            self.rec[idx] = 1
        else:
            if self.rec[idx] == 1:
                self.ones += 1
            self.rec[idx] = 0


    def unfix(self, idx: int) -> None:
        if self.f == 0:
            if self.rec[idx] == 1:
                self.ones -= 1
            self.rec[idx] = 0
        else:
            if self.rec[idx] == 0:
                self.ones -= 1
            self.rec[idx] = 1



    def flip(self) -> None:
        self.f ^= 1
        self.ones = len(self.rec)-self.ones


    def all(self) -> bool:
        return self.ones == len(self.rec)


    def one(self) -> bool:
        return self.ones >= 1


    def count(self) -> int:
        return self.ones 


    def toString(self) -> str:
        if self.f == 0:
            return "".join(map(str,self.rec))
        else:
            temp = [e^1 for e in self.rec]
            return "".join(map(str,temp))
```



# [12. 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)

```python
class Solution:
    def intToRoman(self, num: int) -> str:

        def subMethod(n):
            if 0<=n<=3:
                return "I"*n 
            elif n == 4:
                return "IV"
            elif n == 5:
                return "V"
            elif n <= 8:
                return "V"+subMethod(n-5)
            elif n == 9:
                return "IX"
            elif n == 10:
                return "X"
            elif 11 <= n <= 39:
                return n//10*"X" + subMethod(n%10)
            elif 40 <= n <= 49:
                return "XL" + subMethod(n-40)
            elif 50 <= n <= 89:
                return "L" + subMethod(n-50)
            elif 90 <= n <= 99:
                return "XC" + subMethod(n-90)
            elif 100 <= n <= 399:
                return n//100*"C" + subMethod(n%100)
            elif 400 <= n <= 499:
                return "CD" + subMethod(n-400)
            elif 500 <= n <= 899:
                return "D" + subMethod(n-500)
            elif 900 <= n <= 999:
                return "CM" + subMethod(n-900)
            elif n >= 1000:
                return n//1000*"M" + subMethod(n%1000)
        
        return subMethod(num)
```

# [57. 插入区间](https://leetcode-cn.com/problems/insert-interval/)

给你一个 **无重叠的** *，*按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # 其实可以直接找到横坐标排序后的插入再merge
        # 这里调用merge的做法
        intervals.append(newInterval)
        intervals.sort()
        return self.merge(intervals)
    
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 合并,根据起始点排序
        intervals.sort()
        ans = [intervals[0]]
        for i in range(1,len(intervals)):
            a,b = ans[-1]
            c,d = intervals[i]
            if a <= c <= d <= b: # 直接跳过
                pass 
            elif a <= c <= b <= d: # 修改前一个
                ans[-1][1] = d 
            else:
                ans.append(intervals[i])
        return ans
```

# [89. 格雷编码](https://leetcode-cn.com/problems/gray-code/)

n 位格雷码序列 是一个由 2n 个整数组成的序列，其中：
每个整数都在范围 [0, 2n - 1] 内（含 0 和 2n - 1）
第一个整数是 0
一个整数在序列中出现 不超过一次
每对 相邻 整数的二进制表示 恰好一位不同 ，且
第一个 和 最后一个 整数的二进制表示 恰好一位不同
给你一个整数 n ，返回任一有效的 n 位格雷码序列 。

```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        # 1位格雷码有两个码字
        # (n+1)位格雷码中的前2^n个码字等于n位格雷码的码字，按顺序书写，加前缀0
        # (n+1)位格雷码中的后2^n个码字等于n位格雷码的码字，按逆序书写，加前缀1
        # n+1位格雷码的集合 = n位格雷码集合(顺序)加前缀0 + n位格雷码集合(逆序)加前缀1

        def subMethod(n):
            if n == 1:
                return ['0','1']
            if n > 1:
                temp = subMethod(n-1)
                final = []
                for g in temp:
                    final.append('0'+g)
                for g in temp[::-1]: # 逆序
                    final.append('1'+g)
                return final
        
        def bin_to_int(lst):
            ans = []
            for e in lst:
                temp = list(e[::-1])
                v = 0
                for i in range(len(temp)):
                    v += int(temp[i])*(2**i)
                ans.append(v)
            return ans
                       
        ans = subMethod(n)
        final = bin_to_int(ans)
        return final

```

# [157. 用 Read4 读取 N 个字符](https://leetcode-cn.com/problems/read-n-characters-given-read4/)

给你一个文件，并且该文件只能通过给定的 `read4` 方法来读取，请实现一个方法使其能够读取 n 个字符。

```python
"""
The read4 API is already defined for you.

    @param buf4, a list of characters
    @return an integer
    def read4(buf4):

# Below is an example of how the read4 API can be called.
file = File("abcdefghijk") # File is "abcdefghijk", initially file pointer (fp) points to 'a'
buf4 = [' '] * 4 # Create buffer with enough space to store characters
read4(buf4) # read4 returns 4. Now buf = ['a','b','c','d'], fp points to 'e'
read4(buf4) # read4 returns 4. Now buf = ['e','f','g','h'], fp points to 'i'
read4(buf4) # read4 returns 3. Now buf = ['i','j','k',...], fp points to end of file
"""

class Solution:
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Number of characters to read (int)
        :rtype: The number of actual characters read (int)
        """
        # 这一题难在读题太难
        # py需要开辟一个新的temp_buff
        p = 0
        for i in range(0,n,4):
            temp_buff = [""]*4
            k = read4(temp_buff) # 先读取到temp空间里
            for j in range(k):
                buf[p] = temp_buff[j]
                p += 1
        return min(n,p)
                
```



# [254. 因子的组合](https://leetcode-cn.com/problems/factor-combinations/)

整数可以被看作是其因子的乘积。

例如：

8 = 2 x 2 x 2;
  = 2 x 4.
请实现一个函数，该函数接收一个整数 n 并返回该整数所有的因子组合。

```python
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        #
        def dfs(left,n): # 从left开始，对n进行尝试分解
            ans = []
            for i in range(left,int(sqrt(n))+1):
                if n%i == 0:
                    ans.append([i,n//i]) # 将答案加入
                    for sub in dfs(i,n//i): # 递归处理子问题，尝试对n//i进行分解,分解边界从i开始
                        # 原来的i是一解，sub是对于n//i的解，那么添加方式为
                        temp = [i] + sub 
                        ans.append(temp)
            return ans 
        
        return dfs(2,n)
```

```go
func getFactors(n int) [][]int {
    return dfs(2,n)
}

func dfs(left,n int) [][]int {
    ans := make([][]int,0,0)
    limit := int(math.Sqrt(float64(n)))+1
    for i:=left;i<limit;i++ {
        if n%i == 0 {
            ans = append(ans,[]int{i,n/i})
            for _,subpromblem := range dfs(i,n/i) {
            subpromblem = append(subpromblem,i)
            ans = append(ans,subpromblem)
            }
        }
        
    }
    return ans
}
```

# [280. 摆动排序](https://leetcode-cn.com/problems/wiggle-sort/)

给你一个无序的数组 `nums`, 将该数字 **原地** 重排后使得 `nums[0] <= nums[1] >= nums[2] <= nums[3]...`。

```python
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 拷贝一份列表，按照从小到大，先赋值偶数，再赋值奇数
        cp = nums[:]
        cp.sort()
        p = 0
        for i in range(0,len(nums),2):
            nums[i] = cp[p]
            p += 1
        for i in range(1,len(nums),2):
            nums[i] = cp[p]
            p += 1
        
```

```python
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nums.sort() # 预先排序，不使用额外空间
        # 从第三个开始,索引为2，和前一个交换,间隔为2
        for p in range(2,len(nums),2):
            nums[p],nums[p-1] = nums[p-1],nums[p]
```

```python
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 不预先排序的算法，直接从头到尾排，如果顺序不正确，交换
        for p in range(1,len(nums)):
            if p % 2 == 0:
                if not nums[p] <= nums[p-1]:
                    nums[p],nums[p-1] = nums[p-1],nums[p]
            else:
                if not nums[p] >= nums[p-1]:
                    nums[p],nums[p-1] = nums[p-1],nums[p]
```

# [306. 累加数](https://leetcode-cn.com/problems/additive-number/)

累加数 是一个字符串，组成它的数字可以形成累加序列。

一个有效的 累加序列 必须 至少 包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。

给你一个只包含数字 '0'-'9' 的字符串，编写一个算法来判断给定输入是否是 累加数 。如果是，返回 true ；否则，返回 false 。

说明：累加序列里的数 不会 以 0 开头，所以不会出现 1, 2, 03 或者 1, 02, 3 的情况。

```python
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        # 暴力拆分+判断,python天然支持大数
        def dfs(num,firstnum,secondnum): # num是待切割的字符串
            if num == "": return True 
            total = firstnum + secondnum
            length = len(str(total))
            if str(total) == num[:length]: # 表示长度足够
                return dfs(num[length:],secondnum,total)
            return False 
        
        for i in range(1,len(num)-1): # 枚举分割点， 。。。。i....j.....然后递归检查
            if num[0] == '0' and i > 1: break 
            for j in range(i+1,len(num)):
                if j-i > 1 and num[i] == '0': break  # 第二个数不能是0开头的非0数
                if dfs(num[j:],int(num[:i]),int(num[i:j])):
                    return True 
        return False 
        
```

```python
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        # 枚举分割点，由于py天然支持大数加法，所以直接使用
        # 顺序排列为 first,second,now
        def dfs(now:str,first:int,second:int):
            if now == "":
                return True 
            total = first + second
            length = len(str(total))
            if int(now[:length]) == total:
                return dfs(now[length:],second,total)
            return False 
        
        n = len(num)
        # num[:i],num[i:j],num[j:]
        for i in range(1,n-1):
            if num[0] == '0' and i > 1: break 
            for j in range(i+1,n):
                if j-i > 1 and num[i] == "0": break
                if dfs(num[j:],int(num[:i]),int(num[i:j])):
                    return True 
        return False
```



# [362. 敲击计数器](https://leetcode-cn.com/problems/design-hit-counter/)

设计一个敲击计数器，使它可以统计在过去5分钟内被敲击次数。

每个函数会接收一个时间戳参数（以秒为单位），你可以假设最早的时间戳从1开始，且都是按照时间顺序对系统进行调用（即时间戳是单调递增）。

在同一时刻有可能会有多次敲击。

```python
class HitCounter:
# 双端队列+
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.lst = collections.deque()


    def hit(self, timestamp: int) -> None:
        """
        Record a hit.
        @param timestamp - The current timestamp (in seconds granularity).
        """
        self.lst.append(timestamp)
        g = timestamp - 300
        while self.lst and self.lst[0] <= g:
            self.lst.popleft()


    def getHits(self, timestamp: int) -> int:
        """
        Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity).
        """
        g = timestamp - 300
        while self.lst and self.lst[0] <= g:
            self.lst.popleft()
        return len(self.lst)        
```

# [365. 水壶问题](https://leetcode-cn.com/problems/water-and-jug-problem/)

有两个容量分别为 x升 和 y升 的水壶以及无限多的水。请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？

如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。

你允许：

装满任意一个水壶
清空任意一个水壶
从一个水壶向另外一个水壶倒水，直到装满或者倒空

```python
class Solution:
    def canMeasureWater(self, jug1Capacity: int, jug2Capacity: int, targetCapacity: int) -> bool:
        # 这一题很有意思，分析，假设x >= y
        # 假设两个水壶都为空:
        # 可以方便获得x,y,x+y,(x-y)【有一个空壶，在大壶里有水，如果小壶可以承载这个剩余的，那么大壶还能加x】, 2*y【无法再操作】
        # 两个水壶不能都未满
        x,y,z = jug1Capacity,jug2Capacity,targetCapacity
        if x < y: x,y = y,x # 假设x >= y
        # 需要获取gcd
        def getGCD(a,b):
            while a != 0:
                temp = a 
                a = b%a 
                b = temp 
            return b 
        return z%getGCD(x,y) == 0 and x+y >= z
        # 具体解释见高赞题解
```



# [388. 文件的最长绝对路径](https://leetcode-cn.com/problems/longest-absolute-file-path/)

体力活，描述见链接

```python
class Solution:
    def lengthLongestPath(self, input: str) -> int:
        # 先根据\n切割
        s = input.split("\n")
        # print(s)
        # 然后使用栈，如果这一个的词前缀\t数量和前面的一样多，弹出前一个，如果这个词更多，加入栈，如果这个词少，循环pop弹出
        stack = []
        def countLevel(ss):
            level = 0
            if ss != "" and ss[0] == "\t":
                p = 0
                while p < len(ss):
                    if ss[p] == "\t":
                        level += 1
                    else:
                        break
                    p += 1
            return level 

        preLevel = 0
        maxLength = 0
        nowLength = 0

        for i in range(len(s)):
            nowLevel = countLevel(s[i])
            # print('s[i] = ',s[i],' nowLevel = ',nowLevel)
            while len(stack) != 0:
                preLevel = countLevel(stack[-1])
                if nowLevel <= preLevel:
                    e = stack.pop()
                    nowLength -= 1 + (len(e)-countLevel(e))
                else:
                    break

            stack.append(s[i]) # 将该文件长度计算进去
            nowLength += len(s[i]) + 1 - nowLevel # 需要加一个路径/位，减去制表符占位
            # print(nowLength)
            if "." in s[i]: # 必须要是文件才更新
                maxLength = max(maxLength,nowLength)

        # 返回值注意根目录也加上了路径/位，需要减去
        return maxLength-1 if maxLength != 0 else 0
```



# [401. 二进制手表](https://leetcode-cn.com/problems/binary-watch/)

二进制手表顶部有 4 个 LED 代表 **小时（0-11）**，底部的 6 个 LED 代表 **分钟（0-59）**。每个 LED 代表一个 0 或 1，最低位在右侧。

```python
class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        # 第一排代表小时数量，第二排代表分钟数量
        # 小时必须在0～11，且不会以0开头
        # 分钟必须在0～59，可以以0开头
        ans = []
        lst = [1,2,4,8,1,2,4,8,16,32]
        n = len(lst)
        path1 = [] # 收集前4个
        path2 = [] # 收集后6个
        def judge(stack1,stack2): #传入参数为路径列表
            state1,state2 = False,False
            if len(stack1) == 0 or sum(stack1) <= 11:
                state1 = True
            if len(stack2) == 0 or sum(stack2) <= 59:
                state2 = True 
            if state1 and state2:
                return True
            else:
                return False 
                
        def backtracking(index,remain,path1,path2):
            if remain == 0:
                if judge(path1,path2):
                    hh = str(sum(path1)) if len(path1) != 0 else "0"
                    mm = str(sum(path2)) if len(path2) != 0 else "00"
                    if len(mm) == 1:
                        mm = "0"+mm
                    ans.append(hh+":"+mm)
                return 
            if index == n:
                return 
            if 0<=index<=3:
                path1.append(lst[index])
                backtracking(index+1,remain-1,path1,path2)
                path1.pop()
            backtracking(index+1,remain,path1,path2)
            if index >= 4:
                path2.append(lst[index])
                backtracking(index+1,remain-1,path1,path2)
                path2.pop()

        backtracking(0,turnedOn,[],[])
        return ans
           
```

# [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)

给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。

规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。

请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        # 逆向思考，从边缘冲刷
        m,n = len(heights),len(heights[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        state = [[0 for j in range(n)] for i in range(m)]
        # BFS冲刷
        queue = []
        for i in range(m):
            if not visited[i][0]:
                queue.append([i,0])
                visited[i][0] = True
        for j in range(n):
            if not visited[0][j]:
                queue.append([0,j])
                visited[0][j] = True 

        while len(queue) != 0:
            new_queue = []
            for i,j in queue:
                state[i][j] += 1
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and heights[new_i][new_j] >= heights[i][j]:
                        visited[new_i][new_j] = True 
                        new_queue.append([new_i,new_j])
            queue = new_queue 
        
        visited = [[False for j in range(n)] for i in range(m)] # 重置
        # BFS冲刷
        queue = []
        for i in range(m):
            if not visited[i][n-1]:
                queue.append([i,n-1])
                visited[i][n-1] = True
        for j in range(n):
            if not visited[m-1][j]:
                queue.append([m-1,j])
                visited[m-1][j] = True 

        while len(queue) != 0:
            new_queue = []
            for i,j in queue:
                state[i][j] += 1
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and heights[new_i][new_j] >= heights[i][j]:
                        visited[new_i][new_j] = True 
                        new_queue.append([new_i,new_j])
            queue = new_queue 
        
        ans = []
        for i in range(m):
            for j in range(n):
                if state[i][j] == 2:
                    ans.append([i,j])
        return ans
```

# [444. 序列重建](https://leetcode-cn.com/problems/sequence-reconstruction/)

验证原始的序列 org 是否可以从序列集 seqs 中唯一地重建。序列 org 是 1 到 n 整数的排列，其中 1 ≤ n ≤ 104 。重建是指在序列集 seqs 中构建最短的公共超序列。（即使得所有  seqs 中的序列都是该最短序列的子序列）。请你确定是否只可以从 seqs 重建唯一的序列，且该序列就是 org 。

```python
class Solution:
    def sequenceReconstruction(self, org: List[int], seqs: List[List[int]]) -> bool:
        graph = collections.defaultdict(list)
        inDegree = collections.defaultdict(int)
        n = len(org)
        visited = set()
        # a -> b -> c -> d -> e
        for each in seqs:
            p = 1
            while p < len(each):
                a = each[p-1]
                b = each[p]
                graph[a].append(b)
                inDegree[b] += 1
                p += 1
            for ch in each:
                visited.add(ch)
        # 激活
        for i in range(n):
            graph[i+1]
            inDegree[i+1]
        
        if visited != set(org): return False # 两者集合不相同
        # 然后检查是否每一次都为0
        ans = []
        queue = []
        for key in inDegree:
            if inDegree[key] == 0:
                queue.append(key)
        
        while len(queue):
            new_queue = []
            if len(queue) != 1: return False 
            for node in queue:
                ans.append(node)
                for neigh in graph[node]:
                    inDegree[neigh] -= 1
                    if inDegree[neigh] == 0:
                        new_queue.append(neigh)
            queue = new_queue 
        return ans == org # 检查最后是否一样
```

# [452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。

一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        #贪心：先把气球根据起始位置排序
        points.sort()
        cnt = 1 # 初始化为一根箭
        # 画图辅助
        n = len(points)
        # 贪心的点在于尽量靠右，每次搜索下一个气球的时候，入射点修改成 min(end),
        # 当搜索到某一次start已经大于这个的时候，弓箭数目+1，重置入射点
        p = 0
        end = points[0][1]
        while p < n:
            if points[p][0] <= end:
                end = min(end,points[p][1])
                p += 1
            elif points[p][0] > end:
                end = points[p][1]
                p += 1
                cnt += 1   
        return cnt
            
```

```go
func findMinArrowShots(points [][]int) int {
    cnt := 1
    n := len(points)
    end := points[0][1]
    p := 0
    sort.Slice(points,func(i,j int) bool {return points[i][0] < points[j][0]})
    for p < n {
        if points[p][0] <= end {
            end = min(end,points[p][1])
        } else {
            cnt += 1
            end = points[p][1]
        }
        p += 1
    }
    return cnt
}

func min(a,b int) int {
    if a > b {
        return b
    } else {
        return a
    }
}
```

# [456. 132 模式](https://leetcode-cn.com/problems/132-pattern/)

给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：i < j < k 和 nums[i] < nums[k] < nums[j] 。

如果 nums 中存在 132 模式的子序列 ，返回 true ；否则，返回 false 。

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        # 分析，枚举。思路参考官解 + 三叶
        # 枚举3， 严格维护小于3的1， 然后在3以及右端点区间扫描，看是否存在2 【二分搜索
        if len(nums) < 3:
            return False 
        from sortedcontainers import SortedList

        leftMin = nums[0]
        container = SortedList(nums[1:])
        n = len(nums)
        for j in range(1,n):
            # print(container)
            if leftMin < nums[j]: # 枚举
                index = bisect_left(container,nums[j])-1
                # 只有这个index在范围内且指向的数大于nums[j]的时候，才返回True
                if 0<=index<len(container) and container[index] < nums[j] and container[index] > leftMin:
                    return True 
            # 否则不管怎么样都需要动态修改区间
            if nums[j] < leftMin:
                leftMin = nums[j]
            index = bisect_left(container,nums[j])
            container.pop(index)

        return False
```

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        # 从右到左的单调栈
        
        stack = []
        k = -0xffffffff
        n = len(nums)

        # k是结构中的2
        # 当当前值小于2成立，则返回True
        # k总存着第二大的数

        # 选取单调递减栈，遇到大数字需要更新
        
        for i in range(n-1,-1,-1):
            if nums[i] < k:
                return True 
            while stack and stack[-1] < nums[i]:
                k = max(stack.pop(),k)
            stack.append(nums[i])
        return False 

```

# [468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)

编写一个函数来验证输入的字符串是否是有效的 IPv4 或 IPv6 地址。

如果是有效的 IPv4 地址，返回 "IPv4" ；
如果是有效的 IPv6 地址，返回 "IPv6" ；
如果不是上述类型的 IP 地址，返回 "Neither" 。

```python
class Solution:
    def validIPAddress(self, IP: str) -> str:
        # ipv6允许前导0，且允许多个前导0
        def judgeipv4(s):
            stack = s.split(".")
            if len(stack) != 4:
                return False 
            # 其间的必须是四个数字
            for paragraph in stack:
                if len(paragraph) == 0:
                    return False
                for e in paragraph:
                    if not (48 <= ord(e) <= 57):
                        return False 
            for paragraph in stack:
                if paragraph == "0":
                    continue                   
                if len(str(int(paragraph))) != len(paragraph):
                    return False 
                if not (0<= int(paragraph) <= 255):
                    return False 
            return True 
        
        def judgeipv6(s):
            s = s.lower()
            stack = s.split(":")
            if len(stack) != 8:
                return False 
            alphaSet = set('abcdef')
            # 其间的必须是数字或者"abcdef"
            for paragraph in stack:
                if len(paragraph) == 0:
                    return False
                if len(paragraph) > 4:
                    return False 
                for e in paragraph:
                    state1 =  (48 <= ord(e) <= 57)
                    state2 =  (e in alphaSet)
                    if state1 == False and state2 == False:
                        return False
            return True 
            
        if (judgeipv4(IP)):
            return "IPv4"
        elif (judgeipv6(IP)):
            return "IPv6"
        else:
            return "Neither"
        
```

# [473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/)

还记得童话《卖火柴的小女孩》吗？现在，你知道小女孩有多少根火柴，请找出一种能使用所有火柴拼成一个正方形的方法。不能折断火柴，可以把火柴连接起来，并且每根火柴都要用到。

输入为小女孩拥有火柴的数目，每根火柴用其长度表示。输出即为是否能用所有的火柴拼成正方形。

```python
class Solution:
    def makesquare(self, matchsticks: List[int]) -> bool:
        # 如果长度不是4的倍数，直接gg
        s = sum(matchsticks)
        if s%4 != 0:
            return False 
        limit = s//4
        # 
        # 四个路径的回溯
        n = len(matchsticks)
        state = False
        matchsticks.sort(reverse=True) # 注意这一行
        def backtracking(index,path1,path2,path3,path4,sz1,sz2,sz3,sz4):
            nonlocal state 
            if index == n:
                if sz1 == sz2 == sz3 == sz4:
                    state = True 
                return 
            if state == True:
                return 

            path1.append(matchsticks[index])
            if sz1 + matchsticks[index] <= limit:
                backtracking(index+1,path1,path2,path3,path4,sz1+matchsticks[index],sz2,sz3,sz4)
            path1.pop()

            path2.append(matchsticks[index])
            if sz2 + matchsticks[index] <= limit:
                backtracking(index+1,path1,path2,path3,path4,sz1,sz2+matchsticks[index],sz3,sz4)
            path2.pop()  

            path3.append(matchsticks[index])
            if sz3 + matchsticks[index] <= limit:
                backtracking(index+1,path1,path2,path3,path4,sz1,sz2,sz3+matchsticks[index],sz4)
            path3.pop()    

            path4.append(matchsticks[index])
            if sz4 + matchsticks[index] <= limit:
                backtracking(index+1,path1,path2,path3,path4,sz1,sz2,sz3,sz4+matchsticks[index])
            path4.pop()

        backtracking(0,[],[],[],[],0,0,0,0)
        return state
```



# [474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。

如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        # 每个元素都有选和不选，所以是01背包问题
        # 容量是m,n
        # 先采取朴素的dp。【注意复杂度分析，数据量100*100*600 < 10**8; 600*100 < 10**8),所以可以采取朴素dp】
        # 确定选择方式dp[i][j][k]表示前i个数【包括第i个】，1的上限是j，0的上限是k
        # 初始化dp数组,注意初始化是填0值或者极小值，因为最终要取的是max
        dp = [[[0 for k in range(n+1)] for j in range(m+1)] for i in range(len(strs)+1)]
        # 注意为什么需要开到n+1,因为n+1长度的终止索引才是n -> 而我们需要的是最多得容纳n个
        # 注意为什么要开到m+1,m+1的终止索引是m -> 我们需要容纳m个
        # 注意为什么需要开到len(strs)+1，方便边界处理，可以只开len(strs)，但是边界处理很麻烦
        # 确定状态转移
        # dp[i][j][k] = max(state1,state2) 其中
        # state1 := dp[i-1][j][k] # 不把第i个元素加入
        # state2 := dp[i-1][j-ct0][k-ct1]+1 # 把第i个元素加入
        # 确定遍历顺序，由于dp[i][j][k] 需要从 序号[i-1][j-ct0][k-ct1]转移
        # i >= i-1 ; j >= j-ct0 ; k >= k-ct1 为了使得基础状态已经被计算过，所以需要从小值 -》 向大值填充
        for i in range(1,len(strs)+1):
            # 第i个数的索引是i-1
            ct0 = strs[i-1].count('0')  # 计数0的个数
            ct1 = len(strs[i-1]) - ct0  # 全长减去0的数目就是1的数目
            for j in range(m+1): # 遍历顺序 小 -》 大
                for k in range(n+1): # 遍历顺序 小 -》 大
                    state1 = dp[i-1][j][k] 
                    state2 = dp[i-1][j-ct0][k-ct1]+1 if j >= ct0 and k >= ct1 else 0
                    dp[i][j][k] = max(state1,state2)       
        return dp[-1][-1][-1] # 返回右下角 

```

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        # 根据基础版本，使用滚动数组版本1
        # 首先要知道，滚动数组并不降低时间复杂度，只会降低空间复杂度
        # 但是使用滚动数组，却实实在在能使得运行时间得到一定的提升，为什么？ 因为复用了之前申请的空间，而不采取滚动数组的时候，是纯重复的申请空间，在大O表示的时间复杂度上，数量级虽然不变，但是滚动数组空间复用减少了很多次的空间申请
        # dp[][] 为什么只用两个维度表示，这种表示方法是不适合新手的，因为每一轮最外层循环使得dp的基础含义有变化
        # 例如 第一轮的dp[2][3] 和 第五轮的dp[2][3] 含义是不一样的
        # 所以推荐prev[][]表示
        prev = [[0 for k in range(n+1)] for j in range(m+1)]
        # 滚动次数为times = len(strs) + 1
        for i in range(1,len(strs) + 1):
            now = [[0 for k in range(n+1)] for j in range(m+1)]
            ct0 = strs[i-1].count('0')
            ct1 = len(strs[i-1]) - ct0
            for j in range(m+1):
                for k in range(n+1):
                    state1 = prev[j][k]
                    state2 = prev[j-ct0][k-ct1]+1 if j >= ct0 and k >= ct1 else 0
                    now[j][k] = max(state1,state2)
            prev = now # 将这次结果滚动丢给上次
        
        # 分析空间复杂度，最多只占用了 (m+1)*(n+1)*2次的数量级的空间。
        return prev[-1][-1] # 返回右下角
        
```

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        # 进一步优化，滚动数组的prev是必须的吗？
        # 不是，prev其实已经存在在自身中，只要这个点没有被更新，那么它就是上一轮的值
        # 那么回到了关键问题
        # prev[j][k] <<=== prev[j-ct0][k-ct1]的转移时，一定需要后者是已经计算完成的值，
        #  这一轮             上一轮
        # 为了防止这一轮提前覆盖了上一轮， 需要倒序遍历！

        # 动态规划的关键之一是： ”无后效性【这一点之后做多了慢慢体会】“
        prev = [[0 for k in range(n+1)] for j in range(m+1)]
        for i in range(1,len(strs)+1):
            ct0 = strs[i-1].count('0')
            ct1 = len(strs[i-1]) - ct0 
            for j in range(m,ct0-1,-1): # 注意这个遍历顺序
                for k in range(n,ct1-1,-1): # 注意这个遍历顺序
                    state1 = prev[j][k]
                    state2 = prev[j-ct0][k-ct1]+1 if j >= ct0 and k >= ct1 else 0
                    prev[j][k] = max(state1,state2)
        
        # 复杂度分析，相比于滚动数组1， 没有了每次申请now的开销，空间复杂度数量级不变，常数优化，时间复杂度常数优化
        return prev[-1][-1]
```

# [475. 供暖器](https://leetcode-cn.com/problems/heaters/)

冬季已经来临。 你的任务是设计一个有固定加热半径的供暖器向所有房屋供暖。

在加热器的加热半径范围内的每个房屋都可以获得供暖。

现在，给出位于一条水平线上的房屋 houses 和供暖器 heaters 的位置，请你找出并返回可以覆盖所有房屋的最小加热半径。

说明：所有供暖器都遵循你的半径标准，加热的半径也一样。

```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        ans = 0
        heaters.sort()
        for h in houses:
            # 对每个房屋进行搜索，对每个房屋选择最近的，对所有结果中选择最大的
            j = bisect.bisect_right(heaters,h)
            i = j - 1
            rightDistance = heaters[j]-h if j < len(heaters) else 0xffffffff
            leftDistance = h-heaters[i] if i >=0 else 0xffffffff 
            curDistance = min(leftDistance,rightDistance)
            ans = max(ans,curDistance)
        return ans
```

# [529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/)

给你一个大小为 m x n 二维字符矩阵 board ，表示扫雷游戏的盘面，其中：

'M' 代表一个 未挖出的 地雷，
'E' 代表一个 未挖出的 空方块，
'B' 代表没有相邻（上，下，左，右，和所有4个对角线）地雷的 已挖出的 空白方块，
数字（'1' 到 '8'）表示有多少地雷与这块 已挖出的 方块相邻，
'X' 则表示一个 已挖出的 地雷。
给你一个整数数组 click ，其中 click = [clickr, clickc] 表示在所有 未挖出的 方块（'M' 或者 'E'）中的下一个点击位置（clickr 是行下标，clickc 是列下标）。

根据以下规则，返回相应位置被点击后对应的盘面：

如果一个地雷（'M'）被挖出，游戏就结束了- 把它改为 'X' 。
如果一个 没有相邻地雷 的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的 未挖出 方块都应该被递归地揭露。
如果一个 至少与一个地雷相邻 的空方块（'E'）被挖出，修改它为数字（'1' 到 '8' ），表示相邻地雷的数量。
如果在此次点击中，若无更多方块可被揭露，则返回盘面。

```python
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        # 递归更新
        m,n = len(board),len(board[0])
        direc = [(0,1),(0,-1),(1,0),(-1,0),(-1,-1),(-1,1),(1,-1),(1,1)]

        def method(i,j):
            if board[i][j] == "M":
                board[i][j] = "X"
                return 
            if board[i][j] == "E":
                neigh = []
                neighMine = 0
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n:
                        if board[new_i][new_j] == "M":
                            neighMine += 1
                        elif board[new_i][new_j] == "E":
                            neigh.append([new_i,new_j])
                if neighMine == 0:
                    board[i][j] = "B" 
                    for i,j in neigh:
                        method(i,j)
                else:
                    board[i][j] = str(neighMine) # 更新为数字之后不需要递归更新
                    
        method(click[0],click[1])
        return board
```

# [554. 砖墙](https://leetcode-cn.com/problems/brick-wall/)

你的面前有一堵矩形的、由 n 行砖块组成的砖墙。这些砖块高度相同（也就是一个单位高）但是宽度不同。每一行砖块的宽度之和相等。

你现在要画一条 自顶向下 的、穿过 最少 砖块的垂线。如果你画的线只是从砖块的边缘经过，就不算穿过这块砖。你不能沿着墙的两个垂直边缘之一画线，这样显然是没有穿过一块砖的。

给你一个二维数组 wall ，该数组包含这堵墙的相关信息。其中，wall[i] 是一个代表从左至右每块砖的宽度的数组。你需要找出怎样画才能使这条线 穿过的砖块数量最少 ，并且返回 穿过的砖块数量 。

```python
class Solution:
    def leastBricks(self, wall: List[List[int]]) -> int:
        # 利用前缀和，找到缝隙，
        n = len(wall)
        preDict = collections.defaultdict(int)
        for i in range(n):
            pre = 0
            for t in range(len(wall[i])):
                pre += wall[i][t]
                preDict[pre] += 1
                
        # 避免每次直接更新最大重叠缝隙，注意需要排除最后一条缝隙，则使用for key来更新
        maxGap = 0
        allSum = sum(wall[0])
        del preDict[allSum]

        for key in preDict:
            maxGap = max(maxGap,preDict[key])
        return n-maxGap
```

# [576. 出界的路径数](https://leetcode-cn.com/problems/out-of-boundary-paths/)

给你一个大小为 m x n 的网格和一个球。球的起始坐标为 [startRow, startColumn] 。你可以将球移到在四个方向上相邻的单元格内（可以穿过网格边界到达网格之外）。你 最多 可以移动 maxMove 次球。

给你五个整数 m、n、maxMove、startRow 以及 startColumn ，找出并返回可以将球移出边界的路径数量。因为答案可能非常大，返回对 109 + 7 取余 后的结果。

```python
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        # 假设用记忆化搜索，m*n表示范围,sr,sc是起点
        # 同时需要注意它的出界定义，只能在最后一步出界
        # 之后转dp

        memo = dict()
        mod = 10**9 + 7
        # 状态转移为 dfs(i,j,k) = dfs(...,...,k-1),# 四个方向
        # 基态为，初始化如下
        for i in range(m):
            for j in range(n):
                invalid = 0
                if i == 0:
                    invalid += 1
                if i == m-1:
                    invalid += 1
                if j == 0:
                    invalid += 1
                if j == n-1:
                    invalid += 1
                memo[(i,j,1)] = invalid

        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        def dfs(i,j,k): # 当前坐标点走k步的结果
            if (i,j,k) in memo:
                return memo[(i,j,k)]
            if k == 0:
                return 0
            
            temp = 0
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n:
                    temp += dfs(new_i,new_j,k-1)
            temp %= mod
            memo[(i,j,k)] = temp
            return temp 

        for t in range(1,maxMove+1):
            dfs(startRow,startColumn,t)
        # print(memo)
        ans = 0
        for t in range(1,maxMove+1):
            ans += memo.get((startRow,startColumn,t),0)
        return ans%mod
```

```python
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        # 三维dp
        # dp[i][j][k] ，基态需要考虑dp[i][j][1]
        if maxMove == 0:
            return 0
        mod = 10**9 + 7
        dp = [[[0 for t in range(maxMove+1)] for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0:
                    dp[i][j][1] += 1
                if i == m-1:
                    dp[i][j][1] += 1
                if j == 0:
                    dp[i][j][1] += 1
                if j == n-1:
                    dp[i][j][1] += 1
        # 状态转移为 dp[i][j][k] = dp[...][...][k-1]
        for t in range(2,maxMove+1):
            for i in range(m):
                for j in range(n):
                    state1 = dp[i-1][j][t-1] if i-1>=0 else 0
                    state2 = dp[i+1][j][t-1] if i+1<m else 0
                    state3 = dp[i][j-1][t-1] if j-1>=0 else 0
                    state4 = dp[i][j+1][t-1] if j+1<n else 0
                    dp[i][j][t] = (state1 + state2 + state3 + state4)%mod
        
        ans = 0
        for t in range(1,maxMove+1):
            ans += dp[startRow][startColumn][t]
        
        return ans%mod
```

# [604. 迭代压缩字符串](https://leetcode-cn.com/problems/design-compressed-string-iterator/)

对于一个压缩字符串，设计一个数据结构，它支持如下两种操作： next 和 hasNext。

给定的压缩字符串格式为：每个字母后面紧跟一个正整数，这个整数表示该字母在解压后的字符串里连续出现的次数。

next() - 如果压缩字符串仍然有字母未被解压，则返回下一个字母，否则返回一个空格。
hasNext() - 判断是否还有字母仍然没被解压。

注意：

请记得将你的类在 StringIterator 中 初始化 ，因为静态变量或类变量在多组测试数据中不会被自动清空。更多细节请访问 这里 。

```python
class StringIterator:

    def __init__(self, compressedString: str):
        self.p = 0
        self.numStack = []
        self.charStack = []
        self.n = len(compressedString)
        self.compressedString = compressedString
        # 注意可能有多个数字
        while self.p < self.n and len(self.numStack) == 0:
            if self.compressedString[self.p].isdigit() == False:
                self.charStack.append(self.compressedString[self.p])
                self.p += 1
            else:
                t = self.p 
                while t < self.n and self.compressedString[t].isdigit():
                    t += 1
                self.numStack.append(int(self.compressedString[self.p:t]))
                self.p = t 
        # 此时初始化完毕
        # print(self.numStack,self.charStack)

    def next(self) -> str:
        if self.hasNext():
            self.numStack[0] -= 1
            ans = self.charStack[-1]
            if self.numStack[0] == 0: # 继续扫
                self.numStack = []
                while self.p < self.n and len(self.numStack) == 0:
                    if self.compressedString[self.p].isdigit() == False:
                        self.charStack.append(self.compressedString[self.p])
                        self.p += 1
                    else:
                        t = self.p 
                        while t < self.n and self.compressedString[t].isdigit():
                            t += 1
                        self.numStack.append(int(self.compressedString[self.p:t]))
                        self.p = t 
            return ans 
        else:
            return " "


    def hasNext(self) -> bool:
        if self.numStack == []:
            return False 
        else:
            return True

```

# [609. 在系统中查找重复文件](https://leetcode-cn.com/problems/find-duplicate-file-in-system/)

给你一个目录信息列表 paths ，包括目录路径，以及该目录中的所有文件及其内容，请你按路径返回文件系统中的所有重复文件。答案可按 任意顺序 返回。

一组重复的文件至少包括 两个 具有完全相同内容的文件。

输入 列表中的单个目录信息字符串的格式如下：

"root/d1/d2/.../dm f1.txt(f1_content) f2.txt(f2_content) ... fn.txt(fn_content)"
这意味着，在目录 root/d1/d2/.../dm 下，有 n 个文件 ( f1.txt, f2.txt ... fn.txt ) 的内容分别是 ( f1_content, f2_content ... fn_content ) 。注意：n >= 1 且 m >= 0 。如果 m = 0 ，则表示该目录是根目录。

输出 是由 重复文件路径组 构成的列表。其中每个组由所有具有相同内容文件的文件路径组成。文件路径是具有下列格式的字符串：

"directory_path/file_name.txt"

```python
class Solution:
    def findDuplicate(self, paths: List[str]) -> List[List[str]]:
        # 目标是content<内容>相同
        contDict = collections.defaultdict(list)
        # key是
        for each in paths:
            temp = each.split()
            for content in temp[1:]:
                i = content.index("(")
                key = content[i+1:-1]
                contDict[key].append(temp[0]+"/"+content[:i])
        ans = [v for v in contDict.values() if len(v) >= 2] # 
        return ans
```



# [665. 非递减数列](https://leetcode-cn.com/problems/non-decreasing-array/)

给你一个长度为 n 的整数数组，请你判断在 最多 改变 1 个元素的情况下，该数组能否变成一个非递减数列。

我们是这样定义一个非递减数列的： 对于数组中任意的 i (0 <= i <= n-2)，总满足 nums[i] <= nums[i + 1]。

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        # 找到其最长上升子序列的数目
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
        
        longest = lengthOfLIS(nums)
        if abs(longest-len(nums)) <= 1:
            return True 
        else:
            return False
```

# [686. 重复叠加字符串匹配](https://leetcode-cn.com/problems/repeated-string-match/)

给定两个字符串 a 和 b，寻找重复叠加字符串 a 的最小次数，使得字符串 b 成为叠加后的字符串 a 的子串，如果不存在则返回 -1。

注意：字符串 "abc" 重复叠加 0 次是 ""，重复叠加 1 次是 "abc"，重复叠加 2 次是 "abcabc"。

```python
class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        if b == "":
            return 0
        if len(set(b)) > len(set(a)): # b中有a不存在的元素，直接gg
            return -1

        # 精髓：覆盖b至少需要 ceil(len(b)/len(a))个子串，最多需要ceil(len(b)/len(a))+1个子串
        cnt = 1
        while b not in cnt*a:
            cnt += 1
            if b not in cnt*a and len(a)*cnt > 2*len(b):
                return -1
        return cnt
            
```

# [688. “马”在棋盘上的概率](https://leetcode-cn.com/problems/knight-probability-in-chessboard/)

已知一个 NxN 的国际象棋棋盘，棋盘的行号和列号都是从 0 开始。即最左上角的格子记为 (0, 0)，最右下角的记为 (N-1, N-1)。 

现有一个 “马”（也译作 “骑士”）位于 (r, c) ，并打算进行 K 次移动。 

如下图所示，国际象棋的 “马” 每一步先沿水平或垂直方向移动 2 个格子，然后向与之相垂直的方向再移动 1 个格子，共有 8 个可选的位置。

```python
class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        # k步骤的所有状态数目为 8**k
        # 动态规划，
        # dp[i][j][t], t表示还剩下k次的时候在i,j的次数
        direc = [(-1,-2),(-2,-1),(-2,1),(-1,2),(1,-2),(2,-1),(2,1),(1,2)]
        
        dp = [[[0]*(k+1) for j in range(n)] for i in range(n)]
        # base dp[r][c][0] = 1
        # 状态转移 dp[i][j][t] = sum(dp[i-di[0][j-di[1][t-1]]])
        dp[row][column][0] = 1
        for t in range(1,k+1):
            for i in range(n):
                for j in range(n):
                    for di in direc:
                        new_i = i + di[0]
                        new_j = j + di[1]
                        if 0<=new_i<n and 0<=new_j<n:
                            dp[i][j][t] += dp[new_i][new_j][t-1]
        # print(dp)
        ans = 0
        for i in range(n):
            for j in range(n):
                ans += dp[i][j][k]
        
        return ans / (8**k)
```



# [721. 账户合并](https://leetcode-cn.com/problems/accounts-merge/)

给定一个列表 accounts，每个元素 accounts[i] 是一个字符串列表，其中第一个元素 accounts[i][0] 是 名称 (name)，其余元素是 emails 表示该账户的邮箱地址。

现在，我们想合并这些账户。如果两个账户都有一些共同的邮箱地址，则两个账户必定属于同一个人。请注意，即使两个账户具有相同的名称，它们也可能属于不同的人，因为人们可能具有相同的名称。一个人最初可以拥有任意数量的账户，但其所有账户都具有相同的名称。

合并账户后，按以下格式返回账户：每个账户的第一个元素是名称，其余元素是 按字符 ASCII 顺序排列 的邮箱地址。账户本身可以以 任意顺序 返回。

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
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        # 并查集
        n = len(accounts)
        ufset = UF(n)
        for i in range(n):
            for j in range(i+1,n):
                t1 = set(accounts[i][1:])
                t2 = set(accounts[j][1:])
                if len(t1&t2) != 0:
                    ufset.union(i,j)
        
        userDict = collections.defaultdict(set)
        for i in range(n):
            index = ufset.find(i)
            for e in accounts[i][1:]:
                userDict[index].add(e)
        # print(userDict)
        ans = []
        for key in userDict:
            g = list(userDict[key])
            g.sort()
            ans.append([accounts[key][0]]+ g)
        return ans

```

# [734. 句子相似性](https://leetcode-cn.com/problems/sentence-similarity/)

给定两个句子 words1, words2 （每个用字符串数组表示），和一个相似单词对的列表 pairs ，判断是否两个句子是相似的。

例如，当相似单词对是 pairs = [["great", "fine"], ["acting","drama"], ["skills","talent"]]的时候，"great acting skills" 和 "fine drama talent" 是相似的。

注意相似关系是不具有传递性的。例如，如果 "great" 和 "fine" 是相似的，"fine" 和 "good" 是相似的，但是 "great" 和 "good" 未必是相似的。

但是，相似关系是具有对称性的。例如，"great" 和 "fine" 是相似的相当于 "fine" 和 "great" 是相似的。

而且，一个单词总是与其自身相似。例如，句子 words1 = ["great"], words2 = ["great"], pairs = [] 是相似的，尽管没有输入特定的相似单词对。

最后，句子只会在具有相同单词个数的前提下才会相似。所以一个句子 words1 = ["great"] 永远不可能和句子 words2 = ["doubleplus","good"] 相似。

```python
class Solution:
    def areSentencesSimilar(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        # 暴力检查,不能使用并查集，它的描述有问题，对于句子其实还需要位置上对应
        if len(sentence1) != len(sentence2):
            return False 
        n = len(sentence1)

        pairSet = set(tuple(e) for e in similarPairs)
        for i in range(n):
            w1 = sentence1[i]
            w2 = sentence2[i]
            if w1 != w2 and (w1,w2) not in pairSet and (w2,w1) not in pairSet:
                return False 
        return True
        
```



# [737. 句子相似性 II](https://leetcode-cn.com/problems/sentence-similarity-ii/)

给定两个句子 words1, words2 （每个用字符串数组表示），和一个相似单词对的列表 pairs ，判断是否两个句子是相似的。

例如，当相似单词对是 pairs = [["great", "fine"], ["acting","drama"], ["skills","talent"]]的时候，words1 = ["great", "acting", "skills"] 和 words2 = ["fine", "drama", "talent"] 是相似的。

注意相似关系是 具有 传递性的。例如，如果 "great" 和 "fine" 是相似的，"fine" 和 "good" 是相似的，则 "great" 和 "good" 是相似的。

而且，相似关系是具有对称性的。例如，"great" 和 "fine" 是相似的相当于 "fine" 和 "great" 是相似的。

并且，一个单词总是与其自身相似。例如，句子 words1 = ["great"], words2 = ["great"], pairs = [] 是相似的，尽管没有输入特定的相似单词对。

最后，句子只会在具有相同单词个数的前提下才会相似。所以一个句子 words1 = ["great"] 永远不可能和句子 words2 = ["doubleplus","good"] 相似。

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
    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        # 并查集
        if len(sentence1) != len(sentence2):
            return False 
        indexDict = dict()
        p = 0
        for w in sentence1:
            if indexDict.get(w) == None:
                indexDict[w] = p 
                p += 1
        for w in sentence2:
            if indexDict.get(w) == None:
                indexDict[w] = p 
                p += 1 
        for t1,t2 in similarPairs:
            if indexDict.get(t1) == None:
                indexDict[t1] = p 
                p += 1  
            if indexDict.get(t2) == None:
                indexDict[t2] = p 
                p += 1

        ufset = UF(p) 
        # 须考虑pairs中出现过的不在sen1和sen2中的单词
        for x,y in similarPairs:
            ind1 = indexDict.get(x)
            ind2 = indexDict.get(y)
            if ind1 == None or ind2 == None:
                continue 
            ufset.union(ind1,ind2)

        for i in range(len(sentence1)):
            sentence1[i] = indexDict[sentence1[i]]
            sentence1[i] = ufset.find(sentence1[i])
            sentence2[i] = indexDict[sentence2[i]]
            sentence2[i] = ufset.find(sentence2[i])
        
        sentence1.sort()
        sentence2.sort()
        return sentence1 == sentence2
```

# [738. 单调递增的数字](https://leetcode-cn.com/problems/monotone-increasing-digits/)

给定一个非负整数 N，找出小于或等于 N 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。

（当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。）

```python
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        # 数字本身是单调递增，返回自身
        # 数字如果不是单调递增,找到第一个逆序的地方，尝试让它减1，如果它大于等于前面那个数，则从它之后都变成9
        # 如果它没有大于前面那个数，那游标左移
        stack = list(str(n))
        length = len(stack)
        stack = [int(e) for e in stack] 
        i = 1
        while i < length:
            if stack[i-1] > stack[i]: # 逆序,处理i-1
                p = i-1
                while p >= 0:
                    if (p-1 >= 0 and stack[p-1] <= stack[p]-1) or p == 0:
                        stack[p] -= 1
                        for t in range(p+1,length):
                            stack[t] = 9 
                        break
                    p -= 1
            else:
                pass 
            i += 1
        stack = [str(e) for e in stack]
        ans = int("".join(stack))
        return ans
```

```python
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        # 高赞解答，111...1法
        ones = 111111111
        result = 0
        for t in range(9):
            while result + ones > n:
                ones //= 10
            result += ones
        return result

```



# [754. 到达终点数字](https://leetcode-cn.com/problems/reach-a-number/)

在一根无限长的数轴上，你站在0的位置。终点在target的位置。

每次你可以选择向左或向右移动。第 n 次移动（从 1 开始），可以走 n 步。

返回到达终点需要的最小移动次数。

```python
class Solution:
    def reachNumber(self, target: int) -> int:
        # 先模拟出所有可能性
        # 
        # p = [0]
        # for i in range(10):
        #     temp = []
        #     for e in p:
        #         temp.append(e+i+1)
        #         temp.append(e-(i+1))
        #     p = temp 
        #     print(sorted(list(set(p))))
        
        # 打表发现其规律，两两一组能到达的点的范围，分别为奇、奇、偶、偶
        # 按四个一组分组：
        # n: ... n*(n+1)//2的奇数
        # n+1: ... (n+1)*(n+2)//2的奇数
        # n+2: ... (n+2)*(n+3)//2的偶数
        # n+3: ... (n+3)*(n+4)//2的偶数
        
        target = abs(target) # 转换成为正数

        for i in range(1,10000000,4): # 找到就return
            if target%2 == 0:
                if (i+3)*(i+4)//2 < target: # 不在这一轮里
                    continue 
                if target <= (i+2)*(i+3)//2:
                    return i+2
                else:
                    return i+3
            elif target%2 == 1:
                if (i+1)*(i+2)//2 < target: # 不在这一轮里
                    continue 
                if target <= i*(i+1)//2:
                    return i
                else:
                    return i+1              
```

# [764. 最大加号标志](https://leetcode-cn.com/problems/largest-plus-sign/)

在一个大小在 (0, 0) 到 (N-1, N-1) 的2D网格 grid 中，除了在 mines 中给出的单元为 0，其他每个单元都是 1。网格中包含 1 的最大的轴对齐加号标志是多少阶？返回加号标志的阶数。如果未找到加号标志，则返回 0。

一个 k" 阶由 1 组成的“轴对称”加号标志具有中心网格  grid[x][y] = 1 ，以及4个从中心向上、向下、向左、向右延伸，长度为 k-1，由 1 组成的臂。下面给出 k" 阶“轴对称”加号标志的示例。注意，只有加号标志的所有网格要求为 1，别的网格可能为 0 也可能为 1。

```python
class Solution:
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        # 动态规划，[up,left,down,right]
        dp = [[[0,0,0,0] for j in range(n)] for i in range(n)]
        grid = [[1 for i in range(n)] for j in range(n)]
        for i,j in mines:
            grid[i][j] = 0 
        mines = grid
        for i in range(n):
            for j in range(n):
                if mines[i][j] == 1:
                    # 上方继承
                    if i-1>=0:
                        dp[i][j][0] = dp[i-1][j][0] + 1
                    else:
                        dp[i][j][0] = 1
                    # 左继承
                    if j-1>=0:
                        dp[i][j][1] = dp[i][j-1][1] + 1
                    else:
                        dp[i][j][1] = 1
        # 注意这个扫描顺序
        for i in range(n-1,-1,-1):
            for j in range(n-1,-1,-1):
                if mines[i][j] == 1:
                    # 下继承
                    if i+1<n:
                        dp[i][j][2] = dp[i+1][j][2] + 1
                    else:
                        dp[i][j][2] = 1
                    # 右继承
                    if j+1<n:
                        dp[i][j][3] = dp[i][j+1][3] + 1
                    else:
                        dp[i][j][3] = 1
        
        ans = 0
        for i in range(n):
            for j in range(n):
                ans = max(ans,min(dp[i][j])) # 这里获得的是上下左右臂长度
        
        return ans
```

# [802. 找到最终的安全状态](https://leetcode-cn.com/problems/find-eventual-safe-states/)

在有向图中，以某个节点为起始节点，从该点出发，每一步沿着图中的一条有向边行走。如果到达的节点是终点（即它没有连出的有向边），则停止。

对于一个起始节点，如果从该节点出发，无论每一步选择沿哪条有向边行走，最后必然在有限步内到达终点，则将该起始节点称作是 安全 的。

返回一个由图中所有安全的起始节点组成的数组作为答案。答案数组中的元素应当按 升序 排列。

该有向图有 n 个节点，按 0 到 n - 1 编号，其中 n 是 graph 的节点数。图以下述形式给出：graph[i] 是编号 j 节点的一个列表，满足 (i, j) 是图的一条有向边。

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        # dfs标准版本
        n = len(graph)
        visited = [0 for i in range(n)] # 初始化为0表示未访问，1为正在访问，2为安全

        # 递归解决
        # 如果这个点被访问过了，则判断它是不是安全的
        def dfs(node):
            if visited[node] != 0:
                return visited[node] == 2  # 返回True表示安全
            visited[node] = 1
            for neigh in graph[node]:
                if dfs(neigh) == False: # 只要其中一条有问题，就都有问题
                    return False
            visited[node] = 2
            return True 
        
        for i in range(n):
            dfs(i)
        
        ans = []
        for i in range(n):
            if visited[i] == 2:
                ans.append(i)
        return ans 
```

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        # 拓扑排序，
        # 原先所有出度为0的点一定是安全点
        # 所有单方向到达出度为0的点也是安全点
        # 反向图之后，所有入度为0的点一定是安全点
        # 所有可以被拓扑排序的点一定曾经是入度为0，那么加入结果集

        regraph = collections.defaultdict(list)
        n = len(graph)
        # 构建反向图
        for i in range(n):
            for each in graph[i]:
                regraph[each].append(i)

        # 在反图上进行拓扑排序
        # 找入度为0的点
        # print(regraph)
        # i -> regraph[i]
        inDegree = [0 for i in range(n)]
        for i in range(n):
            for each in regraph[i]:
                inDegree[each] += 1
        
        # print("inDegree",inDegree)
        queue = []
        for i in range(n):
            if inDegree[i] == 0:
                queue.append(i)
        ans = []
        while queue:
            new_queue = []
            for node in queue:
                ans.append(node)
                for neigh in regraph[node]:
                    inDegree[neigh] -= 1
                    if inDegree[neigh] == 0:
                        new_queue.append(neigh)
            queue = new_queue 
        ans.sort()
        return ans 
```



# [825. 适龄的朋友](https://leetcode-cn.com/problems/friends-of-appropriate-ages/)

在社交媒体网站上有 n 个用户。给你一个整数数组 ages ，其中 ages[i] 是第 i 个用户的年龄。

如果下述任意一个条件为真，那么用户 x 将不会向用户 y（x != y）发送好友请求：

age[y] <= 0.5 * age[x] + 7
age[y] > age[x]
age[y] > 100 && age[x] < 100
否则，x 将会向 y 发送一条好友请求。

注意，如果 x 向 y 发送一条好友请求，y 不必也向 x 发送一条好友请求。另外，用户不会向自己发送好友请求。

返回在该社交媒体网站上产生的好友请求总数。

```python
class Solution:
    def numFriendRequests(self, ages: List[int]) -> int:
        # 注意这个条件，当2成立的时候3一定成立
        # x会向y发送请求的条件是
        #  0.5 * ages[x] + 7 < ages[y] <= age[x]
        # 14岁以及以下被ban了
        cnt = 0
        ages.sort() # 预先排序
        n = len(ages)
        for i in range(n):
            if ages[i] <= 14:
                continue
            # 二分搜
            left = bisect.bisect_right(ages,0.5*ages[i]+7) # 比这个大的第一个索引
            right = bisect.bisect_right(ages,ages[i])-1 # 到ages[i]的闭区间
            cnt += right-left # 在这个区间内的都符合，注意每个人都需要排除自身
        return cnt
```

# [842. 将数组拆分成斐波那契序列](https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/)

给定一个数字字符串 S，比如 S = "123456579"，我们可以将它分成斐波那契式的序列 [123, 456, 579]。

形式上，斐波那契式序列是一个非负整数列表 F，且满足：

0 <= F[i] <= 2^31 - 1，（也就是说，每个整数都符合 32 位有符号整数类型）；
F.length >= 3；
对于所有的0 <= i < F.length - 2，都有 F[i] + F[i+1] = F[i+2] 成立。
另外，请注意，将字符串拆分成小块时，每个块的数字一定不要以零开头，除非这个块是数字 0 本身。

返回从 S 拆分出来的任意一组斐波那契式的序列块，如果不能拆分则返回 []。

```python
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        # 暴力拆分+判断,python天然支持大数
        state = False 
        memo = []
        def dfs(num,firstnum,secondnum): # num是待切割的字符串
            nonlocal state
            if num == "": 
                state = True
                memo.append(firstnum)
                memo.append(secondnum)
                return True 
            total = firstnum + secondnum
            length = len(str(total))
            if str(total) == num[:length]: # 表示长度足够
                return dfs(num[length:],secondnum,total)
            return False 
        
        for i in range(1,len(num)-1): # 枚举分割点， 。。。。i....j.....然后递归检查
            if num[0] == '0' and i > 1: break 
            for j in range(i+1,len(num)):
                if j-i > 1 and num[i] == '0': break  # 第二个数不能是0开头的非0数
                if not state:
                    dfs(num[j:],int(num[:i]),int(num[i:j]))
                    
        if len(memo) == 0:
            return []
        # 否则根据memo，然后倒着得到全长，终止条件为计数全场为n
        n = len(num)
        nowLength = len(str(memo[0])) + len(str(memo[1]))
        # a,b,c迭代计算
        c = memo[1]
        b = memo[0]
        ans = [memo[1],memo[0]]
        while nowLength != n:
            # print(ans,nowLength,n)
            a = c-b
            ans.append(a)
            nowLength += int(len(str(a)))
            c = b 
            b = a 
        
        # 需要检查是否每个数爆int
        limit = 2**31 - 1
        for e in ans:
            if e > limit:
                return []
        return ans[::-1]
```

```python
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
    	# 回溯法
        ans = []
        n = len(num)

        def backtracking(index):
            if index == n and len(ans) >= 3:
                return True 
            for i in range(index,n):
                if i > index and num[index] == '0':
                    break 
                now = int(num[index:i+1])
                # if len(str(now)) != len(num[index:i+1]): # 判断是否前导0
                #     break
                if now > 2**31 - 1:
                    break 
                if len(ans) < 2 or now == ans[-2] + ans[-1]:
                    ans.append(now)
                    if backtracking(i+1):
                        return True
                    ans.pop()
                elif len(ans) > 2 and now > ans[-2] + ans[-1]:
                    return False
            return False 
        
        backtracking(0)
        return ans

```



# [851. 喧闹和富有](https://leetcode-cn.com/problems/loud-and-rich/)

有一组 n 个人作为实验对象，从 0 到 n - 1 编号，其中每个人都有不同数目的钱，以及不同程度的安静值（quietness）。为了方便起见，我们将编号为 x 的人简称为 "person x "。

给你一个数组 richer ，其中 richer[i] = [ai, bi] 表示 person ai 比 person bi 更有钱。另给你一个整数数组 quiet ，其中 quiet[i] 是 person i 的安静值。richer 中所给出的数据 逻辑自恰（也就是说，在 person x 比 person y 更有钱的同时，不会出现 person y 比 person x 更有钱的情况 ）。

现在，返回一个整数数组 answer 作为答案，其中 answer[x] = y 的前提是，在所有拥有的钱肯定不少于 person x 的人中，person y 是最安静的人（也就是安静值 quiet[y] 最小的人）。

```python
class Solution:
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
        n = len(quiet)
        graph = collections.defaultdict(list) 
        inDegree = [0 for j in range(n)]
        # richer逻辑自洽，不存在环
        for a,b in richer:
            graph[a].append(b)
            inDegree[b] += 1
        # quiet的所有值互不相同
        
        # 对每个点用dfs
        ans = [i for i in range(n)]

        recordDict = dict() # 根据安静值查人
        for i in range(len(quiet)):
            recordDict[quiet[i]] = i 

        memo = dict()
        def dfs(i,tempMin): # tempMin是当前的最小安静值
            if (i,tempMin) in memo:
                return 
            tempMin = min(tempMin,quiet[i])
            if quiet[ans[i]] > tempMin: # 如果记录的那个的人的安静值大，则需要更新它
                ans[i] = recordDict[tempMin]

            for t in range(tempMin,n): # 比他闹的都全部标记为无须再次访问
                memo[(i,t)] = True # 标记为已经访问过

            for neigh in graph[i]:
                dfs(neigh,tempMin)
        
        queue = []
        for i in range(n):
            if inDegree[i] == 0:
                queue.append(i)
        # print(queue)
        for i in queue: # 只对所有入度为0的进行更新
            dfs(i,quiet[i])

        return ans
```

```python
class Solution:
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
        # 拓扑排序
        n = len(quiet)
        graph = collections.defaultdict(list)
        inDegree = [0 for i in range(n)]
        
        for a,b in richer:
            graph[a].append(b)
            inDegree[b] += 1
        
        # 对所有邻居更新时，复用自身对结果
        # bfs拓扑排序
        queue = []
        for i in range(n):
            if inDegree[i] == 0:
                queue.append(i)

        ans = [i for i in range(n)]
        while len(queue) != 0:
            new_queue = []
            for node in queue:
                for neigh in graph[node]:
                    if quiet[ans[neigh]] > quiet[ans[node]]: # 注意这一行，一种递归思想
                        ans[neigh] = ans[node]
                    inDegree[neigh] -= 1
                    if inDegree[neigh] == 0:
                        new_queue.append(neigh)
            queue = new_queue

        return ans
```

```python
class Solution:
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
        n = len(quiet)
        graph = collections.defaultdict(list)
        for a,b in richer:
            graph[b].append(a)
        
        ans = [-1 for i in range(n)]

        def dfs(i):
            if ans[i] != -1:
                return 
            ans[i] = i
            for neigh in graph[i]:
                dfs(neigh)
                if quiet[ans[i]] > quiet[ans[neigh]]:
                    ans[i] = ans[neigh]
                # dfs必须写在更新前

        
        for i in range(n):
            dfs(i)
        return ans
```

# [873. 最长的斐波那契子序列的长度](https://leetcode-cn.com/problems/length-of-longest-fibonacci-subsequence/)

如果序列 X_1, X_2, ..., X_n 满足下列条件，就说它是 斐波那契式 的：

n >= 3
对于所有 i + 2 <= n，都有 X_i + X_{i+1} = X_{i+2}
给定一个严格递增的正整数数组形成序列 arr ，找到 arr 中最长的斐波那契式的子序列的长度。如果一个不存在，返回  0 。

（回想一下，子序列是从原序列 arr 中派生出来的，它从 arr 中删掉任意数量的元素（也可以不删），而不改变其余元素的顺序。例如， [3, 5, 8] 是 [3, 4, 5, 6, 7, 8] 的一个子序列）

```python
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        # 这一题使用dp，
        # dp[i][j]的含义是 以arr[i],arr[j]结尾的最长序列的长度
        # 状态转移为 dp[i][j] = dp[k][i] + 1 ; 其中arr[k] = arr[j] - arr[i]
        # 为了快速找到kk，使用indexDict
        n = len(arr)
        indexDict = {arr[i]:i for i in range(n)}
        dp = [[2 for j in range(n)] for i in range(n)] # 初始化有效值为2

        ans = 0
        for j in range(n):
            for i in range(j):
                if indexDict.get(arr[j]-arr[i]) != None:
                    k = indexDict.get(arr[j]-arr[i])
                    if k < i:# 注意，不能相等，因为如果是 4，8这样的它还会找到重复的4
                        dp[i][j] = dp[k][i] + 1
                        ans = max(ans,dp[i][j])
        if ans <= 2:
            return 0
        else:
            return ans
```

# [883. 三维形体投影面积](https://leetcode-cn.com/problems/projection-area-of-3d-shapes/)

在 N * N 的网格中，我们放置了一些与 x，y，z 三轴对齐的 1 * 1 * 1 立方体。

每个值 v = grid[i][j] 表示 v 个正方体叠放在单元格 (i, j) 上。

现在，我们查看这些立方体在 xy、yz 和 zx 平面上的投影。

投影就像影子，将三维形体映射到一个二维平面上。

在这里，从顶部、前面和侧面看立方体时，我们会看到“影子”。

返回所有三个投影的总面积。

```python
class Solution:
    def projectionArea(self, grid: List[List[int]]) -> int:
        # 这一题限制长宽相等
        n = len(grid)
        # 计算每横行最大值，每纵列最大值，加上俯视图，俯视图不是n**2，而是实际不为0的块的数量
        view1 = 0
        for line in grid:
            view1 += max(line) # 
        view2 = 0
        for j in range(n):
            temp = []
            for i in range(n):
                temp.append(grid[i][j])
            view2 += max(temp)
        view3 = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] != 0:
                    view3 += 1
        return view1 + view2 + view3
```

# [892. 三维形体的表面积](https://leetcode-cn.com/problems/surface-area-of-3d-shapes/)

给你一个 n * n 的网格 grid ，上面放置着一些 1 x 1 x 1 的正方体。

每个值 v = grid[i][j] 表示 v 个正方体叠放在对应单元格 (i, j) 上。

放置好正方体后，任何直接相邻的正方体都会互相粘在一起，形成一些不规则的三维形体。

请你返回最终这些形体的总表面积。

注意：每个形体的底面也需要计入表面积中。

```python
class Solution:
    def surfaceArea(self, grid: List[List[int]]) -> int:
        # 先计算立方体的数目
        # 然后计算每两列相邻的，取较矮的,他们的相接触的面积减去，为了防止重复，只需要每个对邻居都只需要减一次
        # 而对于高度不为1的，需要减去(heigh-1)*2
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        n = len(grid)
        count = 0
        extra = 0
        for i in range(n):
            for j in range(n):
                count += grid[i][j]
                if grid[i][j] >= 1:
                    extra += (grid[i][j]-1)*2
        area = 6 * count
        area -= extra

        for i in range(n):
            for j in range(n):
                for di in direc:
                    neigh_i = i + di[0]
                    neigh_j = j + di[1]
                    if 0<=neigh_i<n and 0<=neigh_j<n:
                        neigh_Hight = grid[neigh_i][neigh_j]
                        area -= min(grid[i][j],neigh_Hight)
        return area
```



# [916. 单词子集](https://leetcode-cn.com/problems/word-subsets/)

给你两个字符串数组 words1 和 words2。

现在，如果 b 中的每个字母都出现在 a 中，包括重复出现的字母，那么称字符串 b 是字符串 a 的 子集 。

例如，"wrr" 是 "warrior" 的子集，但不是 "world" 的子集。
如果对 words2 中的每一个单词 b，b 都是 a 的子集，那么我们称 words1 中的单词 a 是 通用单词 。

以数组形式返回 words1 中所有的通用单词。你可以按 任意顺序 返回答案。

```python
class Solution:
    def wordSubsets(self, words1: List[str], words2: List[str]) -> List[str]:
        # 对word2进行预处理，统计完每个字母之后都取最高值
        maxMemo = collections.defaultdict(int)
        for w2 in words2:
            tempMemo = collections.defaultdict(int)
            for ch in w2:
                tempMemo[ch] += 1
            for key in tempMemo:
                maxMemo[key] = max(maxMemo[key],tempMemo[key])
        
        ans = []

        for w1 in words1:
            tempMemo = collections.defaultdict(int)
            for ch in w1:
                tempMemo[ch] += 1
            state = True
            for key in maxMemo:
                if tempMemo[key] < maxMemo[key]:
                    state = False
                    break
            if state:
                ans.append(w1)
        return ans
```

# [925. 长按键入](https://leetcode-cn.com/problems/long-pressed-name/)

你的朋友正在使用键盘输入他的名字 name。偶尔，在键入字符 c 时，按键可能会被长按，而字符可能被输入 1 次或多次。

你将会检查键盘输入的字符 typed。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按），那么就返回 True。

```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        # 双指针检查，当name的下一个字符和这一个一样/不一样时分情况讨论
        p1 = 0
        p2 = 0
        n1 = len(name)
        n2 = len(typed)
        while p1 < n1 and p2 < n2:
            if p1 + 1 < n1:
                if name[p1] != name[p1+1]:
                    state = False 
                    while p2 < n2 and name[p1] == typed[p2] :
                        state = True
                        p2 += 1
                    p1 += 1
                    if not state:
                        return False 
                elif name[p1] == name[p1+1]:
                    state = False
                    if p2 < n2 and name[p1] == typed[p2]:
                        p1 += 1
                        p2 += 1
                        state = True
                    if not state:
                        return False 
            elif p1 < n1:
                state = False
                while p2 < n2 and name[p1] == typed[p2]:
                    state = True
                    p2 += 1
                p1 += 1
                if not state:
                    return False 
        
        return p1 == n1 and p2 == n2

```

```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        # 双指针检查，简洁版本
        # 每一个typed中的字符只有两种用途，一种是匹配，一种是重复前一个
        # 如果不符合这两种用途，gg
        i,j = 0,0
        n1 = len(name)
        n2 = len(typed)
        while j < n2:
            if i < n1 and name[i] == typed[j]:
                i += 1
                j += 1
            elif j >= 1 and typed[j] == typed[j-1]:
                j += 1
            else:
                return False 
        return i == n1
```

# [935. 骑士拨号器](https://leetcode-cn.com/problems/knight-dialer/)

国际象棋中的骑士可以按下图所示进行移动：


这一次，我们将 “骑士” 放在电话拨号盘的任意数字键（如上图所示）上，接下来，骑士将会跳 N-1 步。每一步必须是从一个数字键跳到另一个数字键。

每当它落在一个键上（包括骑士的初始位置），都会拨出键所对应的数字，总共按下 N 位数字。

你能用这种方式拨出多少个不同的号码？

因为答案可能很大，所以输出答案模 10^9 + 7。

```python
class Solution:
    def knightDialer(self, n: int) -> int:
        # 写好neighbor表
        mod = 10**9 + 7
        neighbor = {
            1:[6,8],
            2:[7,9],
            3:[4,8],
            4:[3,9,0],
            5:[],
            6:[1,7,0],
            7:[2,6],
            8:[1,3],
            9:[2,4],
            0:[4,6]
        }

        pre = [1 for i in range(10)]
        # 状态转移，
        for i in range(n-1):
            temp = [0 for i in range(10)] # 记录本轮dp
            for i in range(10):
                for each in neighbor[i]:
                    temp[i] += pre[each]
                    temp[i] %= mod
            pre = temp
        return sum(pre)%mod

```

# [959. 由斜杠划分区域](https://leetcode-cn.com/problems/regions-cut-by-slashes/)

在由 1 x 1 方格组成的 N x N 网格 grid 中，每个 1 x 1 方块由 /、\ 或空格构成。这些字符会将方块划分为一些共边的区域。

（请注意，反斜杠字符是转义的，因此 \ 用 "\\" 表示。）。

返回区域的数目。

```python
class UF:
    def __init__(self,size):
        self.root = [i for i in range(size)]
        self.cnt = size # 计算联通分量的数量   
    
    def find(self,x):
        while x != self.root[x]:
            x = self.root[x]
        return x 
    
    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootX] = rootY
            self.cnt -= 1
    
    
class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        # 把方格画成x格，从12点开始顺时针编号,x,x+1,x+2,x+3
        # 如果是/,内连接x,x+3 ; x+1,x+2;
        # 如果是\，内连接x,x+1; x+2;x+3;
        # 然后再每个区域上下左右链接
        
        n = len(grid)

        ufset = UF(n*n*4)
        # print(ufset.root)
        for i in range(n):
            for j in range(n):
                x = 4*(i*n+j) # 作为起始编号x 
                index = i*n+j
                if grid[i][j] == '/':
                    ufset.union(x,x+3)
                    ufset.union(x+1,x+2)
                elif grid[i][j] == '\\':
                    ufset.union(x,x+1)
                    ufset.union(x+2,x+3)
                else:
                    ufset.union(x,x+1)
                    ufset.union(x+1,x+2)
                    ufset.union(x+2,x+3)
                upIndex = (index-n)*4+2 if i > 0 else None
                donwIndex = (index+n)*4 if i+1 < n else None
                leftIndex = (index-1)*4+1 if j > 0 else None 
                rightIndex = (index+1)*4+3 if j+1 < n else None 
                lst = [upIndex,donwIndex,leftIndex,rightIndex]

                if upIndex:
                    ufset.union(x,upIndex)
                if rightIndex:
                    ufset.union(x+1,rightIndex)
                if donwIndex:
                    ufset.union(x+2,donwIndex)
                if leftIndex:
                    ufset.union(x+3,leftIndex)
        
        return ufset.cnt
```



# [967. 连续差相同的数字](https://leetcode-cn.com/problems/numbers-with-same-consecutive-differences/)

返回所有长度为 n 且满足其每两个连续位上的数字之间的差的绝对值为 k 的 非负整数 。

请注意，除了 数字 0 本身之外，答案中的每个数字都 不能 有前导零。例如，01 有一个前导零，所以是无效的；但 0 是有效的。

你可以按 任何顺序 返回答案。

```python
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        # 过滤掉前导0，进行回溯
        ans = []
        def backtracking(path):
            if len(path) == n:
                temp = [str(e) for e in path]
                ans.append("".join(temp))
                return 
            if len(path) == 0:
                for i in range(0,10):
                    path.append(i)
                    backtracking(path)
                    path.pop()
            elif len(path) != 0:
                pre = path[-1]
                if 0<=pre+k<=9:
                    path.append(pre+k)
                    backtracking(path)
                    path.pop()
                if 0<=pre-k<=9:
                    path.append(pre-k)
                    backtracking(path)
                    path.pop()

        backtracking([])   

        final = []
        # print(ans)
        for e in ans:
            if len(str(int(e))) == len(e): # 注意这个判断逻辑，去除前导0
                final.append(int(e))
        final = list(set(final)) # 注意这个去重复逻辑
        return final           
```



# [976. 三角形的最大周长](https://leetcode-cn.com/problems/largest-perimeter-triangle/)

给定由一些正数（代表长度）组成的数组 `A`，返回由其中三个长度组成的、**面积不为零**的三角形的最大周长。

如果不能形成任何面积不为零的三角形，返回 `0`。

```python
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # 假设三条边已经排序，那么充要条件
        # a+b > c, 其中c是最大边
        nums.sort(reverse = True) # 倒序，方便枚举
        n = len(nums)
        # 窗口枚举从大到小，三个三个枚举即可
        pa,pb,pc = 0,1,2
        while pc < n:
            if nums[pa] < nums[pb]+nums[pc]:
                return nums[pa]+nums[pb]+nums[pc]
            pa += 1
            pb += 1
            pc += 1
        return 0
```

# [983. 最低票价](https://leetcode-cn.com/problems/minimum-cost-for-tickets/)

在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。在接下来的一年里，你要旅行的日子将以一个名为 days 的数组给出。每一项是一个从 1 到 365 的整数。

火车票有三种不同的销售方式：

一张为期一天的通行证售价为 costs[0] 美元；
一张为期七天的通行证售价为 costs[1] 美元；
一张为期三十天的通行证售价为 costs[2] 美元。
通行证允许数天无限制的旅行。 例如，如果我们在第 2 天获得一张为期 7 天的通行证，那么我们可以连着旅行 7 天：第 2 天、第 3 天、第 4 天、第 5 天、第 6 天、第 7 天和第 8 天。

返回你想要完成在给定的列表 days 中列出的每一天的旅行所需要的最低消费。

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        # 为了方便边界处理，在days前面加上一个-100
        days = [-100] + days
        # 由于日期有序，所以可以使用二分查找索引，找到该接在哪一天的后面算钱
        # dp[i]表示到以days[i]为止的最低消费
        dp = [0xffffffff for i in range(len(days))]
        dp[0] = 0 # base

        n = len(days)
        for i in range(1,n): # 使用bisect_left
            index1 = bisect.bisect_left(days,days[i]-1)
            # 数字不相等则往前退一格
            if days[index1] != days[i]-1:
                index1 -= 1 

            index7 = bisect.bisect_left(days,days[i]-7)
            if days[index7] != days[i]-7:
                index7 -= 1

            index30 = bisect.bisect_left(days,days[i]-30)
            if days[index30] != days[i]-30:
                index30 -= 1
            dp[i] = min(dp[index1]+costs[0],dp[index7]+costs[1],dp[index30]+costs[2])
        return dp[-1]
```

# [984. 不含 AAA 或 BBB 的字符串](https://leetcode-cn.com/problems/string-without-aaa-or-bbb/)

给定两个整数 A 和 B，返回任意字符串 S，要求满足：

S 的长度为 A + B，且正好包含 A 个 'a' 字母与 B 个 'b' 字母；
子串 'aaa' 没有出现在 S 中；
子串 'bbb' 没有出现在 S 中。

```python
class Solution:
    def strWithout3a3b(self, a: int, b: int) -> str:
        # 给出可行解，使用贪心
        # 先假设A大于等于B
        symbol1 = 'a'
        symbol2 = 'b'
        if a < b:
            symbol1,symbol2 = symbol2,symbol1
            a,b = b,a 
        stack = []
        while a > b and a - 2 >= 0 and b - 1 >= 0:
            stack.append(symbol1*2)
            stack.append(symbol2)
            a -= 2
            b -= 1
        # a = 0,0,1,2,1
        # b = 1,2,0,0,1
        limit = min(a,b)
        for i in range(limit):
            stack.append(symbol1+symbol2)
        a -= limit
        b -= limit
        if a == 0 and b == 1:
            stack.append(symbol2)
        elif a == 0 and b == 2:
            stack = [symbol2*2] + stack
        elif a == 1 and b == 0:
            stack.append(symbol1)
        elif a == 2 and b == 0:
            stack.append(symbol1*2)
        return "".join(stack)
            
        
```



# [985. 查询后的偶数和](https://leetcode-cn.com/problems/sum-of-even-numbers-after-queries/)

给出一个整数数组 A 和一个查询数组 queries。

对于第 i 次查询，有 val = queries[i][0], index = queries[i][1]，我们会把 val 加到 A[index] 上。然后，第 i 次查询的答案是 A 中偶数值的和。

（此处给定的 index = queries[i][1] 是从 0 开始的索引，每次查询都会永久修改数组 A。）

返回所有查询的答案。你的答案应当以数组 answer 给出，answer[i] 为第 i 次查询的答案。

```python
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # 模拟，注意效率，不能尬模
        # 预先求出所有偶数和
        EvenSum = 0
        for e in nums:
            if e%2 == 0:
                EvenSum += e
        n = len(nums)
        ans = [0 for i in range(n)]

        # 看加入的数是奇数还是偶数，再看被加的数是奇数还是偶数
        for i in range(n):
            val,index = queries[i]
            origin = nums[index]
            if val % 2 == 0 and origin % 2 == 0:
                EvenSum += val
                nums[index] += val 
                ans[i] = EvenSum
            elif val % 2 == 0 and origin % 2 == 1: # 加入的是偶数
                # EvenSum = EvenSum # 不变
                nums[index] += val 
                ans[i] = EvenSum 
            elif val % 2 == 1 and origin % 2 == 0:
                EvenSum -= origin # 丢掉原值
                nums[index] += val 
                ans[i] = EvenSum 
            elif val % 2 == 1 and origin % 2 == 1:
                EvenSum += val + origin
                nums[index] += val 
                ans[i] = EvenSum
        return ans          
```

# [1023. 驼峰式匹配](https://leetcode-cn.com/problems/camelcase-matching/)

如果我们可以将小写字母插入模式串 pattern 得到待查询项 query，那么待查询项与给定模式串匹配。（我们可以在任何位置插入每个字符，也可以插入 0 个字符。）

给定待查询列表 queries，和模式串 pattern，返回由布尔值组成的答案列表 answer。只有在待查项 queries[i] 与模式串 pattern 匹配时， answer[i] 才为 true，否则为 false。

```python
class Solution:
    def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:
        # 指针匹配，分析queris中的每个字符会被如何使用
        # 当它是大写字符的时候，尝试匹配pattern,必须匹配
        # 当它是小写字符的时候，尝试匹配pattern,可能匹配
        def subMethod(q,pattern):
            n1,n2 = len(q),len(pattern)
            i,j = 0,0
            # while匹配q
            while i < n1:
                if q[i].upper() == q[i]: # 大写 
                    if j < n2 and q[i] == pattern[j]:
                        i += 1
                        j += 1
                    else:
                        return False 
                elif q[i].lower() == q[i]: # 小写
                    if j < n2 and q[i] == pattern[j]:
                        i += 1
                        j += 1
                    else:
                        i += 1
            return j == n2 
        
        ans = []
        for q in queries:
            t = subMethod(q,pattern)
            ans.append(t)
        return ans

```

# [1129. 颜色交替的最短路径](https://leetcode-cn.com/problems/shortest-path-with-alternating-colors/)

在一个有向图中，节点分别标记为 0, 1, ..., n-1。这个图中的每条边不是红色就是蓝色，且存在自环或平行边。

red_edges 中的每一个 [i, j] 对表示从节点 i 到节点 j 的红色有向边。类似地，blue_edges 中的每一个 [i, j] 对表示从节点 i 到节点 j 的蓝色有向边。

返回长度为 n 的数组 answer，其中 answer[X] 是从节点 0 到节点 X 的红色边和蓝色边交替出现的最短路径的长度。如果不存在这样的路径，那么 answer[x] = -1。

```python
class Solution:
    def shortestAlternatingPaths(self, n: int, red_edges: List[List[int]], blue_edges: List[List[int]]) -> List[int]:
        # 存在自环，存在平行边
        dist = [0xffffffff for i in range(n)]
        dist[0] = 0 
        steps = 0
        queue = [(0,0),(0,1)] # (node,flag) 
        # 0，这一步走的蓝边，下一步走红边，
        # 1，这一步走的红边，下一步走蓝边
        redVisited = [False for i in range(n)]
        blueVisited = [False for i in range(n)]
        redGraph = collections.defaultdict(list)
        for i,j in red_edges:
            redGraph[i].append(j)
        blueGraph = collections.defaultdict(list)
        for i,j in blue_edges:
            blueGraph[i].append(j)
        while queue:
            new_queue = []
            for node,flag in queue:
                dist[node] = min(dist[node],steps)
                if flag == 0:
                    blueVisited[node] = True
                    for neigh in redGraph[node]:
                        if redVisited[neigh] == False:
                            redVisited[neigh] = True 
                            new_queue.append([neigh,1])
                else:
                    redVisited[node] = True 
                    for neigh in blueGraph[node]:
                        if blueVisited[neigh] == False:
                            blueVisited[neigh] = True 
                            new_queue.append([neigh,0])
            steps += 1
            queue = new_queue
        
        for i in range(n):
            if dist[i] == 0xffffffff:
                dist[i] = -1
        return dist
```



# [1171. 从链表中删去总和值为零的连续节点](https://leetcode-cn.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/)

给你一个链表的头节点 head，请你编写代码，反复删去链表中由 总和 值为 0 的连续节点组成的序列，直到不存在这样的序列为止。

删除完毕后，请你返回最终结果链表的头节点。

 

你可以返回任何满足题目要求的答案。

（注意，下面示例中的所有序列，都是对 ListNode 对象序列化的表示。）

```python
class Solution:
    def removeZeroSumSublists(self, head: ListNode) -> ListNode:
        # 链表转数组,再转成链表
        if head == None:
            return head
        lst = []
        cur = head
        while cur:
            lst.append(cur.val)
            cur = cur.next 
        indexDict = dict() # 前缀和记录当前总和,遇见相同值的时候加入待选集合，提升性能选择最后一次加入
        tempChoice = []
        pre = 0
        indexDict[0] = -1
        p = 0
        n = len(lst)
        while p < n:
            pre += lst[p]
            if indexDict.get(pre) == None:
                indexDict[pre] = p 
            elif indexDict.get(pre) != None:
                tempChoice.append((indexDict[pre]+1,p,p-indexDict[pre]-1)) # 第三个参数代表跨度
            p += 1
        # 如果tempChoice不等于[]那么选取其中跨度最长的,中间的点全部删除
        tempChoice.sort(key = lambda x:-x[2])
        state = False # False表示未终止
        if len(tempChoice) == 0: # 没有要删除的 返回链表即可
            state = True 
            return head 
        else:
            # 为了逻辑方便，直接再构建一个数组
            # tempChoice的是闭区间舍弃
            start = tempChoice[0][0]
            end = tempChoice[0][1]
            newLst = lst[:start]+lst[end+1:]
            dummy = ListNode()
            cur = dummy
            p = 0
            n = len(newLst)
            while p < n:
                newNode = ListNode(newLst[p])
                cur.next = newNode
                cur = cur.next
                p += 1
            return self.removeZeroSumSublists(dummy.next) # 递归处理链表
```



# [1238. 循环码排列](https://leetcode-cn.com/problems/circular-permutation-in-binary-representation/)

给你两个整数 n 和 start。你的任务是返回任意 (0,1,2,,...,2^n-1) 的排列 p，并且满足：

p[0] = start
p[i] 和 p[i+1] 的二进制表示形式只有一位不同
p[0] 和 p[2^n -1] 的二进制表示形式也只有一位不同

```python
class Solution:
    def circularPermutation(self, n: int, start: int) -> List[int]:
        # 以0,1为base制造格雷码，然后找到码头重新切割排列
        def method(n):
            if n == 1:
                return ['0','1']
            temp = method(n-1)
            ans = []
            for g in temp:
                ans.append('0'+g)
            for g in temp[::-1]:
                ans.append('1'+g)
            return ans 
        
        def bin_to_int(lst):
            final = []
            for e in lst:
                e = e[::-1]
                v = 0
                for i in range(len(e)):
                    v += int(e[i]) * (2**i)
                final.append(v)
            return final 
        
        lst = method(n)
        lst = bin_to_int(lst)
        
        for i,number in enumerate(lst):
            if number == start:
                break 
        
        final = lst[i:] + lst[:i]
        return final
```

# [1252. 奇数值单元格的数目](https://leetcode-cn.com/problems/cells-with-odd-values-in-a-matrix/)

给你一个 m x n 的矩阵，最开始的时候，每个单元格中的值都是 0。

另有一个二维索引数组 indices，indices[i] = [ri, ci] 指向矩阵中的某个位置，其中 ri 和 ci 分别表示指定的行和列（从 0 开始编号）。

对 indices[i] 所指向的每个位置，应同时执行下述增量操作：

ri 行上的所有单元格，加 1 。
ci 列上的所有单元格，加 1 。
给你 m、n 和 indices 。请你在执行完所有 indices 指定的增量操作后，返回矩阵中 奇数值单元格 的数目。

```python
class Solution:
    def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
        # 方案1:纯模拟
        grid = [[0 for j in range(n)] for i in range(m)]

        for r,c in indices:
            for i in range(len(grid[r])):
                grid[r][i] += 1
            for i in range(m):
                grid[i][c] += 1
        
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]%2 == 1:
                    ans += 1
        return ans
```

```python
class Solution:
    def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
        # 方案，记录每行，每列的增加
        # 只有当行增量+列增量为奇数的时候，它才是奇数
        rowCnt = [0 for i in range(m)]
        colCnt = [0 for i in range(n)]
        cnt = 0
        for ri,ci in indices:
            rowCnt[ri] += 1
            colCnt[ci] += 1

        for i in range(m):
            for j in range(n):
                if (rowCnt[i] + colCnt[j])%2 == 1:
                    cnt += 1
            
        return cnt 
```

# [1130. 叶值的最小代价生成树](https://leetcode-cn.com/problems/minimum-cost-tree-from-leaf-values/) 【需要优化解法，使用区间dp】

给你一个正整数数组 arr，考虑所有满足以下条件的二叉树：

每个节点都有 0 个或是 2 个子节点。
数组 arr 中的值与树的中序遍历中每个叶节点的值一一对应。（知识回顾：如果一个节点有 0 个子节点，那么该节点为叶节点。）
每个非叶节点的值等于其左子树和右子树中叶节点的最大值的乘积。
在所有这样的二叉树中，返回每个非叶节点的值的最小可能总和。这个和的值是一个 32 位整数。

```python
class Solution:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        # 方法1: 纯暴力解法，贪心每次找到最小值，删去他，结果加上它左边或者右边的最小值
        # 注意，节点可能有重复值
        ans = 0
        while len(arr) != 1:
            minVal = min(arr)
            indexGroup = []
            for i,val in enumerate(arr):
                if arr[i] == minVal:
                    indexGroup.append(i)
            deleteIndex = indexGroup[0]
            left = arr[deleteIndex-1] if deleteIndex >= 1 else 0xffffffff
            right = arr[deleteIndex+1] if deleteIndex+1 < len(arr) else 0xffffffff
            minMult = arr[deleteIndex]*min(left,right)
            for index in indexGroup:
                left = arr[index-1] if index >= 1 else 0xffffffff
                right = arr[index+1] if index+1 < len(arr) else 0xffffffff     
                tempMinMult = arr[index]*min(left,right)       
                if tempMinMult < minMult:
                    deleteIndex = index 
                    minMult = tempMinMult
            ans += minMult
            arr.pop(deleteIndex)
        return ans
```

# [1151. 最少交换次数来组合所有的 1](https://leetcode-cn.com/problems/minimum-swaps-to-group-all-1s-together/)

给出一个二进制数组 `data`，你需要通过交换位置，将数组中 **任何位置**上的 1 组合到一起，并返回所有可能中所需 **最少的交换次数**。

```python
class Solution:
    def minSwaps(self, data: List[int]) -> int:
        # 固定长度的滑动窗口，看窗口内最多有多少个1
        n = len(data)
        left = 0
        ss = sum(data)
        right = ss # 初始化
        window = 0
        maxWindow = 0
        for i in range(right):
            window += data[i]
        maxWindow = window # 初始化

        while right < n:
            add = data[right]
            delete = data[left]
            right += 1
            left += 1
            window = window + add - delete
            if window > maxWindow:
                maxWindow = window
        
        return ss - maxWindow
            
```

# [1154. 一年中的第几天](https://leetcode-cn.com/problems/day-of-the-year/)

给你一个字符串 date ，按 YYYY-MM-DD 格式表示一个 现行公元纪年法 日期。请你计算并返回该日期是当年的第几天。

通常情况下，我们认为 1 月 1 日是每年的第 1 天，1 月 2 日是每年的第 2 天，依此类推。每个月的天数与现行公元纪年法（格里高利历）一致

```python
class Solution:
    def dayOfYear(self, date: str) -> int:
        # 特殊处理闰年
        monthDict = [0,0,31,None,31,30,31,30,31,31,30,31,30,31] # 补前两个0，方便判断
        year = int(date[:4])
        isRunNian = False
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            isRunNian = True 
        if isRunNian:
            monthDict[3] = 29
        else:
            monthDict[3] = 28
        pre = 0
        for i in range(len(monthDict)):
            pre += monthDict[i]
            monthDict[i] = pre
        month = int(date[5:7])
        day = int(date[8:])
        ans = monthDict[month] + day 
        return ans

```

# [1166. 设计文件系统](https://leetcode-cn.com/problems/design-file-system/)

你需要设计一个文件系统，你可以创建新的路径并将它们与不同的值关联。

路径的格式是一个或多个连接在一起的字符串，形式为： / ，后面跟着一个或多个小写英文字母。例如， " /leetcode" 和 "/leetcode/problems" 是有效路径，而空字符串 "" 和 "/" 不是。

实现 FileSystem 类:

bool createPath(string path, int value) 创建一个新的 path ，并在可能的情况下关联一个 value ，然后返回 true 。如果路径已经存在或其父路径不存在，则返回 false 。
 int get(string path) 返回与 path 关联的值，如果路径不存在则返回 -1 。

```python
class TrieNode():
    def __init__(self):
        self.children = dict()
        self.val = -1


class FileSystem:

    def __init__(self):
        self.root = TrieNode()


    def createPath(self, path: str, value: int) -> bool:
    # 需要隔开最后一个
        p = path 
        node = self.root 
        p = p.split("/")[1:]
        for w in p[:-1]: # 隔离出最后一个
            if node.children.get(w) == None:
                return False
            node = node.children[w]
        if node.children.get(p[-1]) != None:
            return False
        node.children[p[-1]] = TrieNode()
        node = node.children[p[-1]]
        node.val = value
        return True 


    def get(self, path: str) -> int:
        p = path 
        node = self.root 
        p = p.split("/")[1:]
        for w in p:
            if node.children.get(w) == None:
                return -1
            node = node.children[w]
        
        return node.val     
```

# [1288. 删除被覆盖区间](https://leetcode-cn.com/problems/remove-covered-intervals/)

给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。

只有当 c <= a 且 b <= d 时，我们才认为区间 [a,b) 被区间 [c,d) 覆盖。

在完成所有删除操作后，请你返回列表中剩余区间的数目。

```python
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        # 排序后进行操作
        # 注意这排序原则
        intervals.sort(key = lambda x:(x[0],-x[1]))
        ans = [intervals[0]]
        # 注意这里的a,b,c,d和题目中abcd不一样
        for p in range(1,len(intervals)):
            a,b = ans[-1]
            c,d = intervals[p]
            if d <= b:
                continue 
            ans.append(intervals[p])
        return len(ans)
```

# [1367. 二叉树中的列表](https://leetcode-cn.com/problems/linked-list-in-binary-tree/)

给你一棵以 root 为根的二叉树和一个 head 为第一个节点的链表。

如果在二叉树中，存在一条一直向下的路径，且每个点的数值恰好一一对应以 head 为首的链表中每个节点的值，那么请你返回 True ，否则返回 False 。

一直向下的路径的意思是：从树中某个节点开始，一直连续向下的路径。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        # 利用数组进行
        lst = []
        cur = head
        while cur:
            lst.append(cur.val)
            cur = cur.next 
        n = len(lst)
        state = False 
        def dfs(node,pcur):
            nonlocal state
            if pcur == n:
                state = True 
                return 
            if node == None:
                return 
            if node.val == lst[pcur]:
                dfs(node.left,pcur+1)
                dfs(node.right,pcur+1)
        
        def method(node):
            if node == None:
                return 
            if node.val == lst[0]:
                dfs(node,0)
            if not state:
                method(node.left)
                method(node.right)

        method(root)

        return state 
```



# [1185. 一周中的第几天](https://leetcode-cn.com/problems/day-of-the-week/)

给你一个日期，请你设计一个算法来判断它是对应一周中的哪一天。

输入为三个整数：day、month 和 year，分别表示日、月、年。

您返回的结果必须是这几个值中的一个 {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}。

```python
class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
        days = 0
        # 输入年份之前的年份的天数贡献
        days += 365 * (year - 1971) + (year - 1969) // 4
        # 输入年份中，输入月份之前的月份的天数贡献
        days += sum(monthDays[:month-1])
        if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) and month >= 3:
            days += 1
        # 输入月份中的天数贡献
        days += day

        return week[(days + 3) % 7]

```

# [1229. 安排会议日程](https://leetcode-cn.com/problems/meeting-scheduler/)

你是一名行政助理，手里有两位客户的空闲时间表：slots1 和 slots2，以及会议的预计持续时间 duration，请你为他们安排合适的会议时间。

「会议时间」是两位客户都有空参加，并且持续时间能够满足预计时间 duration 的 最早的时间间隔。

如果没有满足要求的会议时间，就请返回一个 空数组。

「空闲时间」的格式是 [start, end]，由开始时间 start 和结束时间 end 组成，表示从 start 开始，到 end 结束。 

题目保证数据有效：同一个人的空闲时间不会出现交叠的情况，也就是说，对于同一个人的两个空闲时间 [start1, end1] 和 [start2, end2]，要么 start1 > end2，要么 start2 > end1。

```python
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        # 需要足够长的持续时间
        # 判断交集时长最开始符合的
        # 注意预先排序
        n1, n2 = len(slots1), len(slots2)
        slots1.sort()
        slots2.sort()
        p1,p2 = 0,0
        while p1 < n1 and p2 < n2:
            a,b = slots1[p1]
            c,d = slots2[p2]
            if a <= c <= b <= d: # 情况1
                if b - c >= duration:
                    return [c,c + duration]
            elif c <= a <= d <= b: # 情况2
                if d - a  >= duration:
                    return [a,a + duration]
            elif a <= c <= d <= b: # 情况3
                if d - c >= duration:
                    return [c,c + duration]
            elif c <= a <= b <= d: # 情况4
                if b - a >= duration:
                    return [a,a + duration]
            # 指针移动的时候，考虑移动结束时间更早的
            if b <= d: p1 += 1               
            else: p2 += 1               
        return []
```

# [1233. 删除子文件夹](https://leetcode-cn.com/problems/remove-sub-folders-from-the-filesystem/)

你是一位系统管理员，手里有一份文件夹列表 folder，你的任务是要删除该列表中的所有 子文件夹，并以 任意顺序 返回剩下的文件夹。

我们这样定义「子文件夹」：

如果文件夹 folder[i] 位于另一个文件夹 folder[j] 下，那么 folder[i] 就是 folder[j] 的子文件夹。
文件夹的「路径」是由一个或多个按以下格式串联形成的字符串：

/ 后跟一个或者多个小写英文字母。
例如，/leetcode 和 /leetcode/problems 都是有效的路径，而空字符串和 / 不是。

```python
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.isEnd = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,path):
        node = self.root 
        origin = path
        path = path.split("/")[1:]
        for w in path:
            if node.children.get(w) == None:
                node.children[w] = TrieNode()
            node = node.children[w] 
        node.isEnd = origin

    def search(self):
        ans = []
        # 收集所有的isEnd
        def dfs(node):
            if node.isEnd != False:
                ans.append(node.isEnd)
                return 
            for child in node.children:
                dfs(node.children[child])
        dfs(self.root)
        return ans
                

class Solution:
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        tree = Trie()
        for f in folder:
            tree.insert(f)
        ans = tree.search()
        return ans
```

# [1257. 最小公共区域](https://leetcode-cn.com/problems/smallest-common-region/)

给你一些区域列表 regions ，每个列表的第一个区域都包含这个列表内所有其他区域。

很自然地，如果区域 X 包含区域 Y ，那么区域 X  比区域 Y 大。

给定两个区域 region1 和 region2 ，找到同时包含这两个区域的 最小 区域。

如果区域列表中 r1 包含 r2 和 r3 ，那么数据保证 r2 不会包含 r3 。

数据同样保证最小公共区域一定存在。

```python
class TreeNode:
    def __init__(self,name,parent = None,children = []):
        self.name = name 
        self.parent = parent
        self.children = children

class Solution:
    def findSmallestRegion(self, regions: List[List[str]], region1: str, region2: str) -> str:
        # 多叉树找LCA，
        nameDict = dict() # key是名字，val是对应的树节点
        for lst in regions:
            name1 = lst[0]
            if nameDict.get(name1) == None:
                root = TreeNode(name1)
                nameDict[name1] = root
            for child in lst[1:]:
                if nameDict.get(child) == None:
                    childNode = TreeNode(child)
                    nameDict[child] = childNode
                
                nameDict[name1].children.append(nameDict[child])
                nameDict[child].parent = nameDict[name1]
        
        node1 = nameDict[region1]
        node2 = nameDict[region2]

        path1 = []
        while node1 != None:
            path1.append(node1.name)
            node1 = node1.parent 
        path2 = []
        while node2 != None:
            path2.append(node2.name)
            node2 = node2.parent
        
        path1 = path1[::-1]
        path2 = path2[::-1]
        # 找到最后一个公共节点
        p1 = 0
        p2 = 0
        while p1 < len(path1) and p2 < len(path2):
            if path1[p1] == path2[p2]:
                p1 += 1
                p2 += 1
            else:
                break 

        return path1[p1-1] # 回退一个
```

# [1268. 搜索推荐系统](https://leetcode-cn.com/problems/search-suggestions-system/)

给你一个产品数组 products 和一个字符串 searchWord ，products  数组中每个产品都是一个字符串。

请你设计一个推荐系统，在依次输入单词 searchWord 的每一个字母后，推荐 products 数组中前缀与 searchWord 相同的最多三个产品。如果前缀相同的可推荐产品超过三个，请按字典序返回最小的三个。

请你以二维列表的形式，返回在输入 searchWord 每个字母后相应的推荐产品的列表。

```python
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        # 暴力匹配算法
        ans = []
        products.sort()
        for i in range(len(searchWord)):
            template = searchWord[:i+1]
            tempAns = []
            for w in products:
                if w[:i+1] == template:
                    tempAns.append(w)
                    if len(tempAns) >= 3:
                        break 
            ans.append(tempAns)
        return ans
                
```

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.isEnd = False 
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,word):
        node = self.root 
        for ch in word:
            index = ord(ch)-ord('a')
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isEnd = True
    
    # 需要用DFS获取答案，因为要求是字典序
    def search(self,word):
        node = self.root 
        ansLst = []
        for ch in word:
            index = ord(ch)-ord('a')
            if node.children[index] != None: # 有这个字符
                node = node.children[index]
            elif node.children[index] == None: # 没有这个字符
                return []
        # 移动到了公共前缀，从这个节点开始搜答案
        
        def dfs(node,path):
            if len(ansLst) >= 3:
                return 
            if node.isEnd:
                ansLst.append(word+"".join(path))
                # 注意这里不能有return

            for index in range(26):
                if node.children[index] != None:
                    path.append(chr(97+index))
                    dfs(node.children[index],path)
                    path.pop()

        dfs(node,[])
        return ansLst
     
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        tree = Trie()
        for w in products:
            tree.insert(w)
        
        ans = []

        for i in range(len(searchWord)):
            w = searchWord[:i+1]
            t = tree.search(w)
            ans.append(t)
        return ans
```



# [1306. 跳跃游戏 III](https://leetcode-cn.com/problems/jump-game-iii/)

这里有一个非负整数数组 arr，你最开始位于该数组的起始下标 start 处。当你位于下标 i 处时，你可以跳到 i + arr[i] 或者 i - arr[i]。

请你判断自己是否能够跳到对应元素值为 0 的 任一 下标处。

注意，不管是什么情况下，你都无法跳到数组之外。

```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        # BFS + visited
        n = len(arr)
        if arr[start] == 0:
            return True
        visited = [False for i in range(n)]
        queue = [start] 
        while len(queue) != 0:
            new_queue = []
            for node in queue:
                if arr[node] == 0:
                    return True 
                visited[node] = True 
                n1 = node-arr[node]
                n2 = node+arr[node]
                if 0<=n1<n and visited[n1] == False:
                    new_queue.append(n1)
                if 0<=n2<n and visited[n2] == False:
                    new_queue.append(n2)
            queue = new_queue
        return False 
```

# [1333. 餐厅过滤器](https://leetcode-cn.com/problems/filter-restaurants-by-vegan-friendly-price-and-distance/)

给你一个餐馆信息数组 restaurants，其中  restaurants[i] = [idi, ratingi, veganFriendlyi, pricei, distancei]。你必须使用以下三个过滤器来过滤这些餐馆信息。

其中素食者友好过滤器 veganFriendly 的值可以为 true 或者 false，如果为 true 就意味着你应该只包括 veganFriendlyi 为 true 的餐馆，为 false 则意味着可以包括任何餐馆。此外，我们还有最大价格 maxPrice 和最大距离 maxDistance 两个过滤器，它们分别考虑餐厅的价格因素和距离因素的最大值。

过滤后返回餐馆的 id，按照 rating 从高到低排序。如果 rating 相同，那么按 id 从高到低排序。简单起见， veganFriendlyi 和 veganFriendly 为 true 时取值为 1，为 false 时，取值为 0 。

```python
# 读清楚题，自定义排序
class Solution:
    def filterRestaurants(self, restaurants: List[List[int]], veganFriendly: int, maxPrice: int, maxDistance: int) -> List[int]:
        temp = []
        if veganFriendly == 1:
            for theid,rating,isvet,price,distance in restaurants:
                if isvet == veganFriendly and price <= maxPrice and distance <= maxDistance:
                    temp.append([theid,rating])
        else:
            for theid,rating,isvet,price,distance in restaurants:
                if price <= maxPrice and distance <= maxDistance:
                    temp.append([theid,rating])     
                      
        temp.sort(key = lambda x:(-x[1],-x[0]))

        ans = [e[0] for e in temp]
        return ans
```

# [1357. 每隔 n 个顾客打折](https://leetcode-cn.com/problems/apply-discount-every-n-orders/)

超市里正在举行打折活动，每隔 n 个顾客会得到 discount 的折扣。

超市里有一些商品，第 i 种商品为 products[i] 且每件单品的价格为 prices[i] 。

结账系统会统计顾客的数目，每隔 n 个顾客结账时，该顾客的账单都会打折，折扣为 discount （也就是如果原本账单为 x ，那么实际金额会变成 x - (discount * x) / 100 ），然后系统会重新开始计数。

顾客会购买一些商品， product[i] 是顾客购买的第 i 种商品， amount[i] 是对应的购买该种商品的数目。

请你实现 Cashier 类：

Cashier(int n, int discount, int[] products, int[] prices) 初始化实例对象，参数分别为打折频率 n ，折扣大小 discount ，超市里的商品列表 products 和它们的价格 prices 。
double getBill(int[] product, int[] amount) 返回账单的实际金额（如果有打折，请返回打折后的结果）。返回结果与标准答案误差在 10^-5 以内都视为正确结果。

```python
class Cashier:
# 难点在读题:每隔 n 个顾客会得到 discount 的折扣，从1到n,计数到n则打折，然后重制
    def __init__(self, n: int, discount: int, products: List[int], prices: List[int]):
        self.n = n 
        self.discount = discount
        self.table = dict()
        for i in range(len(products)):
            self.table[products[i]] = prices[i]
        self.now = 1


    def getBill(self, product: List[int], amount: List[int]) -> float:
        isDisc = False
        if self.now == self.n:
            isDisc = True 
            self.now = 1
        else:
            self.now += 1
        
        v = 0
        for i,p in enumerate(product):
            v += self.table[p]*amount[i]

        if isDisc:
            v = v - (self.discount * v) / 100 
        else:
            pass 
        return v
```




# [1366. 通过投票对团队排名](https://leetcode-cn.com/problems/rank-teams-by-votes/)

现在有一个特殊的排名系统，依据参赛团队在投票人心中的次序进行排名，每个投票者都需要按从高到低的顺序对参与排名的所有团队进行排位。

排名规则如下：

参赛团队的排名次序依照其所获「排位第一」的票的多少决定。如果存在多个团队并列的情况，将继续考虑其「排位第二」的票的数量。以此类推，直到不再存在并列的情况。
如果在考虑完所有投票情况后仍然出现并列现象，则根据团队字母的字母顺序进行排名。
给你一个字符串数组 votes 代表全体投票者给出的排位情况，请你根据上述排名规则对所有参赛团队进行排名。

请你返回能表示按排名系统 排序后 的所有团队排名的字符串。

```python
class Solution:
    def rankTeams(self, votes: List[str]) -> str:
        n = len(votes)
        bound = len(votes[0])
        # 每个纵列索引扫描，其所有结果存入一个 频次，字符表中，如果最大频次唯一，则将它使用
        # 如果最大频次不唯一，则比较下一个索引，当不存在下一个索引的时候，按照字典序排序
        record = dict()
        ans = []
        # 先收集
        for index in range(bound):
            temp = collections.defaultdict(int)
            record[index] = temp          
            for i in range(n):
                ch = votes[i][index]
                temp[ch] += 1

        used = set(votes[0]) # 表示出现过的字符
        alphaDict = collections.defaultdict(list) # key是字母，val是按照索引排序的频次
        for index in range(bound):
            temp = record[index]
            pivot = []
            for theKey in range(26):
                key = chr(ord("A")+theKey)
                if key in used:
                    alphaDict[key].append(temp[key])

        # print(alphaDict)
        temp = []
        for key in alphaDict:
            tlst = [-e for e in alphaDict[key]]
            temp.append(tlst+[key])
        
        # print(temp)
        temp.sort()
        ans = [e[-1] for e in temp]
        return "".join(ans)
```

# [1395. 统计作战单位数](https://leetcode-cn.com/problems/count-number-of-teams/)

n 名士兵站成一排。每个士兵都有一个 独一无二 的评分 rating 。

每 3 个士兵可以组成一个作战单位，分组规则如下：

从队伍中选出下标分别为 i、j、k 的 3 名士兵，他们的评分分别为 rating[i]、rating[j]、rating[k]
作战单位需满足： rating[i] < rating[j] < rating[k] 或者 rating[i] > rating[j] > rating[k] ，其中  0 <= i < j < k < n
请你返回按上述条件可以组建的作战单位数量。每个士兵都可以是多个作战单位的一部分。

```python
class Solution:
    def numTeams(self, rating: List[int]) -> int:
        # 暴力法1:
        def subMethod(lst):
            n = len(lst)
            less = [0 for i in range(n)] # 记录比他小的有多少个
            more = [0 for i in range(n)] # 记录比他大的有多少个，需要反着扫

            for i in range(n):
                cnt = 0
                for j in range(i):
                    if lst[j] < lst[i]:
                        cnt += 1
                less[i] = cnt 
            
            for i in range(n-1,-1,-1):
                cnt = 0
                for j in range(n-1,i,-1):
                    if lst[j] > lst[i]:
                        cnt += 1
                more[i] = cnt 
            
            ans = 0
            for i in range(n):
                ans += less[i]*more[i]
            return ans
        
        final = subMethod(rating) + subMethod(rating[::-1])
        return final
```

# [1396. 设计地铁系统](https://leetcode-cn.com/problems/design-underground-system/)

地铁系统跟踪不同车站之间的乘客出行时间，并使用这一数据来计算从一站到另一站的平均时间。

实现 UndergroundSystem 类：

void checkIn(int id, string stationName, int t)
通行卡 ID 等于 id 的乘客，在时间 t ，从 stationName 站进入
乘客一次只能从一个站进入
void checkOut(int id, string stationName, int t)
通行卡 ID 等于 id 的乘客，在时间 t ，从 stationName 站离开
double getAverageTime(string startStation, string endStation)
返回从 startStation 站到 endStation 站的平均时间
平均时间会根据截至目前所有从 startStation 站 直接 到达 endStation 站的行程进行计算，也就是从 startStation 站进入并从 endStation 离开的行程
从 startStation 到 endStation 的行程时间与从 endStation 到 startStation 的行程时间可能不同
在调用 getAverageTime 之前，至少有一名乘客从 startStation 站到达 endStation 站
你可以假设对 checkIn 和 checkOut 方法的所有调用都是符合逻辑的。如果一名乘客在时间 t1 进站、时间 t2 出站，那么 t1 < t2 。所有时间都按时间顺序发生。

```python
class UndergroundSystem:

    def __init__(self):
        # 由于id可以会重复，采取的策略是，进入就记录，退出时候根据id搜到这个人的入站，并且加入区间
        self.idrec = dict()
        self.stationRec = collections.defaultdict(list) # key是start+end,val是时间


    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.idrec[id] = [stationName,t]


    def checkOut(self, id: int, stationName: str, t: int) -> None:
        start,startTime = self.idrec[id]
        del self.idrec[id]
        self.stationRec[start+","+stationName].append(t-startTime)


    def getAverageTime(self, startStation: str, endStation: str) -> float:
        key = startStation+","+endStation
        return sum(self.stationRec[key])/len(self.stationRec[key])
```



# [1404. 将二进制表示减到 1 的步骤数](https://leetcode-cn.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/)

给你一个以二进制形式表示的数字 s 。请你返回按下述规则将其减少到 1 所需要的步骤数：

如果当前数字为偶数，则将其除以 2 。

如果当前数字为奇数，则将其加上 1 。

题目保证你总是可以按上述规则将测试用例变为 1 。

```python
class Solution:
    def numSteps(self, s: str) -> int:
        # 加上1之后，从右边往左找，找到第一个为0的位置，它变成1,中间跨越的所有1都变成0
        # 每一个末尾0,都pop掉

        count = 0
        s = list(s)
        while len(s) != 1:
            if s[-1] == "0":
                s.pop()
                count += 1
            elif s[-1] == "1":
                state = False
                for i in range(len(s)-1,-1,-1):
                    if s[i] == "1":
                        s[i] = "0" 
                    elif s[i] == "0":
                        s[i] = "1"
                        state = True # 标志着没有进位
                        break
                if state == False:
                    s = ["1"] + s # 进位加上去
                count += 1
        return count
```

# [1405. 最长快乐字符串](https://leetcode-cn.com/problems/longest-happy-string/)

如果字符串中不含有任何 'aaa'，'bbb' 或 'ccc' 这样的字符串作为子串，那么该字符串就是一个「快乐字符串」。

给你三个整数 a，b ，c，请你返回 任意一个 满足下列全部条件的字符串 s：

s 是一个尽可能长的快乐字符串。
s 中 最多 有a 个字母 'a'、b 个字母 'b'、c 个字母 'c' 。
s 中只含有 'a'、'b' 、'c' 三种字母。
如果不存在这样的字符串 s ，请返回一个空字符串 ""。

```python
# 硬算。。。
class Solution:
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        # 字符不需要用完
        stack = []
        h = []
        if a:
            h.append([a,"a"])
        if b:
            h.append([b,"b"])
        if c:
            h.append([c,"c"])
        h.sort()
        def check(ele):
            if len(stack) >= 2 and stack[-1] == ele and stack[-2] == ele:
                return False 
            return True 
        
        while h:
            times,ele = h.pop()
            if check(ele):
                stack.append(ele)
                times -= 1
                if times != 0:
                    h.append([times,ele])
                    h.sort()
            else:
                if h:
                    times2,ele2 = h.pop()
                    stack.append(ele2)
                    times2 -= 1
                    if times2 != 0:
                        h.append([times2,ele2])
                    h.append([times,ele])
                    h.sort()
                else:
                    break

        return "".join(stack)
        
```



# [1409. 查询带键的排列](https://leetcode-cn.com/problems/queries-on-a-permutation-with-key/)

给你一个待查数组 queries ，数组中的元素为 1 到 m 之间的正整数。 请你根据以下规则处理所有待查项 queries[i]（从 i=0 到 i=queries.length-1）：

一开始，排列 P=[1,2,3,...,m]。
对于当前的 i ，请你找出待查项 queries[i] 在排列 P 中的位置（下标从 0 开始），然后将其从原位置移动到排列 P 的起始位置（即下标为 0 处）。注意， queries[i] 在 P 中的位置就是 queries[i] 的查询结果。
请你以数组形式返回待查数组  queries 的查询结果。

```python
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        # 类似于LRU处理，但是由于数据量小，可以直接强行模拟
        lst = deque(i+1 for i in range(m))
        ans = []
        for q in queries:
            index = lst.index(q)
            lst.remove(q)
            lst.appendleft(q)
            ans.append(index)
        return ans
```

# [1410. HTML 实体解析器](https://leetcode-cn.com/problems/html-entity-parser/)

「HTML 实体解析器」 是一种特殊的解析器，它将 HTML 代码作为输入，并用字符本身替换掉所有这些特殊的字符实体。

HTML 里这些特殊字符和它们对应的字符实体包括：

双引号：字符实体为 &quot; ，对应的字符是 " 。
单引号：字符实体为 &apos; ，对应的字符是 ' 。
与符号：字符实体为 &amp; ，对应对的字符是 & 。
大于号：字符实体为 &gt; ，对应的字符是 > 。
小于号：字符实体为 &lt; ，对应的字符是 < 。
斜线号：字符实体为 &frasl; ，对应的字符是 / 。
给你输入字符串 text ，请你实现一个 HTML 实体解析器，返回解析器解析后的结果。

```python
class Solution:
    def entityParser(self, text: str) -> str:
        # 注意原字符串不一定用空格分割，也不一定只用一个空格分割，所以不能用split方法
        record = dict()
        record["&quot;"] = "\""
        record["&apos;"] = "'"
        record["&amp;"] = "&"
        record["&gt;"] = ">"
        record["&lt;"] = "<"
        record["&frasl;"] = "/"
        ans = []
        temp = []
        p = 0
        n = len(text)
        # 遇到&开始激活temp,中断条件为遇到";"或者遇到下一个&
        while p < n:
            if text[p] != "&":
                ans.append(text[p])
                p += 1
            else:
                temp = ["&"]
                t = 1
                while p+t<n and text[p+t] != ";" and text[p+t] != "&":
                    temp.append(text[p+t])
                    t += 1
                # 此时遇到了终止符
                if p+t<n and text[p+t] == ";":
                    temp.append(text[p+t])
                    key = "".join(temp)
                    # print(key,key in record)
                    if key in record:
                        ans.append(record[key])
                    else:
                        ans.append(key)
                    temp = [] # 清空temp
                    p = p+t
                elif p+t<n: # text[p+t] == "&" 或者越界都没有被收集，都用这一条
                    ans.append("".join(temp))
                    if text[p+t] == "&":
                        temp = ["&"]
                        t -= 1 # 注意这个分支的补丁
                    else:
                        temp = [] # 清空
                    p = p+t
                p += 1
        for e in temp:
            ans.append(e)
        return "".join(ans)
```

# [1428. 至少有一个 1 的最左端列](https://leetcode-cn.com/problems/leftmost-column-with-at-least-a-one/)

我们称只包含元素 0 或 1 的矩阵为二进制矩阵。矩阵中每个单独的行都按非递减顺序排序。

给定一个这样的二进制矩阵，返回至少包含一个 1 的最左端列的索引（从 0 开始）。如果这样的列不存在，返回 -1。

您不能直接访问该二进制矩阵。你只可以通过 BinaryMatrix 接口来访问。

BinaryMatrix.get(row, col) 返回位于索引 (row, col) （从 0 开始）的元素。
BinaryMatrix.dimensions() 返回含有 2 个元素的列表 [rows, cols]，表示这是一个 rows * cols的矩阵。
如果提交的答案调用 BinaryMatrix.get 超过 1000 次，则该答案会被判定为错误答案。提交任何试图规避判定机制的答案将会被取消资格。

下列示例中， mat 为给定的二进制矩阵。您不能直接访问该矩阵。

```python
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
#class BinaryMatrix(object):
#    def get(self, row: int, col: int) -> int:
#    def dimensions(self) -> list[]:

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        m,n = binaryMatrix.dimensions()
        # 根据维度进行二分搜索,找到最左边的1
        temp = []
        for i in range(m):
            left = 0
            right = n-1
            ans = n # 初始化不存在
            while left <= right: # 闭区间二分
                mid = (left+right)//2
                t = binaryMatrix.get(i,mid)
                if t == 0: # 往右边搜索
                    left = mid + 1
                elif t == 1: # 往左边搜索，顺便存
                    right = mid - 1
                    ans = mid 
            temp.append(ans)
        
        final = min(temp)
        if final == n: return -1
        else: return final
```

# [1429. 第一个唯一数字](https://leetcode-cn.com/problems/first-unique-number/)

给定一系列整数，插入一个队列中，找出队列中第一个唯一整数。

实现 FirstUnique 类：

FirstUnique(int[] nums) 用数组里的数字初始化队列。
int showFirstUnique() 返回队列中的 第一个唯一 整数的值。如果没有唯一整数，返回 -1。（译者注：此方法不移除队列中的任何元素）
void add(int value) 将 value 插入队列中。

```python
class FirstUnique:
# 版本1：保留原队列元素
    def __init__(self, nums: List[int]):
        # 标准队列一个，辅助队列一个
        self.queue = nums
        self.hqueue = [] 
        self.ct = collections.defaultdict(int)
        for n in nums:
            self.ct[n] += 1
        for key in self.ct:
            if self.ct[key] == 1:
                self.hqueue.append(key) # 里面存的都是只出现一次的


    def showFirstUnique(self) -> int:
        while len(self.hqueue) != 0:
            if self.ct[self.hqueue[0]] != 1:
                self.hqueue.pop(0)
            else:
                return self.hqueue[0]
        return -1

    def add(self, value: int) -> None:
        self.queue.append(value)
        self.ct[value] += 1
        if self.ct[value] == 1:
            self.hqueue.append(value)
```

```python
class FirstUnique:
# 辅助队列是deque
    def __init__(self, nums: List[int]):
        # 标准队列一个，辅助队列一个
        self.queue = []
        self.hqueue = collections.deque() 
        self.ct = collections.defaultdict(int)
        for n in nums:
            self.ct[n] += 1
        for key in self.ct:
            if self.ct[key] == 1:
                self.hqueue.append(key) # 里面存的都是只出现一次的


    def showFirstUnique(self) -> int:
        while len(self.hqueue) != 0:
            if self.ct[self.hqueue[0]] != 1:
                self.hqueue.popleft()
            else:
                return self.hqueue[0]
        return -1

    def add(self, value: int) -> None:
        self.queue.append(value)
        self.ct[value] += 1
        if self.ct[value] == 1:
            self.hqueue.append(value)
```

# [1461. 检查一个字符串是否包含所有长度为 K 的二进制子串](https://leetcode-cn.com/problems/check-if-a-string-contains-all-binary-codes-of-size-k/)

给你一个二进制字符串 `s` 和一个整数 `k` 。

如果所有长度为 `k` 的二进制字符串都是 `s` 的子串，请返回 `true` ，否则请返回 `false` 。

```python
class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        # 滑动窗口尬算【或者利用状态转移】
        def toInt(win):
            t = 0
            for i in range(len(win)):
                t += int(win[i])*pow(2,i)
            return t 
        
        tset = set()
        for i in range(len(s)-k+1):
            tset.add(toInt(s[i:i+k]))
        return len(tset) == pow(2,k)
```

```python
class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        # rk类似的状态转移
        def toInt(win):
            win = win[::-1]
            t = 0
            for i in range(len(win)):
                t += int(win[i])*pow(2,i)
            return t 
        win = toInt(s[:k])
        tset = set()
        tset.add(win)
        n = len(s)
        g = pow(2,k-1)
        for i in range(k,n):
            add = int(s[i])
            delete = int(s[i-k])
            win -=  delete*g
            win <<= 1 
            win += add
            tset.add(win)
        # print(tset)
        return len(tset) == pow(2,k)
```

# [1466. 重新规划路线](https://leetcode-cn.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/)

n 座城市，从 0 到 n-1 编号，其间共有 n-1 条路线。因此，要想在两座不同城市之间旅行只有唯一一条路线可供选择（路线网形成一颗树）。去年，交通运输部决定重新规划路线，以改变交通拥堵的状况。

路线用 connections 表示，其中 connections[i] = [a, b] 表示从城市 a 到 b 的一条有向路线。

今年，城市 0 将会举办一场大型比赛，很多游客都想前往城市 0 。

请你帮助重新规划路线方向，使每个城市都可以访问城市 0 。返回需要变更方向的最小路线数。

题目数据 保证 每个城市在重新规划路线方向后都能到达城市 0 。

```python
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        # 以0作为起点进行BFS，已知路线网是构成了树
        graph = collections.defaultdict(list)
        gSet = set()
        for (i,j) in connections:
            graph[i].append(j)
            graph[j].append(i)
            gSet.add((i,j))
        queue = [0]
        visited = [False for j in range(n)]
        visited[0] = True
        ans = 0
        while queue:
            new_queue = []
            for node in queue:
                for neigh in graph[node]:
                    if visited[neigh] == True: continue 
                    visited[neigh] = True 
                    if (neigh,node) not in gSet: # 注意这里是反向边
                        ans += 1
                    new_queue.append(neigh)
            queue = new_queue 
        return ans 
```

# [1471. 数组中的 k 个最强值](https://leetcode-cn.com/problems/the-k-strongest-values-in-an-array/)

给你一个整数数组 arr 和一个整数 k 。

设 m 为数组的中位数，只要满足下述两个前提之一，就可以判定 arr[i] 的值比 arr[j] 的值更强：

 |arr[i] - m| > |arr[j] - m|
 |arr[i] - m| == |arr[j] - m|，且 arr[i] > arr[j]
请返回由数组中最强的 k 个值组成的列表。答案可以以 任意顺序 返回。

中位数 是一个有序整数列表中处于中间位置的值。形式上，如果列表的长度为 n ，那么中位数就是该有序列表（下标从 0 开始）中位于 ((n - 1) / 2) 的元素。

例如 arr = [6, -3, 7, 2, 11]，n = 5：数组排序后得到 arr = [-3, 2, 6, 7, 11] ，数组的中间位置为 m = ((5 - 1) / 2) = 2 ，中位数 arr[m] 的值为 6 。
例如 arr = [-7, 22, 17, 3]，n = 4：数组排序后得到 arr = [-7, 3, 17, 22] ，数组的中间位置为 m = ((4 - 1) / 2) = 1 ，中位数 arr[m] 的值为 3 。

```python
class Solution:
    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        arr.sort()
        n = len(arr)-1
        m = arr[n//2]

        arr.sort(key=lambda x:(abs(x-m),x),reverse=True)
        # print(arr)
        return arr[:k]
```

```python
class Solution:
    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        arr.sort()
        n = len(arr)-1
        m = arr[n//2]

        # 双指针
        left,right = 0,len(arr)-1
        ans = []
        while left <= right and k > 0:
            # 取值，
            g1,g2 = abs(arr[left]-m),abs(arr[right]-m)
            k -= 1
            if g1 > g2:
                ans.append(arr[left])
                left += 1
            else:
                ans.append(arr[right])
                right -= 1
        return ans
```

# [1476. 子矩形查询](https://leetcode-cn.com/problems/subrectangle-queries/)

请你实现一个类 SubrectangleQueries ，它的构造函数的参数是一个 rows x cols 的矩形（这里用整数矩阵表示），并支持以下两种操作：

1. updateSubrectangle(int row1, int col1, int row2, int col2, int newValue)

用 newValue 更新以 (row1,col1) 为左上角且以 (row2,col2) 为右下角的子矩形。
2. getValue(int row, int col)

返回矩形中坐标 (row,col) 的当前值。

```python
class SubrectangleQueries:
# 暴力模拟法
    def __init__(self, rectangle: List[List[int]]):       
        self.rec = rectangle

    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        for i in range(row1,row2+1):
            for j in range(col1,col2+1):
                self.rec[i][j] = newValue

    def getValue(self, row: int, col: int) -> int:
        return self.rec[row][col]
        
```

```python
class SubrectangleQueries:
# 倒序查询法
    def __init__(self, rectangle: List[List[int]]):       
        self.rec = rectangle
        self.temp = []

    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        self.temp.append([row1,col1,row2,col2,newValue])


    def getValue(self, row: int, col: int) -> int:
        p = len(self.temp)-1
        while p >= 0:
            row1,col1,row2,col2,newValue = self.temp[p]
            if row1<=row<=row2 and col1<=col<=col2:
                return newValue 
            else:
                p -= 1
        return self.rec[row][col]

```

# [1482. 制作 m 束花所需的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-make-m-bouquets/)

给你一个整数数组 bloomDay，以及两个整数 m 和 k 。

现需要制作 m 束花。制作花束时，需要使用花园中 相邻的 k 朵花 。

花园中有 n 朵花，第 i 朵花会在 bloomDay[i] 时盛开，恰好 可以用于 一束 花中。

请你返回从花园中摘 m 束花需要等待的最少的天数。如果不能摘到 m 束花则返回 -1 。

```python
class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        left, right = min(bloomDay), max(bloomDay)
        ans = -1
        # 需要m束花，
        # 辅助check函数,返回值以k为要求时，能够获得的花束是否大于等于m
        def check(days):
            cnt = 0
            p = 0
            while p < len(bloomDay):
                if days >= bloomDay[p]:
                    t = 0
                    while p+t < len(bloomDay) and days >= bloomDay[p+t] and t < k:
                        t += 1
                    if t == k:
                        cnt += 1
                    p += t
                else:
                    p += 1
            return cnt >= m

        while left <= right: # 闭区间二分
            mid = (left+right)//2
            if check(mid): # 说明可以缩小
                ans = mid 
                right = mid - 1
            else:
                left = mid + 1
        return ans
```

# [1529. 最少的后缀翻转次数](https://leetcode-cn.com/problems/minimum-suffix-flips/)

给你一个长度为 n 、下标从 0 开始的二进制字符串 target 。你自己有另一个长度为 n 的二进制字符串 s ，最初每一位上都是 0 。你想要让 s 和 target 相等。

在一步操作，你可以选择下标 i（0 <= i < n）并翻转在 闭区间 [i, n - 1] 内的所有位。翻转意味着 '0' 变为 '1' ，而 '1' 变为 '0' 。

返回使 s 与 target 相等需要的最少翻转次数。

```python
class Solution:
    def minFlips(self, target: str) -> int:
        n = len(target)
        # 从左到右数有多少次突变
        # 方式是从第一个1开始，从左到右反转
        cnt = 0
        pre = "0"
        for i in range(n):
            if target[i] != pre:
                cnt += 1
            pre = target[i]
        return cnt
```

# [1530. 好叶子节点对的数量](https://leetcode-cn.com/problems/number-of-good-leaf-nodes-pairs/)

给你二叉树的根节点 root 和一个整数 distance 。

如果二叉树中两个 叶 节点之间的 最短路径长度 小于或者等于 distance ，那它们就可以构成一组 好叶子节点对 。

返回树中 好叶子节点对的数量 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        ans = 0
        lvs = []
        def getLeaves(node):
            if node == None:
                return 
            if node.left == None and node.right == None:
                lvs.append(node)
            getLeaves(node.left)
            getLeaves(node.right)    
        getLeaves(root)

        def getLCA(node,p,q):
            if node == None:
                return None 
            if node == p or node == q:
                return node 
            leftPart = getLCA(node.left,p,q)
            rightPart = getLCA(node.right,p,q)
            if leftPart == None and rightPart == None:
                return None 
            elif leftPart == None and rightPart != None:
                return rightPart
            elif leftPart != None and rightPart == None:
                return leftPart
            else:
                return node 
        
        # 然后获得两个节点之间的距离
        def find(n1,n2,path): # n1是固定点，n2是移动点,lca是移动点
            nonlocal p
            if n2 == None:
                return 
            if n2 == n1:
                p = path
                return 
            path += 1
            find(n1,n2.left,path)
            find(n1,n2.right,path)
            path -= 1
        
        for i in range(len(lvs)):
            for j in range(i+1,len(lvs)):
                lca = getLCA(root,lvs[i],lvs[j])
                kk = 0
                p = 0
                find(lvs[i],lca,0)
                kk += p 
                p = 0
                find(lvs[j],lca,0)
                kk += p 
                if kk <= distance:
                    ans += 1
        return ans
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        # 每个节点到他所有能够触碰到的叶子结点的穷举，On2
        def getLeaves(node,ls,path):
            if node == None:
                return 
            if node.left == None and node.right == None:
                ls.append([node,path])
            path += 1
            getLeaves(node.left,ls,path)
            getLeaves(node.right,ls,path)
            path -= 1
        
        ans = 0
        record = collections.defaultdict(lambda :inf) # 记录两个叶子结点最近的距离
        def dfs(node):
            nonlocal ans 
            if node == None:
                return 
            # preOrder穷举
            ls = []
            getLeaves(node,ls,0)
            for i in range(len(ls)):
                for j in range(i+1,len(ls)):
                    key = ls[i][0],ls[j][0]
                    record[key] = min(record[key],ls[i][1]+ls[j][1])              
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        for key in record:
            if record[key] <= distance:
                ans += 1
        return ans
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        # 每个节点到他所有能够触碰到的叶子结点的穷举，On2
        def getLeaves(node,ls,path):
            if node == None:
                return 
            if node.left == None and node.right == None:
                ls.append([node,path])
            path += 1
            getLeaves(node.left,ls,path)
            getLeaves(node.right,ls,path)
            path -= 1
        
        ans = 0
        visited = set() # 记录两个叶子结点最近的距离
        def dfs(node):
            nonlocal ans 
            if node == None:
                return 
                         
            dfs(node.left)
            dfs(node.right)
            # # preOrder穷举
            ls = []
            getLeaves(node,ls,0)
            for i in range(len(ls)):
                for j in range(i+1,len(ls)):
                    key = (ls[i][0],ls[j][0])
                    if ls[i][1]+ls[j][1] <= distance and key not in visited:
                        ans += 1
                        visited.add(key)
        
        dfs(root)
        return ans
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        # 每个节点到他所有能够触碰到的叶子结点的穷举，On2
        def getLeaves(node,ls,path):
            if node == None:
                return 
            if node.left == None and node.right == None:
                ls.append(path)
            path += 1
            getLeaves(node.left,ls,path)
            getLeaves(node.right,ls,path)
            path -= 1
        
        ans = 0
        visited = set() # 记录两个叶子结点最近的距离
        def dfs(node):
            nonlocal ans 
            if node == None:
                return                    
            dfs(node.left)
            dfs(node.right)
            # 
            le = []
            getLeaves(node.left,le,0)
            ri = []
            getLeaves(node.right,ri,0)
            for i in le:
                for j in ri:
                    if i+j <= distance-2:
                        ans += 1
        dfs(root)
        return ans
```

# [1533. 找到最大整数的索引](https://leetcode-cn.com/problems/find-the-index-of-the-large-integer/)

我们有这样一个整数数组 arr ，除了一个最大的整数外，其他所有整数都相等。你不能直接访问该数组，你需要通过 API ArrayReader 来间接访问，这个 API 有以下成员函数：

int compareSub(int l, int r, int x, int y)：其中 0 <= l, r, x, y < ArrayReader.length()， l <= r 且 x <= y。这个函数比较子数组 arr[l..r] 与子数组 arr[x..y] 的和。该函数返回：
1 若 arr[l]+arr[l+1]+...+arr[r] > arr[x]+arr[x+1]+...+arr[y] 。
0 若 arr[l]+arr[l+1]+...+arr[r] == arr[x]+arr[x+1]+...+arr[y] 。
-1 若 arr[l]+arr[l+1]+...+arr[r] < arr[x]+arr[x+1]+...+arr[y] 。
int length()：返回数组的长度。
你最多可以调用函数 compareSub() 20 次。你可以认为这两个函数的时间复杂度都为 O(1) 。

返回 arr 中最大整数的索引。

```python
class Solution:
    def getIndex(self, reader: 'ArrayReader') -> int:
        # 读题+二分搜素
        n = reader.length()
        left,right = 0,n-1
        # compareSub(l,r,x,y); l<=r , x<=y  
        # 闭区间二分搜索      
        while left <= right:
            mid = (left + right)//2    
            # 长度为right-left+1，根据mid分奇偶讨论
            # 如果长度是奇数，则都是mid
            # 如果长度是偶数，则左边mid,右边mid+1
            le = right - left + 1
            # print('le = ',le)
            if le%2 == 1:
                state = reader.compareSub(left,mid,mid,right)
                # print(left,mid,mid,right)
            else:
                state = reader.compareSub(left,mid,mid+1,right)     
                # print(left,mid,mid+1,right)

            if le%2 == 1:
                if state == 0:
                    return mid
                elif state == 1:
                    right = mid - 1
                elif state == -1:
                    left = mid + 1
            else:
                if le == 2:
                    if state == 1:
                        return left 
                    else:
                        return right
                if state == 0:
                    return mid 
                elif state == 1:
                    right = mid 
                elif state == -1:
                    left = mid 
```

# [1541. 平衡括号字符串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertions-to-balance-a-parentheses-string/)

给你一个括号字符串 s ，它只包含字符 '(' 和 ')' 。一个括号字符串被称为平衡的当它满足：

任何左括号 '(' 必须对应两个连续的右括号 '))' 。
左括号 '(' 必须在对应的连续两个右括号 '))' 之前。
比方说 "())"， "())(())))" 和 "(())())))" 都是平衡的， ")()"， "()))" 和 "(()))" 都是不平衡的。

你可以在任意位置插入字符 '(' 和 ')' 使字符串平衡。

请你返回让 s 平衡的最少插入次数。

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        # 栈匹配
        # 左括号直接入栈，右括号只有在相邻的下一个也是右括号时才能pop，否则补充一个再pop
        stack = []
        n = len(s)
        i = 0
        cnt = 0
        while i < n:
            # print('s[i] = ',s[i])
            if s[i] == "(":
                stack.append("(")
            elif s[i] == ")":
                if i+1 < n and s[i+1] == ")":
                    i += 1
                    if len(stack):
                        stack.pop()
                    else:
                        cnt += 1
                else:
                    if len(stack):
                        stack.pop()
                        cnt += 1
                    else:
                        cnt += 2
            i += 1
        # print(stack,cnt)
        return 2*len(stack) + cnt
```

# [1552. 两球之间的磁力](https://leetcode-cn.com/problems/magnetic-force-between-two-balls/)

在代号为 C-137 的地球上，Rick 发现如果他将两个球放在他新发明的篮子里，它们之间会形成特殊形式的磁力。Rick 有 n 个空的篮子，第 i 个篮子的位置在 position[i] ，Morty 想把 m 个球放到这些篮子里，使得任意两球间 最小磁力 最大。

已知两个球如果分别位于 x 和 y ，那么它们之间的磁力为 |x - y| 。

给你一个整数数组 position 和一个整数 m ，请你返回最大化的最小磁力。

```python
class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        position.sort() # 预先排序
        # 尝试以limit距离放入
        left = 0
        right = position[-1]
        while left <= right:
            mid = (left+right)//2
            prePosition = position[0]
            cp_m = m - 1
            for p in position:
                if p-prePosition >= mid:
                    cp_m -= 1
                    prePosition = p
            if cp_m > 0: # 还有剩下的，放不完，需要减小间隔
                right = mid - 1
            elif cp_m <= 0: # 可以放完，可以宽松间隔
                left = mid + 1
                ans = mid
        return ans
```



# [1558. 得到目标数组的最少函数调用次数](https://leetcode-cn.com/problems/minimum-numbers-of-function-calls-to-make-target-array/)

给你一个与 nums 大小相同且初始值全为 0 的数组 arr ，请你调用以上函数得到整数数组 nums 。

请你返回将 arr 变成 nums 的最少函数调用次数。

答案保证在 32 位有符号整数以内。

1. 对某一个数+1
2. 对所有数*2

```python
# 记忆化搜索
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # 找到单个数字中，最复杂的
        # 所有的+1操作必不可少
        memo = dict() # k是n；v是（步骤数，执行+1的次数）
        memo[0] = (0,0)
        memo[1] = (1,1)
        def recur(n):
            if n in memo:
                return memo[n]
            # 如果它是奇数，只能由+1而来
            # 如果它是偶数，能由*2而来
            if n % 2 == 0:
                ops = recur(n//2)
                memo[n] = (ops[0] + 1,ops[1])
                return memo[n]
            else:
                ops = recur(n-1)
                memo[n] = (ops[0] + 1,ops[1]+1)
                return memo[n]
        
        ans = 0
        mult = 0
        for n in nums:
            ops = recur(n)
            ans += ops[1] # 纯加法必不可少
            mult = max(mult,ops[0]-ops[1]) # 乘法数量
        # print(memo)
        ans += mult # 乘法可以同步执行
        
        return ans
```

```python
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # 贪心解法
        # 逆向思考，二进制思考，所有的末尾的1都会先减去1，再消去
        # 除法只需要记录最大值
        n = max(nums)
        if n == 0:
            return 0
        mult = len(bin(n)[2:])-1
        # print(mult)
        ans = 0
        for n in nums:
            ans += bin(n).count('1')
        return ans + mult
```

# [1560. 圆形赛道上经过次数最多的扇区](https://leetcode-cn.com/problems/most-visited-sector-in-a-circular-track/)

给你一个整数 n 和一个整数数组 rounds 。有一条圆形赛道由 n 个扇区组成，扇区编号从 1 到 n 。现将在这条赛道上举办一场马拉松比赛，该马拉松全程由 m 个阶段组成。其中，第 i 个阶段将会从扇区 rounds[i - 1] 开始，到扇区 rounds[i] 结束。举例来说，第 1 阶段从 rounds[0] 开始，到 rounds[1] 结束。

请你以数组形式返回经过次数最多的那几个扇区，按扇区编号 升序 排列。

注意，赛道按扇区编号升序逆时针形成一个圆（请参见第一个示例）。

```python
class Solution:
    def mostVisited(self, n: int, rounds: List[int]) -> List[int]:
        # 纯模拟,注意编号是从1～n
        lst = [0 for i in range(n)]
        # 1->3;3->1;1->2
        for p in range(1,len(rounds)):
            start = rounds[p-1]
            end = rounds[p]
            if end < start:
                end += n
            for i in range(start,end):
                lst[i%n] += 1

        lst[rounds[-1]%n] += 1 # 注意最后一个也要算进去

        k = max(lst)
        # print(lst)
        ans = []
        for i,cnt in enumerate(lst):
            if i == 0 and cnt == k:
                ans.append(n)
            elif cnt == k:
                ans.append(i)
        ans.sort()
        return ans
```

```python
class Solution:
    def mostVisited(self, n: int, rounds: List[int]) -> List[int]:
        # 这一题可以简化成只和起点终点有关
        start = rounds[0]
        end = rounds[-1]
        # 无论中间经过多少圈，如果start <= end, 那么就是start -> end的闭区间
        # 如果start > end ，则除去（start,end)的区间，注意是开区间
        if start <= end:
            return list(i for i in range(start,end+1))
        else:
            l1 = list(i for i in range(1,end+1))
            l2 = list(i for i in range(start,n+1))
            return l1+l2
```



# [1578. 使绳子变成彩色的最短时间](https://leetcode-cn.com/problems/minimum-time-to-make-rope-colorful/)

Alice 把 n 个气球排列在一根绳子上。给你一个下标从 0 开始的字符串 colors ，其中 colors[i] 是第 i 个气球的颜色。

Alice 想要把绳子装扮成 彩色 ，且她不希望两个连续的气球涂着相同的颜色，所以她喊来 Bob 帮忙。Bob 可以从绳子上移除一些气球使绳子变成 彩色 。给你一个下标从 0 开始的整数数组 neededTime ，其中 neededTime[i] 是 Bob 从绳子上移除第 i 个气球需要的时间（以秒为单位）。

返回 Bob 使绳子变成 彩色 需要的 最少时间 。

```python
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        # 贪心，对于一段连续的气球，只保留其中耗时最大的
        cost = 0
        n = len(neededTime)
        i = 0
        while i < n-1: # 检查下一个
            if colors[i] != colors[i+1]:
                i += 1
            else:
                p = i 
                while p < n and colors[p] == colors[i]:
                    p += 1
                # 此时p指向不同色的那一个
                cost += sum(neededTime[i:p]) - max(neededTime[i:p])
                i = p
        return cost

```

# [1580. 把箱子放进仓库里 II](https://leetcode-cn.com/problems/put-boxes-into-the-warehouse-ii/)

给定两个正整数数组 boxes 和 warehouse ，分别包含单位宽度的箱子的高度，以及仓库中n个房间各自的高度。仓库的房间分别从0 到 n - 1自左向右编号，warehouse[i]（索引从 0 开始）是第 i 个房间的高度。

箱子放进仓库时遵循下列规则：

箱子不可叠放。
你可以重新调整箱子的顺序。
箱子可以从任意方向（左边或右边）推入仓库中。
如果仓库中某房间的高度小于某箱子的高度，则这个箱子和之后的箱子都会停在这个房间的前面。
你最多可以在仓库中放进多少个箱子？

```python
class Solution:
    def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
        # 只需要找到可行数量解
        # boxes可能很多
        n = len(warehouse)
        # 推动策略为先格式化仓库形状
        left = [0 for i in range(n)]
        pre = warehouse[0]
        for i in range(n):
            pre = min(pre,warehouse[i])
            left[i] = pre 
        right = [0 for i in range(n)]
        pre = warehouse[-1]
        for i in range(n-1,-1,-1):
            pre = min(pre,warehouse[i])
            right[i] = pre 
        # print(left,right)
        # 从左到右找到第一个等值点，然后加入队列
        ind = warehouse.index(pre)
        queue = left[:ind] + right[ind:]
        # 根据这个队列进行双指针可以完成,这里直接排序+二分了
        queue.sort()
        # print(queue)
        pb = 0
        pq = 0
        cnt = 0
        boxes.sort() # 箱子也要排序
        while pb < len(boxes) and pq < len(queue):
            if boxes[pb] <= queue[pq]:
                cnt += 1
                pq += 1
                pb += 1
            elif boxes[pb] > queue[pq]:
                pq += 1
        return cnt
```

# [1588. 所有奇数长度子数组的和](https://leetcode-cn.com/problems/sum-of-all-odd-length-subarrays/)

给你一个正整数数组 arr ，请你计算所有可能的奇数长度子数组的和。

子数组 定义为原数组中的一个连续子序列。

请你返回 arr 中 所有奇数长度子数组的和 。

```python
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        # 数学方法，一个数组要保持奇数,算上它自己是一个，需要前面的数字个数和后面的数字个数相等
        n = len(arr)
        ans = 0
        # 一个数会被这样用到
        # leftOdd,rightOdd ''' leftEven,rightEven
        # leftOdd 表示左边长度为奇数的数量，leftEven 表示左边长度为偶数的数量
        # right同上
        for i in range(n):
            leftOdd = (i+1)//2
            leftEven = (i)//2 + 1 # +1需要算上0长度
            rightOdd = (n-i)//2
            rightEven = (n-i-1)//2 + 1 # +1是因为需要算上0长度
            ans += arr[i]*(leftOdd*rightOdd+leftEven*rightEven)
        return ans
```

# [1604. 警告一小时内使用相同员工卡大于等于三次的人](https://leetcode-cn.com/problems/alert-using-same-key-card-three-or-more-times-in-a-one-hour-period/)

力扣公司的员工都使用员工卡来开办公室的门。每当一个员工使用一次他的员工卡，安保系统会记录下员工的名字和使用时间。如果一个员工在一小时时间内使用员工卡的次数大于等于三次，这个系统会自动发布一个 警告 。

给你字符串数组 keyName 和 keyTime ，其中 [keyName[i], keyTime[i]] 对应一个人的名字和他在 某一天 内使用员工卡的时间。

使用时间的格式是 24小时制 ，形如 "HH:MM" ，比方说 "23:51" 和 "09:49" 。

请你返回去重后的收到系统警告的员工名字，将它们按 字典序升序 排序后返回。

请注意 "10:00" - "11:00" 视为一个小时时间范围内，而 "23:51" - "00:10" 不被视为一小时内，因为系统记录的是某一天内的使用情况。

```python
class Solution:
    def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
        # 按照每个人作为key，val为时间列表,写一个转换函数
        def toMinute(s):
            hh = int(s[:2])
            mm = int(s[3:])
            return hh*60 + mm 

        nameDict = collections.defaultdict(list)
        n = len(keyName)
        for i in range(n):
            nameDict[keyName[i]].append(toMinute(keyTime[i]))
        for key in nameDict:
            nameDict[key].sort()
        
        # print(nameDict)
        ans = []
        for key in nameDict: # 不采取二分查找
            p = 2
            while p < len(nameDict[key]):
                if nameDict[key][p]-nameDict[key][p-2] <= 60:
                    ans.append(key)
                    break 
                p += 1
        ans.sort()
        return ans
```



# [1612. 检查两棵二叉表达式树是否等价](https://leetcode-cn.com/problems/check-if-two-expression-trees-are-equivalent/)

二叉表达式树是一种表达算术表达式的二叉树。二叉表达式树中的每一个节点都有零个或两个子节点。 叶节点（有 0 个子节点的节点）表示操作数，非叶节点（有 2 个子节点的节点）表示运算符。在本题中，我们只考虑 '+' 运算符（即加法）。

给定两棵二叉表达式树的根节点 root1 和 root2 。如果两棵二叉表达式树等价，返回 true ，否则返回 false 。

当两棵二叉搜索树中的变量取任意值，分别求得的值都相等时，我们称这两棵二叉表达式树是等价的。

```python
# Definition for a binary tree node.
# class Node(object):
#     def __init__(self, val=" ", left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def checkEquivalence(self, root1: 'Node', root2: 'Node') -> bool:
        # 只有加法的时候，收集所有叶子结点
        def dfs(node,path):
            if node == None:
                return 
            if node.left == None and node.right == None:
                path.append(node.val)
            dfs(node.left,path)
            dfs(node.right,path)
        
        path1 = []
        path2 = []
        dfs(root1,path1)
        dfs(root2,path2)
        path1.sort()
        path2.sort()
        return path1 == path2

# 若为减法，则需要分左右记录，左边字符个数+1，右边字符个数-1.利用hashtable即可
```

# [1625. 执行操作后字典序最小的字符串](https://leetcode-cn.com/problems/lexicographically-smallest-string-after-applying-operations/)

给你一个字符串 s 以及两个整数 a 和 b 。其中，字符串 s 的长度为偶数，且仅由数字 0 到 9 组成。

你可以在 s 上按任意顺序多次执行下面两个操作之一：

累加：将  a 加到 s 中所有下标为奇数的元素上（下标从 0 开始）。数字一旦超过 9 就会变成 0，如此循环往复。例如，s = "3456" 且 a = 5，则执行此操作后 s 变成 "3951"。
轮转：将 s 向右轮转 b 位。例如，s = "3456" 且 b = 1，则执行此操作后 s 变成 "6345"。
请你返回在 s 上执行上述操作任意次后可以得到的 字典序最小 的字符串。

如果两个字符串长度相同，那么字符串 a 字典序比字符串 b 小可以这样定义：在 a 和 b 出现不同的第一个位置上，字符串 a 中的字符出现在字母表中的时间早于 b 中的对应字符。例如，"0158” 字典序比 "0190" 小，因为不同的第一个位置是在第三个字符，显然 '5' 出现在 '9' 之前。

```python
class Solution:
    def findLexSmallestString(self, s: str, a: int, b: int) -> str:
        # 设计两个辅助函数
        visited = set()
        # 先试验一下穷举
        def increase(lst,a):
            temp = list(lst)
            for i in range(1,len(lst),2):
                temp[i] = str( (int(temp[i])+a)%10)
            return "".join(temp)
        
        def rotate(lst,b):
            return lst[b:]+lst[:b]
        
        queue = [s]

        while queue:
            new_queue = []
            for lst in queue:
                inc = increase(lst,a)
                rot = rotate(lst,b)
                if inc not in visited:
                    visited.add(inc)
                    new_queue.append(inc)
                if rot not in visited:
                    visited.add(rot)
                    new_queue.append(rot)
            queue = new_queue
        
		# 这里可以换个比较器写法，但是实际上可能还不如sort快
        visited = list(visited)
        visited.sort()
        return visited[0]
```

# [1668. 最大重复子字符串](https://leetcode-cn.com/problems/maximum-repeating-substring/)

给你一个字符串 sequence ，如果字符串 word 连续重复 k 次形成的字符串是 sequence 的一个子字符串，那么单词 word 的 重复值为 k 。单词 word 的 最大重复值 是单词 word 在 sequence 中最大的重复值。如果 word 不是 sequence 的子串，那么重复值 k 为 0 。

给你一个字符串 sequence 和 word ，请你返回 最大重复值 k 。

```python
# 暴力匹配
class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        if word not in sequence:
            return 0
        k = 1
        while k*word in sequence:
            k += 1
        return k-1
```

```python
# dp
class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        m = len(sequence)
        n = len(word)
        dp = [0 for i in range(m)] 
        for i,c in enumerate(sequence):
            if c == word[0] and sequence[i:i+n] == word:
                dp[i+n-1] = dp[i-1]+1
        return max(dp)
```

# [1701. 平均等待时间](https://leetcode-cn.com/problems/average-waiting-time/)

有一个餐厅，只有一位厨师。你有一个顾客数组 customers ，其中 customers[i] = [arrivali, timei] ：

arrivali 是第 i 位顾客到达的时间，到达时间按 非递减 顺序排列。
timei 是给第 i 位顾客做菜需要的时间。
当一位顾客到达时，他将他的订单给厨师，厨师一旦空闲的时候就开始做这位顾客的菜。每位顾客会一直等待到厨师完成他的订单。厨师同时只能做一个人的订单。厨师会严格按照 订单给他的顺序 做菜。

请你返回所有顾客需要等待的 平均 时间。与标准答案误差在 10-5 范围以内，都视为正确结果。

```python
class Solution:
    def averageWaitingTime(self, customers: List[List[int]]) -> float:
        # 已知严格按照订单顺序
        n = len(customers)
        # 对每个顾客，完成的严格时间进行记录
        now = customers[0][0] + customers[0][1]
        cost = customers[0][1] # 先记录总消耗时间
        for i in range(1,n):
            if customers[i][0] >= now:
                # 说明这名顾客无需额外等待
                now = customers[i][0] + customers[i][1]
                cost += customers[i][1]
            elif customers[i][0] < now:
                # 说明这名顾客需要等待
                cost += now - customers[i][0] + customers[i][1]
                now += customers[i][1]
        
        return cost/n
```

# [1717. 删除子字符串的最大得分](https://leetcode-cn.com/problems/maximum-score-from-removing-substrings/)

给你一个字符串 s 和两个整数 x 和 y 。你可以执行下面两种操作任意次。

删除子字符串 "ab" 并得到 x 分。
比方说，从 "cabxbae" 删除 ab ，得到 "cxbae" 。
删除子字符串"ba" 并得到 y 分。
比方说，从 "cabxbae" 删除 ba ，得到 "cabxe" 。
请返回对 s 字符串执行上面操作若干次能得到的最大得分。

```python
class Solution:
    def maximumGain(self, s: str, x: int, y: int) -> int:
        # 根据谁的价值大，定义两个子方法
        # 核心贪心思路：这是因为 'a' 和 'b' 是成对删除的, 不会因为先删除了 "ab" 导致后续少了 "ba".
        def delete_ab(s,x,y): # 表示优先删除ab
            stack = []
            points = 0
            for ch in s:
                stack.append(ch)
                while len(stack) >= 2 and stack[-2]+stack[-1] == "ab":
                    stack.pop()
                    stack.pop()
                    points += x
            new_stack = []
            for ch in stack:
                new_stack.append(ch)
                while len(new_stack) >= 2 and new_stack[-2]+new_stack[-1] == "ba":
                    new_stack.pop()
                    new_stack.pop()
                    points += y 
            return points
        
        def delete_ba(s,x,y): # 优先删除ba
            stack = []
            points = 0
            for ch in s:
                stack.append(ch)
                while len(stack) >= 2 and stack[-2]+stack[-1] == "ba":
                    stack.pop()
                    stack.pop()
                    points += y
            new_stack = []
            for ch in stack:
                new_stack.append(ch)
                while len(new_stack) >= 2 and new_stack[-2]+new_stack[-1] == "ab":
                    new_stack.pop()
                    new_stack.pop()
                    points += x
            return points
        
        if x > y:
            ans = delete_ab(s,x,y)
        elif x == y:
            ans = max(delete_ab(s,x,y),delete_ba(s,x,y))
        elif x < y:
            ans = delete_ba(s,x,y)
        return ans
```

# [1705. 吃苹果的最大数目](https://leetcode-cn.com/problems/maximum-number-of-eaten-apples/)

有一棵特殊的苹果树，一连 n 天，每天都可以长出若干个苹果。在第 i 天，树上会长出 apples[i] 个苹果，这些苹果将会在 days[i] 天后（也就是说，第 i + days[i] 天时）腐烂，变得无法食用。也可能有那么几天，树上不会长出新的苹果，此时用 apples[i] == 0 且 days[i] == 0 表示。

你打算每天 最多 吃一个苹果来保证营养均衡。注意，你可以在这 n 天之后继续吃苹果。

给你两个长度为 n 的整数数组 days 和 apples ，返回你可以吃掉的苹果的最大数目。

```python
class Solution:
    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        cnt = 0
        # 扫描，更新产生的苹果的最后时间和数量,小根堆，时间越早越优先
        lst = []
        apples = apples
        days = days
        n = len(apples)
        for i in range(n):
            if apples[i] != 0:
                heapq.heappush(lst,[i+days[i]-1,apples[i]])
            state = False # 表示今天没有享用苹果
            while not state and len(lst) > 0:
                theDay,amount = heapq.heappop(lst)
                if theDay >= i: # 如果还新鲜
                    amount -= 1
                    cnt += 1
                    state = True # 享用了苹果，跳出循环
                    if amount > 0:
                        heapq.heappush(lst,[theDay,amount])
        # print(i,lst)
        while len(lst):
            i += 1
            state = False
            while not state and len(lst) > 0:
                theDay,amount = heapq.heappop(lst)
                if theDay >= i: # 如果还新鲜
                    amount -= 1
                    cnt += 1
                    state = True # 享用了苹果，跳出循环
                    if amount > 0:
                        heapq.heappush(lst,[theDay,amount])
        return cnt
```

# [1738. 找出第 K 大的异或坐标值](https://leetcode-cn.com/problems/find-kth-largest-xor-coordinate-value/)

给你一个二维矩阵 matrix 和一个整数 k ，矩阵大小为 m x n 由非负整数组成。

矩阵中坐标 (a, b) 的 值 可由对所有满足 0 <= i <= a < m 且 0 <= j <= b < n 的元素 matrix[i][j]（下标从 0 开始计数）执行异或运算得到。

请你找出 matrix 的所有坐标中第 k 大的值（k 的值从 1 开始计数）。

```python
class Solution:
    def kthLargestValue(self, matrix: List[List[int]], k: int) -> int:
        # 所有满足
        m,n = len(matrix),len(matrix[0])
        ans = []
        
        # 先滚动求出每一行
        for line in matrix:
            pre = 0
            temp = []
            for number in line:
                pre ^= number 
                temp.append(pre)
            ans.append(temp)
        # 在累加上每一行

        for i in range(1,m):
            for j in range(n):
                ans[i][j] ^= ans[i-1][j]
        
        lst = []
        for i in range(m):
            for j in range(n):
                lst.append(ans[i][j])
        lst.sort(reverse=True)
        return lst[k-1]
```



# [1750. 删除字符串两端相同字符后的最短长度](https://leetcode-cn.com/problems/minimum-length-of-string-after-deleting-similar-ends/)

给你一个只包含字符 'a'，'b' 和 'c' 的字符串 s ，你可以执行下面这个操作（5 个步骤）任意次：

选择字符串 s 一个 非空 的前缀，这个前缀的所有字符都相同。
选择字符串 s 一个 非空 的后缀，这个后缀的所有字符都相同。
前缀和后缀在字符串中任意位置都不能有交集。
前缀和后缀包含的所有字符都要相同。
同时删除前缀和后缀。
请你返回对字符串 s 执行上面操作任意次以后（可能 0 次），能得到的 最短长度 。

```python
class Solution:
    def minimumLength(self, s: str) -> int:
        # 不能有交集
        # 双指针,处理aba和aa
        # 引入辅助数组
        left = 0
        right = len(s)-1
        d = [True for i in range(len(s))]
        while left < right:
            pivot = s[left]
            if pivot != s[right]:
                break 
            while left < right and s[left] == pivot:
                d[left] = False
                left += 1   
            while left <= right and s[right] == pivot: # 注意这里是小于等于
                d[right] = False
                right -= 1     
        ans = 0
        for i in range(len(s)):
            if d[i]:
                ans += 1
        return ans
       
```

```python
class Solution:
    def minimumLength(self, s: str) -> int:
        left,right = 0,len(s)-1
        while left < right:
            pivot = s[left]
            if pivot != s[right]:
                break 
            while left < right and s[left] == pivot: # 这里left <= right也可以
                left += 1
            while left <= right and s[right] == pivot:
                right -= 1
        
        return right-left+1
```

```python
class Solution:
    def minimumLength(self, s: str) -> int:
        # 不能有交集
        # 双指针,处理aba和aa
        # 引入辅助数组
        left = 0
        right = len(s)-1
        d = [True for i in range(len(s))]
        while left < right:
            pivot = s[left]
            if pivot != s[right]:
                break 
            while left < right and s[left] == pivot:
                d[left] = False
                left += 1   
            while left <= right and s[right] == pivot: # 注意这里是小于等于
                d[right] = False
                right -= 1     
        return sum(d)
```

# [1756. 设计最近使用（MRU）队列](https://leetcode-cn.com/problems/design-most-recently-used-queue/)

设计一种类似队列的数据结构，该数据结构将最近使用的元素移到队列尾部。

实现 MRUQueue 类：

MRUQueue(int n)  使用 n 个元素： [1,2,3,...,n] 构造 MRUQueue 。
fetch(int k) 将第 k 个元素（从 1 开始索引）移到队尾，并返回该元素。

```python
class MRUQueue:
# 方法1:数组强行模拟
    def __init__(self, n: int):
        self.lst = [i+1 for i in range(n)]

    def fetch(self, k: int) -> int:
        e = self.lst[k-1]
        self.lst.pop(k-1)
        self.lst.append(e)
        return e


```

# [1775. 通过最少操作次数使数组的和相等](https://leetcode-cn.com/problems/equal-sum-arrays-with-minimum-number-of-operations/)

给你两个长度可能不等的整数数组 nums1 和 nums2 。两个数组中的所有值都在 1 到 6 之间（包含 1 和 6）。

每次操作中，你可以选择 任意 数组中的任意一个整数，将它变成 1 到 6 之间 任意 的值（包含 1 和 6）。

请你返回使 nums1 中所有数的和与 nums2 中所有数的和相等的最少操作次数。如果无法使两个数组的和相等，请返回 -1 。

```python
class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        # 先判断可行性，先格式化使得len(nums1) <= len(nums2)
        if len(nums1) > len(nums2):
            nums1,nums2 = nums2,nums1
        # 短的全取6，长的全取1，有交集才有解
        lmin = len(nums1)*6 
        rmax = len(nums2)*1
        if rmax > lmin:
            return -1 
        # 然后是看差值是多少，每次其实可以用堆pop或者哈希表
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        if sum1 > sum2:
            nums1,nums2 = nums2,nums1
            sum1,sum2 = sum2,sum1 
        
        # 保证sum1是小的那个,小的那个尽力扩大，大的那个尽力减小
        # 两者有交集，则break
        steps = 0
        ct1 = collections.Counter(nums1)
        ct1Min = min(key for key in ct1)
        ct2 = collections.Counter(nums2)
        ct2Max = max(key for key in ct2)
        # print(ct1Min,ct2Max)
        while sum1 < sum2: # 只要大于等于了，就break
            ct1Min = min(key for key in ct1)
            ct2Max = max(key for key in ct2)
            gap1 = (6-ct1Min)
            gap2 = (ct2Max-1)
            if gap1 > gap2:
                sum1 += gap1 
                ct1[ct1Min] -= 1
                ct1[6] += 1
                if ct1[ct1Min] == 0:
                    del ct1[ct1Min]
            else:
                sum2 -= gap2 
                ct2[ct2Max] -= 1
                ct2[1] += 1
                if ct2[ct2Max] == 0:
                    del ct2[ct2Max]
            steps += 1
        return steps
```

# [1794. 统计距离最小的子串对个数](https://leetcode-cn.com/problems/count-pairs-of-equal-substrings-with-minimum-difference/)

输入数据为两个字符串firstString 和 secondString，两个字符串下标均从0开始，且均只包含小写的英文字符，请计算满足下列要求的下标四元组(i,j,a,b)的个数：

0 <= i <= j < firstString.length
0 <= a <= b < secondString.length
firstString字符串中从i位置到j位置的子串(包括j位置的字符)和secondString字符串从a位置到b位置的子串(包括b位置字符)相等
j-a的数值是所有符合前面三个条件的四元组中可能的最小值
返回符合上述 4 个条件的四元组的 个数 。

```python
class Solution:
    def countQuadruples(self, firstString: str, secondString: str) -> int:
        # 统计所有的j-a的值作为key，构建个列表
        # 贪心：一定是单字符为最优解
        # j-a最小
        indexList1 = [[] for i in range(26)]
        indexList2 = [[] for i in range(26)]
        for i,ch in enumerate(firstString):
            indexList1[ord(ch)-ord('a')].append(i)
        for i,ch in enumerate(secondString):
            indexList2[ord(ch)-ord('a')].append(i)
        
        # 找到j-a的最小时，indexList1取第一个，indexList2取最后一个
        minGap = 0xffffffff
        ans = 0
        for i in range(26):
            if indexList1[i] != [] and indexList2[i] != []:
                gap = indexList1[i][0]-indexList2[i][-1]
                if gap < minGap:
                    minGap = gap 
                    ans = 1
                elif gap == minGap:
                    ans += 1
        return ans
```



# [1845. 座位预约管理系统](https://leetcode-cn.com/problems/seat-reservation-manager/)

请你设计一个管理 n 个座位预约的系统，座位编号从 1 到 n 。

请你实现 SeatManager 类：

SeatManager(int n) 初始化一个 SeatManager 对象，它管理从 1 到 n 编号的 n 个座位。所有座位初始都是可预约的。
int reserve() 返回可以预约座位的 最小编号 ，此座位变为不可预约。
void unreserve(int seatNumber) 将给定编号 seatNumber 对应的座位变成可以预约。

```python
class SeatManager:
# 直接heap的裸题？
    def __init__(self, n: int):
        self.lst = [i+1 for i in range(n)]
        heapq.heapify(self.lst)

    def reserve(self) -> int:
        e = heapq.heappop(self.lst)
        return e

    def unreserve(self, seatNumber: int) -> None:
        heapq.heappush(self.lst,seatNumber)

```

# [1868. 两个行程编码数组的积](https://leetcode-cn.com/problems/product-of-two-run-length-encoded-arrays/)

行程编码（Run-length encoding）是一种压缩算法，能让一个含有许多段连续重复数字的整数类型数组 nums 以一个（通常更小的）二维数组 encoded 表示。每个 encoded[i] = [vali, freqi] 表示 nums 中第 i 段重复数字，其中 vali 是该段重复数字，重复了 freqi 次。

例如， nums = [1,1,1,2,2,2,2,2] 可表示称行程编码数组 encoded = [[1,3],[2,5]] 。对此数组的另一种读法是“三个 1 ，后面有五个 2 ”。
两个行程编码数组 encoded1 和 encoded2 的积可以按下列步骤计算：

将 encoded1 和 encoded2 分别扩展成完整数组 nums1 和 nums2 。
创建一个新的数组 prodNums ，长度为 nums1.length 并设 prodNums[i] = nums1[i] * nums2[i] 。
将 prodNums 压缩成一个行程编码数组并返回之。
给定两个行程编码数组 encoded1 和 encoded2 ，分别表示完整数组 nums1 和 nums2 。nums1 和 nums2 的长度相同。 每一个 encoded1[i] = [vali, freqi] 表示 nums1 中的第 i 段，每一个 encoded2[j] = [valj, freqj] 表示 nums2 中的第 j 段。

返回 encoded1 和 encoded2 的乘积。

注：行程编码数组需压缩成可能的最小长度。

```python
class Solution:
    def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
        # 纯模拟会爆内存，需要双指针优化，木桶原理
        n1 = len(encoded1)
        n2 = len(encoded2)
        p1,p2 = 0,0
        ans = []
        while p1 < n1 and p2 < n2:
            e1,times1 = encoded1[p1]
            e2,times2 = encoded2[p2]
            minTimes = min(times1,times2)
            ans.append([e1*e2,minTimes])
            encoded1[p1][1] -= minTimes
            if encoded1[p1][1] == 0:
                p1 += 1
            encoded2[p2][1] -= minTimes
            if encoded2[p2][1] == 0:
                p2 += 1
        
        # 然后考虑合并
        final = [ans[0]]
        p = 1
        while p < len(ans):
            if ans[p][0] != final[-1][0]:
                final.append(ans[p])
            else:
                final[-1][1] += ans[p][1]
            p += 1

        return final
```

# [1885. 统计数对](https://leetcode-cn.com/problems/count-pairs-in-two-arrays/)

给你两个长度为 n 的整数数组 nums1 和 nums2 ，找出所有满足 i < j 且 nums1[i] + nums1[j] > nums2[i] + nums2[j] 的数对 (i, j) 。

返回满足条件数对的 个数 。

```python
class Solution:
    def countPairs(self, nums1: List[int], nums2: List[int]) -> int:
        # 移项，
        n = len(nums1)
        diff = [nums1[i]-nums2[i] for i in range(n)]
        diff.sort() # 排序
        # 变成了 diff[i]+diff[j] > 0的数目
        # 防止双指针超时,需要用二分搜索
        cnt = 0
        # print(diff)
        for i in range(n):
            target = -diff[i]+1
            # print('target = ',target,end='  ')
            index = bisect.bisect_left(diff,target)
            # print('index = ',index)
            if index > i: # 从它到index-1的闭区间全都符合
                cnt += n-index
            elif index <= i: # 需要刨除自身,且只能找右边的
                cnt += n-i-1
        return cnt
```

# [1901. 找出顶峰元素 II](https://leetcode-cn.com/problems/find-a-peak-element-ii/)

一个 2D 网格中的 顶峰元素 是指那些 严格大于 其相邻格子(上、下、左、右)的元素。

给你一个 从 0 开始编号 的 m x n 矩阵 mat ，其中任意两个相邻格子的值都 不相同 。找出 任意一个 顶峰元素 mat[i][j] 并 返回其位置 [i,j] 。

你可以假设整个矩阵周边环绕着一圈值为 -1 的格子。

要求必须写出时间复杂度为 O(m log(n)) 或 O(n log(m)) 的算法

```python
class Solution:
    def findPeakGrid(self, mat: List[List[int]]) -> List[int]:
        # 如果采用dfs算法很方便
        # 方法1，从任意一个点往上走
        m,n = len(mat),len(mat[0])
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        ans = None 
        def dfs(i,j):
            nonlocal ans
            if ans: return 
            state = True 
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and mat[new_i][new_j] > mat[i][j]:
                    dfs(new_i,new_j)
                    state = False
            if state:
                ans = [i,j]
        
        dfs(0,0)
        return ans
```

```

```



# [1911. 最大子序列交替和](https://leetcode-cn.com/problems/maximum-alternating-subsequence-sum/)

一个下标从 0 开始的数组的 交替和 定义为 偶数 下标处元素之 和 减去 奇数 下标处元素之 和 。

比方说，数组 [4,2,5,3] 的交替和为 (4 + 5) - (2 + 3) = 4 。
给你一个数组 nums ，请你返回 nums 中任意子序列的 最大交替和 （子序列的下标 重新 从 0 开始编号）。

一个数组的 子序列 是从原数组中删除一些元素后（也可能一个也不删除）剩余元素不改变顺序组成的数组。比方说，[2,7,4] 是 [4,2,3,7,2,1,4] 的一个子序列（加粗元素），但是 [2,4,2] 不是。

```python
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        # 动态规划，需要on
        # 按照最后一步选奇/偶数分类dp
        n = len(nums)
        dp = [[0 for i in range(n)] for t in range(2)]
        # dp[0][i] 是以奇数选
        # dp[0][j] 是以偶数选
        dp[0][0] = nums[0]
        dp[1][0] = 0 # 注意这个初始化
        # 状态转移，要么不选这个，继承前面的，要么选这个，交叉继承

        for i in range(1,n):
            dp[0][i] = max(dp[0][i-1],dp[1][i-1]+nums[i])
            dp[1][i] = max(dp[1][i-1],dp[0][i-1]-nums[i])
        
        # print(dp)
        return dp[0][-1]

```

```go
func maxAlternatingSum(nums []int) int64 {
    n := len(nums)
    dp := make([][]int,2,2)
    dp[0] = make([]int,n,n)
    dp[1] = make([]int,n,n)
    
    dp[0][0] = nums[0]
    
    for i:=1;i<n;i++ {
        dp[0][i] = max(dp[0][i-1],dp[1][i-1]+nums[i])
        dp[1][i] = max(dp[1][i-1],dp[0][i-1]-nums[i])
    }
    
    return int64(dp[0][n-1])
}

func max (a,b int) int {
    if a < b {
        return b
    } else {
        return a 
    }
}
```

# [1921. 消灭怪物的最大数量](https://leetcode-cn.com/problems/eliminate-maximum-number-of-monsters/)

你正在玩一款电子游戏，在游戏中你需要保护城市免受怪物侵袭。给你一个 下标从 0 开始 且长度为 n 的整数数组 dist ，其中 dist[i] 是第 i 个怪物与城市的 初始距离（单位：米）。

怪物以 恒定 的速度走向城市。给你一个长度为 n 的整数数组 speed 表示每个怪物的速度，其中 speed[i] 是第 i 个怪物的速度（单位：米/分）。

怪物从 第 0 分钟 时开始移动。你有一把武器，并可以 选择 在每一分钟的开始时使用，包括第 0 分钟。但是你无法在一分钟的中间使用武器。这种武器威力惊人，一次可以消灭任一还活着的怪物。

一旦任一怪物到达城市，你就输掉了这场游戏。如果某个怪物 恰 在某一分钟开始时到达城市，这会被视为 输掉 游戏，在你可以使用武器之前，游戏就会结束。

返回在你输掉游戏前可以消灭的怪物的 最大 数量。如果你可以在所有怪物到达城市前将它们全部消灭，返回  n 。

```python
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        # 使用一个time数组记录每个怪物被消灭前的极限时间    
        time = []
        n = len(dist)
        for i in range(n):
            t = math.ceil(dist[i]/speed[i]) - 1
            time.append(t)
        time.sort() # 排序一下
        
        i = 0
        while i < n:
            if time[i] < i:
                break 
            i += 1
        return i
```

# [1943. 描述绘画结果](https://leetcode-cn.com/problems/describe-the-painting/)

```python
class Solution:
    def splitPainting(self, segments: List[List[int]]) -> List[List[int]]:
        # 区间问题，上下车算法
        ans = []
        up = collections.defaultdict(int)
        for a,b,c in segments:
            up[a] += c 
            up[b] -= c 
        
        lst = [[key,up[key]] for key in up]
        lst.sort()
        n = len(lst)
        pre = lst[0][0]
        now = lst[0][1]
        for p in range(1,n):
            end,gap = lst[p]
            if now != 0:
                ans.append([pre,end,now])
            now += gap 
            pre = end 
        return ans

```



# [1983. Widest Pair of Indices With Equal Range Sum](https://leetcode-cn.com/problems/widest-pair-of-indices-with-equal-range-sum/)

You are given two 0-indexed binary arrays nums1 and nums2. Find the widest pair of indices (i, j) such that i <= j and nums1[i] + nums1[i+1] + ... + nums1[j] == nums2[i] + nums2[i+1] + ... + nums2[j].

The widest pair of indices is the pair with the largest distance between i and j. The distance between a pair of indices is defined as j - i + 1.

Return the distance of the widest pair of indices. If no pair of indices meets the conditions, return 0.

```python
class Solution:
    def widestPairOfIndices(self, nums1: List[int], nums2: List[int]) -> int:
        # 相减，做前缀哈希表
        n = len(nums1)
        lst = [nums1[i]-nums2[i] for i in range(n)]
        preDict = dict()
        preDict[0] = -1
        maxGap = 0
        pre = 0
        for i in range(n):
            pre += lst[i]
            if preDict.get(pre) == None:
                preDict[pre] = i 
            else:
                maxGap = max(maxGap,i-preDict[pre])
        return maxGap
```



# [1999. 最小的仅由两个数组成的倍数](https://leetcode-cn.com/problems/smallest-greater-multiple-made-of-two-digits/)

给你三个整数, k, digit1和 digit2, 你想要找到满足以下条件的 最小 整数：

大于k 且是 k 的倍数
仅由digit1 和 digit2 组成，即 每一位数 均是 digit1 或 digit2
请你返回 最小的满足这两个条件的整数，如果不存在这样的整数，或者最小的满足这两个条件的整数不在32位整数范围（0~231-1），就返回 -1 。 

```python
class Solution:
    def findInteger(self, k: int, digit1: int, digit2: int) -> int:
        limit = 2147483647  # 2**31 - 1
        if digit1 == digit2: # 先构造出1111...
            for i in range(1,11):
                base = (10**i-1)//9
                if (digit1*base) % k == 0 and digit1*base > k and digit1*base <= limit:
                    return digit1*base
            return -1
        # 如果两个数不相等，bfs
        if digit1 > digit2:
            digit1,digit2 = digit2,digit1
        
        digit1 = str(digit1)
        digit2 = str(digit2)
        found = []

        def backtracking(path):
            if len(path) >= 11:
                return 
            if len(path) > 0:
                now = int("".join(path))
                # print(now)
                if now % k == 0 and now > k and now <= limit:
                    found.append(now)
                    return 
                    
            path.append(digit1)
            backtracking(path)
            path.pop()
            path.append(digit2)
            backtracking(path)
            path.pop()
        
        backtracking([])
        found.sort()
        if len(found):
            return found[0]
        else:
            return -1
           
```

# [2008. 出租车的最大盈利](https://leetcode-cn.com/problems/maximum-earnings-from-taxi/)

你驾驶出租车行驶在一条有 n 个地点的路上。这 n 个地点从近到远编号为 1 到 n ，你想要从 1 开到 n ，通过接乘客订单盈利。你只能沿着编号递增的方向前进，不能改变方向。

乘客信息用一个下标从 0 开始的二维数组 rides 表示，其中 rides[i] = [starti, endi, tipi] 表示第 i 位乘客需要从地点 starti 前往 endi ，愿意支付 tipi 元的小费。

每一位 你选择接单的乘客 i ，你可以 盈利 endi - starti + tipi 元。你同时 最多 只能接一个订单。

给你 n 和 rides ，请你返回在最优接单方案下，你能盈利 最多 多少元。

注意：你可以在一个地点放下一位乘客，并在同一个地点接上另一位乘客。

```python
class Solution:
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        # 动态规划
        rides.sort() # 按照起点先后次序排序
        dp = [0 for i in range(n+1)] # dp[i]表示到达i站的时候能拿到的最大值

        cur = 0
        for i in range(len(rides)):
            s,e,tip = rides[i][0],rides[i][1],rides[i][2]
            # 由于更新e的时候，必须要保证s被更新，所以需要一个额外的指针填充s以及s之前的
            while cur <= s:
                dp[cur] = max(dp[cur],dp[cur-1])
                cur += 1
            dp[e] = max(dp[e],e-s+tip+dp[s])

        # print(dp)
        return max(dp)
```

# [2018. 判断单词是否能放入填字游戏内](https://leetcode-cn.com/problems/check-if-word-can-be-placed-in-crossword/)

给你一个 m x n 的矩阵 board ，它代表一个填字游戏 当前 的状态。填字游戏格子中包含小写英文字母（已填入的单词），表示 空 格的 ' ' 和表示 障碍 格子的 '#' 。

如果满足以下条件，那么我们可以 水平 （从左到右 或者 从右到左）或 竖直 （从上到下 或者 从下到上）填入一个单词：

该单词不占据任何 '#' 对应的格子。
每个字母对应的格子要么是 ' ' （空格）要么与 board 中已有字母 匹配 。
如果单词是 水平 放置的，那么该单词左边和右边 相邻 格子不能为 ' ' 或小写英文字母。
如果单词是 竖直 放置的，那么该单词上边和下边 相邻 格子不能为 ' ' 或小写英文字母。
给你一个字符串 word ，如果 word 可以被放入 board 中，请你返回 true ，否则请返回 false 。

```Python
class Solution:
    def placeWordInCrossword(self, board: List[List[str]], word: str) -> bool:
        # 枚举检查： 先检查横行，再检查纵列
        # 枚举# 和 # 中间的字符串,注意正着和倒着都需要检查

        def jugde(s1,s2): # s1为主串，s2为需要判断的串
            if len(s1) != len(s2):
                return False 
            for i in range(len(s1)):
                if s1[i] != " ":
                    if s1[i] != s2[i]:
                        return False 
            return True 

        for line in board:
            # 首尾封端减少边界检查
            temp = "".join(['#'] + line + ["#"]) 
            p1 = 0
            p2 = 1
            while p2 < len(temp):
                while temp[p2] != "#":
                    p2 += 1
                tstring = "".join(temp[p1+1:p2])
                # print("tstring = ", tstring, len(tstring))
                if jugde(tstring,word) or jugde(tstring,word[::-1]):
                    return True 
                p1 = p2 
                p2 += 1
        
        # 构建一个转置矩阵复用代码
        m,n = len(board),len(board[0])
        grid = [[None for j in range(m)] for i in range(n)]
        for i in range(m):
            for j in range(n):
                grid[j][i] = board[i][j] 
        
        board = grid
        # 复用
        for line in board:
            # 首尾封端减少边界检查
            temp = "".join(['#'] + line + ["#"]) 
            p1 = 0
            p2 = 1
            while p2 < len(temp):
                while temp[p2] != "#":
                    p2 += 1
                tstring = "".join(temp[p1+1:p2])
                # print("tstring = ", tstring, len(tstring))
                if jugde(tstring,word) or jugde(tstring,word[::-1]):
                    return True 
                p1 = p2 
                p2 += 1
        return False 
```

# [2021. 街上最亮的位置](https://leetcode-cn.com/problems/brightest-position-on-street/)

一条街上有很多的路灯，路灯的坐标由数组 lights 的形式给出。 每个 lights[i] = [positioni, rangei] 代表坐标为 positioni 的路灯照亮的范围为 [positioni - rangei, positioni + rangei] （包括顶点）。

位置 p 的亮度由能够照到 p的路灯的数量来决定的。

给出 lights, 返回最亮的位置 。如果有很多，返回坐标最小的。

```python
class Solution:
    def brightestPosition(self, lights: List[List[int]]) -> int:
        # 上下车扫描线求过程极大值并且更新时候只严格更新，保留横坐标最小的
        pairs = [[e[0]-e[1],e[0]+e[1]+1] for e in lights] # 【left，right)半开半闭处理
        pairs.sort()
        
        ans = [1,pairs[0][0]] # [次数,索引]
        n = len(pairs)
        now = 0
        up = [e[0] for e in pairs]
        down = [e[1] for e in pairs]
        up.sort()
        down.sort()
        p1 = 0
        p2 = 0
        while p1 < n and p2 < n:
            if up[p1] < down[p2]:
                now += 1
                if now > ans[0]:
                    ans = [now,up[p1]]
                p1 += 1
            else:
                now -= 1
                p2 += 1
        while p1 < n:
            now += 1
            if now > ans[0]:
                ans = [now,up[p1]]
            p1 += 1
        # print(ans[0])
        return ans[1]
```

# [2046. 给按照绝对值排序的链表排序](https://leetcode-cn.com/problems/sort-linked-list-already-sorted-using-absolute-values/)

给你一个链表的头结点 `head` ，这个链表是根据结点的**绝对值**进行**升序**排序, 返回重新根据**节点的值**进行**升序**排序的链表。

```python
class Solution:
    def sortLinkedList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 暴力做法
        lst = []
        cur = head
        while cur:
            lst.append(cur.val)
            cur = cur.next 
        p = 0
        cur = head
        lst.sort()
        while cur:
            cur.val = lst[p]
            p += 1
            cur = cur.next 
        return head
```

```python
# 或者使用头插法、尾插法
class Solution:
    def sortLinkedList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = head, head.next
        while curr:
            if curr.val < 0:
                t = curr.next
                prev.next = t
                curr.next = head
                head = curr
                curr = t
            else:
                prev, curr = curr, curr.next
        return head

作者：lcbin
链接：https://leetcode-cn.com/problems/sort-linked-list-already-sorted-using-absolute-values/solution/tou-cha-fa-chao-jian-ji-javapython3cgo-s-0sl6/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

```python
class Solution:
    def sortLinkedList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 假设第一个点已经排序了
        pre = head
        cur = head.next 
        realHead = head
        # 遇到负数进行头插，否则继续
        while cur != None:
            if cur.val < 0:
                temp = cur.next # 存下它
                pre.next = temp # 由于这个节点被移走了，前一个节点之间连线下一个节点
                cur.next = realHead # 把这个节点提到头部
                realHead = cur  # 将新的头节点的地址存下
                cur = temp # 移动cur指针
            else:
                pre = cur 
                cur = cur.next 
        return realHead
                
```

# [2052. 将句子分隔成行的最低成本](https://leetcode-cn.com/problems/minimum-cost-to-separate-sentence-into-rows/)

给定一个由空格分隔的单词组成的字符串 sentence 和一个整数 k。你的任务是将 sentence 分成多行，每行中的字符数最多为 k。你可以假设 sentence 不以空格开头或结尾，并且 sentence 中的单词由单个空格分隔。

你可以通过在 sentence 中的单词间插入换行来分隔 sentence 。一个单词不能被分成两行。每个单词只能使用一次，并且单词顺序不能重排。同一行中的相邻单词应该由单个空格分隔，并且每行都不应该以空格开头或结尾。

一行长度为 n 的字符串的分隔成本是 (k - n)2 ，总成本就是除开最后一行以外的其它所有行的分隔成本之和。

以 sentence = "i love leetcode" 和k = 12为例：
将sentence 分成 "i", "love", 和"leetcode" 的成本为 (12 - 1)2 + (12 - 4)2 = 185。
将sentence 分成 "i love", 和"leetcode" 的成本为 (12 - 6)2 = 36。
将sentence 分成 "i", 和"love leetcode" 是不可能的，因为 "love leetcode" 的长度大于 k。
返回将sentence分隔成行的最低的可能总成本。

```python
class Solution:
    def minimumCost(self, sentence: str, k: int) -> int:
        # 可以找到重叠子问题，可以动态规划
        # 注意将字符串压缩，压缩后才可以二维dp
        if len(sentence) <= k:
            return 0
        tt = sentence.split()
        for i in range(len(tt)):
            tt[i] = len(tt[i])
        lst = []
        for n in tt:
            lst.append(n)
            lst.append('d') # 占位符
        lst.pop()
        lst = ['d'] + lst
        # 格式化完毕
        # 扫描到d的时候可以分割
        # 分割的时候倒序查索引，当可以分割的时候，加入备选项
        n = len(lst)
        dp = [float('inf') for i in range(n)]
        dp[0] = 0
        # print(lst)
        for i in range(1,n):
            if lst[i] == 'd':
                group = []
                tempSum = 0
                for j in range(i-1,-1,-1):
                    if lst[j] == 'd' and k-tempSum >= 0:
                        group.append([(k-tempSum)**2,j])
                        tempSum += 1
                    else:
                        tempSum += lst[j]
                    if k-tempSum < 0:
                        break
                group.sort() # 找到最小的
                if len(group) != 0:
                    dp[i] = dp[group[0][1]] + group[0][0]           
        # 倒着找到所有的可能切割点
        # print(lst)
        # print(dp)
        prev = 0
        group = []
        for i in range(n-1,-1,-1):
            if lst[i] != 'd':
                prev += lst[i]
            if dp[i] != float('inf') and prev <= k:
                group.append(dp[i])
            if lst[i] == 'd':
                prev += 1
            if prev > k:
                break 
        return min(group)
```

```python
class Solution:
    def minimumCost(self, sentence: str, k: int) -> int:
        A = list(map(len,sentence.split()))
        pre = list(accumulate(A,initial = 0))
        print(pre)
        n = len(A)
        cacl = [pre[-1]-pre[i]+n-i-1<=k for i in range(n+1)]
        print(cacl)

        @lru_cache(None)
        def suffix(i):
            if cacl[i] == True: return 0
            nl = A[i]
            ans = inf
            for j in range(i+1,n+1):
                if nl > k: break 
                ans = min(ans,(k-nl)**2 + suffix(j))
                nl += 1+A[j]
            return ans 
        return suffix(0)

```



# [2061. 扫地机器人清扫过的空间个数](https://leetcode-cn.com/problems/number-of-spaces-cleaning-robot-cleaned/)

一个房间用一个从 0 开始索引的二维二进制矩阵 room 表示，其中 0 表示空闲空间， 1 表示放有物体的空间。在每个测试用例中，房间左上角永远是空闲的。

一个扫地机器人面向右侧，从左上角开始清扫。机器人将一直前进，直到抵达房间边界或触碰到物体时，机器人将会顺时针旋转 90 度并重复以上步骤。初始位置和所有机器人走过的空间都会被它清扫干净。

若机器人持续运转下去，返回被清扫干净的空间数量。

```python
class Solution:
    def numberOfCleanRooms(self, room: List[List[int]]) -> int:
        # 初始向右
        m,n = len(room),len(room[0])
        cnt = 0 
        cnt_turns = 0 # 记录当前转了多少轮，转了四轮则直接返回
        # 清理过的位置置1
        direc = [(0,1),(1,0),(0,-1),(-1,0)]
        p = 0 # 指向direc[0]，作为初始方向
        now_i,now_j = 0,0
        while True:
            if room[now_i][now_j] == 0:
                room[now_i][now_j] = 2 # 更改为非0，1的位置标记
                cnt_turns = 0 # 重置
                cnt += 1
            if 0<=now_i + direc[p][0]<m and 0<=now_j + direc[p][1]<n and (
                room[now_i + direc[p][0]][now_j + direc[p][1]] == 0 or 
                room[now_i + direc[p][0]][now_j + direc[p][1]] == 2
            ):
                now_i += direc[p][0]
                now_j += direc[p][1]
            elif 0<=now_i + direc[p][0]<m and 0<=now_j + direc[p][1]<n and (
                room[now_i + direc[p][0]][now_j + direc[p][1]] == 1  # 注意这里是触到物体才算
            ):
                p = (p+1)%4
                cnt_turns += 1
                if cnt_turns >= 4:
                    return cnt
            elif not (0<=now_i + direc[p][0]<m) or not (0<=now_j + direc[p][1]<n): # 触碰到墙也转弯
                p = (p+1)%4
                cnt_turns += 1
                if cnt_turns >= 4:
                    return cnt
            else:
                return cnt         
```

# [2083. 求以相同字母开头和结尾的子串总数](https://leetcode-cn.com/problems/substrings-that-begin-and-end-with-the-same-letter/)

给你一个仅由小写英文字母组成的，  下标从 0 开始的字符串 s 。返回 s 中以相同字符开头和结尾的子字符串总数。

子字符串是字符串中连续的非空字符序列。

```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        # 每个字符用字典记录序号,list,int都可以
        memo = collections.defaultdict(int)
        for i,ch in enumerate(s):
            memo[ch] += 1
        
        ans = 0
        for key in memo:
            n = memo[key]
            ans += (n+1)*n//2
        return ans
```

# [2096. 从二叉树一个节点到另一个节点每一步的方向](https://leetcode-cn.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/)

给你一棵 二叉树 的根节点 root ，这棵二叉树总共有 n 个节点。每个节点的值为 1 到 n 中的一个整数，且互不相同。给你一个整数 startValue ，表示起点节点 s 的值，和另一个不同的整数 destValue ，表示终点节点 t 的值。

请找到从节点 s 到节点 t 的 最短路径 ，并以字符串的形式返回每一步的方向。每一步用 大写 字母 'L' ，'R' 和 'U' 分别表示一种方向：

'L' 表示从一个节点前往它的 左孩子 节点。
'R' 表示从一个节点前往它的 右孩子 节点。
'U' 表示从一个节点前往它的 父 节点。
请你返回从 s 到 t 最短路径 每一步的方向。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        # 先找到LCA
        path = []
        def findLCA(root,node1_val,node2_val):
            nonlocal path
            if root == None:
                return 
            if root.val == node1_val or root.val == node2_val:
                return root
            leftPart = findLCA(root.left,node1_val,node2_val)
            rightPart = findLCA(root.right,node1_val,node2_val)
            if leftPart == None and rightPart == None:
                return None 
            elif leftPart == None and rightPart != None:
                return rightPart
            elif leftPart != None and rightPart == None:
                return leftPart
            elif leftPart != None and rightPart != None:
                return root 
        
        # 根据LCA找到了最短路径
        # 从这个node开始暴力搜
        lca = findLCA(root,startValue,destValue)
        g1 = []
        def findU(node,path):
            nonlocal g1
            if node == None:
                return 
            if node.val == startValue:
                g1 = path[:]
                return 
            path.append("U")
            findU(node.left,path)
            findU(node.right,path)
            path.pop()
        
        g2 = []
        def findLR(node,path):
            nonlocal g2 
            if node == None:
                return 
            if node.val == destValue:
                g2 = path[:]
                return 
            path.append("L")
            findLR(node.left,path)
            path.pop()
            path.append("R")
            findLR(node.right,path)
            path.pop()
        findU(lca,[])
        findLR(lca,[])
        # print(g1)
        # print(g2)
        return "".join(g1)+"".join(g2)

            
```




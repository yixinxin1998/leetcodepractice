# 22. 括号生成

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

有效括号组合需满足：左括号必须以正确的顺序闭合。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # n对括号，生成n个左括号，n个右括号
        # 每次生成时候检查是否合法
        # 合法要求： 当前括号数量有剩余，且已经生成的左括号数量大于等于右括号数量【即剩下的左数量小于等于右数量】
        ans = []
        path = []
        def backtracking(path,left,right):
            if left > right or left < 0 or right < 0: 
                return 
            if len(path) == 2*n:
                ans.append("".join(path[:]))
                return 

            path.append("(")
            backtracking(path,left-1,right)
            path.pop() # 注意这两个pop
            path.append(")")
            backtracking(path,left,right-1)
            path.pop() # 注意这两个pop
        
        backtracking(path,n,n)
        return ans
```

# 23. 合并K个升序链表

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

```python
# 暴力思路。直接把数据丢到一个数组容器中，根据元素重建链表
# 这样只适用于打周赛这种快速解题。面试这么写凉透了。
```

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 方法1，两两疯狂融合
        if len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]
        a = reduce(self.merge,lists) # 这里采用了reduce语法
        return a

    def merge(self,lst1,lst2): # 传入参数为链表,返回参数为链表，使用reduce
        # 迭代融合
        dummy = ListNode(0)
        cur = dummy
        while lst1 and lst2:
            if lst1.val < lst2.val:
                cur.next = lst1
                lst1 = lst1.next
                cur = cur.next
            else:
                cur.next = lst2
                lst2 = lst2.next
                cur = cur.next
        cur.next = lst1 if lst1 else lst2
        return dummy.next
```

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 方法1，两两疯狂融合
        if len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]
        a = reduce(self.recurMerge,lists) # 这里采用了reduce语法
        return a

    def recurMerge(self,lst1,lst2): # 递归融合
        if lst1 == None: return lst2
        if lst2 == None: return lst1
        if lst1.val < lst2.val:
            lst1.next = self.recurMerge(lst1.next,lst2)
            return lst1
        else:
            lst2.next = self.recurMerge(lst1,lst2.next)
            return lst2

        
```

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 方法2: 多路合并中，每次找到最小的那个添加进来。
        # 但是，如果用普通的找min方法。则每次找min都花费了很长时间。针对这一点进行优化
        # 使用优先级队列【堆】
        # 本来就是困难题了，如果需要再加点挑战可以手写个带容器的堆。但是这个应该不是本题想考的
        # 所以利用内置容器了
        if len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]
        group = [] # 堆容器
        dummy = ListNode(0)
        cur = dummy
        for i in range(len(lists)):
            if lists[i] != None: # 这个防止[[],[],[1,2],[],[1]]这样的空表被加进容器
                heapq.heappush(group,(lists[i].val,i)) 
            # 考虑这里的容器组成应该是什么，需要节点，需要节点值，需要知道它在哪一条链表
            # 上述容器可以简化，因为有了i就可以知道节点链表得到节点，所以节点[ListNode类]不需要放进去
            # 实际上放进去如果不改写容器也比较不了
                    
        # 现在堆里是k条链表。需要注意弹出
        while len(group) != 0: # 弹空
            pair = heapq.heappop(group)
            # 利用弹出来的i值进行处理
            i = pair[1] 
            cur.next = lists[i] # # lists[i]本身就是头节点
            # print(lists[i].val) 检查用
            lists[i] = lists[i].next # 注意头部弹出后更新这个链表，并且尝试再次入堆
            cur = cur.next
            if lists[i] != None:
                heapq.heappush(group,(lists[i].val,i))
        return dummy.next
```

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 归并融合，基于方法1.每次融合一半
        # DAC: devide and conquer
        if len(lists) == 0: return None
        n = len(lists)
        return self.DAC(lists,0,n-1) # 递归的返回值为链表
        #调用递归之后会发现，原来的容器内容爆炸多
        # print(lists)
        # a = self.DAC(lists,0,n-1) # 递归的返回值为链表
        # print(lists)
    
    def DAC(self,lists,left,right): # 闭区间索引，注意递归的返回值是链表
        if left == right:
            return lists[left]
        mid = (left+right)//2
        # 左边融合
        leftPart = self.DAC(lists,left,mid)
        rightPart = self.DAC(lists,mid+1,right)
        return self.merge(leftPart,rightPart) # 这里递归迭代二选一就行

    def recurMerge(self,lst1,lst2): # 递归融合
        if lst1 == None: return lst2
        if lst2 == None: return lst1
        if lst1.val < lst2.val:
            lst1.next = self.recurMerge(lst1.next,lst2)
            return lst1
        else:
            lst2.next = self.recurMerge(lst1,lst2.next)
            return lst2  

    def merge(self,lst1,lst2): # 迭代融合
        dummy = ListNode()
        cur = dummy
        while lst1 and lst2:
            if lst1.val < lst2.val:
                cur.next = lst1
                lst1 = lst1.next
                cur = cur.next
            else:
                cur.next = lst2
                lst2 = lst2.next
                cur = cur.next
        cur.next = lst1 if lst1 else lst2
        return dummy.next
```

# 25. K 个一组翻转链表

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

进阶：

你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        step = 1
        cur = head
        while step < k and cur != None:
            cur = cur.next
            step += 1
        if step != k: # 没有办法走完，不翻转
            return head
        elif step == k: # 有k个，此时cur指向的是需要被翻转的链尾,注意由于链表的尾巴是【val】-》None
        # 所以cur有可能指向None
            if cur == None: return head
            store = cur.next # 存下来  # 递归
            cur.next = None
            cur1 = None
            cur2 = head
            tail = cur2 # 这个cur2是翻转之后的尾巴,尾巴接递归反转之后的头
            while cur2 != None: # 反转当前链
                temp = cur2.next # 存下来
                cur2.next = cur1
                cur1 = cur2
                cur2 = temp
            # 此时cur1是翻转后链表的头
            tail.next = self.reverseKGroup(store,k) # 尾巴接上去
            return cur1 # # 此时cur1是翻转后链表的头

```

# 37. 解数独

编写一个程序，通过填充空格来解决数独问题。

数独的解法需 遵循如下规则：

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
数独部分空格内已填入了数字，空白格用 '.' 表示。

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 原地修改，调用一个检查方法
        # 暴力修改，没有利用已知信息。效率可能过不了
        def check(i,j,theChar):#看是否能够填入theChar
            for t in range(9):
                if board[i][t] == theChar: return False
                if board[t][j] == theChar: return False
            # 九宫格判断逻辑,所处位置的九宫格左上角为
            a = (i//3)*3
            b = (j//3)*3
            for m in range(a,a+3):
                for n in range(b,b+3):
                    if theChar == board[m][n]: return False
            return True
        
        ans = []
        # 暴力填写，如果填完，返回，直接终止
        def backtracking(i,j): # 从左到右，从上到下
            nonlocal ans
            if i == 9:
                ans = deepcopy(board)
                # print(ans)
                return 
            if j == 9: # 换行
                backtracking(i+1,0)
                return 

            if board[i][j] != ".": # 原来就有没我屁事。直接走
                backtracking(i,j+1)
                return 

            for ch in range(1,10):
                ch = str(ch)
                if check(i,j,ch): # 说明可以填
                    board[i][j] = ch
                    backtracking(i,j+1)
                    board[i][j] = "."
        
        backtracking(0,0) # 调用方法
        for i in range(9): # 修改每一行，由于board是引用，需要修改内部的具体每一行
            board[i] = ans[i]

```

# 40. 组合总和 II

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

注意：解集不能包含重复的组合。 

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # 回溯，使用used去重复
        n = len(candidates)
        candidates.sort() # 需要预先排序
        ans = []
        used = [False for i in range(n)]

        def backtracking(choice,path,aim,index): # 参数有index，组合问题用index防止退回来选择
            if aim == 0:
                ans.append(path.copy())
                return 
            if aim < 0:
                return 
            for i in range(index,n):
                if used[i] == True: # 用过就不用了
                    continue 
                if i > 0 and choice[i] == choice[i-1] and used[i-1] == False: # 这里不能是True
                    continue 

                used[i] = True # 标记为已经选择
                path.append(choice[i])
                backtracking(choice,path,aim-choice[i],i+1) # 注意这里是i+1而不是index+1，
                path.pop()
                used[i] = False
        
        backtracking(candidates,[],target,0)
        return ans
```

# 43. 字符串相乘

给定两个以字符串形式表示的非负整数 `num1` 和 `num2`，返回 `num1` 和 `num2` 的乘积，它们的乘积也表示为字符串形式。

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        # 类似于cpu的乘法的ALU处理
        # 取更长的那个作为基数，然后一位位累加
        if num1 == "0" or num2 == "0":
            return "0"
            
        def multSingle(n,single): # 传入类型为字符串
            carry = 0
            ans = ""
            n = n[::-1]
            for i in n:
                p = (int(i)*int(single)+carry) % 10 
                carry = (int(i)*int(single)+carry) // 10 
                ans = str(p) + ans
            if carry != 0:
                ans = str(carry) + ans
            return ans
        
        def merge(up,down,offset): # 传入类型为字符串
            down += "0"*offset
            up = up[::-1]
            down = down[::-1]
            carry = 0
            p = 0
            ans = ""
            while p < len(up) and p < len(down):
                temp = (int(up[p])+int(down[p])+carry)%10
                carry = (int(up[p])+int(down[p])+carry)//10
                ans = str(temp) + ans 
                p += 1
            while p < len(up):
                temp = (int(up[p])+carry)%10
                carry = (int(up[p])+carry)//10
                ans = str(temp) + ans 
                p += 1
            while p < len(down):
                temp = (int(down[p])+carry)%10
                carry = (int(down[p])+carry)//10
                ans = str(temp) + ans 
                p += 1
            if carry != 0:
                ans = str(carry) + ans 
            return ans
        
        offset = 0 # 偏移量
        l = "0"
        for n in num2[::-1]:
            temp = multSingle(num1,n)
            l = merge(l,temp,offset)
            offset += 1
        return l
```

# 51. N 皇后

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 一个判断是否合理的逻辑，回溯
        # python字符串不支持修改，先用数组代替
        graph = [["." for i in range(n)] for j in range(n)]
        # 放置时检查。调用检查函数
        # 每行检查
        ans = [] # 收集结果
        def backtracking(i): # 到达最后一行，ans存储
            if i == n:
                temp = []
                for line in graph: # 注意这里需要转换成题目接收格式
                    temp.append(''.join(line))
                ans.append(temp)
                return 
            for j in range(n):                
                if self.check(graph,i,j,n): # 检查是否可以放置
                    graph[i][j] = "Q" # 放置
                    backtracking(i+1) # 搜索
                    graph[i][j] = "." # 取消放置

        backtracking(0)
        return ans
    
    def check(self,graph,i,j,n):# 检查i，j,由于是从上往下填充，无需检查同行，无需检查下方
        direc = [(-1,0),(-1,-1),(-1,1)] # 只需要检查三个方向
        now_j = i 
        now_i = j
        for di in direc:
            now_i = i # 每轮都要初始化
            now_j = j 
            while 0<=now_i<n and 0<=now_j<n:
                if graph[now_i][now_j] == "Q":
                    return False # 不可放置
                now_i += di[0]
                now_j += di[1]
        return True # 可以放置
            
```

# 52. N皇后 II

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量。

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        # 上一题简化版本。实际上甚至可以用数字的二进制代替图的存储
        # 先暂时用正常存储
        graph = [["." for i in range(n)] for j in range(n)]
        ans = 0 # 记录

        def backtracking(i):
            nonlocal ans
            if i == n:
                ans += 1
                return 
            for j in range(n):
                if self.check(graph,i,j,n):
                    graph[i][j] = "Q"
                    backtracking(i+1)
                    graph[i][j] = "."
        
        backtracking(0)
        return ans
    
    def check(self,graph,i,j,n):
        direc = [(-1,0),(-1,1),(-1,-1)] # 只需要接受三个反向
        for di in direc:
            new_i = i # 每轮次重置
            new_j = j 
            while 0<=new_i<n and 0<=new_j<n:
                if graph[new_i][new_j] == "Q": # 筛选不通过
                    return False
                new_i += di[0]
                new_j += di[1]
        return True # 筛选通过
```



# 60. 排列序列

给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：

"123"
"132"
"213"
"231"
"312"
"321"
给定 n 和 k，返回第 k 个排列。

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        # dfs搜索？纯暴力法
        ans = []
        path = []
        lst = [str(i+1) for i in range(n)] # 构建选择列表

        def backtracking(path,choice,ans): # lst有序搜出来的已经排序好了
            if len(ans) > k: # 剪枝，到了k就收拢
                return 
            if len(path) == n:
                ans.append("".join(path))
                return
            for i in choice:
                cp = choice.copy() # 全排列使用备份法回溯
                cp.remove(i)
                path.append(i)
                backtracking(path,cp,ans)
                path.pop()

        backtracking(path,lst,ans)
        
        return ans[k-1]
            
```

```python
# 优化版,官解
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        def dfs(n, k, index, path):
            if index == n:
                return
            cnt = factorial[n - 1 - index]
            for i in range(1, n + 1):
                if used[i]:
                    continue
                if cnt < k:
                    k -= cnt
                    continue
                path.append(i)
                used[i] = True
                dfs(n, k, index + 1, path)
                # 注意：这里要加 return，后面的数没有必要遍历去尝试了
                return

        if n == 0:
            return ""

        used = [False for _ in range(n + 1)]
        path = []
        factorial = [1 for _ in range(n + 1)]
        for i in range(2, n + 1):
            factorial[i] = factorial[i - 1] * i

        dfs(n, k, 0, path)
        return ''.join([str(num) for num in path])

```

# 68. 文本左右对齐

给定一个单词数组和一个长度 maxWidth，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。

你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。

要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。

文本的最后一行应为左对齐，且单词之间不插入额外的空格。

说明:

单词是指由非空格字符组成的字符序列。
每个单词的长度大于 0，小于等于 maxWidth。
输入单词数组 words 至少包含一个单词。

```python
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # 一个line栈尝试填充，长度被maxWidth限制。记录空格数量进行尽量靠左的分配
        # 每次尝试加入一个单词，判断当前行字母数目+单词数目-1是否越界。不越界则继续加。越界则另取一行
        line = []
        tempLength = 0
        count = 0 # 记录当前这一行
        ans = []
        for w in words:
            tempLength += len(w)
            count += 1 
            if tempLength + count - 1 <= maxWidth: # 尝试加入，如果没有越界，则真加入
                line.append(w)
            elif tempLength + count - 1 > maxWidth: # 越界 另写一行。重置
                # 先收集。l
                ans.append(line)
                tempLength = len(w) # 重置
                count = 1 # 重置
                line = [w]
        if len(line) != 0: # 擦屁股
            ans.append(line)
        
        # 此时每行已经收集好，进行格式化
        def formatList(line,limit):
            # 需要确定单词数目和空格数目
            n = len(line) # 单词数量
            allLength = 0 # 字符总长度
            for w in line:
                allLength += len(w)
            if n == 1:
                temp = str(line[0])
                add = limit - len(temp)
                final = temp+' '*add
                return final
            blanks = limit - allLength # 空格总数
            every = blanks//(n-1) # 先向下取整
            extra = blanks - every*(n-1) # 看盈余多少空格
            final = []
            for i in line:
                if extra > 0:
                    extra -= 1
                    final.append(i)
                    final.append(" "*(every+1))
                else:
                    final.append(i)
                    final.append(" "*every)
            final.pop() # 去掉最后一个
            return ''.join(final)
        
        # 注意最后一行的处理
        for i in range(len(ans)-1):
            ans[i] = formatList(ans[i],maxWidth)
        
        store = ans[-1] # 暂存
        t1 = " ".join(store)
        add = maxWidth - len(t1)
        ans[-1] = t1 + add*" "
        return ans
```

# 72. 编辑距离

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符

```python
# 方法1:基于记忆化递归
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # 基于记忆化递归的编辑距离
        # 首先确定[i,j]是从后往前移动，有效索引为0~n-1闭区间
        memoDict = dict()

        def dp(i,j):
            nonlocal memoDict # dict其实不需要nonlocal
            if (i,j) in memoDict:
                return memoDict[(i,j)]
            if i == -1:
                return j + 1
            if j == -1:
                return i + 1
            # 操作有，插入，删除，替换，和不变化
            if word1[i] == word2[j]:
                memoDict[(i,j)] = dp(i-1,j-1)
            else:
                memoDict[(i,j)] = min(
                    dp(i,j-1) + 1,  # j往前移动，可以理解为word1[i]插入了一个匹配j的。
                    dp(i-1,j) + 1,  # i往前移动，说明word1[i]被匹配，说明可以是删除
                    dp(i-1,j-1) + 1, # 替换
                )
            return memoDict[(i,j)]
        
        return dp(len(word1)-1,len(word2)-1)
```

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)
        dp = [[0xffffffff for j in range(n+1)] for i in range(m+1)] # 初始化为极大值
        # dp[i][j]的意思是长度为i的word1和长度为j的word2的最小编辑距离
        # 由于会达到dp[m][n] 所以开的数组长度为range(n+1)和range(m+1)
        # 基态为dp[0][...] 和 dp[...][0]
        for i in range(m+1):
            dp[i][0] = i 
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1,m+1):
            for j in range(1,n+1):
                # 如果两个字符相同，那么操作次数为去尾巴的操作次数
                # 注意这个word[index-1]
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                elif word1[i-1] != word2[j-1]:
                    dp[i][j] = min(
                        dp[i-1][j] + 1, # 不要word1的最后一个字符，即为删除
                        dp[i][j-1] + 1, # 不要word2的最后一个字符，即在word1加入字符匹配掉它：插入
                        dp[i-1][j-1] + 1 # 替换
                    )
        # 填充完毕之后返回右下角
        return dp[-1][-1]

```



# 93. 复原 IP 地址

给定一个只包含数字的字符串，用以表示一个 IP 地址，返回所有可能从 s 获得的 有效 IP 地址 。你可以按任何顺序返回答案。

有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        # 回溯
        n = len(s) 
        if n < 4 or n > 12: # 直接判空
            return []
        
        ans = [] #收集答案
        path = [] # 总体路径
        part = [] # 分段路径
        
        def judgeValid(lst): # 判断每个part是否合理
            lst = "".join(lst)
            if len(lst) == 0:
                return False
            if len(lst) == 1:
                return True
            if len(lst) >= 2 and lst[0] == "0": # 不能有前导0
                return False
            if 0<=int(lst)<=255:
                return True
            else:
                return False

        def backtracking(path,part,index,times): # 回溯：参数分别为路径，分段，选取索引，加.次数
            if times == 3 and index == n: # 有3个点且收集了全部元素
                # 收集
                part = "".join(part) # 注意part里面有需要的数据，需要合并进temp，然后收集
                temp = "".join(path)
                temp += part
                ans.append(temp)
                return 
            if index >= len(s) :
                return
            # 选择有两种方式，一种是将数字加入part。一种是加点
            # 1. 加数字
             
            part.append(s[index])
            if judgeValid(part): # 合理，继续搜
                backtracking(path,part,index+1,times)
            part.pop()

            # 2. 加点
            if judgeValid(part):
                path += ["".join(part)]
                path += ["."]
                backtracking(path,[],index,times+1)
                path.pop()
                path.pop()
        
        backtracking([],[],0,0)
        ans.sort() # 方便检查答案。不加这一行也没事
        return ans



```

# 96. 不同的二叉搜索树

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

```python
class Solution:
    def numTrees(self, n: int) -> int:
        # 动态规划
        dp = [0 for i in range(n+1)] # dp[i]为以i为头节点的树的数量
        # 以3个节点为例：
        # 以1为头节点，还剩下2个【即递归到了dp[2]]
        # 以2为头节点，左右各自剩下一个【即递归到了 dp[1] + dp[1]]
        # 以3为头节点，左还需要两个【即递归到了dp[2]】
        # 状态转移方程为 dp[i] += dp[left]*dp[right]。如果有一侧树为空，则将值定为1
        dp[0] = 1 # 为了统一语法
        dp[1] = 1
        for i in range(2,n+1):
            for gap in range(0,i):
                left = i - gap - 1
                right = gap
                dp[i] += dp[left]*dp[right]
        return dp[-1]

```

# 124. 二叉树中的最大路径和

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        # 使用后续遍历
        ans = -0xffffffff

        def postOrder(node):
            nonlocal ans
            if node == None:
                return 0
            leftTree = postOrder(node.left)
            rightTree = postOrder(node.right)

            # 如果左子树小于0，则不把它参与计算，如果右子树小于0，则不把它参与计算
            val = node.val
            if leftTree > 0:
                val += leftTree
            if rightTree > 0:
                val += rightTree
            ans = max(ans,val)
            # 到目前为止的路径计算为:
            maxNow = max(node.val+leftTree,node.val+rightTree,0) # 注意这个0。。。
            return maxNow
        
        postOrder(root)
        return ans
```

# 128. 最长连续序列

给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # 方法1:
        # 朴素方法，nlogn
        nums = set(nums)
        nums = [i for i in nums]
        nums.sort()
        if len(nums) == 0:
            return 0

        maxLength = 1            
        prev = nums[0]
        p = 1
        tempLength = 1
        while p < len(nums):
            if nums[p] == prev + 1:
                tempLength += 1
            elif nums[p] != prev + 1:
                tempLength = 1 # 重置

            prev = nums[p]
            if tempLength > maxLength:
                maxLength = tempLength
            p += 1
        return maxLength
```

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

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
    # 方法2: 并查集 但是用到了排序而不是路径压缩
    # nlogn
        # 由于没有按照路径压缩，所以需要使用从小到大的顺序连接。
        nums.sort()
        theDict = {nums[i]:i for i in range(len(nums))}
        # k-v 是 值:索引
        ufSet = UF(len(nums))
        for n in nums:
            if (n-1) in theDict: # 这里x，y顺序不能换
                x = theDict[n-1]
                y = theDict[n]
                ufSet.union(x,y)
            if (n+1) in theDict: # 这里x，y顺序不能换
                x = theDict[n+1]
                y = theDict[n]
                ufSet.union(x,y)
        ct = collections.Counter(ufSet.root)
        maxVal = 0
        for i in ct:
            if ct[i] > maxVal:
                maxVal = ct[i]
        return maxVal

```

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # 进阶枚举 On
        # 朴素暴力解是两层for循环来做,内层每次找+1
        # 优化：从n开始枚举肯定比n-1开始枚举短
        if len(nums) == 0:
            return 0
        nums = set(nums)
        maxLength = 1
        for n in nums:
            if n-1 in nums: # 开启下一轮，因为从n开始枚举肯定比n-1开始枚举短
                continue
            now = n 
            tempLength = 0
            while now in nums:
                tempLength += 1
                now += 1
            maxLength = max(tempLength,maxLength)
        return maxLength

```

# 133. 克隆图

给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

class Node {
    public int val;
    public List<Node> neighbors;
}

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:

    def cloneGraph(self, node: 'Node') -> 'Node':
        # 需要一个非重复的遍历和拷贝传入的是节点而不是原表
        visited = collections.defaultdict(list)

        def dfs(n):
            if n == None:
                return 
            if n.val in visited:
                return 
            c = Node(n.val,[])
            for neigh in n.neighbors:
                visited[n.val].append(neigh.val)
                dfs(neigh)
        #print(node)
        dfs(node) # 调用
        # print(visited) # 此时收集到了全部点
        nodeDict = dict()
        for key in visited:
            nodeDict[key] = Node(key)
        
        # print(nodeDict)
        for key in visited:
            n = visited[key]
            for i in n:
                nodeDict[key].neighbors.append(nodeDict[i])
        if node != None and len(nodeDict) == 0: # 单点图
            return Node(1)
        return nodeDict.get(1) # 防止空图
        
```

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:

    def cloneGraph(self, node: 'Node') -> 'Node':
        # 需要一个非重复的遍历和拷贝传入的是节点而不是原表
        if node == None:
            return None
        visited = dict()

        def dfs(n):
            if n == None:
                return 
            if n.val in visited:
                return visited[n.val] # 注意这一条
            copyNode = Node(n.val,[])
            visited[n.val] = copyNode
            for neigh in n.neighbors:
                copyNode.neighbors.append(dfs(neigh))
            return copyNode

        dfs(node)
        return visited.get(1)
```




# 147. 对链表进行插入排序

插入排序算法：

插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
重复直到所有输入数据插入完为止。

```python
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        # 看动画理解插排
        # 一个主指针，一个scan指针
        if head == None or head.next == None:
            return head

        def insert(main,node):
            if main == None: # 空表插入
                return node

            if main.next == None: # 一个节点的插入
                if node.val <= main.val:
                    node.next = main
                    return node
                else:
                    main.next = node
                    return main

            if node.val <= main.val: # 头插
                node.next = main
                return node

            # 通常插入
            cur1 = main
            cur2 = main.next
            
            while cur2 != None:
                if node.val <= cur2.val:                    
                    cur1.next = node
                    node.next = cur2
                    return main
                else:
                    cur1 = cur1.next
                    cur2 = cur2.next

            cur1.next = node # 比所有元素都大，尾插
            return main
                    
        complete = None # 这边是已经排序的链表
        start = head

        while start != None:
            temp = start.next # 存下
            start.next = None # 变成孤立节点
            complete = insert(complete,start) # 插入
            start = temp

        return complete

```

```python
# 官方解
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        dummyHead = ListNode(0)
        dummyHead.next = head
        lastSorted = head
        curr = head.next

        while curr:
            if lastSorted.val <= curr.val:
                lastSorted = lastSorted.next
            else:
                prev = dummyHead
                while prev.next.val <= curr.val:
                    prev = prev.next
                lastSorted.next = curr.next
                curr.next = prev.next
                prev.next = curr
            curr = lastSorted.next
        
        return dummyHead.next

```



# 148. 排序链表

给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

进阶：

你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 递归合并
        if head == None or head.next == None:
            return head
        cur1 = head
        cur2 = self.devideList(head)

        cur1 = self.sortList(cur1)
        cur2 = self.sortList(cur2)

        return self.mergeList(cur1,cur2)
    

    def devideList(self,head):
        if head == None or head.next == None:
            return head
        slow = head
        fast = head.next
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        temp = slow.next
        slow.next = None
        return temp
    
    def mergeList(self,lst1,lst2):
        if lst1 == None: return lst2
        if lst2 == None: return lst1
        if lst1.val < lst2.val:
            lst1.next = self.mergeList(lst1.next,lst2)
            return lst1
        else:
            lst2.next = self.mergeList(lst1,lst2.next)
            return lst2

```

```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 归并排序
        if head == None or head.next == None:
            return head
        cur1 = head
        cur2 = self.split(head)

        cur1 = self.sortList(cur1)
        cur2 = self.sortList(cur2)

        return self.merge(cur1,cur2)
    
    def split(self,head):
        if head == None:
            return head
        slow = head
        fast = head.next
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        # 还需要断开
        temp = slow.next
        slow.next = None
        return temp
    
    def merge(self,lst1,lst2): # 迭代递归
        dummy = ListNode()
        cur = dummy
        while lst1 and lst2:
            if lst1.val < lst2.val:
                cur.next = lst1
                lst1 = lst1.next
                cur = cur.next
            else:
                cur.next = lst2
                lst2 = lst2.next
                cur = cur.next
        cur.next = lst1 if lst1 else lst2
        return dummy.next 
```

# 161. 相隔为 1 的编辑距离

给定两个字符串 s 和 t，判断他们的编辑距离是否为 1。

注意：

满足编辑距离等于 1 有三种可能的情形：

往 s 中插入一个字符得到 t
从 s 中删除一个字符得到 t
在 s 中替换一个字符得到 t

```python
# 直接计算编辑距离会超时
class Solution:
    def isOneEditDistance(self, s: str, t: str) -> bool:
        # 编辑距离超时
        # 采用朴素解法
        if abs(len(s) - len(t)) > 1:
            return False 
        if len(s) > len(t):
            return self.isOneEditDistance(t,s) # 交换成第一个串比较短
        for i in range(len(s)):
            if s[i] != t[i]:
                if len(s) == len(t): # 长度一样，跳过这一位,代表替换
                    return s[i+1:] == t[i+1:] 
                else: # 长度不一样，且t长一位，那么尝试删除t的那一个
                    return s[i:] == t[i+1:]
        # 如果for里面没有被比较出来，分为两者长度相等和不相等考虑，如果全等也不行。必须差1
        if len(s) == len(t):
            return False
        else:
            return True
```

# 162. 寻找峰值

峰值元素是指其值严格大于左右相邻值的元素。

给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞ 。

你必须实现时间复杂度为 O(log n) 的算法来解决此问题。

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        # 二分搜索
        bound = -0xffffffff
        nums = [bound] + nums + [bound] # 预先处理
        left = 1
        right = len(nums) - 2
        # 返回的索引要记得减去1
        while left <= right: # 注意这个二分条件
            mid = (left+right)//2
            lmid = mid - 1
            rmid = mid + 1
            if nums[lmid] < nums[mid] and nums[mid] > nums[rmid]:
                return mid - 1 
            elif nums[lmid] < nums[mid] < nums[rmid]:
                left = mid + 1
            elif nums[lmid] > nums[mid] > nums[rmid]:
                right = mid - 1
            elif nums[lmid] > nums[mid] and nums[mid] < nums[rmid]:
                left = mid + 1 # right = mid - 1 也可以
```

# 165. 比较版本号

给你两个版本号 version1 和 version2 ，请你比较它们。

版本号由一个或多个修订号组成，各修订号由一个 '.' 连接。每个修订号由 多位数字 组成，可能包含 前导零 。每个版本号至少包含一个字符。修订号从左到右编号，下标从 0 开始，最左边的修订号下标为 0 ，下一个修订号下标为 1 ，以此类推。例如，2.5.33 和 0.1 都是有效的版本号。

比较版本号时，请按从左到右的顺序依次比较它们的修订号。比较修订号时，只需比较 忽略任何前导零后的整数值 。也就是说，修订号 1 和修订号 001 相等 。如果版本号没有指定某个下标处的修订号，则该修订号视为 0 。例如，版本 1.0 小于版本 1.1 ，因为它们下标为 0 的修订号相同，而下标为 1 的修订号分别为 0 和 1 ，0 < 1 。

返回规则如下：

如果 version1 > version2 返回 1，
如果 version1 < version2 返回 -1，
除此之外返回 0。

```python
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        # version不包括负号,两者都非空。
        # 用.分隔之后转为数字
        s1 = version1.split(".")
        s2 = version2.split(".")

        for i in range(len(s1)):
            s1[i] = int(s1[i]) # int去除前导0
        for i in range(len(s2)):
            s2[i] = int(s2[i]) # int去除前导0
        
        # 两者挨个比较
        while len(s1) != 0 and len(s2) != 0:
            p1 = s1.pop(0)
            p2 = s2.pop(0)
            if p1 > p2:
                return 1
            elif p1 < p2:
                return -1
            elif p1 == p2:
                continue
            
        if sum(s1) == sum(s2): # 此时有一个为空，另一个如果里面是全0，则相等
            return 0
        elif sum(s1) > sum(s2):
            return 1
        elif sum(s1) < sum(s2):
            return -1
```

# 166. 分数到小数

给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以 字符串形式返回小数 。

如果小数部分为循环小数，则将循环的部分括在括号内。

如果存在多个答案，只需返回 任意一个 。

对于所有给定的输入，保证 答案字符串的长度小于 104 。

```python
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        if numerator%denominator == 0:
            return str(numerator//denominator)
        
        s = []
        if (numerator) * denominator < 0:
            s.append("-")
        
        # 整数部分
        numerator = abs(numerator)
        denominator = abs(denominator)
        integer = numerator//denominator
        s.append(str(integer))
        s.append(".")

        # 小数部分
        memoDict = {}
        remain = numerator % denominator
        while remain != 0 and remain not in memoDict:
            memoDict[remain] = len(s) # 记录的是长度
            remain *= 10
            s.append(str(remain//denominator))
            remain = remain % denominator
        
        print(memoDict,s)
        if remain != 0:
            insertIndex = memoDict[remain]
            s.insert(insertIndex,"(")
            s.append(")")

        return "".join(s)
```

# 200. 岛屿数量

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # 超级dfs
        ans = 0 # 记录岛屿数量
        area = False # 记录每一次dfs的面积
        m = len(grid)
        n = len(grid[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(1,0),(-1,0)]

        def judgeValid(i,j):
            if 0<=i<m and 0<=j<n and grid[i][j] == "1":
                return True
            return False
        
        def dfs(i,j):
            nonlocal area
            if not judgeValid(i,j): # 不合法 返回
                return 
            if visited[i][j]: # 访问过了返回
                return #
            visited[i][j] = True # 设置为已访问
            area = True
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(new_i,new_j)
        
        for i in range(m):
            for j in range(n):
                dfs(i,j)
                if area != False:
                    ans += 1
                    area = False # 重置
        
        return ans 
```

# 207. 课程表

你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 拓扑排序弱化版本，符合的结果直接在长度上+1
        n = numCourses
        # 有向顺序[a,b]: a <- b
        inDegree = [0 for i in range(n)]
        graph = collections.defaultdict(list)
        for a,b in prerequisites:
            inDegree[a] += 1
            graph[b].append(a)
        queue = collections.deque()
        for i in range(n):
            if inDegree[i] == 0:
                queue.append(i)
        ans = 0 # 记录长度
        while len(queue) != 0:
            e = queue.popleft()
            ans += 1
            # 处理邻居
            for neigh in graph[e]:
                inDegree[neigh] -= 1
                if inDegree[neigh] == 0:
                    queue.append(neigh)
        return n == ans
```



# 210. 课程表 II

现在你总共有 n 门课需要选，记为 0 到 n-1。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]

给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。

可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

```python
# 非建图版拓扑排序,耗时在找邻居
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 拓扑排序
        n = numCourses
        ans = [] # 收集答案
        # 统计所有元素的入度，找到入度为0的
        indegree = [0 for i in range(n)]
        # [a,b],a的入度+1 # 有向顺序为 b -> a
        for a,b in prerequisites:
            indegree[a] += 1
        pre = -1 # 初始化
        while len(ans) != pre:
            queue = collections.deque() # 收集所有入度为0的
            pre = len(ans)
            for i in range(n):
                if indegree[i] == 0:
                    queue.append(i)
            while len(queue) != 0:
                element = queue.popleft()
                indegree[element] = -1 # 注意这一行，代替vistied数组
                ans.append(element) # 加进结果集
                # 还需要根据element拓扑排序，移除b之后，a的入度减少
                for a,b in prerequisites:
                    if b == element:
                        indegree[a] -= 1
        if len(ans) == n:
            return ans
        else:
            return [] 
```

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 拓扑排序
        # 建图版
        n = numCourses
        inDegree = [0 for i in range(n)]
        graph = collections.defaultdict(list)
        ans = [] 
        # a <- b 注意图的方向顺序。 
        for a,b in prerequisites:
            inDegree[a] += 1
            graph[b].append(a)        
        # 把所有入度为0的节点放进队列
        queue = collections.deque()
        for i in range(n):
            if inDegree[i] == 0:
                queue.append(i)
        while len(queue) != 0:
            e = queue.popleft() # 取出
            ans.append(e) # 添加进答案
            for neigh in graph[e]: # 处理邻居的入度
                inDegree[neigh] -= 1
                if inDegree[neigh] == 0:
                    queue.append(neigh)
        # 处理完之后，看是否所有节点都被加入,可以用三目运算符简化，因为我还写go。所以没有用三目的习惯
        if len(ans) == n:
            return ans 
        else:
            return []

```

# 211. 添加与搜索单词 - 数据结构设计

请你设计一个数据结构，支持 添加新单词 和 查找字符串是否与任何先前添加的字符串匹配 。

实现词典类 WordDictionary ：

WordDictionary() 初始化词典对象
void addWord(word) 将 word 添加到数据结构中，之后可以对它进行匹配
bool search(word) 如果数据结构中存在字符串与 word 匹配，则返回 true ；否则，返回  false 。word 中可能包含一些 '.' ，每个 . 都可以表示任何一个字母。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.isEnd = False 
    
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root 
        for ch in word:
            index = ord(ch)-ord("a")
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isEnd = True

    def dfs(self,node,word,wordIndex): # 注意这个dfs,没有剪枝
        if node == None:
            return False 
        if node.isEnd and wordIndex == len(word):
            return True 
        if 0<=wordIndex<len(word):  # index没有越界的时候

            if word[wordIndex] == ".": # 该字符是通配符。dfs调用，只要找到就返回True
                for i in range(26):
                    if self.dfs(node.children[i],word,wordIndex+1) == True:
                        return True
            elif word[wordIndex] != ".":
                index = ord(word[wordIndex]) - ord("a")
                # print(word[wordIndex],index)
                return self.dfs(node.children[index],word,wordIndex+1)
                
        return False # 其余情况，index越界，返回False

    def search(self, word: str) -> bool:
        return self.dfs(self.root,word,0)

```

# 212. 单词搜索 II

给定一个 m x n 二维字符网格 board 和一个单词（字符串）列表 words，找出所有同时在二维网格和字典中出现的单词。

单词必须按照字母顺序，通过 相邻的单元格 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。

```python
# 尬解版，超暴力和不怎么好的剪枝
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        m = len(board)
        n = len(board[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(-1,0),(1,0)]
        ans = []
        visitedWord = dict()
        for w in words:
            visitedWord[w] = False
        
        alphaDict = [False for i in range(26)]
        for i in range(m):
            for j in range(n):
                index = ord(board[i][j]) - ord("a")
                alphaDict[index] = True 

        def toAD(s):
            theDict = [False for i in range(26)]
            for ch in s:
                index = ord(ch) - ord("a")
                theDict[index] = True 
            return theDict

        def dfs(w,index,visited,i,j):
            if index >= len(w):
                return
            if index >= 10:
                return  
            if board[i][j] == w[index]:
                if index == len(w) - 1 and visitedWord[w] == False:
                    ans.append(w)
                    visitedWord[w] = True # 表明已经搜到过了
                    return 
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False:
                        visited[new_i][new_j] = True # 注意这个回溯
                        dfs(w,index+1,visited,new_i,new_j)
                        visited[new_i][new_j] = False # 注意这个回溯

        for i in range(m):
            for j in range(n):
                for w in words:
                    if board[i][j] == w[0] and visitedWord[w] == False: # 以这个开头并且还没有搜到过
                        dic1 = toAD(w)
                        state = True 
                        for t in range(26): #看字母是否过量。过量直接不搜
                            if dic1[t] == True and alphaDict[t] == False:
                                state = False
                                break
                        if state:
                            visited = [[False for j in range(n)] for i in range(m)]
                            visited[i][j] = True
                            dfs(w,0,visited,i,j)

        return ans

```

# 213. 打家劫舍 II

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 强化版本打家劫舍，取决于是否选0
        # 1. 选0则在0~n-2进行dp
        # 2. 不选0则在1～n-1进行dp
        n = len(nums)
        if n == 1:
            return nums[0]
        
        def calc_dp(lst,start,end): # 参数为闭区间参数
            length = len(lst)
            if length == 0:
                return 0
            elif length == 1:
                return lst[0]
            elif length == 2:
                return max(lst[0],lst[1])
            length -= 1
            lst = lst[start:end+1]
            dp = [0 for i in range(length)]
            dp[0] = lst[0]
            dp[1] = max(lst[0],lst[1])
            for i in range(2,length):
                dp[i] = max(dp[i-1],dp[i-2]+lst[i])
            return dp[-1]
        
        situation1 = calc_dp(nums,0,n-2)
        situation2 = calc_dp(nums,1,n-1)
        return max(situation1,situation2)
```

# 220. 存在重复元素 III

给你一个整数数组 nums 和两个整数 k 和 t 。请你判断是否存在 两个不同下标 i 和 j，使得 abs(nums[i] - nums[j]) <= t ，同时又满足 abs(i - j) <= k 。

如果存在则返回 true，不存在返回 false。

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:        
        # 先采用TreeMap做法
        from sortedcontainers import SortedList
        theSet = SortedList()
        n = len(nums)
        for i in range(n):
            index = theSet.bisect_left(nums[i])
            if 0 <= index < len(theSet):
                if abs(theSet[index]-nums[i]) <= t:
                    return True 
            if 0 <= index-1 < len(theSet):
                if abs(theSet[index-1]-nums[i]) <= t:
                    return True 
            theSet.add(nums[i])
            if len(theSet) > k:
                theSet.remove(nums[i-k])
        return False

```

# 223. 矩形面积

给你 二维 平面上两个 由直线构成的 矩形，请你计算并返回两个矩形覆盖的总面积。

每个矩形由其 左下 顶点和 右上 顶点坐标表示：

第一个矩形由其左下顶点 (ax1, ay1) 和右上顶点 (ax2, ay2) 定义。
第二个矩形由其左下顶点 (bx1, by1) 和右上顶点 (bx2, by2) 定义。

```python
class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        # 获取相交区间再去计算
        # 由于ay1,ay2等已定相对大小
        def getIntersection(lst1,lst2):
            # 排序，使得lst1在lst2的左边
            cp = [lst1,lst2]
            cp.sort()
            lst1,lst2 = cp[0],cp[1]
            a = lst1[0]
            b = lst1[1]
            c = lst2[0]
            d = lst2[1]
            # 这里画图辅助理解
            if a <= c <= b <= d:
                return b-c 
            elif a <= c <= d <= b:
                return d-c 
            else:
                return 0
        
        width = getIntersection([ax1,ax2],[bx1,bx2])
        height = getIntersection([ay1,ay2],[by1,by2])

        def getArea(p1,p2):
            wid = abs(p1[0]-p2[0])
            hei = abs(p1[1]-p2[1])
            return wid * hei 

        p1 = (ax1,ay1)
        p2 = (ax2,ay2)

        q1 = (bx1,by1)
        q2 = (bx2,by2)

        return getArea(p1,p2)+getArea(q1,q2) - width * height

```

# 224. 基本计算器

给你一个字符串表达式 `s` ，请你实现一个基本计算器来计算并返回它的值。

- `1 <= s.length <= 3 * 105`

- `s` 由数字、`'+'`、`'-'`、`'('`、`')'`、和 `' '` 组成

- `s` 表示一个有效的表达式

  ```python
  class Solution:
      def calculate(self, s: str) -> int:
          # 没有乘除法，使用递归完成
          # 都基于同一个lst引用进行计算
          # 需要开双端队列防止pop(0)的性能消耗
          
          def submethod(lst):
              stack = []
              num = 0
              symbol = "+"
  
              while len(lst) > 0: # 不用for扫描法是因为难处理,都开if而不是一个if带一堆elif. 分析的时候能发现elif是错误的
                  ch = lst.popleft()
                  if ch.isdigit():
                      num = 10*num + int(ch)
                  if ch == "(":
                      num = submethod(lst) # 对当前已经pop过括号的进行递归计算
  
                  if (ch.isdigit() == False and ch != " ") or len(lst) == 0: # 注意这个层级不是elif
                      if symbol == "+":
                          stack.append(num)
                      elif symbol == "-":
                          stack.append(-num)
                      # elif symbol == "*":
                      #     stack[-1] *= num 
                      # elif symbol == "/":
                      #     stack[-1] = int(stack[-1]/num)
                      symbol = ch 
                      num = 0
                  
                  if ch == ")":
                      break 
              return sum(stack)
  
          s = deque(s)
          return submethod(s)
  ```

# 250. 统计同值子树

给定一个二叉树，统计该二叉树数值相同的子树个数。

同值子树是指该子树的所有节点都拥有相同的数值。

```python
class Solution:
    def countUnivalSubtrees(self, root: TreeNode) -> int:
        # 由于需要看子树，则后续遍历
        # 
        if root == None:
            return 0
        
        ans = 0

        def postOrder(node):
            nonlocal ans
            if node == None:
                return True

            state = False # 默认标记为False，表示该节点左右不同值

            leftState = postOrder(node.left)
            rightState = postOrder(node.right)
            if node.left != None:
                leftState = (leftState and node.left.val == node.val)
            if node.right != None:
                rightState = (rightState and node.right.val == node.val)
            state = (leftState and rightState)
            if state:
                ans += 1
            return state
        
        postOrder(root)
        return ans
```

# 251. 展开二维向量

请设计并实现一个能够展开二维向量的迭代器。该迭代器需要支持 `next`和 `hasNext` 两种操作。

```python
# 迭代器做法
class Vector2D:

    def __init__(self, vec: List[List[int]]):
        self.stack = []
        self.vec = vec 
        if self.vec != []:
            while self.vec != [] and self.vec[0] == []: # 有可能第一个就是空的,也有可能一直是空的，循环pop
                self.vec.pop(0)
            if self.vec != []:
                e = self.vec[0].pop(0)
                self.stack.append(e)
                if self.vec[0] == []:
                    self.vec.pop(0)

    def next(self) -> int:
        if len(self.stack) != 0:
            val = self.stack.pop(0)
        if len(self.stack) == 0:
            if self.vec != []:
                while self.vec != [] and self.vec[0] == []: # 有可能第一个就是空的,也有可能一直是空的，循环pop
                    self.vec.pop(0)
                if self.vec != []:
                    e = self.vec[0].pop(0)
                    self.stack.append(e)
                    if self.vec[0] == []:
                        self.vec.pop(0)

        return val

    def hasNext(self) -> bool:
        return self.stack != []

```



# 256. 粉刷房子

假如有一排房子，共 n 个，每个房子可以被粉刷成红色、蓝色或者绿色这三种颜色中的一种，你需要粉刷所有的房子并且使其相邻的两个房子颜色不能相同。

当然，因为市场上不同颜色油漆的价格不同，所以房子粉刷成不同颜色的花费成本也是不同的。每个房子粉刷成不同颜色的花费是以一个 n x 3 的正整数矩阵 costs 来表示的。

例如，cost\[0][0] 表示第 0 号房子粉刷成红色的成本花费；costs\[1][2] 表示第 1 号房子粉刷成绿色的花费，以此类推。

请计算出粉刷完所有房子最少的花费成本。

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        # dp 关键是用三行dp
        n = len(costs)
        dp = [[0xffffffff for i in range(n)] for j in range(3)] # 初始化为极大值
        for i in range(3):
            dp[i][0] = costs[0][i] # 初始化
        # 状态转移为贪心思路，选取另两行前一个比较小的
        for j in range(1,n):
            dp[0][j] = min(dp[1][j-1],dp[2][j-1]) + costs[j][0]
            dp[1][j] = min(dp[0][j-1],dp[2][j-1]) + costs[j][1]
            dp[2][j] = min(dp[0][j-1],dp[1][j-1]) + costs[j][2]
        
        temp = []
        for i in range(3): # 收集最后一列的三个，返回最小值
            temp.append(dp[i][-1])
        return min(temp)

```

# 281. 锯齿迭代器

给出两个一维的向量，请你实现一个迭代器，交替返回它们中间的元素。

```python
class ZigzagIterator:
    def __init__(self, v1: List[int], v2: List[int]):
        if len(v1) != 0:
            self.state = 0
        else:
            self.state = 1
        self.n1 = len(v1)
        self.n2 = len(v2)
        self.v1 = v1 
        self.v2 = v2
        self.p1 = 0
        self.p2 = 0
        
    def next(self) -> int:
        # 交错进行
        if self.state == 0:
            if self.p1 < self.n1:
                val = self.v1[self.p1]
                self.p1 += 1
            if self.p2 < self.n2:
                self.state = 1
        elif self.state == 1:
            if self.p2 < self.n2:
                val = self.v2[self.p2]
                self.p2 += 1
            if self.p1 < self.n1:
                self.state = 0
        return val
        
    def hasNext(self) -> bool:
        return self.p1 < self.n1 or self.p2 < self.n2
        
```

# 284. 顶端迭代器

奇葩题

给定一个迭代器类的接口，接口包含两个方法： next() 和 hasNext()。设计并实现一个支持 peek() 操作的顶端迭代器 -- 其本质就是把原本应由 next() 方法返回的元素 peek() 出来。

```python
# 方案1:初始化的时候全部取出来
class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.lst = []
        while iterator.hasNext():
            self.lst.append(iterator.next())
        self.cur = 0 # 索引指针

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.lst[self.cur]
        

    def next(self):
        """
        :rtype: int
        """
        val = self.lst[self.cur]
        self.cur += 1
        return val
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.cur < len(self.lst)
        

```

```python
# 方案2:一个个取出来,用一个stack缓存
class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.it = iterator
        self.stack = []
        if self.it.hasNext():
            self.stack.append(self.it.next())

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.stack[-1]
        

    def next(self):
        """
        :rtype: int
        """
        if self.stack:
            val = self.stack.pop()
        if self.it.hasNext():
            self.stack.append(self.it.next())
        return val
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) > 0

```

# 286. 墙与门

你被给定一个 m × n 的二维网格 rooms ，网格中有以下三种可能的初始化值：

-1 表示墙或是障碍物
0 表示一扇门
INF 无限表示一个空的房间。然后，我们用 231 - 1 = 2147483647 代表 INF。你可以认为通往门的距离总是小于 2147483647 的。
你要给每个空房间位上填上该房间到 最近门的距离 ，如果无法到达门，则填 INF 即可。

```python
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        inf = 2**31 - 1
        # 多源最短路径，bfs填充。填充的时候源点时所有的门
        m = len(rooms)
        n = len(rooms[0])
        queue = []
        visited = [[False for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    queue.append((i,j))
                    visited[i][j] = True 
        
        steps = 0
        direc = [(0,1),(0,-1),(1,0),(-1,0)]

        while len(queue) != 0:
            new_queue = []
            for x,y in queue:
              	# 由于是在这里进行的判断，那些数据没有被过滤掉，queue是累存的。
                if rooms[x][y] > 0 and visited[x][y] == False:
                    rooms[x][y] = steps

                visited[x][y] = True
                for di in direc:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if 0<=new_x<m and 0<=new_y<n and visited[new_x][new_y] == False and rooms[new_x][new_y] > 0:
                        new_queue.append((new_x,new_y))
            steps += 1
            queue = new_queue
                    
```

```python

class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        inf = 2**31 - 1
        # 多源最短路径，bfs填充。填充的时候源点时所有的门
        m = len(rooms)
        n = len(rooms[0])
        queue = []
        visited = [[False for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    queue.append((i,j))
                    visited[i][j] = True 
        
        steps = 0
        direc = [(0,1),(0,-1),(1,0),(-1,0)]

        while len(queue) != 0:
            new_queue = []
            for x,y in queue:
            # 优化
                rooms[x][y] = steps
                for di in direc:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if 0<=new_x<m and 0<=new_y<n and visited[new_x][new_y] == False and rooms[new_x][new_y] > 0:
                        new_queue.append((new_x,new_y))
                        visited[new_x][new_y] = True # 优化
            steps += 1
            queue = new_queue
                    
```

# 297. 二叉树的序列化与反序列化

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

# 298. 二叉树最长连续序列

给你一棵指定的二叉树的根节点 root ，请你计算其中 最长连续序列路径 的长度。

最长连续序列路径 是依次递增 1 的路径。该路径，可以是从某个初始节点到树中任意节点，通过「父 - 子」关系连接而产生的任意路径。且必须从父节点到子节点，反过来是不可以的。

```python
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        # 一种后续遍历方案
        # 核心思想是，先处理孩子，再根据孩子处理自己
        maxLength = 0
        def postOrder(node):
            nonlocal maxLength
            if node == None:
                return 0
            length = 1
            left = postOrder(node.left)
            right = postOrder(node.right)
            if node.left != None and node.left.val - 1 == node.val:
                length = max(length,left+1)
            if node.right != None and node.right.val - 1 == node.val:
                length = max(length,right+1)
            maxLength = max(maxLength,length)
            return length
        
        postOrder(root)
        return maxLength
            
```

# 311. 稀疏矩阵的乘法

给你两个 [稀疏矩阵](https://baike.baidu.com/item/稀疏矩阵) **A** 和 **B**，请你返回 **AB** 的结果。你可以默认 **A** 的列数等于 **B** 的行数。

```python
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        # 已知一定符合
        # 结果矩阵初始化
        m1 = len(mat1)
        n1 = len(mat1[0])
        m2 = len(mat2)
        n2 = len(mat2[0])

        ans = [[0 for j in range(n2)] for i in range(m1)]

        for i in range(len(ans)):
            for j in range(len(ans[0])):
                temp = 0
                for d in range(n1):
                    temp += mat1[i][d] * mat2[d][j]
                ans[i][j] = temp
        
        return ans
```

# 320. 列举单词的全部缩写

单词的 广义缩写词 可以通过下述步骤构造：先取任意数量的不重叠的子字符串，再用它们各自的长度进行替换。例如，"abcde" 可以缩写为 "a3e"（"bcd" 变为 "3" ），"1bcd1"（"a" 和 "e" 都变为 "1"），"23"（"ab" 变为 "2" ，"cde" 变为 "3" ）。

给你一个字符串 word ，返回一个由所有可能 广义缩写词 组成的列表。按 任意顺序 返回答案。

```python
class Solution:
    def generateAbbreviations(self, word: str) -> List[str]:
        # 读懂题意的回溯，回溯需要取传入字符长度和明确字符
        # 数字不能相邻

        ans = []
        path = []

        def backtracking(w,path):
            if w == "":
                ans.append("".join(path[:]))
                return 
            
            length = len(w)
            
            for i in range(1,length+1):
                if len(path) == 0 or (len(path) != 0 and path[-1].isalpha()): # 注意这一行的处理
                    path.append(str(i))
                    backtracking(w[i:],path)
                    path.pop()
            path.append(w[0])
            backtracking(w[1:],path)
            path.pop() 

        backtracking(word,[])
        return ans
```



# 331. 验证二叉树的前序序列化

序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如 `#`。上面的二叉树可以被序列化为字符串 "9,3,4,#,#,1,#,#,2,#,6,#,#"，其中 # 代表一个空节点。

给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。

每个以逗号分隔的字符或为一个整数或为一个表示 null 指针的 '#' 。

你可以认为输入格式总是有效的，例如它永远不会包含两个连续的逗号，比如 "1,,3" 。

```python
class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        # 由题解启示： 消消乐法,不断删除叶子节点
        # 每当遇到x,#,# 的时候，消去，并且填充上#
        lst = preorder.split(",")
        stack = []
        for e in lst:
            stack.append(e)
            if len(stack) >= 3:
                while len(stack) >= 3 and  stack[-1] == "#" and stack[-2] == "#" and stack[-3] != "#":
                    for i in range(3):
                        stack.pop()
                    stack.append("#")
        # 消消乐完毕之后，检测栈中是否只是剩下一个“#”
        if len(stack) != 1:
            return False
        return stack[0] == "#"

```

# 333. 最大 BST 子树

给定一个二叉树，找到其中最大的二叉搜索树（BST）子树，并返回该子树的大小。其中，最大指的是子树节点数最多的。

二叉搜索树（BST）中的所有节点都具备以下属性：

左子树的值小于其父（根）节点的值。

右子树的值大于其父（根）节点的值。

注意:

子树必须包含其所有后代。

```python
class Solution:
    def largestBSTSubtree(self, root: TreeNode) -> int:
        # 后续遍历，每个节点return的信息中需要携带的信息是它的左边界和右边界
        if root == None:
            return 0

        maxSize = 0
        def postOrder(node):
            nonlocal maxSize
            if node == None:
                return 0,[0xffffffff,0xffffffff],True
            if node.left == None and node.right == None:
                maxSize = max(maxSize,1)
                return 1,[node.val,node.val],True # size,[小/大],是BST
            leftBound = 0xffffffff
            rightBound = 0xffffffff
            # 根据有无节点分情况讨论。。。好多
            leftTree = postOrder(node.left)
            rightTree = postOrder(node.right)
            if node.left and node.right and leftTree[2] and rightTree[2] and node.val > leftTree[1][1] and node.val < rightTree[1][0]:
                leftBound = leftTree[1][0]
                rightBound = rightTree[1][1]
                maxSize = max(maxSize,1+leftTree[0]+rightTree[0])
                return 1+leftTree[0]+rightTree[0],[leftBound,rightBound],True 
            elif node.left == None and leftTree[2] and rightTree[2] and node.val < rightTree[1][0]:
                rightBound = rightTree[1][1]
                maxSize = max(maxSize,1+rightTree[0])
                return 1+rightTree[0],[node.val,rightBound],True 
            elif node.right == None and leftTree[2] and rightTree[2] and node.val > leftTree[1][1]:
                leftBound = leftTree[1][0]
                maxSize = max(maxSize,1+leftTree[0])
                return 1+leftTree[0],[leftBound,node.val],True 
            else:
                return 1,[leftBound,rightBound],False
        
        postOrder(root)
        return maxSize

```



# 334. 递增的三元子序列

给你一个整数数组 nums ，判断这个数组中是否存在长度为 3 的递增子序列。

如果存在这样的三元组下标 (i, j, k) 且满足 i < j < k ，使得 nums[i] < nums[j] < nums[k] ，返回 true ；否则，返回 false 。

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        if len(nums) < 3:
            return False
        # 三段式
        # 三个数,第一个存最小值。第三个存最大值。第三个数组需要倒序构建得到最大值
        # 需要快速判断是否存在第三个数使得a<b<c
        
        # 存最小值
        minNum = nums[0]
        minList = deque()
        for i in nums:
            if i < minNum:
                minNum = i
            minList.append(minNum)
        # 存最大值
        p = len(nums) - 1
        queue = collections.deque()
        maxNum = nums[-1]
        while p >= 0:
            if nums[p] > maxNum:
                maxNum = nums[p]
            queue.appendleft(maxNum)
            p -= 1
        # print(minList,queue)
        for i in range(1,len(nums)-1): # 枚举中间这个数
            if minList[i-1] < nums[i] < queue[i+1]:
                return True
        return False
```

```
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        if len(nums) < 3:
            return False
        # 双指针更新,注意第一个if
        small = 0xffffffff
        mid = 0xffffffff
        for i in nums:
            if i <= small: # 这里必须有等于号，因为防止mid被提前更新【1，1，-2，6】
                small = i
            elif i < mid: # 注意这个else防止二次更新，更新了small不会连带着把mid更新
                mid = i
            elif i > mid: # 
                return True
        return False        

```

# 354. 俄罗斯套娃信封问题

给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # 预先排序之后找到递增子序列。python看看能不能过。。
        # 以w升序，h降序。对h找递增子序列
        envelopes.sort(key = lambda x:(x[0],-x[1]))
        # 动态规划n^2复杂度找最长递增子序列
        n = len(envelopes)
        dp = [1 for i in range(n)]
        # 状态转移方程：当前的数值大于之前的数值，在其dp上+1更新
        # if envelopes[i][1] > envelopes[j][1]: group.append(dp[j])

        for i in range(n):
            group = []
            for j in range(i):
                if envelopes[i][1] > envelopes[j][1]:
                    group.append(dp[j]) # 这个append操作挺费时间的
            if len(group) != 0: # 为0则代表没有比他大的，不更新
                dp[i] = max(group) + 1
        # 返回序列中的最大值
        return max(dp)
```

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # 预先排序之后找到递增子序列。python看看能不能过。。
        # 以w升序，h降序。对h找递增子序列
        # 可能超时
        envelopes.sort(key = lambda x:(x[0],-x[1]))
        # 动态规划n^2复杂度找最长递增子序列
        n = len(envelopes)
        dp = [1 for i in range(n)]
        # 状态转移方程：当前的数值大于之前的数值，在其dp上+1更新
        # if envelopes[i][1] > envelopes[j][1]: group.append(dp[j])

        for i in range(n):
            tempMax = dp[0]
            for j in range(i):
                if envelopes[i][1] > envelopes[j][1]:
                    tempMax = max(tempMax,dp[j]+1) # 这一步的比较耗能不小
            dp[i] = tempMax 
        # 返回序列中的最大值
        return max(dp)
```

# 361. 轰炸敌人

想象一下炸弹人游戏，在你面前有一个二维的网格来表示地图，网格中的格子分别被以下三种符号占据：

'W' 表示一堵墙
'E' 表示一个敌人
'0'（数字 0）表示一个空位


请你计算一个炸弹最多能炸多少敌人。

由于炸弹的威力不足以穿透墙体，炸弹只能炸到同一行和同一列没被墙体挡住的敌人。

注意：你只能把炸弹放在一个空的格子里

```python
class Solution:
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        # 暴力算,可以炸死一排敌人。。。
        m = len(grid)
        n = len(grid[0])

        def calc(i,j): # 只往上下左右四个方向延伸
            nonlocal theNum
            store = (i,j)
            # 左
            while 0 <= j < n:
                if grid[i][j] == "0":
                    j -= 1
                elif grid[i][j] == "E":
                    j -= 1
                    theNum += 1
                elif grid[i][j] == "W":
                    break
            i,j =  store # 重置
            while 0 <= j < n:
                if grid[i][j] == "0":
                    j += 1
                elif grid[i][j] == "E":
                    theNum += 1
                    j += 1
                elif grid[i][j] == "W":
                    break
            i,j = store
            while 0 <= i < m:
                if grid[i][j] == "0":
                    i += 1
                elif grid[i][j] == "E":
                    theNum += 1
                    i += 1
                elif grid[i][j] == "W":
                    break
            i,j = store
            while 0 <= i < m:
                if grid[i][j] == "0":
                    i -= 1
                elif grid[i][j] == "E":
                    theNum += 1
                    i -= 1
                elif grid[i][j] == "W":
                    break

        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "0":
                    # 调用
                    theNum = 0 # 重置
                    calc(i,j)
                    ans = max(ans,theNum)
        return ans
```

# 370. 区间加法

假设你有一个长度为 n 的数组，初始情况下所有的数字均为 0，你将会被给出 k 个更新的操作。

其中，每个操作会被表示为一个三元组：[startIndex, endIndex, inc]，你需要将子数组 A[startIndex ... endIndex]（包括 startIndex 和 endIndex）增加 inc。

请你返回 k 次操作后的数组。

```python
class Solution:
    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        # 差分数组
        # 方法1: 上车法
        up = [(a[0],a[2]) for a in updates]
        down = [(a[1]+1,-a[2]) for a in updates]
        merge = collections.defaultdict(int)
        for pair in up:
            merge[pair[0]] += pair[1]
        for pair in down:
            merge[pair[0]] += pair[1]
        temp = 0 # 当前人数
        ans = [0 for i in range(length)]
        for i in range(length):
            temp += merge[i]
            ans[i] = temp
        return ans
```

```python
class Solution:
    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        # 差分数组
        # 标准差分数组法
        diff = [0 for i in range(length)]
        pre = 0
        for left,right,increase in updates:
            diff[left] += increase
            if (right+1) < length :
                diff[right+1] -= increase
        #final = list(accumulate(diff)) # accumulate语法，返回的是迭代器，还得强制转换成列表

        # 非accumulate语法
        temp = 0
        final = [0 for i in range(length)]
        temp = 0
        for i in range(length):
            temp += diff[i]
            final[i] = temp
        return final
```

# 372. 超级次方

你的任务是计算 `ab` 对 `1337` 取模，`a` 是一个正整数，`b` 是一个非常大的正整数且会以数组形式给出。

```python
class Solution:
    def superPow(self, a: int, b: List[int]) -> int:
        # 递归解法
        # a ** [1,3,7,4] == (a ** [1,3,7]) ** 10 + a ** 4
        mod = 1337
        if len(b) == 1:
            return a**b[0] % mod 
        else:
            return (((self.superPow(a,b[:-1]))**10)%mod * (a**b[-1] % mod )) % mod
```



# 399. 除法求值

给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。

另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。

返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。

注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # 抽象成图论问题，建立对应关系
        n = len(equations)
        edges = collections.defaultdict(list) # 存节点和权值
        for i in range(n):
            start = equations[i][0]
            end = equations[i][1]
            edges[start].append([end,values[i]])
            edges[end].append([start,1/values[i]])
        
        # 由于需要路径，所以采用dfs
        def dfs(start,end,visited):
            if start not in edges or end not in edges:
                return -1 
            if start == end:
                return 1
            visited.add(start)
            for neigh in edges[start]:
                if neigh[0] not in visited:
                    val = dfs(neigh[0],end,visited)
                    if val > 0:
                        return val * neigh[1]
            visited.remove(start)
            return -1
        
        ans = []
        for q in queries:
            start = q[0]
            end = q[1]
            e = dfs(start,end,set())
            ans.append(e)
        return ans 
            
```

# 402. 移掉 K 位数字

给你一个以字符串表示的非负整数 `num` 和一个整数 `k` ，移除这个数中的 `k` 位数字，使得剩下的数字最小。请你以字符串形式返回这个最小的数字。

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        remain = k  # 允许的pop次数
        stack = []
        # 单调栈,需要单调递增栈
        for ch in num:
            if len(stack) == 0:
                stack.append(ch)
                continue
            elif len(stack) > 0 and int(stack[-1]) > int(ch) and remain > 0:
                while len(stack) > 0 and int(stack[-1]) > int(ch) and remain > 0:
                    stack.pop()
                    remain -= 1
            stack.append(ch)
        
        # 如果没有pop完毕。继续pop
        while remain > 0:
            stack.pop()
            remain -= 1
        # 栈空返回“0”,输出不能有前导0，
        if len(stack) == 0:
            return "0" 
        temp  = "".join(stack)

        return str(int(temp))
```



# 405. 数字转换为十六进制数

给定一个整数，编写一个算法将这个数转换为十六进制数。对于负整数，我们通常使用 补码运算 方法。

注意:

十六进制中所有字母(a-f)都必须是小写。
十六进制字符串中不能包含多余的前导零。如果要转化的数为0，那么以单个字符'0'来表示；对于其他情况，十六进制字符串中的第一个字符将不会是0字符。 
给定的数确保在32位有符号整数范围内。
不能使用任何由库提供的将数字直接转换或格式化为十六进制的方法。

```python
class Solution:
    def toHex(self, num: int) -> str:
        # 补码运算机制是每位取反再+1
        if num == 0:
            return "0"

        if num < 0:
            num += 2**32
            
        ans = ""
        theDict = {10:"a",11:"b",12:"c",13:"d",14:"e",15:"f"}
        while num != 0:
            remain = num%16
            if remain >= 10:
                ans += theDict[remain]
            else:
                ans += str(remain)

            num //= 16
        return ans[::-1]

```



# 408. 有效单词缩写

给一个 非空 字符串 s 和一个单词缩写 abbr ，判断这个缩写是否可以是给定单词的缩写。

字符串 "word" 的所有有效缩写为：

["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]
注意单词 "word" 的所有有效缩写仅包含以上这些。任何其他的字符串都不是 "word" 的有效缩写。

注意:
假设字符串 s 仅包含小写字母且 abbr 只包含小写字母和数字。

```python
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        # 整理，数组中不能出现前导0开头的
        lst = []
        num = ""
        for ch in abbr:
            if ch.isalpha():
                if num != "":
                    lst.append(num)
                    num = ""
                lst.append(ch)
            elif ch.isdigit():
                num += ch
        if num != "": # 擦屁股
            lst.append(num)

        for i in lst:
            if i[0] == "0" : # 不能出现
                return False
                
        p = 0
        cur = 0
        # 双指针。
        while p < len(word) and cur < len(lst):
            if word[p].isalpha() and lst[cur].isalpha():
                if word[p] != lst[cur]:
                    return False
                p += 1
                cur += 1
            else:
                p += int(lst[cur])
                cur += 1

        return p == len(word) and cur == len(lst)
```

# 416. 分割等和子集

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # 子集背包问题
        sumNum = sum(nums)
        if sumNum % 2 == 1:
            return False 
        # dp[i][j]表示，对于前i个物品，当背包容量为j时，如果正好装满，则为true
        # dp[4][9]对与前4个物品，存在一种方法凑出9
        sumNum //= 2 # 只取一半
        n = len(nums)
        dp = [[False for j in range(sumNum+1)] for i in range(n+1)]
        for i in range(n+1):
            dp[i][0] = True 
        # 状态转移为dp[i][j] = dp[i-1][j] # 不选； dp[i][j] = dp[i-1][j-nums[i-1]] # 选
        for i in range(1,n+1):
            for j in range(1,sumNum+1):
                if j - nums[i-1] < 0: # 只能不选
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = (dp[i-1][j] or dp[i-1][j-nums[i-1]])
        return dp[-1][-1]

```

# 421. 数组中两个数的最大异或值

给你一个整数数组 nums ，返回 nums[i] XOR nums[j] 的最大运算结果，其中 0 ≤ i ≤ j < n 。

进阶：你可以在 O(n) 的时间解决这个问题吗？

```python
class TrieNode:
    def __init__(self):
        self.children = [None,None]

class Trie:    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,n):
        node = self.root 
        for i in range(31,-1,-1):
            bit = (n>>i)&1
            if node.children[bit] == None:
                node.children[bit] = TrieNode()
            node = node.children[bit]
    
    def insertAll(self,lst):
        for n in lst:
            self.insert(n) 

class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        # 利用字典树
        tree = Trie()
        tree.insertAll(nums)
        maxXor = 0
        for n in nums:
            node = tree.root 
            xor = 0 # 初始化
            for i in range(31,-1,-1):
                bit = (n>>i)&1
                if node.children[1-bit] != None:
                    node = node.children[1-bit]
                    xor = (xor<<1) + 1
                else:
                    node = node.children[bit]
                    xor = xor<<1
            maxXor = max(maxXor,xor)
        return maxXor
```

# 423. 从英文中重建数字

给定一个非空字符串，其中包含字母顺序打乱的英文单词表示的数字0-9。按升序输出原始的数字。

注意:

输入只包含小写英文字母。
输入保证合法并可以转换为原始的数字，这意味着像 "abc" 或 "zerone" 的输入是不允许的。
输入字符串的长度小于 50,000。

```python
class Solution:
    def originalDigits(self, s: str) -> str:
        # 找特征字母,需要保持顺序
        # x特征对应6; w对应2; u对应4; g对应8; z对应0
        # 二层特征,h 对应3， o对应1 , f对应5，s对应7，i对应9

        firstDict = {
            "x":"six",
            "w":"two",
            "u":"four",
            "g":'eight',
            "z":"zero",
            "h":"three",
            "o":"one",
            "f":"five",
            "s":"seven",
            "i":"nine"
        }
        aimDict = {
            "x":"6",
            "w":"2",
            "u":"4",
            "g":'8',
            "z":"0",            
            "h":"3",
            "o":"1",
            "f":"5",
            "s":"7",
            "i":"9"
        }
        ansList = [0 for i in range(10)] # 记录每个数字的个数
        alphaCount = [0 for i in range(26)]
        for i in s:
            index = ord(i)-ord('a')
            alphaCount[index] += 1

        for key in firstDict:
            while alphaCount[ord(key)-ord("a")] > 0: # 注意这里的while
                ansList[int(aimDict[key])] += 1
                for ch in firstDict[key]:
                    index = ord(ch)-ord("a")
                    alphaCount[index] -= 1
        
        ans = ""
        for i in range(10):
            ans += ansList[i]*str(i)

        return ans
```



# 431. 将 N 叉树编码为二叉树

设计一个算法，可以将 N 叉树编码为二叉树，并能将该二叉树解码为原 N 叉树。一个 N 叉树是指每个节点都有不超过 N 个孩子节点的有根树。类似地，一个二叉树是指每个节点都有不超过 2 个孩子节点的有根树。你的编码 / 解码的算法的实现没有限制，你只需要保证一个 N 叉树可以编码为二叉树且该二叉树可以解码回原始 N 叉树即可。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
"""

# 编码思路，由于节点不一样，所以需要拷贝值，根据值重建节点


class Codec:
    # Encodes an n-ary tree to a binary tree.
    def encode(self, root: 'Node') -> TreeNode:
        # 思路，如果它有多个孩子，则这个节点的左孩子是它的第一个孩子，其余节点轮询为前一个节点的右孩子
        # BFS获取每一层
        if root == None:
            return None
        everyLevel = []
        theMain = TreeNode(root.val)
        queue = [(root,theMain)] # 队列里的是(N树节点，二叉树节点)的元组
        while len(queue) != 0:
            new_queue = []
            for node in queue:
                child_queue = []
                parent = node[1]
                for child in node[0].children:
                    two_tree_node = TreeNode(child.val)
                    child_queue.append(two_tree_node)
                    new_queue.append((child,two_tree_node))
                if len(child_queue) != 0:
                    parent.left = child_queue[0] # 本节点的左孩子指向第一个节点
                    for i in range(len(child_queue)-1): # 每个孩子的右节点指向下一个孩子
                        child_queue[i].right = child_queue[i+1]
            queue = new_queue
        return theMain               

	# Decodes your binary tree to an n-ary tree.
    def decode(self, data: TreeNode) -> 'Node':
        if data == None:
            return 
        # 否则创建拷贝
        root = Node(data.val,[]) # 注意它现在孩子为空，他的左孩子为data的left
        p = data.left # 从二叉树上的left开始
        # 这里画图理解，右劈的一条线都是root的孩子。所以p = p.right。而
        while p != None:
            root.children.append(self.decode(p)) # 这一行很关键
            p = p.right
        return root
```

```python
class Codec:
    # Encodes an n-ary tree to a binary tree.
    def encode(self, root: 'Node') -> TreeNode:
    		# 参考其他人的解法的dfs思路
        # dfs编码思路
        # 如果它有孩子，对每个孩子递归
        if root == None:
            return 
        theRoot = TreeNode(root.val)
        if root.children != None and len(root.children) != 0:
            theRoot.left = self.encode(root.children[0])
        p = theRoot.left
        for node in root.children[1:]:
            p.right = self.encode(node)
            p = p.right
        return theRoot
        
	
	# Decodes your binary tree to an n-ary tree.
    def decode(self, data: TreeNode) -> 'Node':
        if data == None:
            return 
        # 否则创建拷贝
        root = Node(data.val,[]) # 注意它现在孩子为空，他的左孩子为data的left
        p = data.left # 从二叉树上的left开始
        # 这里画图理解，右劈的一条线都是root的孩子。所以p = p.right。而
        while p != None:
            root.children.append(self.decode(p)) # 这一行很关键
            p = p.right
        return root

```

# 435. 无重叠区间

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

注意:

可以认为区间的终点总是大于它的起点。
区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # 已知：起点严格小于终点
        # 贪心：根据后一个坐标升序排列之后，找到最长递增序列。返回全长减去该序列长度
        # 题目限制了区间非空
        n = len(intervals)
        intervals.sort(key = lambda x:x[1])
        prev = intervals[0]
        ans = 1
        for i in intervals[1:]:
            if i[0] >= prev[1]:
                ans += 1
                prev = i
        return n - ans
```

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # 动态规划法,排序后找到最长上升,python的dp超时仅供参考
        n = len(intervals)
        intervals.sort() # 以第一个排序就行
        dp = [1 for i in range(n)]
        for i in range(1,n):
            for j in range(i): 
            # 注意在通常的LIS中，内层需要取最大值.设置一个group.取最大值
            #  但是这里不需要因为这个排序过
                if intervals[i][0] >= intervals[j][1]:
                    dp[i] = dp[j] + 1
        maxLength = max(dp)
        return n - maxLength 
        
```

# 437. 路径总和 III

给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        # 双重dfs，一个是遍历节点的扫法，一个是选取为目标值的扫法
        # 注意参数命名，root和node自己要分清楚
        # 不需要担心path在下一个搜索中会有残留的问题。因为调用顺序，从这个节点开始dfs之后，直到它的dfs执行完毕才会进入pre_order的下一个节点开始深度搜索
        ans = 0 # 收集结果,注意这里ans不用列表收集具体路径，而是遇见合法值就+1
        path = [] # 选择路径

        def dfs(node,targetSum):
            nonlocal ans
            if node == None:
                return                 
            path.append(node.val) # 做选择
            if node.val == targetSum:
                ans += 1
            dfs(node.left,targetSum-node.val)
            dfs(node.right,targetSum-node.val)
            path.pop() # 取消选择

        def pre_order(node):
            if node == None:
                return 
            dfs(node,targetSum)
            pre_order(node.left)
            pre_order(node.right)

        pre_order(root) # 开始搜索
        return ans
```



# 449. 序列化和反序列化二叉搜索树

序列化是将数据结构或对象转换为一系列位的过程，以便它可以存储在文件或内存缓冲区中，或通过网络连接链路传输，以便稍后在同一个或另一个计算机环境中重建。

设计一个算法来序列化和反序列化 二叉搜索树 。 对序列化/反序列化算法的工作方式没有限制。 您只需确保二叉搜索树可以序列化为字符串，并且可以将该字符串反序列化为最初的二叉搜索树。

编码的字符串应尽可能紧凑。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.
        """
        # 普通序列化逻辑
        # 使用先序遍历序列化
        ans = ''
        def code(root):
            nonlocal ans
            if root == None:
                ans += "#,"
                return
            ans += str(root.val)
            ans += ","
            code(root.left)
            code(root.right)
        code(root)
        # 去掉结尾的，
        return ans[:-1]

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        # 可以使用先序遍历和中序遍历重建，也可以直接使用先序遍历重建
        data = data.split(",")
        def decode(nodes):
            if len(nodes) == 0:
                return 
            rootVal = nodes.pop(0)
            if rootVal == "#":
                return None

            root = TreeNode(rootVal)
            
            root.left = decode(nodes)
            root.right = decode(nodes)

            return root # 注意递归返回值是节点

        root = decode(data)
        return root # 这个函数的返回值是节点


```

# 463. 岛屿的周长

给定一个 row x col 的二维网格地图 grid ，其中：grid[i][j] = 1 表示陆地， grid[i][j] = 0 表示水域。

网格中的格子 水平和垂直 方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。

岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。

```python
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        # 找到里面的第一个为1的。顺序搜索即可
        # 算周长基于这样的一个思想： 从岛屿1 跨到0的时候【或者出界点时候】周长+1
        start = None
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    start = (i,j)
                    break

        visited = [[False for j in range(n)] for i in range(m)]
        count = 0 # 记录个数
        direc = [(0,1),(0,-1),(-1,0),(1,0)]
        def valid(i,j): # 判定是否合法
            if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                return True
            return False
        
        def dfs(i,j):
            nonlocal count 
            if not valid(i,j): # 跨出界，则+1
                return 1
            if visited[i][j] == True: # 访问过 啥也不加
                return 0
            visited[i][j] = True
            # 算周长基于这样的一个思想： 从岛屿1 跨到0的时候【或者出界点时候】周长+1
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                temp = dfs(new_i,new_j) # 注意这里
                count += temp
            return 0

        if start == None: # 没有起点
            return 0
        dfs(start[0],start[1])
        return count 

```



# 470. 用 Rand7() 实现 Rand10()

已有方法 rand7 可生成 1 到 7 范围内的均匀随机整数，试写一个方法 rand10 生成 1 到 10 范围内的均匀随机整数。

不要使用系统的 Math.random() 方法。

```python
# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7

class Solution:
    def rand10(self):
        """
        :rtype: int
        """
        # 不能使用累加方法。因为弄出来是中间高两头低的
        # 随机两次，得到49个数。前40个完成映射，后9个重新rand
        # 期望次数约为1.2次【1+9/49+（9/49）**2 …………】 每次调用两次rand
        theSum = (rand7()-1)*7 + rand7()-1 # [0~48]
        if theSum >= 40:
            return self.rand10()
        return theSum//4+1
        
```

# 494. 目标和

给你一个整数数组 nums 和一个整数 target 。

向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # 动态规划，子集01背包问题
        # 假设正数和为p,加负号的和为q,那么求p-q == target
        # 又有p+q == s ,则p == (target+s)//2,
        s = sum(nums)
        if (target+s) % 2 != 0:
            return 0
        p = abs((target+s)//2) # 注意这一行
        n = len(nums)
        dp = [[0 for j in range(p+1)] for i in range(n+1)]
        # dp[i][j]为前i个数组合成j的方案个数,显然dp[i][0] = 1
        for i in range(n+1):
            dp[i][0] = 1
        # 状态转移方程：dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]  #第四个数的索引为3，第i个数的索引为i-1
        for i in range(1,n+1):
            for j in range(0,p+1):
                dp[i][j] = dp[i-1][j]
                if j-nums[i-1] >= 0: 
                    dp[i][j] += dp[i-1][j-nums[i-1]]
        return dp[-1][-1]
```

# 508. 出现次数最多的子树元素和

给你一个二叉树的根结点，请你找出出现次数最多的子树元素和。一个结点的「子树元素和」定义为以该结点为根的二叉树上所有结点的元素之和（包括结点本身）。

你需要返回出现次数最多的子树元素和。如果有多个元素出现的次数相同，返回所有出现次数最多的子树元素和（不限顺序）。

```python
class Solution:
    def findFrequentTreeSum(self, root: TreeNode) -> List[int]:
        # 后序遍历,需要先算出孩子再算自己
        if root == None:
            return []

        countDict = collections.defaultdict(int)

        def postOrder(node):
            if node == None:
                return 0
            valSum = node.val 
            leftVal = postOrder(node.left)
            rightVal = postOrder(node.right)
            valSum = valSum + leftVal + rightVal
            countDict[valSum] += 1
            return valSum
        
        postOrder(root)
        maxTimes = 1
        for key in countDict:
            if countDict[key] > maxTimes:
                maxTimes = countDict[key]
        ans = []
        for key in countDict:
            if countDict[key] == maxTimes:
                ans.append(key)
        return ans

```



# 523. 连续的子数组和

给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：

子数组大小 至少为 2 ，且
子数组元素总和为 k 的倍数。
如果存在，返回 true ；否则，返回 false 。

如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数。0 始终视为 k 的一个倍数。

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        # 预先处理，模
        for i in range(len(nums)):
            nums[i] = nums[i] % k 
        # 前缀和
        # print(nums)
        preSumDict = collections.defaultdict(int) # k-v为当前和：索引
        preSumDict[0] = -1

        tempSum  = 0
        for i,n in enumerate(nums):
            tempSum = (tempSum+n)%k
            # print(preSumDict,tempSum)
            if n != 0:
                if tempSum in preSumDict:
                    return True
                else:
                    preSumDict[tempSum] = i # 只记录第一次出现的即可
            elif n == 0:
                if i - preSumDict[tempSum] >= 2:
                    return True 
        return False
```

# 524. 通过删除字母匹配到字典里最长单词

给你一个字符串 s 和一个字符串数组 dictionary 作为字典，找出并返回字典中最长的字符串，该字符串可以通过删除 s 中的某些字符得到。

如果答案不止一个，返回长度最长且字典序最小的字符串。如果答案不存在，则返回空字符串。

```python
class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        # 超多次双指针
        tempAns = []
        n = len(s)
        for w in dictionary:
            p1 = 0
            p2 = 0
            while p1 < n and p2 < len(w):
                if s[p1] == w[p2]:
                    p1 += 1
                    p2 += 1
                elif s[p1] != w[p2]:
                    p1 += 1
            if p2 == len(w): # p2可以扫完，则收集
                tempAns.append(w)
        # 返回长度最长，字典序还需要最小
        if tempAns == []:
            return ""
        maxLength = 0
        for w in tempAns:
            if len(w) > maxLength:
                maxLength = len(w)
        ans = []
        for w in tempAns:
            if len(w) == maxLength:
                ans.append(w)
        ans.sort() # 这里还要记得排序
        return ans[0]
```

# 528. 按权重随机选择

给定一个正整数数组 w ，其中 w[i] 代表下标 i 的权重（下标从 0 开始），请写一个函数 pickIndex ，它可以随机地获取下标 i，选取下标 i 的概率与 w[i] 成正比。

例如，对于 w = [1, 3]，挑选下标 0 的概率为 1 / (1 + 3) = 0.25 （即，25%），而选取下标 1 的概率为 3 / (1 + 3) = 0.75（即，75%）。

也就是说，选取下标 i 的概率为 w[i] / sum(w) 。

```python
class Solution:

    def __init__(self, w: List[int]):
        # 传入权重数组，构造前缀和
        self.pre = [0 for i in range(len(w))] # 前缀和递增
        temp = 0
        for i in range(len(w)):
            temp += w[i]
            self.pre[i] = temp
        # 例如传入数据为2，4，5。则产生一个11以内，不包括11的随机数。
        # 在闭区间[0,1]对应0；闭区间[2,3,4,5]对应1，闭区间[6,7,8,9,10]对应2
        # 此时pre = [2,6,11],产生一个随机数，例如产生了5，则找到第一个大于它的数字对应的下标
        # print(self.pre)
        

    def pickIndex(self) -> int:
        x = random.randint(0,self.pre[-1]-1)  # random
        def binarySearch(lst,target): # 一定在数组中，且由于w[i]!=0，所以各个数字不相同
            left = 0
            right = len(lst) -1
            while left <= right:
                mid = (left+right)//2
                if lst[mid] == target: # 往右边找，收缩左边界
                    left = mid + 1                
                elif lst[mid] > target: # 往左边找,收缩右边界
                    right = mid - 1
                elif lst[mid] < target: # 
                    left = mid + 1
            return left # 返回值为索引
        ans = binarySearch(self.pre,x)
        # print(x,ans)
        return ans
```

# 531. 孤独像素 I

给定一幅黑白像素组成的图像, 计算黑色孤独像素的数量。

图像由一个由‘B’和‘W’组成二维字符数组表示, ‘B’和‘W’分别代表黑色像素和白色像素。

黑色孤独像素指的是在同一行和同一列不存在其他黑色像素的黑色像素。

```python
class Solution:
    def findLonelyPixel(self, picture: List[List[str]]) -> int:
        # 剪枝，尬模拟
        m = len(picture)
        n = len(picture[0])
        rowVisited = [0 for i in range(m)]
        colVisited = [0 for j in range(n)]
        validRow = set()
        validCol = set()
        black = []
        for i in range(m):
            for j in range(n):
                if picture[i][j] == "B":
                    rowVisited[i] += 1
                    colVisited[j] += 1
                    black.append((i,j))
        for r in range(m):
            if rowVisited[r] == 1:
                validRow.add(r)
        for c in range(n):
            if colVisited[c] == 1:
                validCol.add(c)
        count = 0
        for i,j in black:
            if i in validRow and j in validCol:
                count += 1
        return count
```

# 536. 从字符串生成二叉树

你需要从一个包括括号和整数的字符串构建一棵二叉树。

输入的字符串代表一棵二叉树。它包括整数和随后的 0 ，1 或 2 对括号。整数代表根的值，一对括号内表示同样结构的子树。

若存在左子结点，则从左子结点开始构建。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def str2tree(self, s: str) -> TreeNode:
        # 递归处理,数值可能是负数
        # 利用括号来计算匹配深度
        # 数值不只是一位数。。。
        if len(s) == 0: return None
        left = 0
        indexList = [] # 记录的序号为左右子树相关
        # 如果第一位是负数，需要考虑记录负号
        symbol = 1
        if s[0] == "-":
            symbol = -1
            s = s[1:]
        p = 0 # 处理多位数
        while p < len(s) and s[p] != "(":
            p += 1
        value = symbol * int(s[:p])

        for i,val in enumerate(s):
            if val == "(":
                if left == 0:
                    indexList.append(i)
                left += 1
            elif val == ")":
                left -= 1
        if len(indexList) == 2:
            index1 = indexList[0]
            index2 = indexList[1]
            leftPartString = s[index1+1:index2-1]
            rightPartString = s[index2+1:-1]
        elif len(indexList) == 1:
            index1 = indexList[0]
            leftPartString = s[index1+1:-1]
            rightPartString = ""
        elif len(indexList) == 0:
            leftPartString = ""
            rightPartString = ""
        # 处理完毕，开始递归
        root = TreeNode(value) # 这个值处理在前面
        root.left = self.str2tree(leftPartString)
        root.right = self.str2tree(rightPartString)

        return root
```

# 542. 01 矩阵

给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        # 超级原点的BFSBFS
        # 需要对每个1，找到最近的0. 单源bfs需要调用多次
        # 所以直接反着来。
        # 以所有0找最近的1
        m = len(mat)
        n = len(mat[0])
        visited = [[False for j in range(n)] for i in range(m)]
        ansMat = [[0 for j in range(n)] for i in range(m)]
        queue = []
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append((i,j))
                    visited[i][j] = True 
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        steps = 0
        while len(queue) != 0:
            new_queue = []
            for pair in queue:
                x,y = pair
                if mat[x][y] == 1 and visited[x][y] == False:
                    ansMat[x][y] = steps 
                    visited[x][y] = True 
                for di in direc:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if 0<=new_x<m and 0<=new_y<n and visited[new_x][new_y] == False:
                        new_queue.append((new_x,new_y)) 
            queue = new_queue
            steps += 1
        return ansMat
```

# 543. 二叉树的直径

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        # 求左右子树的最大深度，后续遍历整合
        # 直径 == 左子树最大深度+右子树最大深度
        ans = 0 # 收集结果
        def findDepth(node):
            nonlocal ans
            if node == None:
                return 0
            dleft = findDepth(node.left)
            dright = findDepth(node.right)
            ans = max(ans,dleft+dright)
            return max(dleft,dright) + 1
        
        findDepth(root)
        return ans
    
```

# 549. 二叉树中最长的连续序列

给定一个二叉树，你需要找出二叉树中最长的连续序列路径的长度。

请注意，该路径可以是递增的或者是递减。例如，[1,2,3,4] 和 [4,3,2,1] 都被认为是合法的，而路径 [1,2,4,3] 则不合法。另一方面，路径可以是 子-父-子 顺序，并不一定是 父-子 顺序。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        # 可以是子父子顺序。。。
        # 和298类似
        maxLength = 0

        def postOrder(node):
            nonlocal maxLength
            if node == None:
                return (0,0) # 两个数第一个为递减，第二个为递增.方向从孩子到爹来看
            decrease,increase = 1,1
            left_d,left_i = postOrder(node.left)
            right_d,right_i = postOrder(node.right)
            if node.left != None and node.left.val == node.val - 1: # 孩子比爹小
                increase = left_i + 1
            if node.left != None and node.left.val - 1 == node.val: # 孩子比爹大
                decrease = left_d + 1
            if node.right != None and node.right.val == node.val - 1: # 孩子比爹小  
                increase = max(increase,right_i+1)
            if node.right != None and node.right.val - 1 == node.val: # 孩子比爹大
                decrease = max(decrease,right_d+1)
            maxLength = max(maxLength,increase+decrease-1) # 注意这一行精髓，为上升的和下降的节点数目和减去一个节点
            return (decrease,increase)

        postOrder(root)
        return maxLength


```

# 563. 二叉树的坡度

给定一个二叉树，计算 整个树 的坡度 。

一个树的 节点的坡度 定义即为，该节点左子树的节点之和和右子树节点之和的 差的绝对值 。如果没有左子树的话，左子树的节点之和为 0 ；没有右子树的话也是一样。空结点的坡度是 0 。

整个树 的坡度就是其所有节点的坡度之和。

```python
class Solution:
    def findTilt(self, root: TreeNode) -> int:
        # 后续遍历
        allSum = 0
        
        def postOrder(node):
            nonlocal allSum
            if node == None:
                return 0
            calc = 0
            left = postOrder(node.left)
            right = postOrder(node.right)
            allSum += abs(left-right)
            calc = calc + left + right + node.val
            return calc
        
        postOrder(root)
        return allSum
```

# 565. 数组嵌套

索引从0开始长度为N的数组A，包含0到N - 1的所有整数。找到最大的集合S并返回其大小，其中 S[i] = {A[i], A[A[i]], A[A[A[i]]], ... }且遵守以下的规则。

假设选择索引为i的元素A[i]为S的第一个元素，S的下一个元素应该是A[A[i]]，之后是A[A[A[i]]]... 以此类推，不断添加直到S出现重复的元素。

```python
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        # 为了防止写的菜的并查集。。。直接修改数组
        # 可知里面的元素成环之后不会再访问。
        # 可以用visited数组，可以直接修改原数组
        visited = [False for i in range(len(nums))]
        maxLength = 0
        n = len(nums)
        for i in range(n):
            tempSet = set()
            if visited[i] == True:
                continue 
            now = i 
            while now not in tempSet:
                tempSet.add(now)
                visited[now] = True
                now = nums[now]
            maxLength = max(maxLength,len(tempSet))
        return maxLength

```

# 573. 松鼠模拟

现在有一棵树，一只松鼠和一些坚果。位置由二维网格的单元格表示。你的目标是找到松鼠收集所有坚果的最小路程，且坚果是一颗接一颗地被放在树下。松鼠一次最多只能携带一颗坚果，松鼠可以向上，向下，向左和向右四个方向移动到相邻的单元格。移动次数表示路程。

```python
class Solution:
    def minDistance(self, height: int, width: int, tree: List[int], squirrel: List[int], nuts: List[List[int]]) -> int:
    # 数学思路
        # 建立两个数组，画图辅助，
        # 数组1是所有果子到树的哈夫曼距离
        # 数组2是松鼠到所有果子到距离
        # 最终是找到一个果子，distance(松鼠，果子)+distance(果子，树)+other(sum(distance(果子到树)))最小
        def distance(p1,p2):
            return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])
        arr1 = [distance(nut,tree) for nut in nuts] # 数组1是所有果子到树的哈夫曼距离
        arr2 = [distance(nut,squirrel) for nut in nuts] # 数组2是松鼠到所有果子到距离
        doubleDistance = sum(arr1)*2
        minDistance = 0xffffffff # 初始化为最大值
        n = len(arr1)
        for i in range(n):
            tempDistance = doubleDistance-arr1[i]+arr2[i]
            if tempDistance < minDistance:
                minDistance = tempDistance
        return minDistance

```

# 581. 最短无序连续子数组

给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。

请你找出符合题意的 最短 子数组，并输出它的长度。

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # 做一个副本
        # nlogn
        cp = nums.copy()
        cp.sort()
        n = len(nums)
        for i in range(n):
            cp[i] -= nums[i]
        # 找到从前往后数第一个不为0的，从它开始需要变
        # 找到从后往前数第一个不为0的，从它开始需要变
        left,right = 0,n-1
        while left < n and cp[left] == 0:
            left += 1
        if left == n:
            return 0
        while right >= 0 and cp[right] == 0:
            right -= 1
        return (right-left+1)
```

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # 分成三段 n1 ; n2 ; n3
        # 遍历的时候，找到当前的临时最大值，如果之后有逆序，则记录。没有则不记录
        n = len(nums)
        left,right = 0,n-1 # 记录错乱值
        i,j = 0,n-1 # 循环用的变量
        tempMax = nums[0]
        tempMin = nums[-1]
        while i < n:
            if tempMax < nums[i]: # 到当前为
                tempMax = nums[i]
            elif nums[i] < tempMax: # i它乱序了，记录i
                left = i 
            i += 1
        while j >= 0:
            if tempMin > nums[j]:
                tempMin = nums[j]
            elif nums[j] > tempMin: # j它乱序了，记录j
                right = j 
            j -= 1
        if (left,right) == (0,n-1): # 没更新过
            return 0
        # left会>=right
        return abs(right-left)+1
            
```

```go
func findUnsortedSubarray(nums []int) int {
    n := len(nums)
    arr := make([]int,0,n)
    for _,v := range nums {
        arr = append(arr,v)
    }
    sort.Ints(arr)
    // fmt.Println(arr)
    leftMark := -1
    rightMark := n-1
    for left := 0; left < n; left += 1 {
        if arr[left] != nums[left] {
            leftMark = left
            break
        }
    }
    if leftMark == -1 {
        return 0
    }
    for right := n-1; right >= 0; right -= 1 {
        if arr[right] != nums[right] {
            rightMark = right
            break
        }
    }
    return rightMark-leftMark+1
}
```

```go
func findUnsortedSubarray(nums []int) int {
    n := len(nums)
    leftMark := 0
    rightMark := n-1
    tempMax := nums[0]
    tempMin := nums[n-1]

    for i := 0; i < n; i += 1 { //允许递增
        if nums[i] >= tempMax {
            tempMax = nums[i]
        } else if  nums[i] < tempMax { // 说明它乱序
            leftMark = i
        }
    }    
    for j := n-1; j >=0 ; j -= 1 {
        if nums[j] <= tempMin {
            tempMin = nums[j]
        } else if nums[j] > tempMin {
            rightMark = j
        }
    }

    if leftMark == 0 && rightMark == n - 1{
        return 0
    } else {
        return leftMark - rightMark +1
    }
}
```

# 582. 杀掉进程

系统中存在 n 个进程，形成一个有根树结构。给你两个整数数组 pid 和 ppid ，其中 pid[i] 是第 i 个进程的 ID ，ppid[i] 是第 i 个进程的父进程 ID 。

每一个进程只有 一个父进程 ，但是可能会有 一个或者多个子进程 。只有一个进程的 ppid[i] = 0 ，意味着这个进程 没有父进程 。

当一个进程 被杀掉 的时候，它所有的子进程和后代进程都要被杀掉。

给你一个整数 kill 表示要杀掉进程的 ID ，返回杀掉该进程后的所有进程 ID 的列表。可以按 任意顺序 返回答案。

```python
class Solution:
    def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
        NodeDict = collections.defaultdict(list) # k-v为父节点，子节点
        n = len(pid)
        for i in range(n):
            NodeDict[ppid[i]].append(pid[i])
        # 以kill作为根节点进行广度优先搜索
        queue = [kill]
        ans = [kill]
        while len(queue) != 0:
            new_queue = []
            for i in queue:
                for child in NodeDict[i]:
                    ans.append(child)
                    new_queue.append(child)
            queue = new_queue
        return ans
```

# 583. 两个字符串的删除操作

给定两个单词 *word1* 和 *word2*，找到使得 *word1* 和 *word2* 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # 动态规划
        # 双串问题弄一个0索引
        # dp[i][j]是长度为i,j的单词
        m = len(word1)
        n = len(word2)
        dp = [[0xffffffff for j in range(n+1)] for i in range(m+1)]
        # 初始化
        for i in range(m+1):
            dp[i][0] = i # 只能被疯狂删除来
        for j in range(n+1):
            dp[0][j] = j # 只能被疯狂删除来
        
        for i in range(1,m+1):
            for j in range(1,n+1):
                # 如果字符相同，啥也不用管
                # 如果字符不同，需要删除
                # 注意是word[index-1]
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                elif word1[i-1] != word2[j-1]:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,
                        dp[i][j-1] + 1
                    )
        
        # 返回右下角即可
        return dp[-1][-1]

```

```python
# 另一个思路，使用最长公共子序列。两者和最长公共子序列的差即为答案
```

# 640. 求解方程

求解一个给定的方程，将x以字符串"x=#value"的形式返回。该方程仅包含'+'，' - '操作，变量 x 和其对应系数。

如果方程没有解，请返回“No solution”。

如果方程有无限解，则返回“Infinite solutions”。

如果方程中只有一个解，要保证返回值 x 是一个整数。

```python
class Solution:
    def solveEquation(self, equation: str) -> str:
        # 分类讨论问题，利用等号分隔
        # 注意处理符号问题，左边加上封端符号
        left,right = equation.split("=")
        left = list(left)
        if left[0] != "+" and left[0] != "-":
            left = ["+"] + left
        right = list(right)
        if right[0] != "+" and right[0] != "-":
            right = ["+"] + right
        # 然后处理需要注意正负号，栈处理，注意数字收多位数和收符号
        def getNumAndX(lst):
            countX = []
            countNum = []
            xActive = False
            numActive = False
            temp = []
            while len(lst) != 0:
                if lst[-1] == "x":
                    xActive = True
                    temp.append("1")
                    lst.pop()
                elif lst[-1].isdigit():
                    if xActive == True:
                        if temp == ["1"]: 
                            temp.pop()
                        temp.append(lst.pop())
                    elif xActive == False:
                        numActive = True 
                        temp.append(lst.pop())
                elif lst[-1] in "+-":
                    if lst[-1] == "+":
                        symbol = 1
                    elif lst[-1] == "-":
                        symbol = -1
                    temp = temp[::-1]
                    n = symbol * int(''.join(temp))
                    temp = [] # 清空
                    lst.pop() # 去除符号位
                    if xActive:
                        countX.append(n)
                        xActive = False
                    elif numActive:
                        countNum.append(n)
                        numActive = False
            return sum(countX),sum(countNum) # 返回值为x的系数和数字系数
        
        leftX,leftN = getNumAndX(left)
        rightX,rightN = getNumAndX(right)
        cX = leftX-rightX
        cN = rightN-leftN
        # 题目设置的都是整数
        # 分类讨论，如果是0,0 无穷解。如果是0，非0，无解，否则单解
        if (cX,cN) == (0,0):
            return  "Infinite solutions"
        elif cX == 0 and cN != 0:
            return "No solution"
        else:
            t = cN//cX
            return "x="+str(t)
```

# 646. 最长数对链

给出 n 个数对。 在每一个数对中，第一个数字总是比第二个数字小。

现在，我们定义一种跟随关系，当且仅当 b < c 时，数对(c, d) 才可以跟在 (a, b) 后面。我们用这种形式来构造一个数对链。

给定一个数对集合，找出能够形成的最长数对链的长度。你不需要用到所有的数对，你可以以任何顺序选择其中的一些数对来构造。

```python
# 方法1 n^2
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        # dp 找最长上升子序列的问题，dp[i]为当前为止最长上升子序列的长度
        # 由于要求数对顺序。。。所以pairs预先排序,这样才符合
        pairs.sort()
        n = len(pairs)
        dp = [1 for i in range(n)]
        for i in range(1,n):
            for j in range(0,i): # 遍历扫描之前的
              # 通常的LIS需要一个group取内层的最大值
              # 但是由于这里是排序过的，所以不需要使用group中介
                if pairs[i][0] > pairs[j][1]: # 如果当前的头严格大于之前的尾巴，则长度更新
                    dp[i] = dp[j] + 1
        # 返回值为dp中的最大值
        return max(dp)
```

```python
# 方法2： nlogn
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        # 根据第二个数字排序后，贪心选【题目自带第一个小于第二个数】
        pairs.sort(key = lambda x:x[1])
        ans = 1
        prev = pairs[0]
        for i in pairs[1:]:
            if i[0] > prev[1]:
                ans += 1
                prev = i
        return ans
```

# 650. 只有两个键的键盘

最初记事本上只有一个字符 'A' 。你每次可以对这个记事本进行两种操作：

Copy All（复制全部）：复制这个记事本中的所有字符（不允许仅复制部分字符）。
Paste（粘贴）：粘贴 上一次 复制的字符。
给你一个数字 n ，你需要使用最少的操作次数，在记事本上输出 恰好 n 个 'A' 。返回能够打印出 n 个 'A' 的最少操作次数。

```python
class Solution:
    def minSteps(self, n: int) -> int:
        # 数学思路
        # 质数则直接返回本身
        # 合数计算分解出来的质数和
        if n == 1:
            return 0
        # 预先写了个质数表。。。

        primeList = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

        # primeList = [2] # 也可以朴素打个质数表
        # for i in range(2,n+1):
        #     bound = math.ceil(math.sqrt(i))
        #     valid = True
        #     for t in range(2,bound+1):
        #         if i%t == 0:
        #             valid = False
        #             break
        #     if valid:
        #         primeList.append(i)

        # 先将其分解质因数
        lst = []
        for p in primeList:
            if p > n:
                break 
            if n % p == 0:
                while n % p == 0:
                    n = n // p 
                    lst.append(p)
             
        return sum(lst)
```

```python
# dp思路
class Solution:
    def minSteps(self, n: int) -> int:
        # 动态规划
        # 要想得到dp[i],则必须先得到dp[j],其中j整除i。粘贴次数为times = i//j - 1 ,加上复制次数1次次数为i//j
        # dp[i] = dp[j] + i//j ,dp[i] = dp[i//j] + j # 取较小的那一个
        # 枚举的时候考虑和j成对的那一个数
        dp = [0xffffffff for i in range(n+1)]
        dp[1] = 0 # 基态
        for i in range(2, n + 1):
            j = 1
            while j * j <= i:
                if i % j == 0:
                    dp[i] = min(dp[i], dp[j] + i // j,dp[i // j] + j)
                j += 1
        
        return dp[n]
```

# 661. 图片平滑器

包含整数的二维矩阵 M 表示一个图片的灰度。你需要设计一个平滑器来让每一个单元的灰度成为平均灰度 (向下舍入) ，平均灰度的计算是周围的8个单元和它本身的值求平均，如果周围的单元格不足八个，则尽可能多的利用它们。

```python
class Solution:
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        # 模拟
        m = len(img)
        n = len(img[0])
        def calc_around(i,j):
            direc = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
            valid = 0 # 看有几个格子合法
            theSum = 0
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n:
                    valid += 1
                    theSum += img[new_i][new_j]
            return math.floor(theSum/valid)
        
        cp = [[0 for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                cp[i][j] = calc_around(i,j)
        return cp
```

# 662. 二叉树最大宽度

给定一个二叉树，编写一个函数来获取这个树的最大宽度。树的宽度是所有层中的最大宽度。这个二叉树与满二叉树（full binary tree）结构相同，但一些节点为空。

每一层的宽度被定义为两个端点（该层最左和最右的非空节点，两端点间的null节点也计入长度）之间的长度。

```python
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        # 利用带有额外信息的bfs,额外信息为类满二叉树的序号
        # leftVal = 2*parent rightVal = 2*parent + 1
        # python数值不会爆。。。
        queue = [(root,1)]
        maxWidth = 1

        while len(queue) != 0:
            new_queue = []
            for node,theID in queue:
                if node.left != None:
                    new_queue.append([node.left,theID*2])
                if node.right != None:
                    new_queue.append([node.right,theID*2+1])
            if len(new_queue) != 0:
                maxWidth = max(maxWidth,new_queue[-1][1]-new_queue[0][1]+1)
            queue = new_queue
        return maxWidth 
```

# 669. 修剪二叉搜索树

给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。通过修剪二叉搜索树，使得所有节点的值在[low, high]中。修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，原有的父代子代关系都应当保留）。 可以证明，存在唯一的答案。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

```python
class Solution:
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        # 递归修剪，后续遍历
        if root == None:
            return None

        root.left = self.trimBST(root.left,low,high)
        root.right = self.trimBST(root.right,low,high)

        # 后续部分
        if root.val > high:# 它变成它的左子树，它这个节点不要了
            root = root.left
        elif root.val < low: # 它变成它的右子树
            root = root.right
                  
        return root
```

# 673. 最长递增子序列的个数

给定一个未排序的整数数组，找到最长递增子序列的个数。

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        # 动态规划
        # 两次动态规划,dp[i]的含义是以nums[i]结尾的最长子序列的长度
        # count[i]的含义是以nums[i]结尾的最长子序列的个数
        n = len(nums)
        dp = [1 for i in range(n)]
        count = [1 for i in range(n)]
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]: # 原本应该更新dp[i]
                    if dp[i] < dp[j] + 1:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[i] == dp[j] + 1:
                        count[i] += count[j]

        # 当j < i && nums[j] < nums[i]的时候，就需要去判断当前 dp[j] + 1 > dp[i]，
        #如果为true，说明：dp[j]是不能接在dp[i]前面，递增序列有更大的长度，那么需要更新长度，既然有更大的长度，那么 counts[i] = counts[j]，因为count[i]所以记录的个数已经无效了
        #但如果dp[j] + 1 == dp[i]，说明dp[j]是可以接在dp[i]前面的，所以要 counts[i] += counts[j];

        maxLength = max(dp)
        indexList = []
        ans = 0
        for index in range(n):
            if dp[index] == maxLength:
                indexList.append(index)
        for index in indexList:
            ans += count[index]
        return ans
```

```python
func findNumberOfLIS(nums []int) int {
    // 初始化，golang翻译版本
    n := len(nums)
    dp := make([]int,n)
    count := make([]int,n)
    for i := 0; i < n; i += 1 {
        dp[i] = 1
        count[i] = 1
    }
    for i := 0; i < n; i += 1{
        for j := 0; j < i; j += 1 {
            if nums[i] > nums[j] {
                if dp[i] < dp[j] + 1 {
                    dp[i] = dp[j] + 1
                    count[i] = count[j]
                } else if dp[i] == dp[j] + 1 {
                    count[i] += count[j]
                }
            }
        }
    }
    tempMax := 0
    for i := 0; i < n; i += 1 {
        if tempMax < dp[i] {
            tempMax = dp[i]
        }
    }
    indexList := []int{}
    for i := 0; i < n; i += 1 {
        if dp[i] == tempMax {
            indexList = append(indexList,i)
        }
    }
    ans := 0
    for _,index := range indexList {
        ans += count[index]
    }
    return ans
}
```

# 676. 实现一个魔法字典

设计一个使用单词列表进行初始化的数据结构，单词列表中的单词 互不相同 。 如果给出一个单词，请判定能否只将这个单词中一个字母换成另一个字母，使得所形成的新单词存在于你构建的字典中。

实现 MagicDictionary 类：

MagicDictionary() 初始化对象
void buildDict(String[] dictionary) 使用字符串数组 dictionary 设定该数据结构，dictionary 中的字符串互不相同
bool search(String searchWord) 给定一个字符串 searchWord ，判定能否只将字符串中 一个 字母换成另一个字母，使得所形成的新字符串能够与字典中的任一字符串匹配。如果可以，返回 true ；否则，返回 false 。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.isEnd = False 

class MagicDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
    
    def insert(self,w):
        node = self.root 
        for ch in w:
            index = ord(ch)-ord("a")
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isEnd = True

    def buildDict(self, dictionary: List[str]) -> None:
        for w in dictionary:
            self.insert(w)

    def dfs(self,node,word,Wordindex,times): # 注意传参包含了node。其实word可以不传入
        if node == None: # 
            return False 
        if node.isEnd and Wordindex == len(word) and times == 1: # 节点顺利走完并且变更一次
            return True 

        if 0<=Wordindex<len(word) and times <= 1: # 没有走完的中途
            found = False # 初始化标记，没有找到
            index = ord(word[Wordindex])-ord("a") # 字典树节点索引
           
            for i in range(26): 
                if found: break # 提速
                if index == i: # 如果这个位置有和单词字符相同，则继续往下搜，不消耗次数
                    found = self.dfs(node.children[i],word,Wordindex+1,times)
                elif index != i: # 如果这个位置不和单词字符相同，往下搜的时候消耗一次更改次数
                    found = self.dfs(node.children[i],word,Wordindex+1,times+1)
            return found # 一种后续遍历的感觉
        return False 

    def search(self, searchWord: str) -> bool:
        return self.dfs(self.root,searchWord,0,0)
        
```

#  687. 最长同值路径

给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。 这条路径可以经过也可以不经过根节点。

**注意**：两个节点之间的路径长度由它们之间的边数表示。

```python
class Solution:
    def longestUnivaluePath(self, root: TreeNode) -> int:
        # 还是后续遍历
        maxLength = 0

        def postOrder(node): # 
            nonlocal maxLength
            if node == None:
                return 0
            count = 1 # 表示的最长路径上的节点数目
            theLength = 1 # 考虑左右子树的时候，最长路径上的节点数目
            leftCount = postOrder(node.left)
            rightCount = postOrder(node.right)
            if node.left != None and node.left.val == node.val:
                count = leftCount + 1
                theLength += leftCount
            if node.right != None and node.right.val == node.val:
                count = max(count,rightCount+1)
                theLength += rightCount
            maxLength = max(maxLength,theLength-1) # 路径为节点数-1
            return count 
        
        postOrder(root)
        return maxLength

```

# 695. 岛屿的最大面积

给定一个包含了一些 0 和 1 的非空二维数组 grid 。

一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # dfs
        maxArea = 0
        area = 0 
        m = len(grid)
        n = len(grid[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(-1,0),(1,0)]

        def judgeValid(i,j):
            if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                return True
            return False
        
        def dfs(i,j):
            nonlocal area 
            if not judgeValid(i,j):
                return 
            if visited[i][j]:
                return 
            visited[i][j] = True 
            area += 1
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(new_i,new_j)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    dfs(i,j)
                    if area != 0:
                        maxArea = max(maxArea,area)
                        area = 0 # 重置

        return maxArea
```

# 729. 我的日程安排表 I

实现一个 MyCalendar 类来存放你的日程安排。如果要添加的日程安排不会造成 重复预订 ，则可以存储这个新的日程安排。

当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生 重复预订 。

日程可以用一对整数 start 和 end 表示，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end 。

实现 MyCalendar 类：

MyCalendar() 初始化日历对象。
boolean book(int start, int end) 如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 true 。否则，返回 false 并且不要将该日程安排添加到日历中。

```python
class MyCalendar:

    def __init__(self):
        from sortedcontainers import SortedDict
        self.soDict = SortedDict()  # k-v对为 起始，结束


    def book(self, start: int, end: int) -> bool:
        if len(self.soDict) == 0:
            self.soDict[start] = end 
            return True 
        index = self.soDict.bisect(start)
        # 如果index没有越界
        if 0 < index < len(self.soDict): # 比较前一个和后一个
            if self.soDict.values()[index-1] <= start and end <= self.soDict.keys()[index]:
                self.soDict[start] = end 
                return True 
        if index == 0: # 只需要检查后一个点
            if end <= self.soDict.keys()[index]:
                self.soDict[start] = end 
                return True 
        if index == len(self.soDict): # 只需要检查前一个点
            if self.soDict.values()[index-1] <= start:
                self.soDict[start] = end 
                return True 
        return False
```



# 733. 图像渲染

有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。

给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。

为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为新的颜色值。

最后返回经过上色渲染后的图像。

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        # DFS
        # 使用一个记录矩阵防止走重复
        m = len(image)
        n = len(image[0])
        validMat = [[False for j in range(n)] for i in range(m)]
        direc = [(-1,0),(1,0),(0,1),(0,-1)] # 方向数组
        origin = image[sr][sc]

        def valid(i,j,origin):
            if 0 <= i < m and 0 <= j < n and image[i][j] == origin:
                return True
            return False

        def dfs(i,j,origin,newColor):
            # 终止条件为越界或者不为起始值
            if not valid(i,j,origin):
                return 
            if validMat[i][j] != False:
                return 
            validMat[i][j] = True
            image[i][j] = newColor
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(new_i,new_j,origin,newColor)
        
        dfs(sr,sc,origin,newColor)
        return image
               
```

# 742. 二叉树最近的叶节点

给定一个 每个结点的值互不相同 的二叉树，和一个目标值 k，找出树中与目标值 k 最近的叶结点。 

这里，与叶结点 最近 表示在二叉树中到达该叶节点需要行进的边数与到达其它叶结点相比最少。而且，当一个结点没有孩子结点时称其为叶结点。

在下面的例子中，输入的树以逐行的平铺形式表示。实际上的有根树 root 将以TreeNode对象的形式给出。

```python
class Solution:
    def findClosestLeaf(self, root: TreeNode, k: int) -> int:
        # 树转图进行搜索
        edges = collections.defaultdict(list)
        leaveSet = set()

        def dfs(node,parent): # 这个dfs同时干两件事，收集叶子和建立图
            if node == None:
                return 
            if node.left == None and node.right == None:
                leaveSet.add(node.val)
            if parent != None:
                edges[parent.val].append(node.val)
                edges[node.val].append(parent.val)
            dfs(node.left,node)
            dfs(node.right,node)

        dfs(root,None) # 默认根节点的父节点为None

        # k是起始点，进行bfs,
        queue = [k]
        visitedSet = set() # 防止走重复
        visitedSet.add(k)

        while len(queue) != 0:
            new_queue = []
            for e in queue:
                if e in leaveSet:
                    return e 
                for neigh in edges[e]:
                    if neigh not in visitedSet:
                        visitedSet.add(neigh)
                        new_queue.append(neigh)
            queue = new_queue

```

# 752. 打开转盘锁

你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。

列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        # 死区扩大。死区和visited区合并
        deadends = set(deadends) # 转换成集合
        start = "0000"

        def change(s):
            ans = []
            path = []
            def dfs(path,index,times):
                if index == 4 and times == 0:
                    ans.append("".join(path[:]))
                    return 
                if times < 0:
                    return 
                if index >= 4:
                    return 
                e = int(s[index])
                e1 = (e+1)%10
                e2 = (e-1)%10
                path.append(str(e))
                dfs(path,index+1,times)
                path.pop()
                path.append(str(e1))
                dfs(path,index+1,times-1)
                path.pop()
                path.append(str(e2))
                dfs(path,index+1,times-1)
                path.pop()
            dfs([],0,1)
            return ans
        
        if target == "0000" and target in deadends: return -1
        if target == "0000" and target not in deadends: return 0
        if "0000" in deadends: return -1

        queue = deque()
        queue.append("0000")
        steps = 0
        while len(queue) != 0:
            new_queue = deque()
            while len(queue) != 0:
                s = queue.popleft()
                if s == target:
                    return steps                 
                tempList = change(s)
                for temp in tempList:
                    if temp not in deadends:
                        new_queue.append(temp)
                        deadends.add(temp)
            queue = new_queue
            steps += 1

        return -1
```

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        deadends = set(deadends) # 转换成为集合方便查找
        if target == "0000" and "0000" not in deadends: return 0
        if "0000" in deadends: return -1 

        def change(s):
            ans = []
            lst = [-1,0,1]
            for i in range(4):
                for j in lst:
                    temp = s[:i] + str((int(s[i])+j)%10) + s[i+1:]
                    ans.append(temp)
            return ans 
        
        start = "0000"
        queue = [start]
        steps = 0

        while len(queue) != 0:
            new_queue = []
            for s in queue:
                if s == target:
                    return steps
                tempList = change(s)
                for temp in  tempList:
                    if temp not in deadends:
                        new_queue.append(temp)
                        deadends.add(temp)
            queue = new_queue
            steps += 1
        return -1


```

# 767. 重构字符串

给定一个字符串`S`，检查是否能重新排布其中的字母，使得两相邻的字符不同。

若可行，输出任意可行的结果。若不可行，返回空字符串。

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        # 堆排序思路
        # 最多元素的字符不能超过半数+1，可以等于一半+1
        # ababa . abab
        ct = collections.Counter(s)
        maxVal = 0
        for key in ct:
            if ct[key] > maxVal:
                maxVal = ct[key]
        if maxVal > math.ceil(len(s)/2):
            return ""
        # 然后实时弹出值最多的。或者弹出前两个
        maxHeap = [] # 注意大根堆的负数
        for key in ct:
            heapq.heappush(maxHeap,[-ct[key],key]) # [频次，值]
        ans = ""
        while len(maxHeap) != 0:
            if len(ans) == 0:
                times,key = heapq.heappop(maxHeap)
                times += 1
                ans += key
                if times != 0:
                    heapq.heappush(maxHeap,[times,key])
            elif ans[-1] == maxHeap[0][1]: # 弹两个
                store = heapq.heappop(maxHeap)
                times,key = heapq.heappop(maxHeap)
                times += 1
                ans += key
                if times != 0:
                    heapq.heappush(maxHeap,[times,key])
                heapq.heappush(maxHeap,store)
            elif ans[-1] != maxHeap[0][1]: # 弹一个
                times,key = heapq.heappop(maxHeap)
                times += 1
                ans += key
                if times != 0:
                    heapq.heappush(maxHeap,[times,key])
        return ans

            
```

# 772. 基本计算器 III

实现一个基本的计算器来计算简单的表达式字符串。

表达式字符串只包含非负整数，算符 +、-、*、/ ，左括号 ( 和右括号 ) 。整数除法需要 向下截断 。

你可以假定给定的表达式总是有效的。所有的中间结果的范围为 [-2^31, 2^31 - 1] 。

```python
class Solution:
    def calculate(self, s: str) -> int:
        # 包含乘除法，需要注意除法处理方式
        # 使用递归解决,需要开双端队列不然超时
        def submethod(lst):
            stack = [] # 存数字
            num = 0
            symbol = "+" # 默认符号
            while len(lst) != 0:
                ch = lst.popleft()
                if ch.isdigit():
                    num = 10*num + int(ch)
                if ch == "(":
                    num = submethod(lst)
                if (ch.isdigit() == False and ch != " ") or len(lst) == 0:
                    if symbol == "+":
                        stack.append(num)
                    elif symbol == "-":
                        stack.append(-num)
                    elif symbol == "*":
                        stack[-1] *= num 
                    elif symbol == "/":
                        stack[-1] = int(stack[-1]/num)
                    symbol = ch 
                    num = 0
                    
                if ch == ")":
                    break 
            return sum(stack)

        s = collections.deque(s)
        return submethod(s)
```

# 773. 滑动谜题

在一个 2 x 3 的板上（board）有 5 块砖瓦，用数字 1~5 来表示, 以及一块空缺用 0 来表示.

一次移动定义为选择 0 与一个相邻的数字（上下左右）进行交换.

最终当板 board 的结果是 [[1,2,3],[4,5,0]] 谜板被解开。

给出一个谜板的初始状态，返回最少可以通过多少次移动解开谜板，如果不能解开谜板，则返回 -1 。

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        # BFS广搜。预先处理成字符串好运算
        # 直接索引邻居进行交换
        target = list("123450")
        visitedSet = set()
        start = []
        for i in range(2):
            for j in range(3):
                start.append(str(board[i][j]))
        steps = 0
        queue = [start]
        neighDict = {
            0:[1,3],
            1:[0,2,4],
            2:[1,5],
            3:[0,4],
            4:[1,3,5],
            5:[2,4]
        }
        while len(queue) != 0:
            new_queue = []
            for state in queue: # state是一个一维数组
                if state == target:
                    return steps 
                index = state.index("0")
                for neigh in neighDict[index]:
                    cpState = state.copy() # 复制
                    cpState[index],cpState[neigh] = cpState[neigh],cpState[index] # 交换
                    string =  "".join(cpState) # 判重
                    if string not in visitedSet:
                        visitedSet.add(string)
                        new_queue.append(cpState)
            steps += 1
            queue = new_queue
        return -1
```

# 785. 判断二分图

存在一个 无向图 ，图中有 n 个节点。其中每个节点都有一个介于 0 到 n - 1 之间的唯一编号。给你一个二维数组 graph ，其中 graph[u] 是一个节点数组，由节点 u 的邻接节点组成。形式上，对于 graph[u] 中的每个 v ，都存在一条位于节点 u 和节点 v 之间的无向边。该无向图同时具有以下属性：
不存在自环（graph[u] 不包含 u）。
不存在平行边（graph[u] 不包含重复值）。
如果 v 在 graph[u] 内，那么 u 也应该在 graph[v] 内（该图是无向图）
这个图可能不是连通图，也就是说两个节点 u 和 v 之间可能不存在一条连通彼此的路径。
二分图 定义：如果能将一个图的节点集合分割成两个独立的子集 A 和 B ，并使图中的每一条边的两个节点一个来自 A 集合，一个来自 B 集合，就将这个图称为 二分图 。

如果图是二分图，返回 true ；否则，返回 false 。

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # 染色问题,这一题的图给的很友好，没有自环，且是无向图
        n = len(graph)
        colored = [-1 for i in range(n)] # 初始化为-1表示未染色

        def dfs(i,color): # 传入参数为序号，和需要被染的颜色，dfs
            if colored[i] >= 0:
                return colored[i] == color # 返回值为是否和要被染的那个颜色一致  
            elif colored[i] == -1:
                colored[i] = color
                for neigh in graph[i]:
                    if dfs(neigh,1-color) == False:
                        return False 
                return True

        # DFS染色
        for i in range(n):
            if colored[i] == -1: # 表示未染色，则开始染色
                if dfs(i,0) == False: # 染色失败
                    return False 
        return True
```

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # BFS染色
        n = len(graph)
        colored = [-1 for i in range(n)]

        def bfs(i,color):# 参数为索引和颜色
            colored[i] = color
            queue = deque()
            queue.append(i) # 初始化
            while len(queue) != 0:
                e = queue.popleft()
                for neigh in graph[e]:
                    if colored[neigh] >= 0: # 如果已经被染色，检查
                        if colored[neigh] == colored[e]:
                            return False
                    elif colored[neigh] == -1: # 如果未被染色，染色成邻居的反色
                        queue.append(neigh)
                        colored[neigh] = 1 - colored[e]
            return True 

        for i in range(n):
            if colored[i] == -1: # 开始染色
                if bfs(i,0) == False:
                    return False 
        return True 
```

# 800. 相似 RGB 颜色

RGB 颜色 "#AABBCC" 可以简写成 "#ABC" 。

例如，"#15c" 其实是 "#1155cc" 的简写。
现在，假如我们分别定义两个颜色 "#ABCDEF" 和 "#UVWXYZ"，则他们的相似度可以通过这个表达式 -(AB - UV)^2 - (CD - WX)^2 - (EF - YZ)^2 来计算。

那么给你一个按 "#ABCDEF" 形式定义的字符串 color 表示 RGB 颜色，请你以字符串形式，返回一个与它相似度最大且可以简写的颜色。（比如，可以表示成类似 "#XYZ" 的形式）

任何 具有相同的（最大）相似度的答案都会被视为正确答案。

```python
# 纯尬算
# 实际上由于三者独立，不需要穷举4096次。。。取每个的最优解就行
class Solution:
    def similarRGB(self, color: str) -> str:
        # 穷举，它让我干啥我干啥
        # 穷举三位，原始颜色不为简写
        temp = -0xffffffff
        store = [None,None,None]
        theDict = {"a":10,"b":11,"c":12,"d":13,"e":14,"f":15}
        def toNum(s): # 传入两位字符，得到十进制数
            nonlocal theDict
            n = 0
            if s[1].isalpha():
                n += theDict[s[1]]
            else:
                n += int(s[1])
            if s[0].isalpha():
                n += theDict[s[0]] * 16
            else:
                n += int(s[0]) * 16
            return n 
        
        def calc(strings,i,j,k):
            t1 = toNum(strings[1:3])
            t2 = toNum(strings[3:5])
            t3 = toNum(strings[5:7])
            return -(t1-i*17)**2 - (t2-j*17)**2 - (t3-k*17)**2
        
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    tempAns = calc(color,i,j,k)
                    if tempAns > temp:
                        temp = tempAns
                        store = [i,j,k]
        
        Mirror = {10:"a",11:"b",12:"c",13:"d",14:"e",15:"f"}

        final = "#"
        for ele in store:
            if Mirror.get(ele):
                final += Mirror.get(ele)*2
            else:
                final += str(ele)*2
        return final
```

# 807. 保持城市天际线

在二维数组grid中，grid[i][j]代表位于某处的建筑物的高度。 我们被允许增加任何数量（不同建筑物的数量可能不同）的建筑物的高度。 高度 0 也被认为是建筑物。

最后，从新数组的所有四个方向（即顶部，底部，左侧和右侧）观看的“天际线”必须与原始数组的天际线相同。 城市的天际线是从远处观看时，由所有建筑物形成的矩形的外部轮廓。 请看下面的例子。

建筑物高度可以增加的最大总和是多少？

```python
class Solution:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        # 类似于脑筋急转弯
        slide = [max(grid[i]) for i in range(len(grid))] # 侧边
        view = [-1 for j in range(len(grid[0]))]
        # 填充底部数组
        m,n = len(grid),len(grid[0])
        for j in range(n):
            temp = grid[0][j]
            for i in range(m):
                if grid[i][j] > temp:
                    temp = grid[i][j]
            view[j] = temp
        
        # 看它比限制值差多少
        ans = 0
        for i in range(m):
            for j in range(n):
                extra = min(slide[i],view[j])-grid[i][j] 
                ans += extra
        return ans
                
```

```go
class Solution:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        # 类似于脑筋急转弯
        slide = [max(grid[i]) for i in range(len(grid))] # 侧边
        view = [-1 for j in range(len(grid[0]))]
        # 填充底部数组
        m,n = len(grid),len(grid[0])
        for j in range(n):
            temp = grid[0][j]
            for i in range(m):
                if grid[i][j] > temp:
                    temp = grid[i][j]
            view[j] = temp
        
        # 看它比限制值差多少
        ans = 0
        for i in range(m):
            for j in range(n):
                extra = min(slide[i],view[j])-grid[i][j] 
                ans += extra
        return ans
                
```



# 816. 模糊坐标

我们有一些二维坐标，如 "(1, 3)" 或 "(2, 0.5)"，然后我们移除所有逗号，小数点和空格，得到一个字符串S。返回所有可能的原始字符串到一个列表中。

原始的坐标表示法不会存在多余的零，所以不会出现类似于"00", "0.0", "0.00", "1.0", "001", "00.01"或一些其他更小的数来表示坐标。此外，一个小数点前至少存在一个数，所以也不会出现“.1”形式的数字。

最后返回的列表可以是任意顺序的。而且注意返回的两个数字中间（逗号之后）都有一个空格。

```python
class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        # 回溯
        s = s[1:-1]
        ans = []
        n = len(s)

        def backtracking(path,index,countDot,countGap):
            if index == n and countGap == 0:
                if judge(path):
                    temp = "(" + "".join(path) + ")"
                    ans.append(temp)
                return 
            if index == n and countGap != 0:
                return 
            if countDot < 0 or countGap < 0:
                return 
            
            # 选字母
            path.append(s[index])
            backtracking(path,index+1,countDot,countGap)
            path.pop()
            # 选小数点
            if len(path) != 0 and path[-1] != "." and path[-1] != ", ":
                path.append(".")
                backtracking(path,index,countDot-1,countGap)
                path.pop()
            # 选逗号
            if len(path) != 0 and path[-1] != ".":
                path.append(", ")
                backtracking(path,index,countDot,countGap-1)
                path.pop()

        def judge(temp):  # 还需要合法化的筛
            temp = ''.join(temp) # 传入的是个列表,已经去括号了
            n = len(temp)
            for i in range(n):
                if temp[i] == ",":
                    index = i 
                    break 
            left = "".join(temp[:index])
            right = "".join(temp[index+2:])
            leftDot = 0
            rightDot = 0
            for ch in left:
                if ch == ".":
                    leftDot += 1
            if leftDot > 1: return False
            for ch in right:
                if ch == ".":
                    rightDot += 1
            if rightDot > 1: return False 
            # 需要去掉01.23这样的
            if len(left) > 1 and left[0] == "0" and left[1] != ".":
                return False 
            if len(right) > 1 and right[0] == "0" and right[1] != ".":
                return False
            leftSum = 0
            state = False
            countZero = 0
            for ch in left:
                if ch.isdigit():
                    leftSum += int(ch)
                if ch == ".":
                    state = True 
                if state and ch.isdigit():
                    countZero += int(ch)
            if leftSum == 0 and len(left) != 1 or (state and countZero == 0):
                return False 
            if state and left[-1] == "0":
                return False
            rightSum = 0
            state = False
            countZero = 0
            for ch in right:
                if ch.isdigit():
                    rightSum += int(ch)
                if ch == ".":
                    state = True 
                if state and ch.isdigit():
                    countZero += int(ch)
            if rightSum == 0 and len(right) != 1 or (state and countZero == 0):
                return False
            if state and right[-1] == "0":
                return False
            # 还需要去掉1.0等点后面是0的。0.10也不行

            return True

        backtracking([],0,2,1)
        ans.sort()
        return ans
            
```

# 817. 链表组件

给定链表头结点 head，该链表上的每个结点都有一个 唯一的整型值 。

同时给定列表 G，该列表是上述链表中整型值的一个子集。

返回列表 G 中组件的个数，这里对组件的定义为：链表中一段最长连续结点的值（该值必须在列表 G 中）构成的集合。

```python
class Solution:
    def numComponents(self, head: ListNode, nums: List[int]) -> int:
        # 题意很魔鬼
        # 1，由于数组G是链表head所有元素值的子集，所以数组G中的任何元素都能在链表中找到（这TM不是废话？）;
        # 2，因此G中的每个元素就可以看做是链表head的一个子链表，即G中的每个元素都是链表head的组件；
        # 3，但是此时的组件还不敢称之为真正的组件，因为完全存在这样一种可能：
        #3.1 G中任意组合的两个元素a, b构成了一个更长的head的子链表 a->b ，
        #3.2 此时根据题意 a->b 比 a 和 b 都要长，所以 a->b 包涵了 a、b 成为真正的组件，原来的a、b 就不能算组件了，
        #3.3 如此一来问题变成了 对于给定的集合G，G中所有的元素能构成多少个head中相连的子链表？
        # 如果G里有链表中连续的元素，则为1个组件，否则也为1个组件

        count = 0
        # 把g，集合化。每次遍历。如果下一个元素在set中，count不变，不在set中，count+1
        numsSet = set(nums)
        cur = head
        size = 0
        active = False # 注意这个变量的设计
        while cur != None:
            if cur.val in numsSet:
                active = True
            else:
                if active == True:
                    count += 1
                    active = False
            cur = cur.next
            size += 1
        # 最后如果active，没有被消除也要加上
        if active:
            count += 1
        return count 
```

# 826. 安排工作以达到最大收益

有一些工作：difficulty[i] 表示第 i 个工作的难度，profit[i] 表示第 i 个工作的收益。

现在我们有一些工人。worker[i] 是第 i 个工人的能力，即该工人只能完成难度小于等于 worker[i] 的工作。

每一个工人都最多只能安排一个工作，但是一个工作可以完成多次。

举个例子，如果 3 个工人都尝试完成一份报酬为 1 的同样工作，那么总收益为 $3。如果一个工人不能完成任何工作，他的收益为 $0 。

我们能得到的最大收益是多少？

```python
class Solution:
    def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:
        # 预先处理二分。注意那些费力不讨好的任务，预处理那些费力不讨好的
        difficultyDict = collections.defaultdict(int)
        n = len(difficulty)
        for i in range(n):
            if profit[i] > difficultyDict[difficulty[i]]:
                difficultyDict[difficulty[i]] = profit[i]
        theMap = []
        for key in difficultyDict:
            theMap.append([key,difficultyDict[key]])
        theMap.sort()
        p = 0
        nowMax = theMap[0][1]
        while p < len(theMap):
            nowMax = max(nowMax,theMap[p][1])
            if theMap[p][1] < nowMax:
                theMap[p][1] = nowMax
            p += 1
        # 由于现在不存在重复的，只需要找小于等于n的最大值，二分预处理
        def binarySearch(lst,n):
            left = 0
            right = len(lst) - 1
            while left <= right:
                mid = (left + right)//2
                if lst[mid][0] == n:
                    return mid 
                elif lst[mid][0] > n: # 缩小
                    right = mid - 1
                elif lst[mid][0] < n: # 增大
                    left = mid + 1
            return right 
        
        profits = 0
        minD = min(difficulty)
        for n in worker:
            if n < minD:
                continue 
            index = binarySearch(theMap,n)
            profits += theMap[index][1]
        return profits

```

# 836. 矩形重叠

矩形以列表 [x1, y1, x2, y2] 的形式表示，其中 (x1, y1) 为左下角的坐标，(x2, y2) 是右上角的坐标。矩形的上下边平行于 x 轴，左右边平行于 y 轴。

如果相交的面积为 正 ，则称两矩形重叠。需要明确的是，只在角或边接触的两个矩形不构成重叠。

给出两个矩形 rec1 和 rec2 。如果它们重叠，返回 true；否则，返回 false 。

```python
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        # 判距法,先排序到rec1在左边
        ax1,ay1,ax2,ay2 = rec1
        bx1,by1,bx2,by2 = rec2
        def getIntersection(lst1,lst2):
            # 排序，使得lst1在lst2的左边
            cp = [lst1,lst2]
            cp.sort()
            lst1,lst2 = cp[0],cp[1]
            a = lst1[0]
            b = lst1[1]
            c = lst2[0]
            d = lst2[1]
            if a <= c <= b <= d:
                return b-c 
            elif a <= c <= d <= b:
                return d-c 
            else:
                return 0
        
        width = getIntersection([ax1,ax2],[bx1,bx2])
        height = getIntersection([ay1,ay2],[by1,by2])

        return width*height != 0
```

# 841. 钥匙和房间

有 N 个房间，开始时你位于 0 号房间。每个房间有不同的号码：0，1，2，...，N-1，并且房间里可能有一些钥匙能使你进入下一个房间。

在形式上，对于每个房间 i 都有一个钥匙列表 rooms[i]，每个钥匙 rooms[i][j] 由 [0,1，...，N-1] 中的一个整数表示，其中 N = rooms.length。 钥匙 rooms[i][j] = v 可以打开编号为 v 的房间。

最初，除 0 号房间外的其余所有房间都被锁住。

你可以自由地在房间之间来回走动。

如果能进入每个房间返回 true，否则返回 false。

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        # BFS
        n = len(rooms)
        visited = [False for i in range(n)] # 代表是否访问过
        visited[0] = True
        queue = [0] # 放的是房间号
        while len(queue) != 0:
            new_queue = []
            for r in queue: # 遍历收集每一个房间的钥匙
                for nei in rooms[r]: # 收集这个房间的所有钥匙
                    if not visited[nei]:
                        visited[nei] = True # 标记为访问
                        new_queue.append(nei)
            queue = new_queue
        for every in visited:
            if every == False:
                return False 
        return True
```

# 849. 到最近的人的最大距离

给你一个数组 seats 表示一排座位，其中 seats[i] = 1 代表有人坐在第 i 个座位上，seats[i] = 0 代表座位 i 上是空的（下标从 0 开始）。

至少有一个空座位，且至少有一人已经坐在座位上。

亚历克斯希望坐在一个能够使他与离他最近的人之间的距离达到最大化的座位上。

返回他到离他最近的人的最大距离。

```python
class Solution:
    def maxDistToClosest(self, seats: List[int]) -> int:
        # 求每个1到最近的1的距离
        indexList = [] # 记录的是1
        n = len(seats)
        for i in range(n):
            if seats[i] == 1:
                indexList.append(i)
        
        # 地板除处以相邻的
        prev = 0
        maxGap = 1
        if indexList[0] != 0:
            maxGap = indexList[0] - 0
        if indexList[-1] != n-1:
            maxGap = max(maxGap,n-1-indexList[-1])
        # 然后求两两差值即可
        p = 1
        while p < len(indexList):
            maxGap = max(maxGap,(indexList[p]-indexList[p-1])//2) # 注意这里是地板除
            p += 1
        return maxGap
```

# 859. 亲密字符串

给定两个由小写字母构成的字符串 A 和 B ，只要我们可以通过交换 A 中的两个字母得到与 B 相等的结果，就返回 true ；否则返回 false 。

交换字母的定义是取两个下标 i 和 j （下标从 0 开始），只要 i!=j 就交换 A[i] 和 A[j] 处的字符。例如，在 "abcd" 中交换下标 0 和下标 2 的元素可以生成 "cbad" 。

```python
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        # 必须要交换。不交换就相等不一定符合，要看是否有某个字符重复两次以上
        if len(s) != len(goal):
            return False 
        n = len(s)
        alphaDict1 = [0 for i in range(26)]
        active = False # 标记是否有数字达到过2以上
        for i in range(n):
            index = ord(s[i]) - ord("a")
            alphaDict1[index] += 1
            if alphaDict1[index] == 2:
                active = True 
                break 
        # 一一比对。
        memo = []
        for i in range(n):
            if s[i] != goal[i]:
                memo.append((s[i],goal[i]))
        if len(memo) == 2: # 
            return memo[0][1] == memo[1][0] and memo[0][0] == memo[1][1]
        if len(memo) == 0:
            return active
        return False

```

# 861. 翻转矩阵后的得分

有一个二维矩阵 A 其中每个元素的值为 0 或 1 。

移动是指选择任一行或列，并转换该行或列中的每一个值：将所有 0 都更改为 1，将所有 1 都更改为 0。

在做出任意次数的移动后，将该矩阵的每一行都按照二进制数来解释，矩阵的得分就是这些数字的总和。

返回尽可能高的分数。

```python
class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        # 数学分析，先把最高位翻转到1，然后每一列取0/1中数量较多的
        def toNum(lst):# 传入一个列表，获得二进制数
            lst = lst[::-1]
            n = len(lst)
            num = 0
            for i in range(n):
                num += 2**i * lst[i]
            return num 
        
        # 修改原则，先把每一行最高位置1
        for line in grid:
            if line[0] != 1:
                for i in range(len(line)):
                    line[i] ^= 1
        
        # 再统计每一列中是0多还是1多
        m = len(grid)
        n = len(grid[0])
        theList = []
        for j in range(n):
            oneCounts = 0
            for i in range(m):
                if grid[i][j] == 1:
                    oneCounts += 1
            theList.append(max(oneCounts,m-oneCounts))

        ans = toNum(theList)
        return ans
```

# 863. 二叉树中所有距离为 K 的结点

给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。

返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # 树转图，然后bfs
        graph = collections.defaultdict(list) # 树的节点值唯一且范围很小

        def preOrder(node,parent):
            if node == None:
                return 
            if parent != None:
                graph[node.val].append(parent.val)
                graph[parent.val].append(node.val)
            preOrder(node.left,node)
            preOrder(node.right,node)

        preOrder(root,None) # 调用树转图

        # BFS搜
        queue = [target.val]
        ans = []
        steps = -1
        visited = set()
        visited.add(target.val)
        while steps != k:
            new_queue = []
            steps += 1
            for e in queue:
                if steps == k:
                    ans.append(e)
                for neigh in graph[e] :
                    if neigh not in visited:
                        visited.add(neigh)
                        new_queue.append(neigh)
            queue = new_queue
        return ans

```

# 884. 两句话中的不常见单词

给定两个句子 A 和 B 。 （句子是一串由空格分隔的单词。每个单词仅由小写字母组成。）

如果一个单词在其中一个句子中只出现一次，在另一个句子中却没有出现，那么这个单词就是不常见的。

返回所有不常用单词的列表。

您可以按任何顺序返回列表。

```python
class Solution:
    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        # 暴力尬算法
        lst1 = s1.split(" ")
        lst2 = s2.split(" ")
        ct1 = collections.Counter(lst1)
        ct2 = collections.Counter(lst2)
        ans = []
        for key in ct1:
            if ct1[key] == 1 and key not in ct2:
                ans.append(key)
        for key in ct2:
            if ct2[key] == 1 and key not in ct1:
                ans.append(key)
        return ans
```

# 886. 可能的二分法

给定一组 N 人（编号为 1, 2, ..., N）， 我们想把每个人分进任意大小的两组。

每个人都可能不喜欢其他人，那么他们不应该属于同一组。

形式上，如果 dislikes[i] = [a, b]，表示不允许将编号为 a 和 b 的人归入同一组。

当可以用这种方法将所有人分进两组时，返回 true；否则返回 false。

```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        colors = [-1 for i in range(n)]
        edges = collections.defaultdict(list)
        # 注意双向,自己构建一个无向图
        for a,b in dislikes: # 系数都减1
            edges[a-1].append(b-1)
            edges[b-1].append(a-1)

        def dfs(i,color): # 染色
            if colors[i] >= 0:
                return colors[i] == color 
            elif colors[i] == -1:
                colors[i] = color 
                for neigh in edges[i]:
                    if dfs(neigh,1-color) == False: # 染色失败
                        return False 
                return True

        for i in range(n):
            if colors[i] == -1: # 未染色
                if dfs(i,0) == False: # 染色失败
                    return False 
        return True 
```

# 901. 股票价格跨度

编写一个 StockSpanner 类，它收集某些股票的每日报价，并返回该股票当日价格的跨度。

今天股票价格的跨度被定义为股票价格小于或等于今天价格的最大连续日数（从今天开始往回数，包括今天）。

例如，如果未来7天股票的价格是 [100, 80, 60, 70, 60, 75, 85]，那么股票跨度将是 [1, 1, 1, 2, 1, 4, 6]。

```python
class StockSpanner:
# 单调栈，维护单调递减栈
# 栈内为元组，有两个数据，一个是价格数据，一个是距离第一个比他大的数据的天数
# 有一种动态规划的思想
# 遇到递增则特殊处理
    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        gap = 1  #默认为1
        while len(self.stack) != 0 and self.stack[-1][0] <= price: 
            gap += self.stack.pop()[1]
        self.stack.append((price,gap))
        return self.stack[-1][1]
```

# 914. 卡牌分组

给定一副牌，每张牌上都写着一个整数。

此时，你需要选定一个数字 X，使我们可以将整副牌按下述规则分成 1 组或更多组：

每组都有 X 张牌。
组内所有的牌上都写着相同的整数。
仅当你可选的 X >= 2 时返回 true。

```python
class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        # ct之后找到所有数的gcd,最大公因数大于等于2
        def findGCD(a,b):
            while a != 0:
                temp = a
                a = b % a 
                b = temp
            return b 
        ct = collections.Counter(deck)
        lst = []
        for key in ct: # 列表中只需要记录频次
            lst.append(ct[key])
        t = (reduce(findGCD,lst))
        return t >= 2
```

# 957. N 天后的牢房

8 间牢房排成一排，每间牢房不是有人住就是空着。

每天，无论牢房是被占用或空置，都会根据以下规则进行更改：

如果一间牢房的两个相邻的房间都被占用或都是空的，那么该牢房就会被占用。
否则，它就会被空置。
（请注意，由于监狱中的牢房排成一行，所以行中的第一个和最后一个房间无法有两个相邻的房间。）

我们用以下方式描述监狱的当前状态：如果第 i 间牢房被占用，则 cell[i]==1，否则 cell[i]==0。

根据监狱的初始状态，在 N 天后返回监狱的状况（和上述 N 种变化）。

```python
# 其实数学方法可以证出来是14天一个循环。。。但是不知道怎么证
class Solution:
    def prisonAfterNDays(self, cells: List[int], n: int) -> List[int]:
        # 记忆化递归。一个哈希表搜。k是当前状态，v是下一次状态
        memo = dict()
        circleTimes = None
        countTimes = -1 # 这个参数是print大法调出来的。。。
        timesDict = dict()

        def memo_recur(lst):
            nonlocal memo
            nonlocal countTimes
            nonlocal circleTimes
            countTimes += 1
            key = ""
            for i in lst:
                key += chr(i)
            if key in memo: # 如果已经计算过,算上次出现的时候次数的差值
                if circleTimes == None:
                    circleTimes = countTimes - timesDict[key]
                return memo[key]
            # 否则
            temp = lst.copy()
            p = 0
            temp[0],temp[7] = 0,0
            for i in range(8):
                if 0<=i-1<=7 and 0<=i+1<=7:
                    if lst[i-1]^lst[i+1] == 0:
                        temp[i] = 1
                    else:
                        temp[i] = 0
            memo[key] = temp
            timesDict[key] = countTimes
            return memo[key]

        # 需要找到最小循环节
        while n > 0:
            if circleTimes == None or n < circleTimes:
                cells = memo_recur(cells)            
                n -= 1
            elif circleTimes != None and n >= circleTimes:
                n = n % circleTimes

            # cells = memo_recur(cells)            
            # n -= 1
            # print(cells,circleTimes,n)
        return cells
```

# 990. 等式方程的可满足性

给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。

只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。 

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
        return self.find(x) == self.find(y)

class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        # 并查集，
        theUFset = UF(26) # 创建大小为26的并查集
        # 整理，所有先链接好所有链接的

        for equa in equations: # 先链接
            x = ord(equa[0]) - ord("a")
            y = ord(equa[3]) - ord("a")
            if equa[1:3] == "==":
                theUFset.union(x,y)

        for equa in equations: # 再判断不允许链接的部分
            x = ord(equa[0]) - ord("a")
            y = ord(equa[3]) - ord("a")                
            if equa[1:3] == "!=":
                if theUFset.isConnect(x,y) == False:
                    pass
                else:
                    return False 

        return True
```

# 994. 腐烂的橘子

在给定的网格中，每个单元格可以有以下三个值之一：

值 0 代表空单元格；
值 1 代表新鲜橘子；
值 2 代表腐烂的橘子。
每分钟，任何与腐烂的橘子（在 4 个正方向上）相邻的新鲜橘子都会腐烂。

返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        # 预先遍历求出橘子总数，把烂橘子加进队列
        # BFS
        allOrange = 0
        m = len(grid)
        n = len(grid[0])
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        visited = [[False for j in range(n)] for i in range(m)]
        queue = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] > 0:
                    allOrange += 1
                if grid[i][j] == 2:
                    queue.append((i,j))
                    visited[i][j] = True 

        if allOrange == 0: # 开始没有橘子
            return 0
            
        steps = -1
        while len(queue) != 0:
            new_queue = []
            steps += 1
            for i,j in queue:
                allOrange -= 1 
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and grid[new_i][new_j] == 1:
                        visited[new_i][new_j] = True 
                        grid[new_i][new_j] = 2
                        new_queue.append((new_i,new_j))
            queue = new_queue
        
        if allOrange != 0:
            return -1
        else:
            return steps
```

# 999. 可以被一步捕获的棋子数

在一个 8 x 8 的棋盘上，有一个白色的车（Rook），用字符 'R' 表示。棋盘上还可能存在空方块，白色的象（Bishop）以及黑色的卒（pawn），分别用字符 '.'，'B' 和 'p' 表示。不难看出，大写字符表示的是白棋，小写字符表示的是黑棋。

你现在可以控制车移动一次，请你统计有多少敌方的卒处于你的捕获范围内（即，可以被一步捕获的棋子数）。

```python
class Solution:
    def numRookCaptures(self, board: List[List[str]]) -> int:
        # 我只有一辆车：进行四个方向的搜索
        for i in range(8):
            for j in range(8):
                if board[i][j] == "R":
                    R = (i,j)
        
        direc = [(-1,0),(1,0),(0,1),(0,-1)]

        ans = 0
        for di in direc:
            i = R[0]+di[0]
            j = R[1]+di[1]
            tempAns = 0
            while 0 <= i < 8 and 0 <= j < 8 and board[i][j] == ".":
                i += di[0]
                j += di[1]
            if 0 <= i < 8 and 0 <= j < 8 and board[i][j] == "p":
                ans += 1
        return ans 
```

# 1014. 最佳观光组合

给你一个正整数数组 values，其中 values[i] 表示第 i 个观光景点的评分，并且两个景点 i 和 j 之间的 距离 为 j - i。

一对景点（i < j）组成的观光组合的得分为 values[i] + values[j] + i - j ，也就是景点的评分之和 减去 它们两者之间的距离。

返回一对观光景点能取得的最高分。

```python
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        # 最优解问题，考虑动态规划
        # 根据数据规模考虑 n 级别的动态规划
        n = len(values)
        # (value[i]+i) + (value[j]-j) 其中 i < j , 对第一个括号内进行分析讨论,使得它最大
        origin = values[0] + 0
        
        dp = [0 for i in range(n)]
        for i in range(1,n):
            dp[i] = (values[i]-i) + origin
            if values[i]+i > origin: # 这个更新需要放在dp[i]的后面，因为i必须在j的前面
                origin = values[i] + i 
        return max(dp)

```

```go
func maxScoreSightseeingPair(values []int) int {
    n := len(values)
    dp := make([]int,n)
    origin := values[0] + 0
    for i := 1; i < n; i += 1 {
        dp[i] = values[i]-i + origin
        if values[i] + i > origin {
            origin = values[i] + i 
        }
    }
    return max(dp)
}

func max(arr []int) int {
    n := arr[0]
    for _,v := range arr {
        if v > n {
            n = v
        }
    }
    return n
}
```

# 1027. 最长等差数列

给定一个整数数组 A，返回 A 中最长等差子序列的长度。

回想一下，A 的子序列是列表 A[i_1], A[i_2], ..., A[i_k] 其中 0 <= i_1 < i_2 < ... < i_k <= A.length - 1。并且如果 B[i+1] - B[i]( 0 <= i < B.length - 1) 的值都相同，那么序列 B 是等差的。

```python
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        # 设计一个函数，对存在的每种差都计算一遍
        # 不能排序计算最小差。因为不一定是在相邻的里面差出来的
        gapSet = set()
        n = len(nums)
        for i in range(n): # 
            for j in range(i+1,n):
                gap = nums[j]-nums[i]
                gapSet.add(gap)
        
        temp = [] # 记录每种差值的最长
        for gap in gapSet:
            tempLength = self.findArr(nums,gap)
            temp.append(tempLength)
        return max(temp)
                
    def findArr(self,nums,gap): #  辅助函数，找最长定差序列
        # 使用dp和哈希
        n = len(nums)
        dp = [1 for i in range(n)]
        hashDict = dict()
        # dp[i]的含义为到i为止，最长长度，
        # 状态转移为： nums[i]-gap是否在之前出现过，出现过则为它的长度+1.否则没必要改变
        hashDict[nums[0]] = 0 # k-v 为 数：索引
        for i in range(1,n):
            if nums[i]-gap in hashDict:
                dp[i] = dp[hashDict[nums[i]-gap]] + 1
            hashDict[nums[i]] = i 
        return max(dp) # 返回最大值

```

# 1029. 两地调度

公司计划面试 2n 人。给你一个数组 costs ，其中 costs[i] = [aCosti, bCosti] 。第 i 人飞往 a 市的费用为 aCosti ，飞往 b 市的费用为 bCosti 。

返回将每个人都飞到 a 、b 中某座城市的最低费用，要求每个城市都有 n 人抵达。

```python
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        # 需要用到的思想，差值为额外负担
        # 假设所有人去同一个地方,然后按照权重排序
        n = len(costs)
        person = [(costs[i][0]-costs[i][1],costs[i]) for i in range(n)]
        person.sort()
        needCost = 0 # 假设全去b,
        for i in range(n):
            needCost += costs[i][1]
        # 然后取前一半,他们加上差额

        for i in range(n//2):
            needCost += person[i][0]
        return needCost
```



# 1030. 距离顺序排列矩阵单元格

给出 R 行 C 列的矩阵，其中的单元格的整数坐标为 (r, c)，满足 0 <= r < R 且 0 <= c < C。

另外，我们在该矩阵中给出了一个坐标为 (r0, c0) 的单元格。

返回矩阵中的所有单元格的坐标，并按到 (r0, c0) 的距离从最小到最大的顺序排，其中，两单元格(r1, c1) 和 (r2, c2) 之间的距离是曼哈顿距离，|r1 - r2| + |c1 - c2|。（你可以按任何满足此条件的顺序返回答案。）

```python
class Solution:
    def allCellsDistOrder(self, rows: int, cols: int, rCenter: int, cCenter: int) -> List[List[int]]:
        gridList = []
        for i in range(rows):
            for j in range(cols):
                gridList.append([i,j])            
        gridList.sort(key = lambda x:abs(x[0]-rCenter)+abs(x[1]-cCenter))
        return gridList
```

# 1052. 爱生气的书店老板

今天，书店老板有一家店打算试营业 customers.length 分钟。每分钟都有一些顾客（customers[i]）会进入书店，所有这些顾客都会在那一分钟结束后离开。

在某些时候，书店老板会生气。 如果书店老板在第 i 分钟生气，那么 grumpy[i] = 1，否则 grumpy[i] = 0。 当书店老板生气时，那一分钟的顾客就会不满意，不生气则他们是满意的。

书店老板知道一个秘密技巧，能抑制自己的情绪，可以让自己连续 X 分钟不生气，但却只能使用一次。

请你返回这一天营业下来，最多有多少客户能够感到满意。

```python
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        # 先预先计算原本是0的时候到全部顾客人数
        total = 0
        n = len(customers)
        for i in range(n):
            if grumpy[i] == 0:
                total += customers[i]
        # 此时total为全部人数。然后定窗口长度从左到右，计算最大增量。只对那些为1的有影响
        window = 0
        for i in range(minutes): # 初始化窗口
            if grumpy[i] == 1:
                window += customers[i]

        left = 0
        right = minutes
        maxWindow = window # 初始化为第一个窗口值
        while right < n: # 开始滑动
            add = grumpy[right]
            if add == 1:
                window += customers[right]
            right += 1
            delete = grumpy[left]
            if delete == 1:
                window -= customers[left]
            left += 1
            maxWindow = max(maxWindow,window)
        
        # window是增量，最终返回total+增量
        return total + maxWindow
            

```

# 1054. 距离相等的条形码

在一个仓库里，有一排条形码，其中第 i 个条形码为 barcodes[i]。

请你重新排列这些条形码，使其中两个相邻的条形码 不能 相等。 你可以返回任何满足该要求的答案，此题保证存在答案。

```python
class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        # 堆，每次添加必须都要以最多重复次数的打头
        ct = collections.Counter(barcodes)
        maxHeap = []
        for key in ct:
            maxHeap.append((-ct[key],key)) # 频次，元素
        heapq.heapify(maxHeap)
        ans = []
        times,val = heapq.heappop(maxHeap)
        ans.append(val)
        times += 1 # 注意原先的频次是负数
        if times != 0:
            heapq.heappush(maxHeap,(times,val))
        while len(maxHeap) != 0:
            if ans[-1] != maxHeap[0][1]:
                    times,val = heapq.heappop(maxHeap)
                    ans.append(val)
                    times += 1 # 注意原先的频次是负数
                    if times != 0:
                        heapq.heappush(maxHeap,(times,val))
            elif ans[-1] == maxHeap[0][1]: # 则先暂存
                store = heapq.heappop(maxHeap) ## 则先暂存
                times,val = heapq.heappop(maxHeap)
                ans.append(val)
                times += 1 # 注意原先的频次是负数
                if times != 0:
                    heapq.heappush(maxHeap,(times,val))
                heapq.heappush(maxHeap,store) # 暂存的丢回去
        return ans           
```

# 1058. 最小化舍入误差以满足目标

给定一系列价格 [p1,p2...,pn] 和一个目标 target，将每个价格 pi 舍入为 Roundi(pi) 以使得舍入数组 [Round1(p1),Round2(p2)...,Roundn(pn)] 之和达到给定的目标值 target。每次舍入操作 Roundi(pi) 可以是向下舍 Floor(pi) 也可以是向上入 Ceil(pi)。

如果舍入数组之和无论如何都无法达到目标值 target，就返回 -1。否则，以保留到小数点后三位的字符串格式返回最小的舍入误差，其定义为 Σ |Roundi(pi) - (pi)|（ i 从 1 到 n ）。

```python
class Solution:
    def minimizeError(self, prices: List[str], target: int) -> str:
        # 先判断是否可达
        # 判断target是否在down,up

        down = sum(math.floor(float(i)) for i in prices)
        up = sum(math.ceil(float(i)) for i in prices)
        
        if target < down or target > up:
            return "-1"
        
        # 然后统计cost
        # 先假设全取小
        cost = 0
        # 看需要取多少个大的
        bigger = target - down 
        smaller = len(prices)-bigger
        # print("smaller = ",smaller,"bigger",bigger,"")
        prices.sort(key = lambda x: round(float(x)-math.floor(float(x)),3))
        # print(prices)
        # 根据小数部分排序，
        # 取小的开销为round(float(i)-math.floor(float(i)),3）
        # 取大的开销为round(math.ceil(float(i)-float(i)),3）
        ans = 0
        # 前smaller个取开销方法1，其他的取开销方法2
        ans += sum(round(float(i)-math.floor(float(i)),3)for i in prices[:smaller])
        ans += sum(round(math.ceil(float(i))-float(i),3)for i in prices[smaller:])
        
        ans = str(round(float(ans),3))
        ans = ans.split(".")
        extra = 3-len(ans[1])
        return ans[0]+"."+ans[1]+extra*"0"


```

# 1078. Bigram 分词

给出第一个词 first 和第二个词 second，考虑在某些文本 text 中可能以 "first second third" 形式出现的情况，其中 second 紧随 first 出现，third 紧随 second 出现。

对于每种这样的情况，将第三个词 "third" 添加到答案中，并返回答案。

```python
class Solution:
    def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        # 尬算
        lst = text.split(" ")
        p = 0
        ans = []
        while p < len(lst):
            if p-1>=0 and p-2>= 0:
                if lst[p-2] == first and lst[p-1] == second:
                    ans.append(lst[p])
            p += 1
        return ans
```



# 1086. 前五科的均分

给你一个不同学生的分数列表 items，其中 items[i] = [IDi, scorei] 表示 IDi 的学生的一科分数，你需要计算每个学生 最高的五科 成绩的 平均分。

返回答案 result 以数对数组形式给出，其中 result[j] = [IDj, topFiveAveragej] 表示 IDj 的学生和他 最高的五科 成绩的 平均分。result 需要按 IDj  递增的 顺序排列 。

学生 最高的五科 成绩的 平均分 的计算方法是将最高的五科分数相加，然后用 整数除法 除以 5 。

```python
class Solution:
    def highFive(self, items: List[List[int]]) -> List[List[int]]:
        # 字典哈希
        theDict = collections.defaultdict(list)
        for _id,score in items:
            theDict[_id].append(score)
        
        ans = []
        for key in theDict:
            theDict[key].sort(reverse = True)
            ans.append([key,sum(theDict[key][:5])//5])
        
        ans.sort(key = lambda x:x[0])
        return ans

```

# 1087. 花括号展开

我们用一个特殊的字符串 S 来表示一份单词列表，之所以能展开成为一个列表，是因为这个字符串 S 中存在一个叫做「选项」的概念：

单词中的每个字母可能只有一个选项或存在多个备选项。如果只有一个选项，那么该字母按原样表示。

如果存在多个选项，就会以花括号包裹来表示这些选项（使它们与其他字母分隔开），例如 "{a,b,c}" 表示 ["a", "b", "c"]。

例子："{a,b,c}d{e,f}" 可以表示单词列表 ["ade", "adf", "bde", "bdf", "cde", "cdf"]。

请你按字典顺序，返回所有以这种方式形成的单词。

```python
class Solution:
    def expand(self, s: str) -> List[str]:
        # dfs
        # 先要分隔开所有组,在括号前后+ “#
        ans = ""
        for ch in s:
            if ch == "{":
                ans += "#{"
            elif ch == "}":
                ans += "}#"
            else:
                ans += ch
        lst = ans.split("#")
        temp = []
        for cp in lst:
            if len(cp) != 0:
                if cp[0] == "{":
                    length = len(cp)
                    th = cp[1:length-1].split(",")
                    temp.append(th)
                else:
                    temp.append([cp]) # 注意添加为列表
        # 此时temp格式化完毕，dfs搜
        l = len(temp)
        ans = []
        path = []
        def dfs(path,index):
            if len(path) == l:
                ans.append("".join(path))
                return
            for ch in temp[index]:
                path.append(ch)
                dfs(path,index+1)
                path.pop()
        
        dfs(path,0)
        ans.sort()
        return ans
```

# 1091. 二进制矩阵中的最短路径

给你一个 n x n 的二进制矩阵 grid 中，返回矩阵中最短 畅通路径 的长度。如果不存在这样的路径，返回 -1 。

二进制矩阵中的 畅通路径 是一条从 左上角 单元格（即，(0, 0)）到 右下角 单元格（即，(n - 1, n - 1)）的路径，该路径同时满足下述要求：

路径途经的所有单元格都的值都是 0 。
路径中所有相邻的单元格应当在 8 个方向之一 上连通（即，相邻两单元之间彼此不同且共享一条边或者一个角）。
畅通路径的长度 是该路径途经的单元格总数。

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        # 最短的第一反应是用BFS
        # 起点非法或者终点非法直接返回False
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1

        visited = [[False for j in range(n)] for i in range(n)]
        visited[0][0] = True
        direc = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] # 方向数组
        queue = [(0,0)]
        steps = 1
        while len(queue) != 0:
            new_queue = []
            for i,j in queue:
                if (i,j) == (n-1,n-1):
                    return steps
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<n and 0<=new_j<n and visited[new_i][new_j] == False and grid[new_i][new_j] == 0: # 为0才加入路径
                        visited[new_i][new_j] = True 
                        new_queue.append((new_i,new_j))
            steps += 1
            queue = new_queue
        return -1
```



# 1101. 彼此熟识的最早时间

在一个社交圈子当中，有 N 个人。每个人都有一个从 0 到 N-1 唯一的 id 编号。

我们有一份日志列表 logs，其中每条记录都包含一个非负整数的时间戳，以及分属两个人的不同 id，logs[i] = [timestamp, id_A, id_B]。

每条日志标识出两个人成为好友的时间，友谊是相互的：如果 A 和 B 是好友，那么 B 和 A 也是好友。

如果 A 是 B 的好友，或者 A 是 B 的好友的好友，那么就可以认为 A 也与 B 熟识。

返回圈子里所有人之间都熟识的最早时间。如果找不到最早时间，就返回 -1 。

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
        return self.find(x) == self.find(y)

class Solution:
    def earliestAcq(self, logs: List[List[int]], n: int) -> int:
        # 找最小生成树
        # 生成了n-1条边的时候
        # Kruskal算法
        ufSet = UF(n) # n个人，顶点为n个
        edges = 0
        logs.sort(key = lambda x:x[0]) # 按照时间排序，
        for time,A,B in logs:
            if not ufSet.isConnect(A,B): # 没有在同一集合里，并一下
                edges += 1 # 并成功了，边+1
                ufSet.union(A,B)
            if edges == n-1: # 满足最小生成树，返回时间
                return time
        return -1 # 搜完了都没并起来
        
```

# 1102. 得分最高的路径

给你一个 R 行 C 列的整数矩阵 A。矩阵上的路径从 [0,0] 开始，在 [R-1,C-1] 结束。

路径沿四个基本方向（上、下、左、右）展开，从一个已访问单元格移动到任一相邻的未访问单元格。

路径的得分是该路径上的 最小 值。例如，路径 8 →  4 →  5 →  9 的值为 4 。

找出所有路径中得分 最高 的那条路径，返回其 得分。

```python
class Solution:
    def maximumMinimumPath(self, grid: List[List[int]]) -> int:
        # 二分+bfs
        left = 0
        right = 10**9+1
        
        m,n = len(grid),len(grid[0])
        direc = [(0,1),(0,-1),(1,0),(-1,0)]

        def bfs(start_i,start_j,limit):
            queue = [(0,0)]
            visited = [[False for j in range(n)] for i in range(m)] # 每轮重置
            if grid[0][0] >= limit:
                visited[0][0] = True
            if visited[0][0] == False:
                return False
            while len(queue) != 0:
                new_queue = []
                for i,j in queue:
                    if i == m-1 and j == n-1:
                        return True 
                    for di in direc:
                        new_i = i + di[0]
                        new_j = j + di[1]
                        if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and grid[new_i][new_j] >= limit:
                            visited[new_i][new_j] = True 
                            new_queue.append((new_i,new_j))
                queue = new_queue
            return False 


        while left <= right:
            mid = (left+right)//2
            state = bfs(0,0,mid)
            if state: # 说明可以到达，尝试搜更大的
                left = mid + 1
            elif not state:
                right = mid - 1
        
        return right

```



# 1109. 航班预订统计

这里有 n 个航班，它们分别从 1 到 n 进行编号。

有一份航班预订表 bookings ，表中第 i 条预订记录 bookings[i] = [firsti, lasti, seatsi] 意味着在从 firsti 到 lasti （包含 firsti 和 lasti ）的 每个航班 上预订了 seatsi 个座位。

请你返回一个长度为 n 的数组 answer，其中 answer[i] 是航班 i 上预订的座位总数。

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        # 看作上下车问题
        lst = [0 for i in range(n)]
        # a时刻上车，b时刻之后下车 [a,b,c]
        up = [[a[0]-1,a[2]] for a in bookings] # 上车，正数
        down = [[a[1],-a[2]] for a in bookings] # 下车，负数

        merge = up + down # 融合作为集合，或者是合并之后排序用指针扫
        # merge.sort()
        timedict = collections.defaultdict(int)
        for pair in merge:
            timedict[pair[0]] += pair[1]

        accum = 0 # 累计人数
        for i in range(n):
            accum += timedict[i]
            lst[i] = accum
        return lst 
```

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        # 差分数组方法
        # 这一题根据 [[1,2,10]] n = 5 来构建的时候 标准答案为[10,10,0,0,0]
        # diff[] = [[10,0,-10,0,0]]
        # 前缀和为 [10,10,0,0,0]
        # 其中diff的构建为 对于[left,right,increase] ,注意索引和数值的关系，
        # diff[left-1] += increase 
        # diff[right] -= increase [注意防止越界]
        diff = [0 for i in range(n)]
        for left,right,increase in bookings:
            diff[left-1] += increase
            if right < n:
                diff[right] -= increase
        # 然后根据差分数组构建前缀和
        prefix = [0 for i in range(n)]
        tempSum = 0
        for i in range(len(diff)):
            tempSum += diff[i]
            prefix[i] = tempSum
        return prefix1

```

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        nums = [0] * n
        for left, right, inc in bookings:
            nums[left - 1] += inc
            if right < n:
                nums[right] -= inc
    
        for i in range(1, n): # 直接在差分数组上从左到右求和
            nums[i] += nums[i - 1]
        
        return nums
```

# 1110. 删点成林

给出二叉树的根节点 root，树上每个节点都有一个不同的值。

如果节点值在 to_delete 中出现，我们就把该节点从树上删去，最后得到一个森林（一些不相交的树构成的集合）。

返回森林中的每棵树。你可以按任意顺序组织答案。

```python
class Solution:
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        deleteSet = set(to_delete)

        ans = []
        if root == None:
            return []
            
        if root.val not in deleteSet:
            ans.append(root)

        # 使用后续遍历，带父亲信息
        def postOrder(node,parent=None):
            if node == None:
                return 
            leftNode = postOrder(node.left,node)
            rightNode = postOrder(node.right,node)
            # 先处理完孩子才能处理自己，所以必须是后续遍历
            if node.val in deleteSet:
                if parent != None and parent.left == node:
                    parent.left = None
                if parent != None and parent.right == node:
                    parent.right = None
                if node.left != None:
                    ans.append(node.left)
                if node.right != None:
                    ans.append(node.right)
        
        postOrder(root)
        return ans
```

# 1118. 一月有多少天

指定年份 `Y` 和月份 `M`，请你帮忙计算出该月一共有多少天。

```python
class Solution:
    def numberOfDays(self, year: int, month: int) -> int:
        # 判断闰年用
        mDict = {1:31,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
        if month != 2:
            return mDict[month]
        
        if month == 2:
            if year % 400 == 0:
                return 29
            elif year % 4 == 0 and year % 100 != 0:
                return 29
            else:
                return 28
```

# 1120. 子树的最大平均值

给你一棵二叉树的根节点 root，找出这棵树的 每一棵 子树的 平均值 中的 最大 值。

子树是树中的任意节点和它的所有后代构成的集合。

树的平均值是树中节点值的总和除以节点数。

```python
class Solution:
    def maximumAverageSubtree(self, root: TreeNode) -> float:
        # 后续遍历，先收集，然后求平均，传入参数需要包含数目
        avgList = []
        def postOrder(node,value,size):
            if node == None:
                return [0,0]
            cpSum = value
            leftVal,leftSize = postOrder(node.left,value,size)
            rightVal,rightSize = postOrder(node.right,value,size)
            cpSum = leftVal + rightVal + cpSum + node.val
            size = size + leftSize + rightSize + 1
            avgList.append(cpSum/size)
            return cpSum,size # 返回值要返回size
        postOrder(root,0,0) # 调用

        return max(avgList)
```

# 1136. 平行课程

已知有 N 门课程，它们以 1 到 N 进行编号。

给你一份课程关系表 relations[i] = [X, Y]，用以表示课程 X 和课程 Y 之间的先修关系：课程 X 必须在课程 Y 之前修完。

假设在一个学期里，你可以学习任何数量的课程，但前提是你已经学习了将要学习的这些课程的所有先修课程。

请你返回学完全部课程所需的最少学期数。

如果没有办法做到学完全部这些课程的话，就返回 -1。

```python
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        # 拓扑排序，计量入度和出度,都预先-1
        inDegree = [0 for i in range(n)]
        edges = collections.defaultdict(list)
        for a,b in relations:
            inDegree[b-1] += 1
            edges[a-1].append(b-1)
        # 找到所有入度为0的作为初始节点
        
        queue = []
        for i in range(n):
            if inDegree[i] == 0:
                queue.append(i)
        times = 0
        while len(queue) != 0:
            new_queue = []
            for e in queue: # 对应邻居的入度减1
                for neigh in edges[e]:
                    inDegree[neigh] -= 1
                    if inDegree[neigh] == 0: # 代替了visited数组的效果
                        new_queue.append(neigh)
            queue = new_queue 
            times += 1
        
        for i in range(n):
            if inDegree[i] != 0:
                return -1
        return times
```

# 1138. 字母板上的路径

我们从一块字母板上的位置 (0, 0) 出发，该坐标对应的字符为 board[0][0]。

在本题里，字母板为board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]，如下所示。

我们可以按下面的指令规则行动：

如果方格存在，'U' 意味着将我们的位置上移一行；
如果方格存在，'D' 意味着将我们的位置下移一行；
如果方格存在，'L' 意味着将我们的位置左移一列；
如果方格存在，'R' 意味着将我们的位置右移一列；
'!' 会把在我们当前位置 (r, c) 的字符 board[r][c] 添加到答案中。
（注意，字母板上只存在有字母的位置。）

返回指令序列，用最小的行动次数让答案和目标 target 相同。你可以返回任何达成目标的路径。

```python
class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        # 得到字符之间的最小曼哈顿距离，然后转换成合法的序列，注意转移方向。 注意到z的处理。
        target = "a" + target
        def getManhatum(s1,s2): # 注意需要特殊处理含有z的时候 z是[5][0]
            ind1 = ord(s1) - ord("a")
            # p 是 （行，列）
            p1 = (ind1//5,ind1%5)
            ind2 = ord(s2) - ord("a")
            p2 = (ind2//5,ind2%5)
            inst = ""
            gapX = p2[0]-p1[0]
            gapY = p2[1]-p1[1]
            if p1 != (5,0) and p2 != (5,0):                
                if gapY >= 0:
                    inst += gapY*"R"
                elif gapY < 0:
                    inst += abs(gapY)*"L"
                if gapX >= 0:
                    inst += gapX*"D"
                elif gapX < 0:
                    inst += abs(gapX)*"U"
            elif p1 == (5,0): # 优先往上，只会是上右
                if gapX <= 0:
                    inst += abs(gapX)*"U"
                if gapY >= 0:
                    inst += gapY*"R"

            elif p2 == (5,0): # 优先往左，只会是下左
                if gapY <= 0:
                    inst += abs(gapY)*"L"
                if gapX >= 0:
                    inst += gapX*"D"

            inst += "!"
            return inst
        
        ans = ""
        p = 1
        while p < len(target):
            s1 = target[p-1]
            s2 = target[p]
            ans += getManhatum(s1,s2)
            p += 1
        return ans
```

# 1155. 掷骰子的N种方法

这里有 d 个一样的骰子，每个骰子上都有 f 个面，分别标号为 1, 2, ..., f。

我们约定：掷骰子的得到总点数为各骰子面朝上的数字的总和。

如果需要掷出的总点数为 target，请你计算出有多少种不同的组合情况（所有的组合情况总共有 f^d 种），模 10^9 + 7 后返回。

```python
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # 动态规划，d是轮数，f是骰子点数
        # 越界返回0
        lst = [0 for i in range(f*d - (d-1))] # 初始化为全0
        for i in range(f):
            lst[i] = 1 # 初始化基态，f个1
        # 设当前状态为[...n...],下一个状态为叠加。为当前值往右偏移叠加
        for times in range(d-1): # 叠加轮数
            base = lst.copy()
            for offset in range(1,f):
                for i in range(len(lst)):
                    if offset + i < len(lst):
                        lst[offset+i] += base[i]
        # 最终lst[0]代表数值和为d,那么target对应的索引为[target-d]
        # 如果越界，返回0
        if target - d < 0 or target - d >= len(lst):
            return 0
        else:
            return lst[target-d] % (10**9 + 7)

```

# 1160. 拼写单词

给你一份『词汇表』（字符串数组） words 和一张『字母表』（字符串） chars。

假如你可以用 chars 中的『字母』（字符）拼写出 words 中的某个『单词』（字符串），那么我们就认为你掌握了这个单词。

注意：每次拼写（指拼写词汇表中的一个单词）时，chars 中的每个字母都只能用一次。

返回词汇表 words 中你掌握的所有单词的 长度之和。

```python
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
      # 模拟
        charsDict = [0 for i in range(26)]
        for ch in chars:
            index = ord(ch)-ord("a")
            charsDict[index] += 1
        def check(s,lst): # 检查
            cp = lst.copy()
            for ch in s:
                index = ord(ch) - ord("a")
                cp[index] -= 1
                if cp[index] < 0:
                    return False
            return True
        ans = 0
        for word in words:
            if check(word,charsDict):
                ans += len(word) # 注加的是长度
        return ans
```

# 1162. 地图分析

你现在手里有一份大小为 N x N 的 网格 grid，上面的每个 单元格 都用 0 和 1 标记好了。其中 0 代表海洋，1 代表陆地，请你找出一个海洋单元格，这个海洋单元格到离它最近的陆地单元格的距离是最大的。

我们这里说的距离是「曼哈顿距离」（ Manhattan Distance）：(x0, y0) 和 (x1, y1) 这两个单元格之间的距离是 |x0 - x1| + |y0 - y1| 。

如果网格上只有陆地或者海洋，请返回 -1。

```python
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        # 超级原点BFS
        # 以每个1为起点，搜0的BFS。更新型搜索，一次BFS
        n = len(grid)
        allSum = 0
        queue = []
        visited = [[False for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                allSum += grid[i][j]
                if grid[i][j] == 1:
                    queue.append((i,j))
                    visited[i][j] = True 

        if allSum == 0 or allSum == n**2:
            return -1
        
        steps = 0
        ans = 0
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        while len(queue) != 0:
            new_queue = []
            for pair in queue:
                x,y = pair
                if grid[x][y] == 0 and visited[x][y] == False:
                    visited[x][y] = True
                    ans = max(ans,steps)
                for di in direc:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if 0<=new_x<n and 0<=new_y<n and visited[new_x][new_y] == False:
                        new_queue.append((new_x,new_y))
            queue = new_queue
            steps += 1
        return ans
```

# 1167. 连接棒材的最低费用

为了装修新房，你需要加工一些长度为正整数的棒材 。棒材以数组 sticks 的形式给出，其中 sticks[i] 是第 i 根棒材的长度。

如果要将长度分别为 x 和 y 的两根棒材连接在一起，你需要支付 x + y 的费用。 由于施工需要，你必须将所有棒材连接成一根。

返回你把所有棒材 sticks 连成一根所需要的最低费用。注意你可以任意选择棒材连接的顺序。

```python
class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        # 贪心，借助堆，小顶堆
        # 直接排序的思路不行。因为这个时候贪心的时候不能保证新棍子还是最小的，除非满足斐波那契以上
        cost = 0
        minHeap = [i for i in sticks]
        heapq.heapify(minHeap)
        while len(minHeap) != 1:
            e1 = heapq.heappop(minHeap)
            e2 = heapq.heappop(minHeap)
            cost += e1 
            cost += e2 
            heapq.heappush(minHeap,e1+e2)
        return cost
```

# 1176. 健身计划评估

你的好友是一位健身爱好者。前段日子，他给自己制定了一份健身计划。现在想请你帮他评估一下这份计划是否合理。

他会有一份计划消耗的卡路里表，其中 calories[i] 给出了你的这位好友在第 i 天需要消耗的卡路里总量。

为了更好地评估这份计划，对于卡路里表中的每一天，你都需要计算他 「这一天以及之后的连续几天」 （共 k 天）内消耗的总卡路里 T：

如果 T < lower，那么这份计划相对糟糕，并失去 1 分； 
如果 T > upper，那么这份计划相对优秀，并获得 1 分；
否则，这份计划普普通通，分值不做变动。
请返回统计完所有 calories.length 天后得到的总分作为评估结果。

注意：总分可能是负数。

```python
class Solution:
    def dietPlanPerformance(self, calories: List[int], k: int, lower: int, upper: int) -> int:
        # 固定窗口大小
        # 这一题题目描述有问题，不足k的应该不算，算了反而错误
        ans = [] # 每日打分
        window = sum(calories[:k])

        def judge(win,ans):
            if window > upper: ans.append(1)
            elif window < lower: ans.append(-1)
            else: ans.append(0)

        judge(window,ans) # 初始化传入
        left = 0
        right = k
        n = len(calories)
        while right < n:
            add = calories[right]
            delete = calories[left]
            window = window + add - delete
            judge(window,ans)
            left += 1
            right += 1

        return sum(ans)

```

# 1181. 前后拼接

给你一个「短语」列表 phrases，请你帮忙按规则生成拼接后的「新短语」列表。

「短语」（phrase）是仅由小写英文字母和空格组成的字符串。「短语」的开头和结尾都不会出现空格，「短语」中的空格不会连续出现。

「前后拼接」（Before and After puzzles）是合并两个「短语」形成「新短语」的方法。我们规定拼接时，第一个短语的最后一个单词 和 第二个短语的第一个单词 必须相同。

返回每两个「短语」 phrases[i] 和 phrases[j]（i != j）进行「前后拼接」得到的「新短语」。

注意，两个「短语」拼接时的顺序也很重要，我们需要同时考虑这两个「短语」。另外，同一个「短语」可以多次参与拼接，但「新短语」不能再参与拼接。

请你按字典序排列并返回「新短语」列表，列表中的字符串应该是 不重复的 。

```python
class Solution:
    def beforeAndAfterPuzzles(self, phrases: List[str]) -> List[str]:
        # 字典映射法
        theDict = dict() # k-v为单词，索引
        mirror = dict()
        p = 0
        for l in phrases:
            temp = l.split()
            for word in temp:
                if word not in theDict:
                    theDict[word] = p
                    mirror[p] = word
                    p += 1

        # 转换成数字，考虑两两是否可能连接
        for i in range(len(phrases)):
            temp = phrases[i].split(" ")
            for j in range(len(temp)):
                temp[j] = theDict[temp[j]]
            phrases[i] = temp 
        
        # 连接
        n = len(phrases)
        tempSet = set()
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if phrases[i][-1] == phrases[j][0]:
                    tp = tuple(phrases[i][:-1] + phrases[j])
                    tempSet.add(tp)
        # 收集答案
        ans = []
        for tp in tempSet:
            t = ""
            for n in tp:
                t = t + mirror[n] + " "
            ans.append(t[:-1])
        ans.sort()
        return ans
```

# 1197. 进击的骑士

一个坐标可以从 -infinity 延伸到 +infinity 的 无限大的 棋盘上，你的 骑士 驻扎在坐标为 [0, 0] 的方格里。

骑士的走法和中国象棋中的马相似，走 “日” 字：即先向左（或右）走 1 格，再向上（或下）走 2 格；或先向左（或右）走 2 格，再向上（或下）走 1 格。

每次移动，他都可以按图示八个方向之一前进。

现在，骑士需要前去征服坐标为 [x, y] 的部落，请你为他规划路线。

最后返回所需的最小移动次数即可。本题确保答案是一定存在的。

```python
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        # bfs,调用bfs看谁能最先到0，0
        # 还有一种数学解
        if x == y == 0:
            return 0

        update = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
        queue = deque()
        queue.append([x,y])
        steps = 0
        visitedSet = set() # 不走重复路

        def calc_distance(a,b): # 要保证移动后不会离终点更远
            return abs(a)+abs(b)

        while queue[0] != [0,0]:
            steps += 1
            new_queue = deque()
            while len(queue) != 0:
                x,y = queue.popleft()
                if x == y == 0:
                    return steps-1
                for di in update:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if calc_distance(new_x,new_y)-2 < calc_distance(x,y) and (new_x,new_y) not in visitedSet: # 没有重复走过，才计算,并且需要稍微近一点才加入队列
                        new_queue.append([new_x,new_y])
                        visitedSet.add((new_x,new_y))
            queue = new_queue
```

# 1202. 交换字符串中的元素

给你一个字符串 s，以及该字符串中的一些「索引对」数组 pairs，其中 pairs[i] = [a, b] 表示字符串中的两个索引（编号从 0 开始）。

你可以 任意多次交换 在 pairs 中任意一对索引处的字符。

返回在经过若干次交换后，s 可以变成的按字典序最小的字符串。

```python
class UF: # 这一题不优化并查集卡常。。
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


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        # 并查集，串上的集排序
        n = len(s)
        ufSet = UF(n)
        for x,y in pairs:            
            ufSet.union(x,y)

        theQueue = collections.defaultdict(list)
        theFind = [i for i in range(n)] # 判断每个点在哪个集里
        for i in range(n):
            e = ufSet.find(i)
            theQueue[e].append((i,s[i]))
            theFind[i] = e

        for key in theQueue:
            theQueue[key].sort(key = lambda x:x[1],reverse = True)
        
        ans = "" # 收集结果
        for target in theFind:
            ans += theQueue[target].pop()[1]
        return ans
```



# 1218. 最长定差子序列

给你一个整数数组 arr 和一个整数 difference，请你找出并返回 arr 中最长等差子序列的长度，该子序列中相邻元素之间的差等于 difference 。

子序列 是指在不改变其余元素顺序的情况下，通过删除一些元素或不删除任何元素而从 arr 派生出来的序列。

```python
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        # 如果diff为0，则直接计数        
        diff = difference
        if diff == 0:
            ct = collections.Counter(arr)
            maxVal = 0
            for key in ct:
                if ct[key] > maxVal:
                    maxVal = ct[key] # 统计长度
            return maxVal 
        # 不能改变数组元素相对顺序，使用dp
        else:
            n = len(arr)
            # dp[i]的含义是到i为止最长等差
            # dp[i]看nums[i]-diff是否在之前出现过，如果没有出现过，则还是1，如果出现过。为它对应长度+1
            hashSet = dict() # 遇到相同元素时
            dp = [1 for i in range(n)]
            hashSet[arr[0]] = 0 # key-v 为数:索引
            for i in range(1,n):
                if arr[i]-diff in hashSet:
                    dp[i] = dp[hashSet[arr[i]-diff]] + 1
                else:
                    pass 
                # 只需要记录元素第一次出现的位置
                hashSet[arr[i]] = i 
            # print(dp)
            return max(dp) # 返回数组中的最大值
```

# 1219. 黄金矿工

你要开发一座金矿，地质勘测学家已经探明了这座金矿中的资源分布，并用大小为 m * n 的网格 grid 进行了标注。每个单元格中的整数就表示这一单元格中的黄金数量；如果该单元格是空的，那么就是 0。

为了使收益最大化，矿工需要按以下规则来开采黄金：

每当矿工进入一个单元，就会收集该单元格中的所有黄金。
矿工每次可以从当前位置向上下左右四个方向走。
每个单元格只能被开采（进入）一次。
不得开采（进入）黄金数目为 0 的单元格。
矿工可以从网格中 任意一个 有黄金的单元格出发或者是停止。

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        # 多次调用dfs
        m = len(grid)
        n = len(grid[0])
        get = 0
        final = 0

        ans = 0

        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        def dfs(i,j,vi):
            nonlocal get 
            nonlocal final
            if not (0<=i<m and 0<=j<n and vi[i][j] != 0):
                return 
            memo = vi[i][j]
            get += vi[i][j]
            final = max(get,final)
            vi[i][j] = 0 # 清空
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(new_i,new_j,vi)
             # 注意这个回溯
            vi[i][j] = memo
            get -= vi[i][j]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] != 0: # 每次重置
                    visited = grid.copy()
                    final = 0
                    get = 0
                    dfs(i,j,visited) # 调用
                    ans = max(ans,final)
        
        return ans
            
```

# 1243. 数组变换

首先，给你一个初始数组 arr。然后，每天你都要根据前一天的数组生成一个新的数组。

第 i 天所生成的数组，是由你对第 i-1 天的数组进行如下操作所得的：

假如一个元素小于它的左右邻居，那么该元素自增 1。
假如一个元素大于它的左右邻居，那么该元素自减 1。
首、尾元素 永不 改变。
过些时日，你会发现数组将会不再发生变化，请返回最终所得到的数组。

```python
class Solution:
    def transformArray(self, arr: List[int]) -> List[int]:
        # 纯模拟
        if len(arr) <= 2:
            return arr 
        while True:
            pre = arr.copy()
            for p in range(1,len(arr)-1):
                if pre[p-1] > pre[p] and pre[p+1] > pre[p]:
                    arr[p] += 1  #注意这里是arr
                elif pre[p-1] < pre[p] and pre[p+1] < pre[p]:
                    arr[p] -= 1  #注意这里是arr
            if arr == pre:
                break
        return arr
        
```

# 1245. 树的直径

给你这棵「无向树」，请你测算并返回它的「直径」：这棵树上最长简单路径的 边数。

我们用一个由所有「边」组成的数组 edges 来表示一棵无向树，其中 edges[i] = [u, v] 表示节点 u 和 v 之间的双向边。

树上的节点都已经用 {0, 1, ..., edges.length} 中的数做了标记，每个节点上的标记都是独一无二的。

```python
class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        # 注意，形成的是无向树，利用结论
        # 两次BFS，第一次任选一个点，最后到达的点作为下一次的起点
        # 然后第二次BFS获取路径长度
        # 先转换成邻接链表的结构
        if len(edges) == 0:
            return 0
        graph = collections.defaultdict(list)
        for a,b in edges:
            graph[a].append(b)
            graph[b].append(a)
        # 
        start = edges[0][0]
        queue = [start]
        last = None
        visited = set()
        visited.add(start)
        while len(queue) != 0:
            new_queue = []
            for i in queue:
                for nei in graph[i]:
                    if nei not in visited:
                        visited.add(nei)
                        new_queue.append(nei)
            if new_queue == []:
                last = queue[-1]
            queue = new_queue
        # 然后此时以last作为起点
        steps = 0
        queue = [last]
        visited = set()
        visited.add(last)
        while len(queue) != 0:
            new_queue = []
            for i in queue:
                for nei in graph[i]:
                    if nei not in visited:
                        visited.add(nei)
                        new_queue.append(nei)
            if new_queue == []:
                return steps
            steps += 1
            queue = new_queue
```

# 1249. 移除无效的括号

给你一个由 '('、')' 和小写字母组成的字符串 s。

你需要从字符串中删除最少数目的 '(' 或者 ')' （可以删除任意位置的括号)，使得剩下的「括号字符串」有效。

请返回任意一个合法字符串。

有效「括号字符串」应当符合以下 任意一条 要求：

空字符串或只包含小写字母的字符串
可以被写作 AB（A 连接 B）的字符串，其中 A 和 B 都是有效「括号字符串」
可以被写作 (A) 的字符串，其中 A 是一个有效的「括号字符串」

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        # 记录左右括号层级
        # 每次添加右括号时候，检查左括号栈是否有元素
        leftStack = 0 # 数字记录即可
        ans = []
        pair = 0 # 看有多少对括号
        for i in s:
            if i == "(":
                leftStack += 1
                ans.append(i)
            elif i == ")":
                if leftStack > 0:
                    leftStack -= 1
                    ans.append(i)
                    pair += 1
            else:
                ans.append(i)
        final = ""
        for i in ans: # 只加入允许的成对括号
            if i == "(" and pair > 0:
                pair -= 1
                final += i
            elif i == ")":
                final += i
            elif i.isalpha() == True:
                final += i
        return final
```

# 1253. 重构 2 行二进制矩阵

给你一个 2 行 n 列的二进制数组：

矩阵是一个二进制矩阵，这意味着矩阵中的每个元素不是 0 就是 1。
第 0 行的元素之和为 upper。
第 1 行的元素之和为 lower。
第 i 列（从 0 开始编号）的元素之和为 colsum[i]，colsum 是一个长度为 n 的整数数组。
你需要利用 upper，lower 和 colsum 来重构这个矩阵，并以二维整数数组的形式返回它。

如果有多个不同的答案，那么任意一个都可以通过本题。

如果不存在符合要求的答案，就请返回一个空的二维数组。

```python
class Solution:
    def reconstructMatrix(self, upper: int, lower: int, colsum: List[int]) -> List[List[int]]:
        n = len(colsum)
        firstLine = [0 for i in range(n)]
        secondLine = [0 for i in range(n)]
        # 已经限定了，那么直接枚举
        for i in range(n):
            if upper >= 0 and lower >= 0:
                if colsum[i] == 2:
                    firstLine[i] = 1
                    secondLine[i] = 1
                    upper -= 1
                    lower -= 1
                elif colsum[i] == 1: # 只有一个为1,优先消耗多的那个
                    if upper >= lower:
                        firstLine[i] = 1
                        upper -= 1
                    elif upper < lower:
                        secondLine[i] = 1
                        lower -= 1
            else:
                return []
        if upper == lower == 0:
            # lst = [(firstLine[i]+secondLine[i]) for i in range(n)]
            # print(lst) # 检查答案用
            return [firstLine,secondLine]
        else:
            return []
```



# 1254. 统计封闭岛屿的数目

有一个二维矩阵 grid ，每个位置要么是陆地（记号为 0 ）要么是水域（记号为 1 ）。

我们从一块陆地出发，每次可以往上下左右 4 个方向相邻区域走，能走到的所有陆地区域，我们将其称为一座「岛屿」。

如果一座岛屿 完全 由水域包围，即陆地边缘上下左右所有相邻区域都是水域，那么我们将其称为 「封闭岛屿」。

请返回封闭岛屿的数目。

```python
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        # 即dfs的时候不能碰到边界
        m = len(grid)
        n = len(grid[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(1,0),(-1,0),(0,1),(0,-1)]

        def judgeValid(i,j): # 判断这个格子是否合法
            if 0<=i<m and 0<=j<n and grid[i][j] == 0:
                return True
            else:
                return False
        
        validArea = 0 # 非大框边界面积
        realArea = 0 # 真实面积

        def dfs(i,j):
            nonlocal validArea
            nonlocal realArea
            nonlocal visited
            if not judgeValid(i,j):
                return 
            if visited[i][j] == True:
                return 
            visited[i][j] = True 
            if i != 0 and i != m-1 and j != 0 and j != n-1: # 如果不在大框边界上
                validArea += 1 # 合法面积+1
            realArea += 1
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(new_i,new_j)
        
        count = 0 # 计算合法岛屿数目
        for i in range(m):
            for j in range(n):
                dfs(i,j)
                if validArea == realArea and validArea != 0: # 搜完之后面积不为0且合法面积等于真实面积
                    count += 1    
                else: # 逻辑清晰占位
                    pass 
                # 搜完都需要重置           
                validArea = 0 # 重置
                realArea = 0 # 重置
        
        return count
```

# 1256. 加密数字

给你一个非负整数 `num` ，返回它的「加密字符串」。

加密的过程是把一个整数用某个未知函数进行转化，你需要从下表推测出该转化函数：

| 原始 | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 变化 | “”   | 0    | 1    | 00   | 01   | 10   | 11   | 000  |

```python
class Solution:
    def encode(self, num: int) -> str:
        # 规律为偏移的二进制
        # 3～6 共四个数对应00 01 10 11
        # 7～14 共八个数对应 000 001 010 。。。
        # 2^n - 1 ~ 2^(n+1) - 2 ;n为字符串长度
        # num+1 取对数
        # 然后减去基数变成二进制，不足位用0补齐
        # 
        if num == 0:
            return ""
        def getLength(num):
            n = int(math.log2(num+1))
            return n
        n = getLength(num)
        base = 2**n - 1 # 基数
        offset = num - base
        offset = bin(offset)[2:]
        diff = n - len(offset)
        ans = diff*"0" + offset
        return ans
```

```python
class Solution:
    def encode(self, num: int) -> str:
        return bin(num + 1)[3: ]

作者：Hanxin_Hanxin
链接：https://leetcode-cn.com/problems/encode-number/solution/cpython3-shu-xue-ji-suan-wei-yun-suan-by-fks4/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

# 1267. 统计参与通信的服务器

这里有一幅服务器分布图，服务器的位置标识在 m * n 的整数矩阵网格 grid 中，1 表示单元格上有服务器，0 表示没有。

如果两台服务器位于同一行或者同一列，我们就认为它们之间可以进行通信。

请你统计并返回能够与至少一台其他服务器进行通信的服务器的数量。

```python
class Solution:
    def countServers(self, grid: List[List[int]]) -> int:
        # 行列计数，预处理
        m = len(grid)
        n = len(grid[0])
        visitedRow = set() # 横行
        visitedCol = set() # 纵行
        rowNum = [0 for i in range(m)]
        colNum = [0 for j in range(n)]
        theOne = set()
        counts = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    counts += 1
                    rowNum[i] += 1
                    colNum[j] += 1
                    theOne.add((i,j))
        # 处理后找rowNum和colNum为1的
        for i in range(m):
            if rowNum[i] == 1:
                visitedRow.add(i)
        for j in range(n):
            if colNum[j] == 1:
                visitedCol.add(j)
        for i,j in theOne: # 只在合法集合里面搜
            if i in visitedRow and j in visitedCol:
                counts -= 1
        return counts
```

# 1271. 十六进制魔术数字

你有一个十进制数字，请按照此规则将它变成「十六进制魔术数字」：首先将它变成字母大写的十六进制字符串，然后将所有的数字 0 变成字母 O ，将数字 1  变成字母 I 。

如果一个数字在转换后只包含 {"A", "B", "C", "D", "E", "F", "I", "O"} ，那么我们就认为这个转换是有效的。

给你一个字符串 num ，它表示一个十进制数 N，如果它的十六进制魔术数字转换是有效的，请返回转换后的结果，否则返回 "ERROR" 。

```python
class Solution:
    def toHexspeak(self, num: str) -> str:
        # 先把数字转换成正常数字，然后转16进制
        n = int(num)
        ans = []
        while n != 0:
            e = n % (16)
            ans.append(e)
            n //= 16
        theDict = {0:"O",1:"I",10:"A",11:"B",12:"C",13:"D",14:"E",15:"F"}
        ans = ans[::-1] # 倒置
        final = ""
        for n in ans:
            if n not in theDict:
                return "ERROR"
            final += theDict[n]
        return final
```

# 1272. 删除区间

实数集合可以表示为若干不相交区间的并集，其中每个区间的形式为 [a, b)（左闭右开），表示满足 a <= x < b 的所有实数  x 的集合。如果某个区间 [a, b) 中包含实数 x ，则称实数 x 在集合中。

给你一个 有序的 不相交区间列表 intervals 和一个要删除的区间 toBeRemoved 。intervals 表示一个实数集合，其中每一项 intervals[i] = [ai, bi] 都表示一个区间 [ai, bi) 。

请你 intervals 中任意区间与 toBeRemoved 有交集的部分都删除。返回删除所有交集区间后， intervals 剩余部分的 有序 列表。换句话说，返回实数集合，并满足集合中的每个实数 x 都在 intervals 中，但不在 toBeRemoved 中。

```python
class Solution:
    def removeInterval(self, intervals: List[List[int]], toBeRemoved: List[int]) -> List[List[int]]:
        # 先排序intervals
        intervals.sort()
        # 然后对每个interval进行判断。
        ans = []

        def judge(a,b,c,d): # a,b为intervals,删除仅仅删除一个区间c,d
            # 分类讨论，
            if a <= b <= c <=d:
                return [a,b]
            if a <= c <= b <=d:
                return [a,c]
            if a <= c <= d <= b:
                return [a,c],[d,b]
            if c <= a<= b<= d:
                return []
            if c <= a <= d <= b:
                return [d,b]
            if c <= d <= a <= b:
                return [a,b]

        c,d = toBeRemoved
        for a,b in intervals:
            temp = judge(a,b,c,d)
            if temp == []:
                continue
            elif len(temp) == 2 and type(temp[0]) == list:
                if temp[0][0] != temp[0][1]:
                    ans.append(temp[0])
                if temp[1][0] != temp[1][1]:
                    ans.append(temp[1])
            else:
                if temp[0] != temp[1]:
                    ans.append(temp)

        return ans

```

# 1273. 删除树节点

给你一棵以节点 0 为根节点的树，定义如下：

节点的总数为 nodes 个；
第 i 个节点的值为 value[i] ；
第 i 个节点的父节点是 parent[i] 。
请你删除节点值之和为 0 的每一棵子树。

在完成所有删除之后，返回树中剩余节点的数目。

```python
class Solution:
    def deleteTreeNodes(self, nodes: int, parent: List[int], value: List[int]) -> int:
        # 需要计算完所有孩子节点之后才能考虑自身是否删除
        # 先转成n叉树图
        graph = collections.defaultdict(list)
        for i in range(nodes):
            graph[parent[i]].append(i)

        root = graph[-1][0]
        # 
        def postOrder(node):
            nonlocal nodes
            if node == None:
                return 
            val = value[node]
            counts = 1
            childrenVal = []
            for child in graph[node]:
                pVal,pcounts = postOrder(child)
                val += pVal
                counts += pcounts
            if val == 0:
                nodes -= counts # 减去它的所有孩子和自己  
                counts = 0 # 防止重复计算,被算过就清空 
            return val,counts
        
        postOrder(root)
        return nodes
```

# 1283. 使结果不超过阈值的最小除数

给你一个整数数组 nums 和一个正整数 threshold  ，你需要选择一个正整数作为除数，然后将数组里每个数都除以它，并对除法结果求和。

请你找出能够使上述结果小于等于阈值 threshold 的除数中 最小 的那个。

每个数除以除数后都向上取整，比方说 7/3 = 3 ， 10/2 = 5 。

题目保证一定有解。

```python
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        # 调用一个方法进行二分查找
        # 调用一个方法计算值
        def calc(lst,n):
            theSum = 0
            for i in lst:
                theSum += ceil(i/n)
            return theSum
        # 题目保证有解
        # 初始化范围就弄成最大范围

        left = 1
        right = max(nums)

        while left <= right:
            mid = (left+right)//2
            e = calc(nums,mid)
            # print("left = ",left,"right = ",right,'mid = ',mid,"e = ",e)
            if e == threshold: # 值和墙相等，尝试继续缩小，不影响left的值，
                right = mid - 1
            elif e > threshold: # 值大于墙，要增大除数。
                left = mid + 1
            elif e < threshold: # 值小于墙，尝试继续缩小
                right = mid - 1
        
        return left 
```

# 1286. 字母组合迭代器

请你设计一个迭代器类，包括以下内容：

一个构造函数，输入参数包括：一个 有序且字符唯一 的字符串 characters（该字符串只包含小写英文字母）和一个数字 combinationLength 。
函数 next() ，按 字典序 返回长度为 combinationLength 的下一个字母组合。
函数 hasNext() ，只有存在长度为 combinationLength 的下一个字母组合时，才返回 True；否则，返回 False。

```python
class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        # 非迭代器思想
        self.lst = []
        self.p = 0

        def backtracking(characters,combinationLength,path,index):
            if len(path) == combinationLength:
                self.lst.append("".join(path[:]))
                return             
            for i in range(index,len(characters)):
                path.append(characters[i])
                backtracking(characters,combinationLength,path,i+1)
                path.pop()
        
        backtracking(characters,combinationLength,[],0)
        
    def next(self) -> str:
        val = self.lst[self.p]
        self.p += 1
        return val

    def hasNext(self) -> bool:
        return self.p < len(self.lst)
```

```

```




# 1319. 连通网络的操作次数

用以太网线缆将 n 台计算机连接成一个网络，计算机的编号从 0 到 n-1。线缆用 connections 表示，其中 connections[i] = [a, b] 连接了计算机 a 和 b。

网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。

给你这个计算机网络的初始布线 connections，你可以拔开任意两台直连计算机之间的线缆，并用它连接一对未直连的计算机。请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回 -1 。 

```python
class UF:
    def __init__(self,size):
        self.root = [i for i in range(size)]
    
    def union(self,x,y):
        findX = self.find(x)
        findY = self.find(y)
        if findX != findY:
            self.root[findX] = findY
    
    def find(self,x):
        while x != self.root[x]:
            x = self.root[x]
        return x 
    
    def isConnect(self,x,y):
        return self.find(x) == self.find(y)

class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        # 连通n个需要n-1条边
        # 并查集搜
        if len(connections) < n - 1:
            return -1
        unionSet = UF(n)

        for cn in connections:
            unionSet.union(cn[0],cn[1]) # 链接完毕之后

        # 链接完毕之后找有几组
        theSet = set()
        for i in range(n): # 对每个节点寻根，看有多少个不同的根
            theSet.add(unionSet.find(i))
        # 有k组，则需要k-1个线将它们链接在一起
        return len(theSet) - 1
        
```



# 1324. 竖直打印单词

给你一个字符串 s。请你按照单词在 s 中的出现顺序将它们全部竖直返回。
单词应该以字符串列表的形式返回，必要时用空格补位，但输出尾部的空格需要删除（不允许尾随空格）。
每个单词只能放在一列上，每一列中也只能有一个单词。

```python
class Solution:
    def printVertically(self, s: str) -> List[str]:
        # 模拟
        # 先转换成矩阵
        temp = s.split(" ")
        mat = [a for a in temp]
        maxLength = 0
        for line in mat:
            maxLength = max(maxLength,len(line))
        # 长度不足的都补齐
        for i in range(len(mat)):
            diff = maxLength - len(mat[i])
            mat[i] += diff*" "
        # 然后行转列，列转行
        allString = ""
        for j in range(len(mat[0])): # 外循环遍历列
            for i in range(len(mat)): # 内循环遍历行
                allString += mat[i][j]
        width = len(mat)
        height = maxLength
        ans = []
        for i in range(0,len(allString),width):
            ans.append(allString[i:i+width])
        # 再对每一行进行空格检查
        for i in range(len(ans)):
            while ans[i][-1] == " ":
                ans[i] = ans[i][:-1]
        return ans
```

# 1351. 统计有序矩阵中的负数

给你一个 `m * n` 的矩阵 `grid`，矩阵中的元素无论是按行还是按列，都以非递增顺序排列。 

请你统计并返回 `grid` 中 **负数** 的数目。

```python
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        # 二分法找出每一行的第一个负数
        count = 0
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] < 0:
                    count += 1
        return count 
```

```python
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        # 双指针法
        m = len(grid)
        n = len(grid[0])

        r = 0
        p = n-1
        ans = 0

        while r < m:
            while p >= 0 and grid[r][p] < 0:
                p -= 1
            # print(n-p-1)
            ans += n - p - 1
            r += 1
            
        return ans
```

# 1373. 二叉搜索子树的最大键值和

给你一棵以 root 为根的 二叉树 ，请你返回 任意 二叉搜索子树的最大键值和。

二叉搜索树的定义如下：

任意节点的左子树中的键值都 小于 此节点的键值。
任意节点的右子树中的键值都 大于 此节点的键值。
任意节点的左子树和右子树都是二叉搜索树。

```python
class Solution:
    def maxSumBST(self, root: TreeNode) -> int:
        # 后续遍历收集信息，后续遍历需要携带的信息有边界和是否是BSTBST
        theAns = 0 # 初始化为0。。。
        
        def postOrder(node):
            nonlocal theAns
            if node == None:
                return 0,[0xffffffff,0xffffffff],True 
            leftTree = postOrder(node.left)
            rightTree = postOrder(node.right)
            if node.left == None and node.right == None:
                theAns = max(theAns,node.val)
                return node.val,[node.val,node.val],True 
            val = node.val 
            leftBound = 0xffffffff
            rightBound = 0xffffffff
            if node.left and node.right and leftTree[2] and rightTree[2] and node.val > leftTree[1][1] and node.val < rightTree[1][0]:
                leftBound = leftTree[1][0]
                rightBound = rightTree[1][1]
                theAns = max(theAns,leftTree[0]+rightTree[0]+node.val)
                return leftTree[0]+rightTree[0]+node.val,[leftBound,rightBound],True 
            elif node.left == None and leftTree[2] and rightTree[2] and node.val < rightTree[1][0]:
                leftBound = node.val 
                rightBound = rightTree[1][1]
                theAns = max(theAns,rightTree[0]+node.val)
                return rightTree[0]+node.val,[leftBound,rightBound],True 
            elif node.right == None and leftTree[2] and rightTree[2] and node.val > leftTree[1][1]:
                leftBound = leftTree[1][0]
                rightBound = node.val 
                theAns = max(theAns,leftTree[0]+node.val)
                return leftTree[0]+node.val,[leftBound,rightBound],True 
            else:
                return 0,[leftBound,rightBound],False
        
        postOrder(root)
        return theAns

```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxSumBST(self, root: TreeNode) -> int:
        min_,max_=float('-inf'),float('inf')
        def dfs(root):
            if not root:
                return max_,min_,0
            
            lmin,lmax,lsum=dfs(root.left)
            rmin,rmax,rsum=dfs(root.right)
            #能构成二叉搜索树
            if lmax<root.val<rmin:
                nonlocal res
                res=max(res,root.val+lsum+rsum)
                return min(lmin,root.val),max(rmax,root.val),root.val+lsum+rsum
            return min_,max_,0
        
        res=0
        dfs(root)
        return res

作者：yim-6
链接：https://leetcode-cn.com/problems/maximum-sum-bst-in-binary-tree/solution/python3-zi-di-xiang-shang-di-gui-by-yim-lwn0e/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

```python
class Solution:
    def maxSumBST(self, root: TreeNode) -> int:
        floor = -0xffffffff
        ceiling = 0xffffffff
        ans = 0

        def postOrder(node):
            nonlocal ans
            if node == None:
                return ceiling,floor,0 # 注意这一行
            
            leftTree = postOrder(node.left)
            rightTree = postOrder(node.right)

            if leftTree[1]<node.val<rightTree[0]:
                ans = max(ans,leftTree[2]+rightTree[2]+node.val)
                return min(leftTree[0],node.val),max(rightTree[1],node.val),leftTree[2]+rightTree[2]+node.val # 还需要注意这一行，不能直接用leftTree[0]
            return floor,ceiling,0 # 注意这一行

        postOrder(root)
        return ans
```

# 1376. 通知所有员工所需的时间

公司里有 n 名员工，每个员工的 ID 都是独一无二的，编号从 0 到 n - 1。公司的总负责人通过 headID 进行标识。

在 manager 数组中，每个员工都有一个直属负责人，其中 manager[i] 是第 i 名员工的直属负责人。对于总负责人，manager[headID] = -1。题目保证从属关系可以用树结构显示。

公司总负责人想要向公司所有员工通告一条紧急消息。他将会首先通知他的直属下属们，然后由这些下属通知他们的下属，直到所有的员工都得知这条紧急消息。

第 i 名员工需要 informTime[i] 分钟来通知它的所有直属下属（也就是说在 informTime[i] 分钟后，他的所有直属下属都可以开始传播这一消息）。

返回通知所有员工这一紧急消息所需要的 分钟数 。

```python
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        # DFS，最慢一定出现在叶子节点
        totalTime = 0
        edges = collections.defaultdict(list)
        for i in range(n): # 建立图，由于树的特性，无需使用visited
            edges[manager[i]].append(i)
        
        def dfs(theID,nowTime):
            nonlocal totalTime
            if edges[theID] == []: # 叶子
                totalTime = max(totalTime,nowTime)
                return 
            for sub in edges[theID]:
                dfs(sub,nowTime+informTime[theID])
        
        dfs(headID,0)
        return totalTime

```



# 1391. 检查网格中是否存在有效路径

给你一个 m x n 的网格 grid。网格里的每个单元都代表一条街道。grid[i][j] 的街道可以是：

1 表示连接左单元格和右单元格的街道。
2 表示连接上单元格和下单元格的街道。
3 表示连接左单元格和下单元格的街道。
4 表示连接右单元格和下单元格的街道。
5 表示连接左单元格和上单元格的街道。
6 表示连接右单元格和上单元格的街道。

你最开始从左上角的单元格 (0,0) 开始出发，网格中的「有效路径」是指从左上方的单元格 (0,0) 开始、一直到右下方的 (m-1,n-1) 结束的路径。该路径必须只沿着街道走。

注意：你 不能 变更街道。

如果网格中存在有效的路径，则返回 true，否则返回 false 。

```python
class Solution:
    def hasValidPath(self, grid: List[List[int]]) -> bool:
        # 限定方向的dfs,在这个格子上限定下一次走的方向
        direc = {
            1:[(0,1),(0,-1)],
            2:[(-1,0),(1,0)],
            3:[(0,-1),(1,0)],
            4:[(1,0),(0,1)],
            5:[(0,-1),(-1,0)],
            6:[(-1,0),(0,1)]
        }
        # 1，不能到2，2不能到1 在dfs里限定一下
        m = len(grid)
        n = len(grid[0])
        visited1 = [[False for j in range(n)] for i in range(m)]
        visited2 = [[False for j in range(n)] for i in range(m)]

        def dfs(i,j,visited,lasti,lastj): # 传入参数还要传入上一步
            nonlocal direc
            if not (0<=i<m and 0<=j<n):
                return 
            if visited[i][j]: return 
            if grid[i][j] == 1 and grid[lasti][lastj] == 2:
                return 
            if grid[i][j] == 2 and grid[lasti][lastj] == 1:
                return 
            visited[i][j] = True 
            di = direc[grid[i][j]]
            for every in di:
                new_i = i + every[0]
                new_j = j + every[1]
                dfs(new_i,new_j,visited,i,j)
        
        dfs(0,0,visited1,0,0) # 调用
        dfs(m-1,n-1,visited2,m-1,n-1) # 调用

        # 然后看最后一个格子是否被访问
        return visited1[-1][-1] == True and visited2[0][0] == True
            

```

# 1400. 构造 K 个回文字符串

给你一个字符串 s 和一个整数 k 。请你用 s 字符串中 所有字符 构造 k 个非空 回文串 。

如果你可以用 s 中所有字符构造 k 个回文字符串，那么请你返回 True ，否则返回 False 。

```python
class Solution:
    def canConstruct(self, s: str, k: int) -> bool:
        # 只需要考虑可行
        ct = collections.Counter(s)
        # 回文串要求，奇数个数的字母少于等于1个。
        oddNum = 0
        extra = 0
        for key in ct:
            if ct[key]%2 == 1:
                oddNum += 1
                extra += ct[key]-1
            else:
                extra += ct[key]
        # 奇数字母大于k,则不能。奇数字母等于k。可以。奇数字母小于k，需要看偶数字母是否能够补齐。
        if oddNum > k:
            return False
        elif oddNum == k:
            return True 
        diff = k - oddNum # 看差多少个
        # 看diff是否少于extra
        if diff <= extra:
            return True
        else:
            return False 
```

# 1422. 分割字符串的最大得分

给你一个由若干 0 和 1 组成的字符串 s ，请你计算并返回将该字符串分割成两个 非空 子字符串（即 左 子字符串和 右 子字符串）所能获得的最大得分。

「分割字符串的得分」为 左 子字符串中 0 的数量加上 右 子字符串中 1 的数量。

```python
class Solution:
    def maxScore(self, s: str) -> int:
        # 滑动窗口,要求非空
        p = 0
        leftWindow = collections.defaultdict(int)
        rightWindow = collections.defaultdict(int)
        for i in s:
            rightWindow[i] += 1
        maxPoints = 0
        while p < len(s)-1:
            leftWindow[s[p]] += 1
            rightWindow[s[p]] -= 1
            maxPoints = max(maxPoints,(leftWindow["0"]+rightWindow["1"]))
            p += 1
        return maxPoints
```

# 1423. 可获得的最大点数

几张卡牌 排成一行，每张卡牌都有一个对应的点数。点数由整数数组 cardPoints 给出。

每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 k 张卡牌。

你的点数就是你拿到手中的所有卡牌的点数之和。

给你一个整数数组 cardPoints 和整数 k，请你返回可以获得的最大点数。

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        # 逆向思考，剩下的数组一定是连续的，宽度为n-k，最大化拿到的点数即最小化剩余的点数
        n = len(cardPoints)
        size = n - k
        if size == 0: # 返回全部值
            return sum(cardPoints)
        left = 0
        right = size
        window = 0 # 需要初始化
        for i in cardPoints[:size]:
            window += i 
        minWindow = window # 初始化
        while right < n: # 定长窗口滑动
            add = cardPoints[right]
            right += 1
            delete = cardPoints[left]
            left += 1
            window = window + add - delete # 窗口变化
            if window < minWindow: # 这一种耗时少很多，因为不需要每次比较之后都赋值
                minWindow = window
            # minWindow = min(window,minWindow) # 收集，这一步很耗时
        return sum(cardPoints)-minWindow
```

# 1424. 对角线遍历 II

给你一个列表 `nums` ，里面每一个元素都是一个整数列表。请你依照下面各图的规则，按顺序返回 `nums` 中对角线上的整数。

```python
# 数据量限制，纯模拟超时
class Solution:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
      # 改进
        ans = []
        theDict = collections.defaultdict(list)
        # 利用横纵坐标和为定制分组
        # 由于扫描方向为从左到右从上到下
        # 那么收集到的元素在添加到答案前需要反向
        # 由于theDict有序，所以直接按序遍历。。
        for i in range(len(nums)):
            for j in range(len(nums[i])):
                theDict[i+j].append(nums[i][j])
        for key in theDict:
            theDict[key] = theDict[key][::-1]
        ans = []
        for key in theDict:
            for n in theDict[key]:
                ans.append(n)
        return ans 
```

# 1442. 形成两个异或相等数组的三元组数目

给你一个整数数组 arr 。

现需要从数组中取三个下标 i、j 和 k ，其中 (0 <= i < j <= k < arr.length) 。

a 和 b 定义如下：

a = arr[i] ^ arr[i + 1] ^ ... ^ arr[j - 1]
b = arr[j] ^ arr[j + 1] ^ ... ^ arr[k]
注意：^ 表示 按位异或 操作。

请返回能够令 a == b 成立的三元组 (i, j , k) 的数目。

```python
class Solution:
    def countTriplets(self, arr: List[int]) -> int:
        # 直接尬算,位运算的进出窗口很好算
        count = 0
        n = len(arr)
        # 注意i不必从左端点开始，k不必从右端点开始
        # 全闭区间思路
        for start in range(n): # start在0~n-1
            for end in range(start+1,n): # end 在0~n-1
                left = 0
                right = 0
                for t in range(start,end+1): # 注意这个区间,结束为end+1,因为要取end
                    right ^= arr[t]
                # print("start = ",start,"end = ",end)
                # print("left = ",left,"right = ",right)

                for j in range(start+1,end+1): # 注意j不能为左端点。注意这个区间,结束为end+1,因为要取end
                    left ^= arr[j]
                    right ^= arr[j]
                    if left == right:
                        count += 1
        return count 
```

# 1448. 统计二叉树中好节点的数目

给你一棵根为 root 的二叉树，请你返回二叉树中好节点的数目。

「好节点」X 定义为：从根到该节点 X 所经过的节点中，没有任何节点的值大于 X 的值。

```python
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        # dfs,先序遍历即可
        count = 0
        def dfs(node,tempMax):
            nonlocal count 
            if node == None:
                return 
            if node.val > tempMax:
                tempMax = node.val 
            if node.val == tempMax:
                count += 1
            dfs(node.left,tempMax)
            dfs(node.right,tempMax)
        dfs(root,root.val) # 调用
        return count
```

# 1475. 商品折扣后的最终价格

给你一个数组 prices ，其中 prices[i] 是商店里第 i 件商品的价格。

商店里正在进行促销活动，如果你要买第 i 件商品，那么你可以得到与 prices[j] 相等的折扣，其中 j 是满足 j > i 且 prices[j] <= prices[i] 的 最小下标 ，如果没有满足条件的 j ，你将没有任何折扣。

请你返回一个数组，数组中第 i 个元素是折扣后你购买商品 i 最终需要支付的价格。

```python
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        # 模拟，n**2
        ans = []
        n = len(prices)
        for i in range(n):
            active = False
            for j in range(i+1,n):
                if prices[j] <= prices[i]:
                    ans.append(prices[i]-prices[j])
                    active = True
                    break
            if not active:
                ans.append(prices[i])
        return ans

```

```python
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        # 单调栈，单调递增栈，遇到第一个递减的特殊处理 O(n)
        # 记录的是下标
        stack = []
        n = len(prices)
        ans = prices.copy() # 初始化为原值，没有改变则不改变
        for i in range(n):
            if len(stack) == 0:
                stack.append(i)
            elif prices[stack[-1]] < prices[i]: # 保持单调递增
                stack.append(i)
            elif prices[stack[-1]] >= prices[i]: # 遇到了递减，需要处理
                while len(stack) > 0 and prices[stack[-1]] >= prices[i]:
                    index = stack.pop() # 此时这个索引遇到了需要改变的值
                    ans[index] = prices[index] - prices[i]
                stack.append(i)
        return ans 
```

# 1481. 不同整数的最少数目

给你一个整数数组 `arr` 和一个整数 `k` 。现需要从数组中恰好移除 `k` 个元素，请找出移除后数组中不同整数的最少数目。

```python
class Solution:
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        # 优先级队列+ Counter
        ct = collections.Counter(arr)
        minHeap = [] # 尽量移除元素少的，所以使用小根堆
        origin = len(ct)
        for key in ct:
            minHeap.append(ct[key])
        heapq.heapify(minHeap)
        times = 0
        while k > 0: # 尝试移除
            e = heapq.heappop(minHeap)
            if k - e >= 0:
                k -= e
                times += 1
            else:
                break
        return origin - times
```

# 1497. 检查数组对是否可以被 k 整除

给你一个整数数组 arr 和一个整数 k ，其中数组长度是偶数，值为 n 。

现在需要把数组恰好分成 n / 2 对，以使每对数字的和都能够被 k 整除。

如果存在这样的分法，请返回 True ；否则，返回 False 。

```python
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        # 预先模运算之后转行成哈希表找两数之和
        # 注意有负数。。
        newArr = [i%k for i in arr]
        cnt = collections.Counter(newArr)

        for key in cnt: # 这一步是由于会互相消除，所以*2
            cnt[key] *= 2

        # print(newArr)
        # print(cnt)
        for e in newArr: # 注意0不算
            if k-e in cnt:
                if cnt[e] > 0 and cnt[k-e] > 0:
                    cnt[e] -= 1
                    cnt[k-e] -= 1
                else:
                    return False
        theSum = 0

        if cnt.get(key) != None: # 注意这个补丁。。。
                if cnt[key] % 4 != 0:
                    return False

        for key in cnt:            
            if key != 0:
                theSum += cnt[key]

        if theSum == 0:
            return True
        else:
            return False
                
```

# 1506. 找到 N 叉树的根节点

给定一棵 [N 叉树](https://slack-redir.net/link?url=https%3A%2F%2Fleetcode.com%2Farticles%2Fintroduction-to-n-ary-trees) 的所有节点在一个数组 `Node[] tree` 中，树中每个节点都有 **唯一的值** 。

找到并返回 N 叉树的 **根节点** 。

```python
class Solution:
    def findRoot(self, tree: List['Node']) -> 'Node':
        # dfs不标记本身,最后会有一个节点不被标记
        nodeDict = dict() # k-v 为数值-节点,不包含根节点
        valDict = dict() # 记录所有出现过的节点

        def dfs(node,state):
            if node == None:
                return 
            if node.val in nodeDict: # 提前剪枝回来
                return 
            if state != True:
                nodeDict[node.val] = node 
            valDict[node.val] = node
            for child in node.children:
                dfs(child,False)
        
        for node in tree:
            if node.val not in nodeDict:
                dfs(node,True)
        
        # print(nodeDict,valDict)
        for key in valDict:
            if key not in nodeDict:
                return valDict[key]
```

```python
# 0神的异或解法
class Solution:
    def findRoot(self, tree: List['Node']) -> 'Node':
        # 遍历所有节点的时候，只有根节点被访问一次，其他的都是偶数次
        xorNum = 0
        for node in tree:
            xorNum ^= node.val
            for child in node.children:
                xorNum ^= child.val
        
        for node in tree:
            if node.val == xorNum:
                return node
```

# 1522. N 叉树的直径

给定一棵 N 叉树的根节点 root ，计算这棵树的直径长度。

N 叉树的直径指的是树中任意两个节点间路径中 最长 路径的长度。这条路径可能经过根节点，也可能不经过根节点。

（N 叉树的输入序列以层序遍历的形式给出，每组子节点用 null 分隔）

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def diameter(self, root: 'Node') -> int:
        """
        :type root: 'Node'
        :rtype: int
        """
        ans = 0

        # 求孩子的深度
        def getDepth(node):
            nonlocal ans
            if node == None: # 空节点为0
                return 0
            dList = []
            for child in node.children:
                dList.append(getDepth(child))
            if len(dList) == 0: # 叶子节点为1
                return 1
            elif len(dList) == 1:
                ans = max(ans,dList[0]) # 注意这一行
            elif len(dList) > 1:
                dList.sort() # 排序，找到两个深度最大的，更新ans
                ans = max(ans,dList[-1]+dList[-2])
            return dList[-1] + 1
        
        getDepth(root)
        return ans

```

# 1514. 概率最大的路径

给你一个由 n 个节点（下标从 0 开始）组成的无向加权图，该图由一个描述边的列表组成，其中 edges[i] = [a, b] 表示连接节点 a 和 b 的一条无向边，且该边遍历成功的概率为 succProb[i] 。

指定两个节点分别作为起点 start 和终点 end ，请你找出从起点到终点成功概率最大的路径，并返回其成功概率。

如果不存在从 start 到 end 的路径，请 返回 0 。只要答案与标准答案的误差不超过 1e-5 ，就会被视作正确答案。

```python
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        # dj模版题
        graph = collections.defaultdict(list)
        length = len(edges)
        for i in range(length):
            a,b = edges[i]
            graph[a].append((b,succProb[i]))
            graph[b].append((a,succProb[i]))

        # print(graph)
        distance = [0xffffffff for i in range(n)] 
        distance[start] = -1
        queue = [(-1,start)] # （距离,节点编号）,初始化为-1,方便处理.注意用start
        state = False 

        while len(queue) != 0:
            nowDistance,cur = heapq.heappop(queue)
            if cur == end:
                state = True
                break
            for neigh,prob in graph[cur]:
                new_distance = prob*nowDistance
                if distance[neigh] > new_distance:
                    distance[neigh] = new_distance
                    heapq.heappush(queue,(new_distance,neigh))

        if state:
            return abs(nowDistance)
        else:
            return 0
```

# 1557. 可以到达所有点的最少点数目

给你一个 有向无环图 ， n 个节点编号为 0 到 n-1 ，以及一个边数组 edges ，其中 edges[i] = [fromi, toi] 表示一条从点  fromi 到点 toi 的有向边。

找到最小的点集使得从这些点出发能到达图中所有点。题目保证解存在且唯一。

你可以以任意顺序返回这些节点编号。

```python
class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        # 找到所有入度为0的点。和一个visited
        # 所有入度为0的点先加入
        visited = [False for i in range(n)]
        inDegree = [0 for i in range(n)]
        graph = collections.defaultdict(list)
        for a,b in edges:
            graph[a].append(b)
            inDegree[b] += 1
        # BFS
        queue = []
        ans = []
        for i in range(n):
            if inDegree[i] == 0:
                queue.append(i)
                ans.append(i)
                visited[i] = True

        while len(queue) != 0:
            new_queue = []
            for i in queue:
                for neigh in graph[i]:
                    if visited[neigh] == False: # 没有访问过，就标记为访问，并且添加进新队列
                        new_queue.append(neigh)
                        visited[neigh] = True 
            queue = new_queue
        
        for i in range(n):
            if visited[i] == False:
                ans.append(i)
        return ans
```



# 1564. 把箱子放进仓库里 I

给定两个正整数数组 boxes 和 warehouse ，分别包含单位宽度的箱子的高度，以及仓库中 n 个房间各自的高度。仓库的房间分别从 0 到 n - 1 自左向右编号， warehouse[i] （索引从 0 开始）是第 i 个房间的高度。

箱子放进仓库时遵循下列规则：

箱子不可叠放。
你可以重新调整箱子的顺序。
箱子只能从左向右推进仓库中。
如果仓库中某房间的高度小于某箱子的高度，则这个箱子和之后的箱子都会停在这个房间的前面。
你最多可以在仓库中放进多少个箱子？

```python
class Solution:
    def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
        # boxes倒序排序
        # 仓库形状整理成类似单调栈。赋值为当前的最小值
        minVol = warehouse[0]
        for i in range(len(warehouse)):
            minVol = min(minVol,warehouse[i])
            warehouse[i] = minVol
        # 倒序整理仓库
        warehouse = warehouse[::-1]
        boxes.sort()
        count = 0
        # 由于不能放弃箱子，所以可能会挡住
        pBox = 0
        pW = 0
        while pBox < len(boxes) and pW < len(warehouse):
            # 如果箱子矮，则两个都加1，并且计数器加一
            if boxes[pBox] <= warehouse[pW]:
                pBox += 1
                pW += 1
                count += 1
            elif boxes[pBox] > warehouse[pW]: # pW移动到直到大于等于为止,注意防止越界
                while pW < len(warehouse) and warehouse[pW] < boxes[pBox]:
                    pW += 1
        return count

```

# 1576. 替换所有的问号

给你一个仅包含小写英文字母和 '?' 字符的字符串 s，请你将所有的 '?' 转换为若干小写字母，使最终的字符串不包含任何 连续重复 的字符。

注意：你 不能 修改非 '?' 字符。

题目测试用例保证 除 '?' 字符 之外，不存在连续重复的字符。

在完成所有转换（可能无需转换）后返回最终的字符串。如果有多个解决方案，请返回其中任何一个。可以证明，在给定的约束条件下，答案总是存在的。

```python
class Solution:
    def modifyString(self, s: str) -> str:
        # 所有的问号先尝试填a,然后检查是否合法
        stack = []
        start = 0
        check = []
        for i in range(len(s)):
            if s[i] == "?":
                stack.append(chr(start+97))
                start = (start+1)%26
                check.append(i)
            elif s[i] != "?":
                stack.append(s[i])

        for index in check:
            if index-1>=0 and index < len(s)-1:
                while stack[index] == stack[index-1] or stack[index] == stack[index+1]:
                    e = stack[index]
                    e = chr(((ord(e)-97)+10)%26+97)
                    stack[index] = e
            elif index == 0 and len(s)>1:
                while stack[0] == stack[1]:
                    e = stack[index]
                    e = chr(((ord(e)-97)+10)%26+97)
                    stack[index] = e
            elif index == len(s)-1 and len(s)>1:
                while stack[index-1] == stack[index]:
                    e = stack[index]
                    e = chr(((ord(e)-97)+10)%26+97)
                    stack[index] = e
        
        return ''.join(stack)

            
```

# 1598. 文件夹操作日志搜集器

每当用户执行变更文件夹操作时，LeetCode 文件系统都会保存一条日志记录。

下面给出对变更操作的说明：

"../" ：移动到当前文件夹的父文件夹。如果已经在主文件夹下，则 继续停留在当前文件夹 。
"./" ：继续停留在当前文件夹。
"x/" ：移动到名为 x 的子文件夹中。题目数据 保证总是存在文件夹 x 。
给你一个字符串列表 logs ，其中 logs[i] 是用户在 ith 步执行的操作。

文件系统启动时位于主文件夹，然后执行 logs 中的操作。

执行完所有变更文件夹操作后，请你找出 返回主文件夹所需的最小步数 。

```python
class Solution:
    def minOperations(self, logs: List[str]) -> int:
        # 模拟栈操作,允许嵌套同名文件夹
        stack = []
        for op in logs:
            if stack == []:
                if op == "../" or op == "./":
                    pass
                else:
                    stack.append(op)
            elif stack != []:
                if op == "../":
                    stack.pop()
                elif op == "./":
                    pass
                else:
                    stack.append(op)
        return len(stack)
```

# 1629. 按键持续时间最长的键

LeetCode 设计了一款新式键盘，正在测试其可用性。测试人员将会点击一系列键（总计 n 个），每次一个。

给你一个长度为 n 的字符串 keysPressed ，其中 keysPressed[i] 表示测试序列中第 i 个被按下的键。releaseTimes 是一个升序排列的列表，其中 releaseTimes[i] 表示松开第 i 个键的时间。字符串和数组的 下标都从 0 开始 。第 0 个键在时间为 0 时被按下，接下来每个键都 恰好 在前一个键松开时被按下。

测试人员想要找出按键 持续时间最长 的键。第 i 次按键的持续时间为 releaseTimes[i] - releaseTimes[i - 1] ，第 0 次按键的持续时间为 releaseTimes[0] 。

注意，测试期间，同一个键可以在不同时刻被多次按下，而每次的持续时间都可能不同。

请返回按键 持续时间最长 的键，如果有多个这样的键，则返回 按字母顺序排列最大 的那个键。

```python
class Solution:
    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        prev = 0
        n = len(releaseTimes)
        record = collections.defaultdict(int)
        for i in range(n):
            time = releaseTimes[i]-prev 
            if time > record[keysPressed[i]]:
                record[keysPressed[i]] = time 
            prev = releaseTimes[i]
        
        maxNum = -1
        tempAns = []
        for key in record:
            if maxNum < record[key]:
                maxNum = record[key]
        for key in record:
            if record[key] == maxNum:
                tempAns.append(key)
        tempAns.sort()
        return tempAns[-1]
```

# 1634. 求两个多项式链表的和

多项式链表是一种特殊形式的链表，每个节点表示多项式的一项。

每个节点有三个属性：

coefficient：该项的系数。项 9x4 的系数是 9 。
power：该项的指数。项 9x4 的指数是 4 。
next：指向下一个节点的指针（引用），如果当前节点为链表的最后一个节点则为 null 。
例如，多项式 5x3 + 4x - 7 可以表示成如下图所示的多项式链表：

多项式链表必须是标准形式的，即多项式必须 严格 按指数 power 的递减顺序排列（即降幂排列）。另外，系数 coefficient 为 0 的项需要省略。

给定两个多项式链表的头节点 poly1 和 poly2，返回它们的和的头节点。

PolyNode 格式：

输入/输出格式表示为 n 个节点的列表，其中每个节点表示为 [coefficient, power] 。例如，多项式 5x3 + 4x - 7 表示为： [[5,3],[4,1],[-7,0]] 。

```python
# Definition for polynomial singly-linked list.
# class PolyNode:
#     def __init__(self, x=0, y=0, next=None):
#         self.coefficient = x
#         self.power = y
#         self.next = next

class Solution:
    def addPoly(self, poly1: 'PolyNode', poly2: 'PolyNode') -> 'PolyNode':
        # 根据链表的power融合
        cur = self.merge(poly1,poly2) # 融合完毕之后修剪链表。修剪系数为0的
        dummy = PolyNode()
        dummy.next = cur

        def pruneList(head): # 修剪方法，修剪系数为0的
            slow = head
            fast = head.next
            while fast != None:
                if fast.coefficient == 0:
                    slow.next = fast.next
                    fast = fast.next
                else:
                    slow = slow.next
                    fast = fast.next

        pruneList(dummy) # 调用修剪方法
        return dummy.next
        

    def merge(self,lst1,lst2):
        if lst1 == None: return lst2
        if lst2 == None: return lst1
        if lst1.power > lst2.power:
            lst1.next = self.merge(lst1.next,lst2)
            return lst1
        elif lst1.power == lst2.power:
            lst1.coefficient += lst2.coefficient
            lst1.next = self.merge(lst1.next,lst2.next)
            return lst1
        elif lst1.power < lst2.power:
            lst2.next = self.merge(lst1,lst2.next)
            return lst2
```

# 1641. 统计字典序元音字符串的数目

给你一个整数 n，请返回长度为 n 、仅由元音 (a, e, i, o, u) 组成且按 字典序排列 的字符串数量。

字符串 s 按 字典序排列 需要满足：对于所有有效的 i，s[i] 在字母表中的位置总是与 s[i+1] 相同或在 s[i+1] 之前。

```python
class Solution:
    def countVowelStrings(self, n: int) -> int:
        # 使用数学思路
        # 以五行dp来做
        # 以abcde表示的话。例如第一行就是长度为i以a结尾的，
        # 递推的时候是少一个字符然后递推。
        dp = [[0 for j in range(n+1)] for i in range(5)]
        for i in range(5): # 初始化
            dp[i][1] = 1
        for j in range(1,n+1): # 初始化
            dp[0][j] = 1
        # 状态转移方程，非压缩版
        for j in range(2,n+1):
            dp[1][j] = dp[0][j-1] + dp[1][j-1]
            dp[2][j] = dp[0][j-1] + dp[1][j-1] + dp[2][j-1]
            dp[3][j] = dp[0][j-1] + dp[1][j-1] + dp[2][j-1] + dp[3][j-1]
            dp[4][j] = dp[0][j-1] + dp[1][j-1] + dp[2][j-1] + dp[3][j-1] + dp[4][j-1]
        ans = 0       
        for i in range(5):
            ans += dp[i][-1]
        return ans
```

# 1657. 确定两个字符串是否接近

如果可以使用以下操作从一个字符串得到另一个字符串，则认为两个字符串 接近 ：

操作 1：交换任意两个 现有 字符。
例如，abcde -> aecdb
操作 2：将一个 现有 字符的每次出现转换为另一个 现有 字符，并对另一个字符执行相同的操作。
例如，aacabb -> bbcbaa（所有 a 转化为 b ，而所有的 b 转换为 a ）
你可以根据需要对任意一个字符串多次使用这两种操作。

给你两个字符串，word1 和 word2 。如果 word1 和 word2 接近 ，就返回 true ；否则，返回 false 。

```python
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        # 两字符串长度不等一定False
        if len(word1) != len(word2):
            return False 
        # 计数然后看是否所有字符数量一样，顺序不用管
        # 但是需要激活的是现有字符所以还需要激活数组
        
        ct1 = collections.Counter(word1)
        ct2 = collections.Counter(word2)

        active1 = [False for i in range(26)]
        active2 = [False for j in range(26)]

        lst1 = []
        lst2 = []
        for key in ct1:
            lst1.append(ct1[key])
            index = ord(key)-ord("a")
            active1[index] = True
        for key in ct2:
            lst2.append(ct2[key])
            index = ord(key)-ord("a")
            active2[index] = True
        lst1.sort()
        lst2.sort()

        if lst1 == lst2 and active1 == active2:
            return True
        else:
            return False
```

# 1658. 将 x 减到 0 的最小操作数

给你一个整数数组 nums 和一个整数 x 。每一次操作时，你应当移除数组 nums 最左边或最右边的元素，然后从 x 中减去该元素的值。请注意，需要 修改 数组以供接下来的操作使用。

如果可以将 x 恰好 减到 0 ，返回 最小操作数 ；否则，返回 -1 。

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # 需要修改数组是啥意思
        # 逆向思考：最后数组中留下的是一串连续值。看total-window是否为0
        # 返回的是最小操作数目。即全长减去window长度为操作次数。取最小的

        # 看total-window == 0,代表window正好，收集n-windowSize
        # 窗口移动逻辑为total-window > 0 则代表window少了，还需要取，右扩张
        # total - window < 0 。代表window大了，需要收缩，左收缩
        total = sum(nums) # 数目
        left = 0
        right = 0
        window = 0 # 数总和
        windowSize = 0 # 窗口大小
        n = len(nums) # 全长
        minOperation = n + 1 # 初始化为满次数+1，如果没有被更新，返回-1
        while right < n:
            add = nums[right]
            window += add
            windowSize += 1
            right += 1
            if total - window == x:
                minOperation = min(minOperation,n-windowSize)
            while left < right and total - window < x:
                delete = nums[left]
                window -= delete
                windowSize -= 1
                left += 1
                if total - window == x:
                    minOperation = min(minOperation,n-windowSize)
        # 全过程没有被更新
        # 返回-1
        return minOperation if minOperation != n+1 else -1
 
```

# 1673. 找出最具竞争力的子序列

给你一个整数数组 nums 和一个正整数 k ，返回长度为 k 且最具 竞争力 的 nums 子序列。

数组的子序列是从数组中删除一些元素（可能不删除元素）得到的序列。

在子序列 a 和子序列 b 第一个不相同的位置上，如果 a 中的数字小于 b 中对应的数字，那么我们称子序列 a 比子序列 b（相同长度下）更具 竞争力 。 例如，[1,3,4] 比 [1,3,5] 更具竞争力，在第一个不相同的位置，也就是最后一个位置上， 4 小于 5 。

```python
class Solution:
    def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
        # 先存储一个单调栈和索引
        # 根据可以pop的次数来决定

        stack = []
        remain = len(nums)-k # 可以pop的次数
        for i in range(len(nums)):
            if len(stack) == 0:
                stack.append(i)
                continue
            elif len(stack) > 0 and nums[stack[-1]] > nums[i] and remain > 0:
                while len(stack) > 0 and nums[stack[-1]] > nums[i] and remain > 0:
                    stack.pop()
                    remain -= 1
            stack.append(i)
        
        # print(stack)
        # 此时，stack里面是索引
        # 如果没有remain没有pop完，继续pop
        while remain > 0:
            stack.pop()
            remain -= 1

        ans = [nums[i] for i in stack[:k]]
        return ans
```

```go
func mostCompetitive(nums []int, k int) []int {
    remain := len(nums) - k
    stack := make([]int,0,0)
    for _,v := range(nums) {
        if len(stack) == 0 {
            stack = append(stack,v)
            continue
        } else if len(stack) > 0 && stack[len(stack)-1] > v && remain > 0 {
            for len(stack) > 0 && stack[len(stack)-1] > v && remain > 0 {
                stack = stack[:len(stack)-1]
                remain -= 1
            }
        }
        stack = append(stack,v)
    }
    
    for remain > 0 {
        remain -= 1
        stack = stack[:len(stack)-1]
    }
    
    return stack[:k]
}
```

# 1695. 删除子数组的最大得分

给你一个正整数数组 nums ，请你从中删除一个含有 若干不同元素 的子数组。删除子数组的 得分 就是子数组各元素之 和 。

返回 只删除一个 子数组可获得的 最大得分 。

如果数组 b 是数组 a 的一个连续子序列，即如果它等于 a[l],a[l+1],...,a[r] ，那么它就是 a 的一个子数组。

```python
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        # 滑动窗口板子题
        # 数组数据全正数
        maxPoints = 0
        n = len(nums)
        left = 0
        right = 0
        window = 0
        windowDict = collections.defaultdict(int)
        windowSize = 0
        while right < n:
            add = nums[right]
            windowDict[add] += 1
            window += add
            windowSize += 1
            right += 1
            if windowSize == len(windowDict): # 收集结果
                maxPoints = max(maxPoints,window)
            while left < right and windowSize > len(windowDict): # 说明有重复
                delete = nums[left]
                windowDict[delete] -= 1
                if windowDict[delete] == 0: del windowDict[delete]
                window -= delete
                windowSize -= 1
                left += 1
        return maxPoints
        
```

# 1708. 长度为 K 的最大子数组

在数组 A 和数组 B 中，对于第一个满足 A[i] != B[i] 的索引 i ，当 A[i] > B[i] 时，数组 A 大于数组 B。

例如，对于索引从 0 开始的数组：

[1,3,2,4] > [1,2,2,4] ，因为在索引 1 上， 3 > 2。
[1,4,4,4] < [2,1,1,1] ，因为在索引 0 上， 1 < 2。
一个数组的子数组是原数组上的一个连续子序列。

给定一个包含不同整数的整数类型数组 nums ，返回 nums 中长度为 k 的最大子数组。

```python
class Solution:
    def largestSubarray(self, nums: List[int], k: int) -> List[int]:
        # 固定窗口大小
        # 定义一个比较器
        window = nums[:k]
        window = deque(window) # 双端队列提速
        ans = window.copy()
        for i in nums[k:]:
            window.popleft() # 注意pop左边。不用双端队列的时间成本很高
            window.append(i)
            if self.compare(ans,window) < 0:
                ans = window.copy()

        ans = list(ans)
        return ans
    
    def compare(self,lst1,lst2):
        if lst1[0] > lst2[0]:
            return 1
        else:
            return -1
```



# 1720. 解码异或后的数组

未知 整数数组 arr 由 n 个非负整数组成。

经编码后变为长度为 n - 1 的另一个整数数组 encoded ，其中 encoded[i] = arr[i] XOR arr[i + 1] 。例如，arr = [1,0,2,1] 经编码后得到 encoded = [1,2,3] 。

给你编码后的数组 encoded 和原数组 arr 的第一个元素 first（arr[0]）。

请解码返回原数组 arr 。可以证明答案存在并且是唯一的。

```python
class Solution:
    def decode(self, encoded: List[int], first: int) -> List[int]:
        # [a,b,c,d,e] -> [a^b,b^c,c^d,d^e]
        # b = a ^ a^ b
        ans = []
        ans.append(first)
        prev = first
        for i in encoded:
            prev = prev ^ i
            ans.append(prev)
        return ans
```

# 1730. 获取食物的最短路径

你现在很饿，想要尽快找东西吃。你需要找到最短的路径到达一个食物所在的格子。

给定一个 m x n 的字符矩阵 grid ，包含下列不同类型的格子：

'*' 是你的位置。矩阵中有且只有一个 '*' 格子。
'#' 是食物。矩阵中可能存在多个食物。
'O' 是空地，你可以穿过这些格子。
'X' 是障碍，你不可以穿过这些格子。
返回你到任意食物的最短路径的长度。如果不存在你到任意食物的路径，返回 -1。

```python
class Solution:
    def getFood(self, grid: List[List[str]]) -> int:
        # 找到自己的开始位置
        # 所有的x添加进不可访问。
        # 可能没有饼
        m = len(grid)
        n = len(grid[0])
        visited = set()
        end = set()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "X":
                    visited.add((i,j))
                if grid[i][j] == "*":
                    start = (i,j)
                if grid[i][j] == "#":
                    end.add((i,j))
        
        # BFS搜
        direc = [(1,0),(-1,0),(0,1),(0,-1)]
        queue = [(start)]
        steps = 0
        while len(queue) != 0:
            new_queue = []
            for pair in queue:
                x,y = pair
                if (x,y) in end: # 可能有好几个饼，找到一个饼就停下来
                    return steps
                for di in direc:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if 0<= new_x < m and 0 <= new_y < n and (new_x,new_y) not in visited:
                        new_queue.append((new_x,new_y))
                        visited.add((new_x,new_y))
            queue = new_queue
            steps += 1
        return -1
```

# 1759. 统计同构子字符串的数目

给你一个字符串 s ，返回 s 中 同构子字符串 的数目。由于答案可能很大，只需返回对 109 + 7 取余 后的结果。

同构字符串 的定义为：如果一个字符串中的所有字符都相同，那么该字符串就是同构字符串。

子字符串 是字符串中的一个连续字符序列。

```python
class Solution:
    def countHomogenous(self, s: str) -> int:
        mod = 10**9 + 7
        # 双指针，注意边界条件处理
        # 当前长度为k,总结果ans += (k+1)*k/2
        p = 0
        n = len(s)
        mark = s[0]
        ans = 0
        while p < n:
            memo = p
            while p < n and s[p] == mark:
                p += 1
            k = p - memo
            ans += (k+1)*k//2
            if p < n:
                mark = s[p]
        return ans % mod
```

# 1765. 地图中的最高点

给你一个大小为 m x n 的整数矩阵 isWater ，它代表了一个由 陆地 和 水域 单元格组成的地图。

如果 isWater[i][j] == 0 ，格子 (i, j) 是一个 陆地 格子。
如果 isWater[i][j] == 1 ，格子 (i, j) 是一个 水域 格子。
你需要按照如下规则给每个单元格安排高度：

每个格子的高度都必须是非负的。
如果一个格子是是 水域 ，那么它的高度必须为 0 。
任意相邻的格子高度差 至多 为 1 。当两个格子在正东、南、西、北方向上相互紧挨着，就称它们为相邻的格子。（也就是说它们有一条公共边）
找到一种安排高度的方案，使得矩阵中的最高高度值 最大 。

请你返回一个大小为 m x n 的整数矩阵 height ，其中 height[i][j] 是格子 (i, j) 的高度。如果有多种解法，请返回 任意一个 。

```python
class Solution:
    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        # 扩散问题,BFS可能在python超时
        # 注意优化
        m = len(isWater)
        n = len(isWater[0])
        #BFS扩散
        queue = []
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        ans = [[0 for j in range(n)] for i in range(m)]

        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    queue.append((i,j))
                    visited[i][j] = True
        
        steps = 0
        while len(queue) != 0:
            new_queue = []
            for x,y in queue:
                ans[x][y] = steps
                for di in direc:
                    new_x = x + di[0] 
                    new_y = y + di[1]
                    if 0<=new_x<m and 0<=new_y<n and visited[new_x][new_y] == False:
                        new_queue.append((new_x,new_y))
                        visited[new_x][new_y] = True
            queue = new_queue
            steps += 1
        
        return ans
```

# 1769. 移动所有球到每个盒子所需的最小操作数

有 n 个盒子。给你一个长度为 n 的二进制字符串 boxes ，其中 boxes[i] 的值为 '0' 表示第 i 个盒子是 空 的，而 boxes[i] 的值为 '1' 表示盒子里有 一个 小球。

在一步操作中，你可以将 一个 小球从某个盒子移动到一个与之相邻的盒子中。第 i 个盒子和第 j 个盒子相邻需满足 abs(i - j) == 1 。注意，操作执行后，某些盒子中可能会存在不止一个小球。

返回一个长度为 n 的数组 answer ，其中 answer[i] 是将所有小球移动到第 i 个盒子所需的 最小 操作数。

每个 answer[i] 都需要根据盒子的 初始状态 进行计算。

```python
class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        # ？数据量这么小的尬算？
        ans = []
        n = len(boxes)
        for i in range(n):
            temp = 0
            for j in range(n):
                if i == j:
                    continue 
                if boxes[j] == "1":
                    temp += abs(i-j)
            ans.append(temp)
        return ans
```

```

```



# 1779. 找到最近的有相同 X 或 Y 坐标的点

给你两个整数 x 和 y ，表示你在一个笛卡尔坐标系下的 (x, y) 处。同时，在同一个坐标系下给你一个数组 points ，其中 points[i] = [ai, bi] 表示在 (ai, bi) 处有一个点。当一个点与你所在的位置有相同的 x 坐标或者相同的 y 坐标时，我们称这个点是 有效的 。

请返回距离你当前位置 曼哈顿距离 最近的 有效 点的下标（下标从 0 开始）。如果有多个最近的有效点，请返回下标 最小 的一个。如果没有有效点，请返回 -1 。

两个点 (x1, y1) 和 (x2, y2) 之间的 曼哈顿距离 为 abs(x1 - x2) + abs(y1 - y2) 。

```python
class Solution:
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        p1 = (x,y)
        # 返回的是下标最小的，保证严格更新就可以了
        ans = -1 # 默认为-1
        mindistance = 0xffffffff
        for i in range(len(points)):
            p2 = points[i]
            if p1[0] == p2[0] or p1[1] == p2[1]:
                theDistance = self.getManhatum(p1,p2)
                if theDistance < mindistance: # 严格更新
                    mindistance = theDistance
                    ans = i
        return ans


    def getManhatum(self,p1,p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
```

# 1823. 找出游戏的获胜者

共有 n 名小伙伴一起做游戏。小伙伴们围成一圈，按 顺时针顺序 从 1 到 n 编号。确切地说，从第 i 名小伙伴顺时针移动一位会到达第 (i+1) 名小伙伴的位置，其中 1 <= i < n ，从第 n 名小伙伴顺时针移动一位会回到第 1 名小伙伴的位置。

游戏遵循如下规则：

从第 1 名小伙伴所在位置 开始 。
沿着顺时针方向数 k 名小伙伴，计数时需要 包含 起始时的那位小伙伴。逐个绕圈进行计数，一些小伙伴可能会被数过不止一次。
你数到的最后一名小伙伴需要离开圈子，并视作输掉游戏。
如果圈子中仍然有不止一名小伙伴，从刚刚输掉的小伙伴的 顺时针下一位 小伙伴 开始，回到步骤 2 继续执行。
否则，圈子中最后一名小伙伴赢得游戏。
给你参与游戏的小伙伴总数 n ，和一个整数 k ，返回游戏的获胜者。

```python
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        # 约瑟夫环问题,只关注活着的人的下标变化
        # f(n,k)为最终活着的那个人的当前轮索引
        # 例如第一轮： 当前坐标为u,下一轮坐标为v
        # (u-k%n) == v
        # 移动项： u = v + k%n  从而可以从小的推到大的 
        # 递推公式
        # f(n+1,k) = (k%(n+1)+f(n,k))%(n+1)
                #  = (k+f(n,k))%(n+1) 
        # dp[i+1] = (k+dp[i])%(i+1)
        # 基态为f(1,k) = 0
        # 用s的写法如下
        # s = 0
        # for i in range(1,n):
        #     s = (k+s)%(i+1)
        # return (s+1)
        # dp写法为，初始化dp[1] = 0,求dp[n]
        dp = [0 for i in range(n+1)]
        for i in range(1,n):
            dp[i+1] = (k+dp[i])%(i+1)
        return (dp[-1]+1)

```

# 1852. 每个子数组的数字种类数

给你一个整数数组 nums与一个整数 k，请你构造一个长度 n-k+1 的数组 ans，这个数组第i个元素 ans[i] 是每个长度为k的子数组 nums[i:i+k-1] = [nums[i], nums[i+1], ..., nums[i+k-1]]中数字的种类数。

返回这个数组 ans。

```python
class Solution:
    def distinctNumbers(self, nums: List[int], k: int) -> List[int]:
        # 滑动窗口板子题
        window = collections.defaultdict(int)
        ans = [] # 收集答案
        for i in nums[:k]:
            window[i] += 1
        ans.append(len(window))
        # 固定窗口大小
        left = 0
        right = k
        n = len(nums)
        while right < n:
            add = nums[right]
            window[add] += 1
            delete = nums[left]
            window[delete] -= 1
            if window[delete] == 0: del window[delete] # 注意要删除
            ans.append(len(window)) 
            left += 1
            right +=1
        return ans
```

# 1854. 人口最多的年份

给你一个二维整数数组 logs ，其中每个 logs[i] = [birthi, deathi] 表示第 i 个人的出生和死亡年份。

年份 x 的 人口 定义为这一年期间活着的人的数目。第 i 个人被计入年份 x 的人口需要满足：x 在闭区间 [birthi, deathi - 1] 内。注意，人不应当计入他们死亡当年的人口中。

返回 人口最多 且 最早 的年份。

```python
class Solution:
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        # 差分数组。上车模型
        up = collections.defaultdict(int)
        down = collections.defaultdict(int)
        for birth,death in logs:
            up[birth-1950] += 1
            down[death-1950] += 1
        n = max(down)
        # 申请数组长度为最大值
        arr = [0 for i in range(n)]
        now = 0
        for i in range(n):
            if i in up:
                now += up[i]
            if i in down:
                now -= down[i]
            arr[i] = now
        maxPeople = max(arr)
        for i in range(n):
            if arr[i] == maxPeople:
                return i+1950

```

# 1858. 包含所有前缀的最长单词

给定一个字符串数组 words，找出 words 中所有的前缀都在 words 中的最长字符串。

例如，令 words = ["a", "app", "ap"]。字符串 "app" 含前缀 "ap" 和 "a" ，都在 words 中。
返回符合上述要求的字符串。如果存在多个（符合条件的）相同长度的字符串，返回字典序中最小的字符串，如果这样的字符串不存在，返回 ""。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.isEnd = False # 默认为False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,word):
        node = self.root
        for ch in word:
            index = ord(ch)-ord("a")
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isEnd = True
    
    def insertAll(self,lst):
        for word in lst:
            self.insert(word)
    
    def find(self): # 收路径上可行节点
        ans = []
        path = []
        def dfs(path,node,length):
            if node == None:
                return 
            if node.isEnd: # 
                ans.append("".join(path[:])) # 可以成功走到节点。不能是叶子节点式的判断
                # 因为会出现 [a,b,abcde],a也是合法的，不能断链

            for c in range(26):
                char = chr(97+c)
                path.append(char)
                if node.children[c] != None and node.children[c].isEnd: # 路径需要被激活过
                    dfs(path,node.children[c],length+1)
                path.pop()

        dfs(path,self.root,0)
        return ans

class Solution:
    def longestWord(self, words: List[str]) -> str:
        tree = Trie()
        tree.insertAll(words)
        ans = tree.find()
        # 此时ans里面有一堆字符串，需要找到最长的
        ans.sort(key = lambda x:(-len(x),x)) # 长度排序之后字典序排序,最长的排在前面
        # 或者用过滤器
        if len(ans) == 0:
            return ""
        return ans[0]

```

# 1865. 找出和为指定值的下标对

给你两个整数数组 nums1 和 nums2 ，请你实现一个支持下述两类查询的数据结构：

累加 ，将一个正整数加到 nums2 中指定下标对应元素上。
计数 ，统计满足 nums1[i] + nums2[j] 等于指定值的下标对 (i, j) 数目（0 <= i < nums1.length 且 0 <= j < nums2.length）。
实现 FindSumPairs 类：

FindSumPairs(int[] nums1, int[] nums2) 使用整数数组 nums1 和 nums2 初始化 FindSumPairs 对象。
void add(int index, int val) 将 val 加到 nums2[index] 上，即，执行 nums2[index] += val 。
int count(int tot) 返回满足 nums1[i] + nums2[j] == tot 的下标对 (i, j) 数目。

```python
class FindSumPairs:
# 哈希表两数之和法
# 数据量比较小，不需要使用TreeSet
    def __init__(self, nums1: List[int], nums2: List[int]):
        self.cnt1 = collections.Counter(nums1)
        self.arr = nums2
        self.cnt2 = collections.Counter(nums2)

    def add(self, index: int, val: int) -> None:
        origin = self.arr[index]
        self.cnt2[origin] -= 1
        self.arr[index] += val
        now = self.arr[index]
        self.cnt2[now] += 1

    def count(self, tot: int) -> int:
        sums = 0
        for key in self.cnt1:
            if tot-key in self.cnt2:
                sums += self.cnt1[key]*self.cnt2[tot-key]
        return sums
```




# 1887. 使数组元素相等的减少操作次数

给你一个整数数组 nums ，你的目标是令 nums 中的所有元素相等。完成一次减少操作需要遵照下面的几个步骤：

找出 nums 中的 最大 值。记这个值为 largest 并取其下标 i （下标从 0 开始计数）。如果有多个元素都是最大值，则取最小的 i 。
找出 nums 中的 下一个最大 值，这个值 严格小于 largest ，记为 nextLargest 。
将 nums[i] 减少到 nextLargest 。
返回使 nums 中的所有元素相等的操作次数。

```python
class Solution:
    def reductionOperations(self, nums: List[int]) -> int:
        nums.sort(key = lambda x:-x)
        # 指针扫描[5,5,5,3,3,3,1,1,1]
        # 找到下一个不同的数，用操作数加上它的下标
        ops = 0
        now = 0
        p = 0
        n = len(nums)
        while p < n:
            while p < n and nums[p] == nums[now]:
                p += 1
            if p < n:
                ops += p 
            now = p
        return ops

```

# 1905. 统计子岛屿

给你两个 m x n 的二进制矩阵 grid1 和 grid2 ，它们只包含 0 （表示水域）和 1 （表示陆地）。一个 岛屿 是由 四个方向 （水平或者竖直）上相邻的 1 组成的区域。任何矩阵以外的区域都视为水域。

如果 grid2 的一个岛屿，被 grid1 的一个岛屿 完全 包含，也就是说 grid2 中该岛屿的每一个格子都被 grid1 中同一个岛屿完全包含，那么我们称 grid2 中的这个岛屿为 子岛屿 。

请你返回 grid2 中 子岛屿 的 数目 。

```python
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        # 只有点既在2中也在1中才有效
        # 那么做一个图3.对图3dfs。如果图3dfs的面积等于图2dfs的面积，计数器+1
        m = len(grid1)
        n = len(grid1[0])
        visited2 = [[False for j in range(n)] for i in range(m)]
        visited3 = [[False for j in range(n)] for i in range(m)]

        direc = [(-1,0),(1,0),(0,1),(0,-1)]
        grid3 = [[(grid1[i][j]&grid2[i][j]) for j in range(n)] for i in range(m)]
        
        def judgeValid(grid,i,j): # 需要传入参数是哪一张表
            if 0<=i<m and 0<=j<n and grid[i][j] == 1:
                return True
            else:
                return False 
        
        def dfs(area,grid,visited,i,j): # 传入参数area用的列表，当作全局变量使用，因为我需要调用的之后改变它的值
            if judgeValid(grid,i,j) == False:
                return 
            if visited[i][j] == True:
                return 
            visited[i][j] = True
            area[0] += 1
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(area,grid,visited,new_i,new_j)

        count = 0 # 计数器
        for i in range(m):
            for j in range(n):
                area2 = [0] # 调用前重置
                area3 = [0]
                dfs(area2,grid2,visited2,i,j)
                dfs(area3,grid3,visited3,i,j)
                if area3 == area2 and area2 != [0]:
                    count += 1 
        return count
```

# 1914. 循环轮转矩阵

给你一个大小为 m x n 的整数矩阵 grid ，其中 m 和 n 都是 偶数 ；另给你一个整数 k 。

矩阵由若干层组成，如下图所示，每种颜色代表一层：

```python
class Solution:
    def rotateGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        # 模拟，体力活，抽出条状
        m,n = len(grid),len(grid[0])
        limitCol = m//2
        limitRow = n//2

        def rotate(start_i,start_j): # 抽成带状，重新填充
            temp = []
            for j in range(start_j,n-start_j):
                temp.append(grid[start_i][j])
            temp.pop() # 除去右上角
            for i in range(start_i,m-start_i):
                temp.append(grid[i][n-start_j-1])
            temp.pop() # 除去右下角
            for j in range(n-start_j-1,start_j-1,-1):
                temp.append(grid[m-start_i-1][j])
            temp.pop() # 除去左下角
            for i in range(m-start_i-1,start_i-1,-1):
                temp.append(grid[i][start_j])
            temp.pop() # 除去左上角
            
            length = len(temp)
            times = k%length
            temp = temp[times:]+temp[:times]

            # 重新填充
            p = 0
            for j in range(start_j,n-start_j):
                grid[start_i][j] = temp[p]
                p += 1
            p -= 1
            for i in range(start_i,m-start_i):
                grid[i][n-start_j-1] = temp[p]
                p += 1           
            p -= 1
            for j in range(n-start_j-1,start_j-1,-1): 
                grid[m-start_i-1][j] = temp[p]
                p += 1
            p -= 1
            for i in range(m-start_i-1,start_i,-1): # 注意这个的收尾变化
                grid[i][start_j] = temp[p]
                p += 1   
                    
        now_x = 0
        now_y = 0
        while now_x < limitRow and now_y < limitCol:
            rotate(now_x,now_y)
            now_x += 1
            now_y += 1
        return grid

```

# 1926. 迷宫中离入口最近的出口

给你一个 m x n 的迷宫矩阵 maze （下标从 0 开始），矩阵中有空格子（用 '.' 表示）和墙（用 '+' 表示）。同时给你迷宫的入口 entrance ，用 entrance = [entrancerow, entrancecol] 表示你一开始所在格子的行和列。

每一步操作，你可以往 上，下，左 或者 右 移动一个格子。你不能进入墙所在的格子，你也不能离开迷宫。你的目标是找到离 entrance 最近 的出口。出口 的含义是 maze 边界 上的 空格子。entrance 格子 不算 出口。

请你返回从 entrance 到最近出口的最短路径的 步数 ，如果不存在这样的路径，请你返回 -1 。

```python
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        # 入口不算出口,BFS找到第一个边界点
        m = len(maze)
        n = len(maze[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        visited[entrance[0]][entrance[1]] = True 
        queue = [(entrance[0],entrance[1])]

        steps = 0
        while len(queue) != 0:
            new_queue = []
            for i,j in queue:
                if (i==0 or i ==m-1 or j == 0 or j == n-1) and (i,j) != (entrance[0],entrance[1]):
                    return steps
                for di in direc:
                    new_i = i + di[0]
                    new_j = j + di[1]
                    if 0<=new_i<m and 0<=new_j<n and visited[new_i][new_j] == False and maze[new_i][new_j] == ".":
                        visited[new_i][new_j] = True 
                        new_queue.append((new_i,new_j))
            steps += 1
            queue = new_queue
        return -1
```

# 1976. 到达目的地的方案数

你在一个城市里，城市由 n 个路口组成，路口编号为 0 到 n - 1 ，某些路口之间有 双向 道路。输入保证你可以从任意路口出发到达其他任意路口，且任意两个路口之间最多有一条路。

给你一个整数 n 和二维整数数组 roads ，其中 roads[i] = [ui, vi, timei] 表示在路口 ui 和 vi 之间有一条需要花费 timei 时间才能通过的道路。你想知道花费 最少时间 从路口 0 出发到达路口 n - 1 的方案数。

请返回花费 最少时间 到达目的地的 路径数目 。由于答案可能很大，将结果对 109 + 7 取余 后返回。

```python
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        # 先dj求出0到任意点点时间花费
        # dj算法,注意最终答案记得取模
        graph = collections.defaultdict(list)
        for a,b,c in roads:
            graph[a].append((c,b))
            graph[b].append((c,a))

        distance = [float("inf") for i in range(n)]
        queue = [(0,0)] # （距离，节点编号）
        distance[0] = 0
        dp = [0 for i in range(n)] # dp数组dp[i]的意思是到
        dp[0] = 1

        while len(queue) != 0:
            nowDistance,cur = heapq.heappop(queue)
            if distance[cur] < nowDistance:
                continue
            for node in graph[cur]:
                addDistance = node[0]
                neigh = node[1]
                if distance[neigh] > addDistance + nowDistance:
                    distance[neigh] = addDistance + nowDistance
                    dp[neigh] = dp[cur] # 继承且重置
                    heapq.heappush(queue,(addDistance + nowDistance,neigh))
                elif distance[neigh] == addDistance + nowDistance:
                    dp[neigh] += dp[cur] # 累加
        
        # print(distance)
        # print(dp)
        # 此时的distance数组表示起点到任意点的最短时间
        # 进一步用动态规划
        return dp[-1]%(10**9+7) 

```

```python
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        # 精简版本
        distance = [float("inf") for i in range(n)]
        queue = [(0,0)]
        graph = collections.defaultdict(list)
        for a,b,c in roads:
            graph[a].append((c,b))
            graph[b].append((c,a))
        dp = [0 for i in range(n)]
        dp[0] = 1
        while len(queue) != 0:
            nowDistance,cur = heapq.heappop(queue)
            if distance[cur] < nowDistance:
                continue 
            for node in graph[cur]:
                addDistance = node[0]
                neigh = node[1]
                if distance[neigh] > addDistance + nowDistance:
                    distance[neigh] = addDistance + nowDistance
                    dp[neigh] = dp[cur]
                    heapq.heappush(queue,(addDistance+nowDistance,neigh))
                elif distance[neigh] == addDistance + nowDistance:
                    dp[neigh] += dp[cur]
        return dp[-1]%(10**9+7)
```



# 1992. 找到所有的农场组

给你一个下标从 0 开始，大小为 m x n 的二进制矩阵 land ，其中 0 表示一单位的森林土地，1 表示一单位的农场土地。

为了让农场保持有序，农场土地之间以矩形的 农场组 的形式存在。每一个农场组都 仅 包含农场土地。且题目保证不会有两个农场组相邻，也就是说一个农场组中的任何一块土地都 不会 与另一个农场组的任何一块土地在四个方向上相邻。

land 可以用坐标系统表示，其中 land 左上角坐标为 (0, 0) ，右下角坐标为 (m-1, n-1) 。请你找到所有 农场组 最左上角和最右下角的坐标。一个左上角坐标为 (r1, c1) 且右下角坐标为 (r2, c2) 的 农场组 用长度为 4 的数组 [r1, c1, r2, c2] 表示。

请你返回一个二维数组，它包含若干个长度为 4 的子数组，每个子数组表示 land 中的一个 农场组 。如果没有任何农场组，请你返回一个空数组。可以以 任意顺序 返回所有农场组。

```python
class Solution:
    def findFarmland(self, land: List[List[int]]) -> List[List[int]]:
        # 调用dfs改变
        m = len(land)
        n = len(land[0])
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        # 每次搜索把所有值都置0,可以dfs的时候就计数1次
        ans = []
        def judgeValid(i,j):
            if 0<=i<m and 0<=j<n and land[i][j] == 1:
                return True
            else:
                return False 

        def dfs(i,j):
            nonlocal tempMax_i
            nonlocal tempMax_j
            if not judgeValid(i,j):
                return 
            land[i][j] = 0
            tempMax_i = max(tempMax_i,i) # 记录过程中达到的右下角
            tempMax_j = max(tempMax_j,j)
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(new_i,new_j)
        
        for i in range(m):
            for j in range(n):
                if land[i][j] == 1:
                    # 每次调用需要重置
                    tempMax_i = 0
                    tempMax_j = 0
                    dfs(i,j)
                    ans.append([i,j,tempMax_i,tempMax_j])
        return ans 

```

# [5846. 找到数组的中间位置](https://leetcode-cn.com/problems/find-the-middle-index-in-array/)

给你一个下标从 0 开始的整数数组 nums ，请你找到 最左边 的中间位置 middleIndex （也就是所有可能中间位置下标最小的一个）。

中间位置 middleIndex 是满足 nums[0] + nums[1] + ... + nums[middleIndex-1] == nums[middleIndex+1] + nums[middleIndex+2] + ... + nums[nums.length-1] 的数组下标。

如果 middleIndex == 0 ，左边部分的和定义为 0 。类似的，如果 middleIndex == nums.length - 1 ，右边部分的和定义为 0 。

请你返回满足上述条件 最左边 的 middleIndex ，如果不存在这样的中间位置，请你返回 -1 。

```python
class Solution:
    def findMiddleIndex(self, nums: List[int]) -> int:
        # 直接强行搜
        leftWindow = 0
        rightWindow = sum(nums)
        # 不算自身的位置
        ans = -1 # 默认
        for i in range(len(nums)):
            # 先退right
            rightWindow -= nums[i]
            if leftWindow == rightWindow:
                ans = i 
                return ans 
            leftWindow += nums[i]
        return ans
```

# [5859. 差的绝对值为 K 的数对数目](https://leetcode-cn.com/problems/count-number-of-pairs-with-absolute-difference-k/)

给你一个整数数组 nums 和一个整数 k ，请你返回数对 (i, j) 的数目，满足 i < j 且 |nums[i] - nums[j]| == k 。

|x| 的值定义为：

如果 x >= 0 ，那么值为 x 。
如果 x < 0 ，那么值为 -x 。

```python
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i+1,n):
                if abs(nums[i]-nums[j]) == k:
                    ans += 1
        return ans
```

# [5860. 从双倍数组中还原原数组](https://leetcode-cn.com/problems/find-original-array-from-doubled-array/)

一个整数数组 original 可以转变成一个 双倍 数组 changed ，转变方式为将 original 中每个元素 值乘以 2 加入数组中，然后将所有元素 随机打乱 。

给你一个数组 changed ，如果 change 是 双倍 数组，那么请你返回 original数组，否则请返回空数组。original 的元素可以以 任意 顺序返回。

```python
class Solution:
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        n = len(changed)
        if n % 2 == 1:
            return []
        if sum(changed) == 0:
            return [0 for i in range(n//2)]
        changed.sort()
        ct = collections.Counter(changed)

        ans = []
        for key in ct:
            if key == 0:
                if ct[key] % 2 != 0:
                    return []
                else:
                    times = ct[key]//2
                    ct[key] = 0
                    for i in range(times):
                        ans.append(key)
                    continue
            if ct[key] != 0:
                if key * 2 in ct:  
                    times = ct[key]
                    for i in range(times):
                        ans.append(key)
                    ct[key*2] -= ct[key]
                    ct[key] = 0

        for key in ct:
            if ct[key] != 0:
                return []
        return ans
```

# [5861. 出租车的最大盈利](https://leetcode-cn.com/problems/maximum-earnings-from-taxi/)

你驾驶出租车行驶在一条有 n 个地点的路上。这 n 个地点从近到远编号为 1 到 n ，你想要从 1 开到 n ，通过接乘客订单盈利。你只能沿着编号递增的方向前进，不能改变方向。

乘客信息用一个下标从 0 开始的二维数组 rides 表示，其中 rides[i] = [starti, endi, tipi] 表示第 i 位乘客需要从地点 starti 前往 endi ，愿意支付 tipi 元的小费。

每一位 你选择接单的乘客 i ，你可以 盈利 endi - starti + tipi 元。你同时 最多 只能接一个订单。

给你 n 和 rides ，请你返回在最优接单方案下，你能盈利 最多 多少元。

注意：你可以在一个地点放下一位乘客，并在同一个地点接上另一位乘客。

```python
class Solution:
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        # 动态规划
        # pair = [a,b,c]
        # 状态转移方程为dp[b] = max(dp[b],dp[a]+b-a+c)
        # dp[i]的意思是，以i为终点为止，可以获得的最大收益
        rides.sort()
        dp = [0 for i in range(n+1)]
        i = 0
        for a,b,c in rides:
            while i < a: # 填充所有a之前的数
                dp[i+1] = max(dp[i],dp[i+1])
                i += 1
            dp[b] = max(dp[b],dp[a]+b-a+c)
        return max(dp)
```

# [5863. 统计特殊四元组](https://leetcode-cn.com/problems/count-special-quadruplets/)

给你一个 下标从 0 开始 的整数数组 nums ，返回满足下述条件的 不同 四元组 (a, b, c, d) 的 数目 ：

nums[a] + nums[b] + nums[c] == nums[d] ，且
a < b < c < d

```python
class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        # 。。。暴力强搜
        ans = 0
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                for k in range(j+1,n):
                    for d in range(k+1,n):
                        if nums[i]+nums[j]+nums[k] == nums[d]:
                            ans += 1
        return ans 
```

# [5864. 游戏中弱角色的数量](https://leetcode-cn.com/problems/the-number-of-weak-characters-in-the-game/)

你正在参加一个多角色游戏，每个角色都有两个主要属性：攻击 和 防御 。给你一个二维整数数组 properties ，其中 properties[i] = [attacki, defensei] 表示游戏中第 i 个角色的属性。

如果存在一个其他角色的攻击和防御等级 都严格高于 该角色的攻击和防御等级，则认为该角色为 弱角色 。更正式地，如果认为角色 i 弱于 存在的另一个角色 j ，那么 attackj > attacki 且 defensej > defensei 。

返回 弱角色 的数量。

```python
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        properties.sort(key = lambda x:(x[0],-x[1]))
        # 数据量需要n1
        # 单调栈,单调递减栈。有递增的则处理，处理的时候数值+1
        stack = []
        count = 0
        for cp in properties:
            if len(stack) == 0:
                stack.append(cp[1])
            elif stack[-1] >= cp[1]:
                stack.append(cp[1])
            elif stack[-1] < cp[1]:
                while len(stack) > 0 and stack[-1] < cp[1]:
                    stack.pop()
                    count += 1
                stack.append(cp[1])
        return count
```

# [5867. 反转单词前缀](https://leetcode-cn.com/problems/reverse-prefix-of-word/)

给你一个下标从 0 开始的字符串 word 和一个字符 ch 。找出 ch 第一次出现的下标 i ，反转 word 中从下标 0 开始、直到下标 i 结束（含下标 i ）的那段字符。如果 word 中不存在字符 ch ，则无需进行任何操作。

例如，如果 word = "abcdefd" 且 ch = "d" ，那么你应该 反转 从下标 0 开始、直到下标 3 结束（含下标 3 ）。结果字符串将会是 "dcbaefd" 。
返回 结果字符串 。

```python
class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        start = None
        for i in range(len(word)):
            if word[i] == ch:
                start = i+1
                break
        if start == None:
            return word
        sli = word[:start]
        sli = sli[::-1]
        ans = sli + word[start:]
        return ans
```

# [5868. 可互换矩形的组数](https://leetcode-cn.com/problems/number-of-pairs-of-interchangeable-rectangles/)

用一个下标从 0 开始的二维整数数组 rectangles 来表示 n 个矩形，其中 rectangles[i] = [widthi, heighti] 表示第 i 个矩形的宽度和高度。

如果两个矩形 i 和 j（i < j）的宽高比相同，则认为这两个矩形 可互换 。更规范的说法是，两个矩形满足 widthi/heighti == widthj/heightj（使用实数除法而非整数除法），则认为这两个矩形 可互换 。

计算并返回 rectangles 中有多少对 可互换 矩形。

```python
class Solution:
    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        # 约分之之后统计
        def findGCD(a,b):
            while a != 0:
                temp = a
                a = b%a
                b = temp
            return b
        
        for i in range(len(rectangles)):
            x,y = rectangles[i]
            gcd = findGCD(x,y)
            x,y = x//gcd,y//gcd
            rectangles[i] = [x,y]

        theDict = collections.defaultdict(int)
        for x,y in rectangles:
            theDict[x,y] += 1

        ans = 0

        for key in theDict:
            e = theDict[key] 
            ans += e*(e-1)//2  

        return ans             
```

# [5869. 两个回文子序列长度的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/)

给你一个字符串 s ，请你找到 s 中两个 不相交回文子序列 ，使得它们长度的 乘积最大 。两个子序列在原字符串中如果没有任何相同下标的字符，则它们是 不相交 的。

请你返回两个回文子序列长度可以达到的 最大乘积 。

子序列 指的是从原字符串中删除若干个字符（可以一个也不删除）后，剩余字符不改变顺序而得到的结果。如果一个字符串从前往后读和从后往前读一模一样，那么这个字符串是一个 回文字符串 。

```python
class Solution:
    def maxProduct(self, s: str) -> int:
        # 三路径回溯？
        path1 = []
        path2 = []
        path3 = []
        ans = 0

        def backtracking(p1,p2,p3,index):
            nonlocal ans
            if index == len(s):
                if p2 == p2[::-1] and p3 == p3[::-1] and p2 != [] and p3 != []:
                    ans = max(ans,len(p2)*len(p3))
                return            

            p1.append(s[index])
            backtracking(p1,p2,p3,index+1)
            p1.pop()

            p2.append(s[index])
            backtracking(p1,p2,p3,index+1)
            p2.pop()

            p3.append(s[index])
            backtracking(p1,p2,p3,index+1)
            p3.pop()
        
        backtracking(path1,path2,path3,0)
        return ans
```

# [5871. 将一维数组转变成二维数组](https://leetcode-cn.com/problems/convert-1d-array-into-2d-array/)

给你一个下标从 0 开始的一维整数数组 original 和两个整数 m 和  n 。你需要使用 original 中 所有 元素创建一个 m 行 n 列的二维数组。

original 中下标从 0 到 n - 1 （都 包含 ）的元素构成二维数组的第一行，下标从 n 到 2 * n - 1 （都 包含 ）的元素构成二维数组的第二行，依此类推。

请你根据上述过程返回一个 m x n 的二维数组。如果无法构成这样的二维数组，请你返回一个空的二维数组。

```python
class Solution:
    def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
        if len(original) != m*n:
            return []
        ans = []
        # n是列的数目
        p = 0
        while p < len(original):
            ans.append(original[p:p+n])
            p += n
        return ans
        
```

# [5872. 连接后等于目标字符串的字符串对](https://leetcode-cn.com/problems/number-of-pairs-of-strings-with-concatenation-equal-to-target/)

给你一个 数字 字符串数组 nums 和一个 数字 字符串 target ，请你返回 nums[i] + nums[j] （两个字符串连接）结果等于 target 的下标 (i, j) （需满足 i != j）的数目。

```python
class Solution:
    def numOfPairs(self, nums: List[str], target: str) -> int:
        ans = 0
        n = len(nums)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if nums[i]+nums[j] == target:
                    ans += 1
        return ans
```



# [5873. 考试的最大困扰度](https://leetcode-cn.com/problems/maximize-the-confusion-of-an-exam/)

一位老师正在出一场由 n 道判断题构成的考试，每道题的答案为 true （用 'T' 表示）或者 false （用 'F' 表示）。老师想增加学生对自己做出答案的不确定性，方法是 最大化 有 连续相同 结果的题数。（也就是连续出现 true 或者连续出现 false）。

给你一个字符串 answerKey ，其中 answerKey[i] 是第 i 个问题的正确结果。除此以外，还给你一个整数 k ，表示你能进行以下操作的最多次数：

每次操作中，将问题的正确答案改为 'T' 或者 'F' （也就是将 answerKey[i] 改为 'T' 或者 'F' ）。
请你返回在不超过 k 次操作的情况下，最大 连续 'T' 或者 'F' 的数目。

```python
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        # 两次dp，一次对Tdp，一次对Fdp
        left = 0
        right = 0
        size = 0
        max_size = 0
        countZero = 0
        while right < len(answerKey):
            add = answerKey[right]
            right += 1
            if add != "T":
                size += 1
                max_size = max(max_size,size)
            elif add == "T":
                size += 1
                countZero += 1
                if countZero <= k:
                    max_size = max(max_size,size)
            while left < right and countZero > k:
                delete = answerKey[left]
                if delete == "T":
                    countZero -= 1
                size -= 1
                left += 1
        
        left = 0
        right = 0
        size = 0    
        countZero = 0
        while right < len(answerKey):
            add = answerKey[right]
            right += 1
            if add != "F":
                size += 1
                max_size = max(max_size,size)
            elif add == "F":
                size += 1
                countZero += 1
                if countZero <= k:
                    max_size = max(max_size,size)
            while left < right and countZero > k:
                delete = answerKey[left]
                if delete == "F":
                    countZero -= 1
                size -= 1
                left += 1
        
        return max_size
```

# [5875. 执行操作后的变量值](https://leetcode-cn.com/problems/final-value-of-variable-after-performing-operations/)

存在一种仅支持 4 种操作和 1 个变量 X 的编程语言：

++X 和 X++ 使变量 X 的值 加 1
--X 和 X-- 使变量 X 的值 减 1
最初，X 的值是 0

给你一个字符串数组 operations ，这是由操作组成的一个列表，返回执行所有操作后， X 的 最终值 。

```python
class Solution:
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        ans = 0
        theDict = dict()
        theDict["X++"] = 1
        theDict["++X"] = 1
        theDict["--X"] = -1
        theDict["X--"] = -1
        for ch in operations:
            ans += theDict[ch]
        return ans
        
```

# [5876. 数组美丽值求和](https://leetcode-cn.com/problems/sum-of-beauty-in-the-array/)

给你一个下标从 0 开始的整数数组 nums 。对于每个下标 i（1 <= i <= nums.length - 2），nums[i] 的 美丽值 等于：

2，对于所有 0 <= j < i 且 i < k <= nums.length - 1 ，满足 nums[j] < nums[i] < nums[k]
1，如果满足 nums[i - 1] < nums[i] < nums[i + 1] ，且不满足前面的条件
0，如果上述条件全部不满足
返回符合 1 <= i <= nums.length - 2 的所有 nums[i] 的 美丽值的总和 。

```python
class Solution:
    def sumOfBeauties(self, nums: List[int]) -> int:
        ans = 0
        n = len(nums)
        tempMax = nums[0]
        cpList = [0 for i in range(n)] # 前缀最大值
        for t in range(n):
            if tempMax < nums[t]:
                tempMax = nums[t]
            cpList[t] = tempMax
        tempMin = nums[-1]
        cpList2 = [0 for i in range(n)] # 后缀最小值
        p = n-1
        while p >= 0:
            if tempMin > nums[p]:
                tempMin = nums[p]
            cpList2[p] = tempMin
            p -= 1
        
        for i in range(1,n-1):
            if cpList[i-1] < nums[i] < cpList2[i+1]:
                # print("a")
                ans += 2
            elif nums[i-1] < nums[i] < nums[i+1]:
                # print('b')
                ans += 1

        # print(cpList)
        # print(cpList2)
        return ans
                
```

# [5877. 检测正方形](https://leetcode-cn.com/problems/detect-squares/)

给你一个在 X-Y 平面上的点构成的数据流。设计一个满足下述要求的算法：

添加 一个在数据流中的新点到某个数据结构中。可以添加 重复 的点，并会视作不同的点进行处理。
给你一个查询点，请你从数据结构中选出三个点，使这三个点和查询点一同构成一个 面积为正 的 轴对齐正方形 ，统计 满足该要求的方案数目。
轴对齐正方形 是一个正方形，除四条边长度相同外，还满足每条边都与 x-轴 或 y-轴 平行或垂直。

实现 DetectSquares 类：

DetectSquares() 使用空数据结构初始化对象
void add(int[] point) 向数据结构添加一个新的点 point = [x, y]
int count(int[] point) 统计按上述方式与点 point = [x, y] 共同构造 轴对齐正方形 的方案数。

```python
class DetectSquares:

    def __init__(self):
        # 注意分为直线 y = x + k 和 直线 y = -x - b
        # 不是要求找矩形的时候点在左下方。卡了一个小时
        self.pointDict = dict()
        self.gapDict = collections.defaultdict(dict)
        self.gapDict2 = collections.defaultdict(dict)


    def add(self, point: List[int]) -> None:
        x = point[0]
        y = point[1]
        if self.pointDict.get((x,y)) == None:
            self.pointDict[(x,y)] = 1
        else:
            self.pointDict[(x,y)] += 1
            
        gap = x - y
                    
        if self.gapDict[gap].get((x,y)) == None:
            self.gapDict[gap][(x,y)] = 1
        elif self.gapDict[gap].get((x,y)) != None:
            self.gapDict[gap][(x,y)] += 1
        
        gap2 = x + y
        if self.gapDict2[gap2].get((x,y)) == None:
            self.gapDict2[gap2][(x,y)] = 1
        elif self.gapDict2[gap2].get((x,y)) != None:
            self.gapDict2[gap2][(x,y)] += 1
        
                    

    def count(self, point: List[int]) -> int:
        x = point[0]
        y = point[1]
        gap = x - y
        gap2 = x + y
        ans = 0

        for tx,ty in self.gapDict[gap]:
            if tx == x or ty == y:
                continue 
            c1 = self.gapDict[gap][(tx,ty)]
            p = x - tx 
            c2 = self.pointDict.get((x-p,y))
            c3 = self.pointDict.get((x,y-p))

            if c1 != None and c2 != None and c3 != None:
                ans += c1*c2*c3

        for tx,ty in self.gapDict2[gap2]:
            if tx == x or ty == y:
                continue 
            c1 = self.gapDict2[gap2][(tx,ty)]
 
            c2 = self.pointDict.get((tx,y))
            c3 = self.pointDict.get((x,ty))

            if c1 != None and c2 != None and c3 != None:
                ans += c1*c2*c3
        
        return ans
```

# [5881. 增量元素之间的最大差值](https://leetcode-cn.com/problems/maximum-difference-between-increasing-elements/)

给你一个下标从 0 开始的整数数组 nums ，该数组的大小为 n ，请你计算 nums[j] - nums[i] 能求得的 最大差值 ，其中 0 <= i < j < n 且 nums[i] < nums[j] 。

返回 最大差值 。如果不存在满足要求的 i 和 j ，返回 -1 。 

```python
class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        maxDiff = -1
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                if nums[i] < nums[j]:
                    maxDiff = max(maxDiff,nums[j]-nums[i])
        return maxDiff
```

# [5890. 转换字符串的最少操作次数](https://leetcode-cn.com/problems/minimum-moves-to-convert-string/)

给你一个字符串 s ，由 n 个字符组成，每个字符不是 'X' 就是 'O' 。

一次 操作 定义为从 s 中选出 三个连续字符 并将选中的每个字符都转换为 'O' 。注意，如果字符已经是 'O' ，只需要保持 不变 。

返回将 s 中所有字符均转换为 'O' 需要执行的 最少 操作次数。

```
class Solution:
    def minimumMoves(self, s: str) -> int:
        ops = 0
        p = 0
        n = len(s)
        while p < n:
            if s[p] == "X":
                p += 3
                ops += 1
            else:
                p += 1
        return ops
```

# [5891. 找出缺失的观测数据](https://leetcode-cn.com/problems/find-missing-observations/)

现有一份 n + m 次投掷单个 六面 骰子的观测数据，骰子的每个面从 1 到 6 编号。观测数据中缺失了 n 份，你手上只拿到剩余 m 次投掷的数据。幸好你有之前计算过的这 n + m 次投掷数据的 平均值 。

给你一个长度为 m 的整数数组 rolls ，其中 rolls[i] 是第 i 次观测的值。同时给你两个整数 mean 和 n 。

返回一个长度为 n 的数组，包含所有缺失的观测数据，且满足这 n + m 次投掷的 平均值 是 mean 。如果存在多组符合要求的答案，只需要返回其中任意一组即可。如果不存在答案，返回一个空数组。

k 个数字的 平均值 为这些数字求和后再除以 k 。

注意 mean 是一个整数，所以 n + m 次投掷的总和需要被 n + m 整除。

```python
class Solution:
    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        m = len(rolls)
        allNum = mean * (m+n)
        s = sum(rolls)
        remain = allNum - s 
        ans = []
        times = remain//n 
        avg = remain//n
        el = remain%n
        # print(remain,times,el)
        for i in range(n-el):
            ans.append(avg)
        for i in range(el):
            ans.append(avg+1)
        for i in ans:
            if i > 6 or i < 1:
                return []
        return ans
```


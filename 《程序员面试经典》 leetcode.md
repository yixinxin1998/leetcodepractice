# 《程序员面试经典》 leetcode

# 面试题 01.01. 判定字符是否唯一

实现一个算法，确定一个字符串 `s` 的所有字符是否全都不同。

```python
class Solution:
    def isUnique(self, astr: str) -> bool:
        # 不使用额外的数据结构
        # 方法1:申请一个bool数组长度为256，按照字符索引构造映射 时间On
        # 方法2:排序，看上一位是否和下一位不同
        temp = sorted(astr)
        if len(temp)<=1: # 长度小于等于1位一定是True
            return True
        p = 1
        while p < len(temp):
            if temp[p] == temp[p-1]:
                return False
            p += 1
        return True
```

# 面试题 01.02. 判定是否互为字符重排

给定两个字符串 `s1` 和 `s2`，请编写一个程序，确定其中一个字符串的字符重新排列后，能否变成另一个字符串。

```python
class Solution:
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False
        dict1 = collections.defaultdict(int)
        dict2 = collections.defaultdict(int)
        for i in s1:
            dict1[i] += 1
        for i in s2:
            dict2[i] += 1
        return dict1 == dict2
```

# 面试题 01.03. URL化

URL化。编写一种方法，将字符串中的空格全部替换为%20。假定该字符串尾部有足够的空间存放新增字符，并且知道字符串的“真实”长度。（注：用Java实现的话，请使用字符数组实现，以便直接在数组上操作。）

```python
class Solution:
    def replaceSpaces(self, S: str, length: int) -> str:
        S = list(S)
        ans = ''
        p = 0
        while p < length:
            if S[p] == ' ':
                ans += '%20'
            else:
                ans += S[p]
            p += 1
        return ans
```

# 面试题 01.04. 回文排列

给定一个字符串，编写一个函数判定其是否为某个回文串的排列之一。

回文串是指正反两个方向都一样的单词或短语。排列是指字母的重新排列。

回文串不一定是字典当中的单词。

```python
class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        # 其中奇数频次字符不大于1
        dict1 = collections.defaultdict(int)
        for i in s: # 初始化字典
            dict1[i] += 1
        # 检查字典
        count = 0
        print(dict1)
        for i in dict1:
            if dict1[i]%2 == 1:
                count += 1
            if count >= 2:
                return False
        return True
```

# 面试题 01.05. 一次编辑

字符串有三种编辑操作:插入一个字符、删除一个字符或者替换一个字符。 给定两个字符串，编写一个函数判定它们是否只需要一次(或者零次)编辑。

```python
class Solution:
    def oneEditAway(self, first: str, second: str) -> bool:
        # 直接计算编辑距离。。。
        m = len(first)
        n = len(second)
        dp = [[0xffffffff for j in range(n+1)] for i in range(m+1)] # 初始化为极大值
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1,m+1):
            for j in range(1,n+1):
                if first[i-1] == second[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                elif first[i-1] != second[j-1]:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,
                        dp[i][j-1] + 1,
                        dp[i-1][j-1] + 1
                    )

        return dp[-1][-1] <= 1
```



# 面试题 01.06. 字符串压缩

字符串压缩。利用字符重复出现的次数，编写一种方法，实现基本的字符串压缩功能。比如，字符串aabcccccaaa会变为a2b1c5a3。若“压缩”后的字符串没有变短，则返回原先的字符串。你可以假设字符串中只包含大小写英文字母（a至z）。

```python
class Solution:
    def compressString(self, S: str) -> str:
        if len(S) == 0:
            return ''
        origin_lenth = len(S)
        # 扫描指针+填充指针
        lst = [] # 接收列表
        p = 0
        mark = S[p]
        times = 0 # 记录重复次数
        while p < len(S):
            if S[p] == mark:
                times += 1
                p += 1
            elif S[p] != mark:
                # 收集结果
                lst.append(mark+str(times))
                mark = S[p] # 重置
                times = 1 # 重置
                p += 1
        # 最后一次没有切换状态，没有进入结果收集，处理尾巴
        lst.append(mark+str(times))
        ans = ''.join(lst)
        if len(ans) < origin_lenth:
            return ans
        else:
            return S
```

# 面试题 01.07. 旋转矩阵

给你一幅由 `N × N` 矩阵表示的图像，其中每个像素的大小为 4 字节。请你设计一种算法，将图像旋转 90 度。

不占用额外内存空间能否做到？

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # 翻转前一半的行，然后做关于对角线的交换
        # 其是方阵
        n = len(matrix)
        for i in range(n//2):
            matrix[i],matrix[n-i-1] = matrix[n-i-1],matrix[i]
        # 转化对角线
        for i in range(n):
            for j in range(i,n): # 注意这个起点
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
```

# 面试题 01.08. 零矩阵

编写一种算法，若M × N矩阵中某个元素为0，则将其所在的行与列清零。

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        if matrix == []: # 不合理情况排除
            return 
        # 第一轮先记录所有0的横纵坐标
        record_row = set()
        record_col = set()
        for i in range(len(matrix[0])): # 有几列
            for j in range(len(matrix)): # 有几行
                if matrix[j][i] == 0:
                    record_row.add(j) # 记录行序号
                    record_col.add(i) # 记录列序号
        # 记录哪些行有0
        # 记录哪些列有0
        # 可以提高效率
        for x in record_row: # 把标记的行序号的那一行置0
            matrix[x] = [0]*len(matrix[0])
        for y in record_col: # 把标记列序号的那一列置0
            for q in range(len(matrix)):
                matrix[q][y] = 0
```



# 面试题 01.09. 字符串轮转

字符串轮转。给定两个字符串`s1`和`s2`，请编写代码检查`s2`是否为`s1`旋转而成（比如，`waterbottle`是`erbottlewat`旋转后的字符串）。

```python
class Solution:
    def isFlipedString(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False
        elif len(s1) == len(s2):
            s1 = s1 + s1
            if s2 in s1: # 这里了采用了内部的KMP
                return True
            else:
                return False
```

# 面试题 02.01. 移除重复节点

编写代码，移除未排序链表中的重复节点。保留最开始出现的节点。

```python
class Solution:
    def removeDuplicateNodes(self, head: ListNode) -> ListNode:
        # 用一个集合记录已经访问过的值，如果访问值存在，则查询下一个值
        visited = set()
        # 用一个列表记录有效节点 或者 直接用双指针        
        # 用哑节点简化边界条件处理
        dummy = ListNode(-1) # 初始化一个不存在的元素
        dummy.next = head
        cur1 = dummy
        cur2 = dummy.next
        while cur2 != None:
            if cur2.val in visited:
                cur2 = cur2.next
                cur1.next = cur2
            elif cur2.val not in visited:
                visited.add(cur2.val)
                cur1.next = cur2
                cur2 = cur2.next
                cur1 = cur1.next
        return dummy.next
```

# 面试题 02.02. 返回倒数第 k 个节点

实现一种算法，找出单向链表中倒数第 k 个节点。返回该节点的值。给定的 *k* 保证是有效的。

```python
class Solution:
    def kthToLast(self, head: ListNode, k: int) -> int:
        fast = head
        while k >= 1:
            fast = fast.next
            k -= 1
        slow = head
        while fast != None:
            fast = fast.next
            slow = slow.next
        return slow.val
```

# 面试题 02.03. 删除中间节点

若链表中的某个节点，既不是链表头节点，也不是链表尾节点，则称其为该链表的「中间节点」。

假定已知链表的某一个中间节点，请实现一种算法，将该节点从链表中删除。

例如，传入节点 c（位于单向链表 a->b->c->d->e->f 中），将其删除后，剩余链表为 a->b->d->e->f

```python
class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        #链表：a-b-c-d-e 删c；
        #只知道这个结点c，说明知道这个结点c的地址，但无法删除这个结点c；
        #这时，我们可以考虑用一个能删的替换这个一定不能删的结点c；
        #即用下一个结点d的数据域覆盖这个不能删的结点c的数据域，此时链表变成：
        #a-b-d-d-e
        #这个结点c在逻辑上就成了下一个结点d；
        #此时只要删掉第二个结点，即相当于实现了删除；

        node.val = node.next.val
        node.next = node.next.next
```

# 面试题 02.04. 分割链表

给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。

```python
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # 创建两个链，一个小链，一个大于等于链
        # 创建两个链的哑节点
        small = ListNode()
        big = ListNode()
        small_head  = small
        big_head = big
        cur = head
        while cur != None:
            if cur.val < x: # 进小链
                small.next = cur
                small = small.next
            elif cur.val >= x: # 进大链
                big.next = cur
                big = big.next
            cur = cur.next
        # 此时小链和大链已经构造，调整俩链的尾节点
        small.next = big_head.next # 
        big.next = None
        return small_head.next
```

# 面试题 02.05. 链表求和

给定两个用链表表示的整数，每个节点包含一个数位。

这些数位是反向存放的，也就是个位排在链表首部。

编写函数对这两个整数求和，并用链表形式返回结果。

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 按照数位处理原则进行加和
        # 用一个carry位表示进位
        # 创建一个新链表来存储各个数位
                # 默认进位为0，然后开始计算
        carry = 0
        temp_cur = ListNode()
        head = temp_cur
        while l1 and l2: # 两者都不为空时
            val = l1.val + l2.val
            real_val = (val + carry) % 10
            carry = (val + carry) // 10
            temp_node = ListNode(real_val)
            temp_cur.next = temp_node
            temp_cur = temp_cur.next
            l1 = l1.next
            l2 = l2.next

        # 还剩下一个栈，将所有值弹出
        lst = l1 if l1 else l2
        while lst:
            val = lst.val
            real_val = (val + carry) % 10
            carry = (val + carry) // 10
            temp_node = ListNode(real_val)
            temp_cur.next = temp_node
            temp_cur = temp_cur.next
            lst = lst.next
        # 如果还有进位 要把进位加进去
        if carry > 0:
            temp_node = ListNode(carry)
            temp_cur.next = temp_node
            temp_cur = temp_cur.next
        return head.next #
```

# 面试题 02.06. 回文链表

编写一个函数，检查输入的链表是否是回文的。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        # 快慢指针，先找到中点
        # 反转中点之后的链表
        # 再从中点开始对比
        # 处理空节点
        if head == []:
            return True
        slow = head
        fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        # 反转fast之后的链表
        cur1 = None
        cur2 = slow
        while cur2 != None:
            temp = cur2.next # 临时存储下一个节点
            cur2.next = cur1
            cur1 = cur2 # 交换完之后，让cur1指着原来的cur2，如此cur1永远指着逻辑最左边
            cur2 = temp # 交换完之后，cur2指着需要交换的下一个元素
        # 循环结束后此时的cur1之后的链表为中点反转后的链表
        # 比较cur1和头节点，循环
        compare = head
        while cur1 != None:
            if cur1.val == compare.val:
                cur1 = cur1.next
                compare = compare.next
            else:
                return False
        return True
```

# 面试题 02.07. 链表相交

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `null` 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 双指针，跳跃一次。
        # 不会跳跃第二次，因为长度 a+b = b+a
        # 如果中途碰头则为交汇点，如果末尾碰头则无交汇
        cur1 = headA
        cur2 = headB
        while cur1 != cur2:
            cur1 = cur1.next if cur1 else headB
            cur2 = cur2.next if cur2 else headA
        return cur1
```

# 面试题 02.08. 环路检测

给定一个链表，如果它是有环链表，实现一个算法返回环路的开头节点。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        # 运用到数学方法
        # 先双指针求相遇，快慢指针，速率比为1:2
        # slow = a + b
        # fast = a + b + c + b
        # 2*(a+b) = a + b + c + b
        # c = a
        # 相遇之后把任意一个指针重置成头节点，两者用相同的速度跑，再次相遇点为入环点
        slow = head
        fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if fast == slow: # 如果相遇，进入内层循环
                fast = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return fast
        return None # 如果fast越界，则无环
```

# 面试题 03.01. 三合一

三合一。描述如何只用一个数组来实现三个栈。

你应该实现push(stackNum, value)、pop(stackNum)、isEmpty(stackNum)、peek(stackNum)方法。stackNum表示栈下标，value表示压入的值。

构造函数会传入一个stackSize参数，代表每个栈的大小。

示例1:

 输入：
["TripleInOne", "push", "push", "pop", "pop", "pop", "isEmpty"]
[[1], [0, 1], [0, 2], [0], [0], [0], [0]]
 输出：
[null, null, null, 1, -1, -1, true]
说明：当栈为空时`pop, peek`返回-1，当栈满时`push`不压入元素。

```python
class TripleInOne:

    def __init__(self, stackSize: int):
        # 传入的三个栈一样大
        self.stack = [None for i in range(3*stackSize)]
        # self.stack1_limit = stackSize - 1
        # self.stack2_limit = 2 * stackSize - 1
        # self.stack3_limit = 3 * stackSize - 1
        # 游标指向三个栈的位置
        self.mark = [0,stackSize,2 * stackSize] # 将坐标集成成数组方便操作
        # mark指针指向的是要加入的那个元素的空位
        self.uplimit = [stackSize - 1,2 * stackSize - 1,3 * stackSize - 1] # 闭区间
        self.downlimit = [0,stackSize,2 * stackSize] # 不能小于这个数

    def push(self, stackNum: int, value: int) -> None: # 当栈满时`push`不压入元素。
        if self.mark[stackNum] <= self.uplimit[stackNum]: # push只需要看是否到了上界
            self.stack[self.mark[stackNum]] = value
            self.mark[stackNum] += 1    

    def pop(self, stackNum: int) -> int:
        # 游标的前一个在范围内中时
        if self.mark[stackNum]-1 >= self.downlimit[stackNum]:
            e = self.stack[self.mark[stackNum]-1]
            self.stack[self.mark[stackNum]-1] = None # 然后删除
            self.mark[stackNum] -= 1
            return e # 返回删除值
        return -1 # 删不了返回-1

    def peek(self, stackNum: int) -> int:
        if self.mark[stackNum]-1 >= self.downlimit[stackNum]:
            e = self.stack[self.mark[stackNum]-1]
            return e # 返回值
        return -1 # 越界了返回-1

    def isEmpty(self, stackNum: int) -> bool:
        return self.mark[stackNum] == self.downlimit[stackNum]
```

# 面试题 03.02. 栈的最小值

请设计一个栈，除了常规栈支持的pop与push函数以外，还支持min函数，该函数返回栈元素中的最小值。执行push、pop和min操作的时间复杂度必须为O(1)。

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        # 需要主栈，辅助栈
        # 这里不节约辅助栈空间，每次入最小栈时候，比较最小栈的栈顶和当前元素，把小的加进去
        # 这一题不需要判断操作是否有效，即不考虑empty问题？
        self.stack = []
        self.helper = [0xffffffff] # 辅助栈,初始化一个极大值，它在正常执行过程中不会为空

    def push(self, x: int) -> None:
        self.stack.append(x)
        self.helper.append(min(x,self.helper[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.helper.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.helper[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

# 面试题 03.04. 化栈为队

实现一个MyQueue类，该类用两个栈来实现一个队列。

```python
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []
        self.temp = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.stack.append(x) 
        return 

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        while len(self.stack) != 1:
            self.temp.append(self.stack.pop())
        element = self.stack.pop()
        while len(self.temp) != 0:
            self.stack.append(self.temp.pop())
        return element

    def peek(self) -> int:
        """
        Get the front element.
        """
        return self.stack[0]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.stack) == 0


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

# 面试题 03.05. 栈排序

栈排序。 编写程序，对栈进行排序使最小元素位于栈顶。最多只能使用一个其他的临时栈存放数据，但不得将元素复制到别的数据结构（如数组）中。该栈支持如下操作：push、pop、peek 和 isEmpty。当栈为空时，peek 返回 -1。

```python
class SortedStack:
    # 两个栈，主栈是已经排序好的
    # 辅助栈是用来辅助排序的
    # 如果即将入栈元素小于等于主栈栈顶，
    # 则元素直接入栈
    # 如果元素大于栈顶
    # 则主栈pop，元素转移到辅助栈，直到主栈为空或者入栈元素小于栈顶
    # 效率一般
    def __init__(self):
        self.main_stack = []
        self.support_stack = []

    def push(self, val: int) -> None:
        if len(self.main_stack) == 0 :
            self.main_stack.append(val) # 为空的时候直接入栈
            return 
        # 如果即将入栈元素小于等于主栈栈顶，
        # 则元素直接入栈
        if val <= self.main_stack[-1]:
            self.main_stack.append(val)
            return
        # 如果元素大于栈顶
        # 则主栈pop，元素转移到辅助栈，直到主栈为空或者入栈元素小于栈顶
        while len(self.main_stack)>0 and val > self.main_stack[-1]:
            self.support_stack.append(self.main_stack.pop())
        self.main_stack.append(val)
        while len(self.support_stack) != 0:
            self.main_stack.append(self.support_stack.pop())


    def pop(self) -> None:
        if self.isEmpty():
            return 
        self.main_stack.pop()

    def peek(self) -> int:
        if self.isEmpty():
            return -1
        return self.main_stack[-1]

    def isEmpty(self) -> bool:
        return len(self.main_stack) == 0


# Your SortedStack object will be instantiated and called as such:
# obj = SortedStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.peek()
# param_4 = obj.isEmpty()
```

# 面试题 03.06. 动物收容所

动物收容所。有家动物收容所只收容狗与猫，且严格遵守“先进先出”的原则。在收养该收容所的动物时，收养人只能收养所有动物中“最老”（由其进入收容所的时间长短而定）的动物，或者可以挑选猫或狗（同时必须收养此类动物中“最老”的）。换言之，收养人不能自由挑选想收养的对象。请创建适用于这个系统的数据结构，实现各种操作方法，比如enqueue、dequeueAny、dequeueDog和dequeueCat。允许使用Java内置的LinkedList数据结构。

enqueue方法有一个animal参数，animal[0]代表动物编号，animal[1]代表动物种类，其中 0 代表猫，1 代表狗。

dequeue*方法返回一个列表[动物编号, 动物种类]，若没有可以收养的动物，则返回[-1,-1]。

```python
class AnimalShelf:

    def __init__(self):
        self.dogqueue = []
        self.catqueue = []
        self.timestamp = 0  #时间戳

    def enqueue(self, animal: List[int]) -> None:
        if animal[1] == 0:
            self.catqueue.append(animal+[self.timestamp])
            self.timestamp += 1
        elif animal[1] == 1:
            self.dogqueue.append(animal+[self.timestamp])
            self.timestamp += 1

    def dequeueAny(self) -> List[int]:
        if len(self.dogqueue) == 0 and len(self.catqueue) == 0:
            return [-1,-1]
        elif len(self.dogqueue) == 0:
            return self.catqueue.pop(0)[:2]
        elif len(self.catqueue) == 0:
            return self.dogqueue.pop(0)[:2]
        
        e1 = self.catqueue[0][2]
        e2 = self.dogqueue[0][2]
        if e1 < e2:
            return self.dequeueCat()
        else:
            return self.dequeueDog()


    def dequeueDog(self) -> List[int]:
        if len(self.dogqueue) == 0:
            return [-1,-1]
        return self.dogqueue.pop(0)[:2]


    def dequeueCat(self) -> List[int]:
        if len(self.catqueue) == 0:
            return [-1,-1]
        return self.catqueue.pop(0)[:2]

```

```python
class AnimalShelf:

    def __init__(self):
        self.dogqueue = collections.deque()
        self.catqueue = collections.deque()
        self.timestamp = 0

    def enqueue(self, animal: List[int]) -> None:
        if animal[1] == 0:
            self.catqueue.append(animal+[self.timestamp])
            self.timestamp += 1
        elif animal[1] == 1:
            self.dogqueue.append(animal+[self.timestamp])
            self.timestamp += 1

    def dequeueAny(self) -> List[int]:
        if len(self.dogqueue) == 0 and len(self.catqueue) == 0:
            return [-1,-1]
        elif len(self.dogqueue) == 0:
            return self.catqueue.popleft()[:2]
        elif len(self.catqueue) == 0:
            return self.dogqueue.popleft()[:2]
        
        e1 = self.catqueue[0][2]
        e2 = self.dogqueue[0][2]
        if e1 < e2:
            return self.dequeueCat()
        else:
            return self.dequeueDog()


    def dequeueDog(self) -> List[int]:
        if len(self.dogqueue) == 0:
            return [-1,-1]
        return self.dogqueue.popleft()[:2]


    def dequeueCat(self) -> List[int]:
        if len(self.catqueue) == 0:
            return [-1,-1]
        return self.catqueue.popleft()[:2]
```



# 面试题 04.01. 节点间通路

节点间通路。给定有向图，设计一个算法，找出两个节点之间是否存在一条路径。

```python
class Solution:
    def findWhetherExistsPath(self, n: int, graph: List[List[int]], start: int, target: int) -> bool:
        # bfs
        # 先去除自环和重复边
        edges = collections.defaultdict(list)
        acc_edges = [set() for i in range(n)]

        for a,b in graph:
            if a != b and b not in acc_edges[a]:
                edges[a].append(b)
                acc_edges[a].add(b)
        
        visited = [False for i in range(n)]
        # 开始bfs
        queue = [start]
        visited[start] = True # 初始化标记为访问过
        while len(queue) != 0:
            new_queue = []
            for e in queue:
                if e == target:
                    return True   
                for neigh in edges[e]:
                    if visited[neigh] == False:
                        new_queue.append(neigh)
                        visited[neigh] = True 
            queue = new_queue
        return False
```

# 面试题 04.02. 最小高度树

给定一个有序整数数组，元素各不相同且按升序排列，编写一个算法，创建一棵高度最小的二叉搜索树。

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if len(nums) == 0:
            return None
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
```

# 面试题 04.03. 特定深度节点链表

给定一棵二叉树，设计一个算法，创建含有某一深度上所有节点的链表（比如，若一棵树的深度为 `D`，则会创建出 `D` 个链表）。返回一个包含所有深度的链表的数组。

```python
class Solution:
    def listOfDepth(self, tree: TreeNode) -> List[ListNode]:
        # BFS先得到每一层的节点，并且同时串联
        if tree == None:
            return []
        ans = [] # 存储的是链表的头节点
        queue = [tree] # BFS用到的队列
        while len(queue) != 0:
            new_queue = [] # 下一次队列做准备
            Dummy_head = ListNode(-1) # 先存哑节点
            cur = Dummy_head # 预备指针指向Dummy
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            for i in queue:
                if i.val:
                    node1 = ListNode(i.val)
                    cur.next = node1 # 给指针指向新节点
                    cur = cur.next # 移动指针
            queue = new_queue
            ans.append(Dummy_head.next)
        return ans
```



# 面试题 04.04. 检查平衡性

实现一个函数，检查二叉树是否平衡。在这个问题中，平衡树的定义如下：任意一个节点，其两棵子树的高度差不超过 1。

```python
# 复杂的前序遍历法
class Solution:
    def __init__(self):
        self.lst = []
        
    def isBalanced(self, root: TreeNode) -> bool:
        if root == None:
            return True
        ans = self.pre_order_check(root)
        count = 0
        for i in self.lst:
            if i <= 1:
                count += 1
        return count == len(self.lst) 

    def pre_order_check(self,node):
        if node == None:
            return 
        val = (self.getDepth(node.left) - self.getDepth(node.right))
        self.lst.append(abs(val))
        self.pre_order_check(node.left)
        self.pre_order_check(node.right)
    
    def getDepth(self,root): # 获取每个节点的深度
        if root == None:
            return 0
        else:
            return max(self.getDepth(root.left),self.getDepth(root.right)) + 1

```

```python
# 精简版前序遍历法
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if root == None:
            return True
        # 否则判断该节点是否平衡且左右子树是否平衡
        return abs(self.getDepth(root.left)-self.getDepth(root.right)) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)
    
    def getDepth(self,node): # 获取每个节点的深度
        if node == None:
            return 0
        leftDepth = self.getDepth(node.left)
        rightDepth = self.getDepth(node.right)
        return max(leftDepth,rightDepth) + 1
```

```python
# 后序遍历法，官方解答
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def height(root: TreeNode) -> int:
            if not root:
                return 0
            leftHeight = height(root.left)
            rightHeight = height(root.right)
            if leftHeight == -1 or rightHeight == -1 or abs(leftHeight - rightHeight) > 1:
                return -1
            else:
                return max(leftHeight, rightHeight) + 1

        return height(root) >= 0

```

# 面试题 04.05. 合法二叉搜索树

实现一个函数，检查一棵二叉树是否为二叉搜索树。

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        # 方法1: 常规中序遍历法
        inorder_lst = []
        def in_order(node):
            if node == None:
                return
            in_order(node.left)
            inorder_lst.append(node.val)
            in_order(node.right)
        # 调用方法之后开始检查
        in_order(root) 
        if len(inorder_lst) <= 1:
            return True
        # 否则挨个检查，这一题BST不允许节点值相同
        p = 1
        while p < len(inorder_lst):
            if inorder_lst[p] - inorder_lst[p-1] <= 0:
                return False
            p += 1
        return True

```

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        # 方法2： 后序遍历法
        ans = self.helper(root)
        return ans

    def helper(self,node,low_limit = -0xffffffff,up_limit = 0xffffffff):
        if node == None:
            return True
        # 看这个这个节点值是否在界线范围内
        if node.val <= low_limit or node.val >= up_limit:
            return False
        if self.helper(node.left,low_limit,node.val) == False:
            return False# 左子树上界变成这个节点的值
        if self.helper(node.right,node.val,up_limit) == False:
            return False# 右子树下界变成这个节点的值
        return True

```

# 面试题 04.06. 后继者

设计一个算法，找出二叉搜索树中指定节点的“下一个”节点（也即中序后继）。

如果指定节点没有对应的“下一个”节点，则返回`null`。

```python
class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> TreeNode:
        # 没有parent指针，直接中序遍历
        # 这一题没有考虑节点不在树中
        lst = []
        def inorder(root):
            if root == None:
                return 
            inorder(root.left)
            lst.append(root)
            inorder(root.right)
        inorder(root)
        for i in range(len(lst)):
            if lst[i] == p:
                if i != len(lst)-1:
                    return lst[i+1]
                else:
                    return None
```

# 面试题 04.08. 首个共同祖先

设计并实现一个算法，找出二叉树中某两个节点的第一个共同祖先。不得将其他的节点存储在另外的数据结构中。注意：这不一定是二叉搜索树。

```python
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # 节点都在书中，常规后序遍历
        if root == None: return root
        if root == p or root == q: return root
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)
        if left == None and right == None: return None
        if left != None and right != None: return root
        if left != None and right == None: return left
        if left == None and right != None: return right
```

# 面试题 04.10. 检查子树

检查子树。你有两棵非常大的二叉树：T1，有几万个节点；T2，有几万个节点。设计一个算法，判断 T2 是否为 T1 的子树。

如果 T1 有这么一个节点 n，其子树与 T2 一模一样，则 T2 为 T1 的子树，也就是说，从节点 n 处把树砍断，得到的树与 T2 完全相同。

```python
class Solution:
    def checkSubTree(self, t1: TreeNode, t2: TreeNode) -> bool:
        if t2 == None: # 这一题默认一个空树是任何树的子树
            return True 
        return self.check_method(t1,t2)
    
    # 一个遍历检查方法
    def check_method(self,node,t2): # node是移动的，t2是固定的
        if node == None:
            return False
        if self.isSameTree(node,t2):
            return True
        # 用or链接即可，只要有遍历到True就行
        return self.check_method(node.left,t2) or self.check_method(node.right,t2)

    # 一个判断子树方法
    def isSameTree(self,p,q):
        if p == None and q == None: # 对比完了
            return True 
        if p != None and q == None: 
            return False
        if p == None and q != None:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
```

# 面试题 04.12. 求和路径

给定一棵二叉树，其中每个节点都含有一个整数数值(该值或正或负)。设计一个算法，打印节点数值总和等于某个给定值的所有路径的数量。注意，路径不一定非得从二叉树的根节点或叶节点开始或结束，但是其方向必须向下(只能从父节点指向子节点方向)。

示例:
给定如下二叉树，以及目标和 sum = 22，

```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        # dfs两次，一次先序遍历扫路径，一次dfs扫是否有符合的结果，dfs不能剪枝
        ans = [] #收集答案
        path = [] # 收集路径
        def dfs(root,sum):
            if root == None:
                return
            path.append(root.val)
            if root.val == sum:
                ans.append(path[:])
            dfs(root.left,sum-root.val)
            dfs(root.right,sum-root.val)
            path.pop()
        def pre_order(node):
            if node == None:
                return 
            dfs(node,sum)
            pre_order(node.left)
            pre_order(node.right)
        pre_order(root)
        return len(ans)
```

# 面试题 05.03. 翻转数位

给定一个32位整数 `num`，你可以将一个数位从0变为1。请编写一个程序，找出你能够获得的最长的一串1的长度。

```python
class Solution:
    def reverseBits(self, num: int) -> int:
    # 注意python的bin处理
        neg = False
        if num < 0:
            num += 2**31
            neg = True

        longest = 0
        s = bin(num)[2:]
        if neg:
            s = "1" + s 
        elif not neg:
            s = "0" + s
        
        # 只有32位
        # print(len(s))
        # print(s)
        dp = [[0 for i in range(len(s))] for k in range(2)]
        # dp[0][i]表示未使用这个机会，dp[1][i]表示使用这个机会
        dp[0][0] = 0 if s[0] == "0" else 1
        dp[1][0] = 1 if s[0] == "0" else 0
        for i in range(1,len(s)):
            dp[0][i] = dp[0][i-1]+1 if s[i] == "1" else 0
            if s[i] == "1":
                dp[1][i] = max(dp[0][i-1],dp[1][i-1]) + 1
            elif s[i] == "0":
                dp[1][i] = dp[0][i-1] + 1
        
        return max(max(dp[0]),max(dp[1]))
```

# 面试题 05.06. 整数转换

整数转换。编写一个函数，确定需要改变几个位才能将整数A转成整数B。

示例1:

 输入：A = 29 （或者0b11101）, B = 15（或者0b01111）
 输出：2

```python
class Solution:
    def convertInteger(self, A: int, B: int) -> int:
        # 位运算统计有几个位不同
        count = 0
        for i in range(32):
            count += 0 if (A>>i)&1 == (B>>i)&1 else 1
        return count
```

# 面试题 05.07. 配对交换

配对交换。编写程序，交换某个整数的奇数位和偶数位，尽量使用较少的指令（也就是说，位0与位1交换，位2与位3交换，以此类推）。

示例1:

 输入：num = 2（或者0b10）
 输出 1 (或者 0b01)

```python
class Solution:
    def exchangeBits(self, num: int) -> int:
        # 位运算
        # 提取奇数位和偶数位置，利用掩码的方法
        # 10101010……的16进制为0xaaaaaaaa
        # 01010101……的16进制为0x55555555
        # 奇数位右移位一位 偶数位左移位一位
        # 然后做按位或
        odd = num&0xaaaaaaaa
        even = num&0x55555555
        odd = odd>>1
        even = even<<1
        ans = odd|even
        return ans
```

# 面试题 08.01. 三步问题

三步问题。有个小孩正在上楼梯，楼梯有n阶台阶，小孩一次可以上1阶、2阶或3阶。实现一种方法，计算小孩有多少种上楼梯的方式。结果可能很大，你需要对结果模1000000007。

```python
class Solution:
    def waysToStep(self, n: int) -> int:
        # 泰波那契数列
        # 用dp做或者用递归做,递归超时。。大数越界 
        # import sys
        # sys.setrecursionlimit(1000000)
        # def recur(n,a=1,b=2,c=4):
        #     if n == 1:
        #         return a
        #     else:
        #         temp = a + b
        #         a = b % 1000000007
        #         b = c % 1000000007
        #         c = (c + temp) % 1000000007
        #         return recur(n-1,a,b,c)
        # return recur(n) % 1000000007
        # dp做法，
        if n == 1: return 1
        if n == 2: return 2
        if n == 3: return 4
        # 如果需要用dp填表，则dp = [1 for i in range(n+1)]
        a = 1
        b = 2
        c = 4
        for i in range(4,n+1):
            temp = a + b
            a = b % 1000000007
            b = c % 1000000007
            c = (c + temp) % 1000000007
        return c
```

# 面试题 08.02. 迷路的机器人

设想有个机器人坐在一个网格的左上角，网格 r 行 c 列。机器人只能向下或向右移动，但不能走到一些被禁止的网格（有障碍物）。设计一种算法，寻找机器人从左上角移动到右下角的路径。

```python
class Solution:
    def pathWithObstacles(self, obstacleGrid: List[List[int]]) -> List[List[int]]:
        # 先动态规划，计算能够到当前为止的方案数
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        if m == 0 or n == 0:
            return []
        if obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1] == 1:
            return []
        # 不加一圈外圈
        dp = [[0 for j in range(n)] for i in range(m)]
        # dp[i][j] == 如果这个格子不为0，则
        # dp[i][j] = dp[i-1][j]+dp[i][j-1]
        dp[0][0] = 1 # 起点
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] != 1:
                    if i-1 >=0 and j-1 >= 0:
                        dp[i][j] = dp[i-1][j]+dp[i][j-1]
                    elif i-1>=0 and j >= 0:
                        dp[i][j] = dp[i-1][j]
                    elif i >= 0 and j-1>=0:
                        dp[i][j] = dp[i][j-1]
        path = []
        if dp[-1][-1] == 0:
            return []

        # 从终点倒着搜
        now = [m-1,n-1]
        while now != [0,0]:
            if dp[now[0]][now[1]] > 0:
                path.append(now.copy())
            if now[0]-1 >= 0 and dp[now[0]-1][now[1]] > 0:
                now[0] -= 1
            elif now[1]-1 >= 0 and dp[now[0]][now[1]-1] > 0:
                now[1] -= 1
        
        path.append([0,0])
        return path[::-1]
```

# 面试题 08.04. 幂集

幂集。编写一种方法，返回某集合的所有子集。集合中**不包含重复的元素**。

说明：解集不能包含重复的子集。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # 回溯
        ans = [] # 收集结果
        stack = [] # 回溯路径
        def backtracking(nums,stack,k):#选择列表，做选择,收集长度
            # 收集结果的条件
            if len(stack) == k:
                ans.append(stack[:]) #传引用
            p = 0
            while p < len(nums):
                stack.append(nums[p]) # 做选择
                backtracking(nums[p+1:],stack,k) # 注意这个p+1
                stack.pop() # 取消选择
                p += 1
        for i in range(len(nums)+1):
            backtracking(nums,stack,i)
        return ans
```

# 面试题 08.07. 无重复字符串的排列组合

无重复字符串的排列组合。编写一种方法，计算某字符串的所有排列组合，字符串每个字符均不相同。

```python
class Solution:
    def permutation(self, S: str) -> List[str]:
        # 回溯
        ans = []
        stack = []
        S = list(S)
        n = len(S)
        def backtracking(S,stack):  # 选择列表，选择路径
            if len(stack) == n: # 注意这里不能写len(S),因为S是传入的列表，当递归时候，传入的temp是短于要求的
                ans.append(stack[:])
                return 
            p = 0
            while p < len(S):
                temp = S.copy()
                e = temp.pop(p)
                stack.append(e)
                backtracking(temp,stack)
                stack.pop()
                p += 1
        backtracking(S,stack)
        new_ans = []
        for i in ans:
            s = ''.join(i)
            new_ans.append(s)
        
        return new_ans
```

# 面试题 08.08. 有重复字符串的排列组合

有重复字符串的排列组合。编写一种方法，计算某字符串的所有排列组合。

```python
class Solution:
    def permutation(self, S: str) -> List[str]:
        # 将其转换成字母后进行映射
        # 集合去重复暴力算
        S = list(S)
        i = 1
        theDict = dict()
        mirror = dict()
        for ch in S:
            if theDict.get(ch) == None:
                theDict[ch] = i 
                mirror[i] = ch
                i += 1
            # elif theDict.get(ch) != None:
            #     pass
        
        lst = [theDict[ch] for ch in S]
        lst.sort() # 变成一个数列,预先排列
        n = len(lst)
        ans = []

        def backtracking(choice,path):
            if len(choice) == 0:
                ans.append(tuple(path[:]))
                return 
            for i in range(len(choice)):
                cp = choice.copy()
                e = choice[i]
                cp.remove(e)
                path.append(e)
                backtracking(cp,path)
                path.pop()
        
        backtracking(lst,[])
        
        theSet = set(ans)
        final = []
        for pair in theSet:
            temp = ""
            for i in pair:
                temp += mirror[i]
            final.append(temp)
        # final.sort()
        return final        
```

```python
class Solution:
    def permutation(self, S: str) -> List[str]:
        # 使用used数组去重复
        lst = list(S)
        lst.sort()
        n = len(lst)
        used = [False for i in range(n)]
        ans = []

        def backtracking(choice,path):
            if len(path) == n:
                ans.append("".join(path))
                return 
            for i in range(n):
                if used[i] == True: # 代表这个数字被使用过了
                    continue 
                if i > 0 and choice[i] == choice[i-1] and used[i-1] == True:
                    # 这个剪and used[i-1] == False:或者是True都能过
                    # 一个是顺序选，一个是倒序选,顺序剪更快
                    continue 
                path.append(choice[i])
                used[i] = True
                backtracking(choice,path)
                used[i] = False 
                path.pop()
        
        backtracking(lst,[])

        return ans
        
```

# 面试题 08.09. 括号

括号。设计一种算法，打印n对括号的所有合法的（例如，开闭一一对应）组合。

说明：解集不能包含重复的子集。

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

# 面试题 08.10. 颜色填充

编写函数，实现许多图片编辑软件都支持的「颜色填充」功能。

待填充的图像用二维数组 image 表示，元素为初始颜色值。初始坐标点的行坐标为 sr 列坐标为 sc。需要填充的新颜色为 newColor 。

「周围区域」是指颜色相同且在上、下、左、右四个方向上存在相连情况的若干元素。

请用新颜色填充初始坐标点的周围区域，并返回填充后的图像。

```
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        # dfs
        m = len(image)
        n = len(image[0])
        # 一个visited矩阵
        visited = [[False for j in range(n)] for i in range(m)]
        origin = image[sr][sc] # 记录原始颜色
        direc = [(-1,0),(1,0),(0,1),(0,-1)] # 方向数组

        def valid(i,j):
            if 0<=i<m and 0<=j<n and image[i][j] == origin:
                return True
            else:
                return False
        
        def dfs(i,j,origin,newColor): # 注意不是回溯，不需要撤销
            if not valid(i,j): # 不合法，返回
                return 
            if visited[i][j] == True:
                return 
            visited[i][j] = True 
            image[i][j] = newColor
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(new_i,new_j,origin,newColor)
        
        dfs(sr,sc,origin,newColor)
        return image

```

# 面试题 08.11. 硬币

硬币。给定数量不限的硬币，币值为25分、10分、5分和1分，编写代码计算n分有几种表示法。(结果可能会很大，你需要将结果模上1000000007)

```python
class Solution:
    def waysToChange(self, n: int) -> int:
        coins = [1,5,10,25]
        # 需要求的是组合数，那么外层遍历物品，内层遍历价值
        dp = [0 for i in range(n+1)]
        dp[0] = 1
        for coin in coins:
            for j in range(coin,n+1):
                dp[j] += dp[j-coin]
        return dp[-1] % 1000000007
```

# 面试题 08.12. 八皇后

设计一种算法，打印 N 皇后在 N × N 棋盘上的各种摆法，其中每个皇后都不同行、不同列，也不在对角线上。这里的“对角线”指的是所有的对角线，不只是平分整个棋盘的那两条对角线。

注意：本题相对原题做了扩展

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 题目描述比较模糊。实际上就是普通八皇后问题
        graph = [["." for i in range(n)] for j in range(n)]
        ans = []
        def backtracking(i):
            if i == n: # 注意收集格式
                temp = []
                for line in graph:
                    temp.append("".join(line))
                ans.append(temp)
                return 
            for j in range(n):
                if self.check(graph,i,j,n):
                    graph[i][j] = "Q"
                    backtracking(i+1)
                    graph[i][j] = "."
        backtracking(0)
        return ans
    
    def check(self,graph,i,j,n):
        direc = [(-1,0),(-1,1),(-1,-1)] # 只需要搜仨方向即可
        for di in direc:
            now_i = i 
            now_j = j 
            while 0 <= now_i < n and 0 <= now_j < n:
                if graph[now_i][now_j] == "Q":
                    return False
                now_i += di[0]
                now_j += di[1]
        return True
```

# 面试题 10.01. 合并排序的数组

给定两个排序后的数组 A 和 B，其中 A 的末端有足够的缓冲空间容纳 B。 编写一个方法，将 B 合并入 A 并排序。

初始化 A 和 B 的元素数量分别为 m 和 n。

```python
class Solution:
    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
        """
        Do not return anything, modify A in-place instead.
        """
        # 实际上就是原地归并排序
        i = m - 1
        j = n - 1
        while (i >=0 and j >= 0):
            if A[i] > B[j]:
                A[i+j+1] = A[i]
                i -= 1
            elif A[i] <= B[j]:
                A[i+j+1] = B[j]
                j -= 1
        if i == -1:
            while j >= 0:
                A[j] = B[j]
                j -= 1
```

# 面试题 10.02. 变位词组

编写一种方法，对字符串数组进行排序，将所有变位词组合在一起。变位词是指字母相同，但排列不同的字符串。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 创建字典，字典的键为排序后的字符串，字典的值为列表，该列表可变
        dict1 = dict()
        # 给每个键先默认附加值为列表,注意
        for i in strs:
            key = ''.join(sorted(i))
            # if not dict1.get(key): 这一行写不写都行
            dict1[key] = []
        # 然后给每个键填充值
        for i in strs:
            key = ''.join(sorted(i))
            dict1[key].append(i)
        ans = [] # 收集答案
        for i in dict1:
            ans.append(dict1[i])
        return ans
```



# 面试题 10.03. 搜索旋转数组

搜索旋转数组。给定一个排序后的数组，包含n个整数，但这个数组已被旋转过很多次了，次数不详。请编写代码找出数组中的某个元素，假设数组元素原先是按升序排列的。若有多个相同元素，返回索引值最小的一个。

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
                # 如果left和right相等且是结果值，直接返回最左边
                return left
            while arr[left] == arr[right]: # 当他们不是target的时候
            # 由于是返回最左边的那个,所以right收缩
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

# 面试题 10.05. 稀疏数组搜索

稀疏数组搜索。有个排好序的字符串数组，其中散布着一些空字符串，编写一种方法，找出给定字符串的位置。

```python
class Solution:
    def findString(self, words: List[str], s: str) -> int:
        # 注意：数组是排好序的
        # 效率不稳定的二分搜索
        left = 0
        right = len(words)-1
        while left <= right:
            mid = (left+right)//2
            if words[mid] == '': # 为空字符串，看右端点能否收缩,只要为空就不进入比较
                if words[right] != s: # 这一行极为重要，不管是空还是其他值，只要不等于目标值就收缩
                    right -= 1
                else:
                    return right
            elif words[mid] == s:
                return mid # 返回索引
            elif words[mid] < s: # 则在右边查找
                left = mid + 1
            elif words[mid] > s: # 则在左边查找
                right = mid - 1
        # 找不到则返回-1
        return -1

```

# 面试题 10.09. 排序矩阵查找

给定M×N矩阵，每一行、每一列都按升序排列，请编写代码找出某元素。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 以右上角作为二叉搜索树的模式来看待这个矩阵
        # 处理空矩阵
        if len(matrix) == 0:
            return False
        m = len(matrix) # m 是第几行 , 这里表示有多少行
        n = len(matrix[0]) # n是第几列， 这里表示有多少列
        # matrix[m][n] # 表示m行第n个
        i = n-1 # 被n限制
        j = 0 # 被m限制
        while 0 <= i < n and 0 <= j < m:
            print(matrix[j][i])
            if matrix[j][i] == target:
                return True
            elif matrix[j][i] > target: # 如果矩阵数值大于目标值，向左找
                i -= 1
            elif matrix[j][i] < target: # 如果矩阵数值小于目标值，向下找
                j += 1
        return False # 越界则终止循环，返回False
```



# 面试题 16.01. 交换数字

编写一个函数，不用临时变量，直接交换`numbers = [a, b]`中`a`与`b`的值。

```python
class Solution:
    def swapNumbers(self, numbers: List[int]) -> List[int]:
        numbers[0] += numbers[1] # a = a+b ,此时 a == a+b
        numbers[1] -= numbers[0] # 此时 b == -a
        numbers[0] += numbers[1] # 此时 a == b
        numbers[1] = -1*numbers[1] # 此时 b == aa
        return numbers
```

# 面试题 16.02. 单词频率

设计一个方法，找出任意指定单词在一本书中的出现频率。

你的实现应该支持如下操作：

WordsFrequency(book)构造函数，参数为字符串数组构成的一本书
get(word)查询指定单词在书中出现的频率

```python
# 字典树
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.isWord = False
        self.word_count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self,word):
        node = self.root
        for char in word:
            index = ord(char) - ord("a")
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isWord = True
        node.word_count += 1
    
    def insert_all(self,words):
        for word in words:
            self.insert(word)
    
    def cacl(self,word):
        node = self.root
        for char in word:
            index = ord(char) - ord("a")
            if node.children[index] == None:
                return 0
            node = node.children[index]
        if node.isWord:
            return node.word_count
        else:
            return 0
        
class WordsFrequency:

    def __init__(self, book: List[str]):
        self.Trie = Trie() # 构造字典树
        self.Trie.insert_all(book) # 全部插入

    def get(self, word: str) -> int:
        return self.Trie.cacl(word) # 计算

```

# 面试题 16.05. 阶乘尾数

设计一个算法，算出 n 阶乘有多少个尾随零。

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        # 看其中包含多少个5
        count = 0
        while n >= 5:
            count += n//5
            n //= 5
        return count
```

# 面试题 16.06. 最小差

给定两个整数数组`a`和`b`，计算具有最小差绝对值的一对数值（每个数组中取一个值），并返回该对数值的差

```python
class Solution:
    def smallestDifference(self, a: List[int], b: List[int]) -> int:
        # 双排指针，初始化最小值为极大值
        min_gap = 0xffffffff
        # 排序后初始化指针为首位
        a.sort()
        b.sort()
        pa,pb = 0,0
        # 指针移动逻辑，移动较小的那个使得两数更接近
        while pa < len(a) and pb < len(b):
            gap = abs(a[pa]-b[pb])
            min_gap = min(gap,min_gap)
            if a[pa] < b[pb]:
                pa += 1
            else:
                pb += 1
        return min_gap
```

# 面试题 16.07. 最大数值

编写一个方法，找出两个数字`a`和`b`中最大的那一个。不得使用if-else或其他比较运算符。

```python
class Solution:
    def maximum(self, a: int, b: int) -> int:
        # 考察的是绝对值的思想
        # 数学思路
        return (a+b+abs(a-b))//2
```

# 面试题 16.08. 整数的英语表示

给定一个整数，打印该整数的英文描述。

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

# 面试题 16.10. 生存人数

给定 N 个人的出生年份和死亡年份，第 i 个人的出生年份为 birth[i]，死亡年份为 death[i]，实现一个方法以计算生存人数最多的年份。

你可以假设所有人都出生于 1900 年至 2000 年（含 1900 和 2000 ）之间。如果一个人在某一年的任意时期处于生存状态，那么他应该被纳入那一年的统计中。例如，生于 1908 年、死于 1909 年的人应当被列入 1908 年和 1909 年的计数。

如果有多个年份生存人数相同且均为最大值，输出其中最小的年份。

```python
class Solution:
    def maxAliveYear(self, birth: List[int], death: List[int]) -> int:
        # 上下车模型做差分
        up = collections.defaultdict(int)
        n = len(birth)
        lst = [[birth[i]-1900,death[i]-1900+1] for i in range(n)]
        for u,d in lst:
            up[u] += 1
            up[d] -= 1
        length = max(death)-1900+1
        # 然后按照顺序填充
        people = [0 for i in range(length)]
        now = 0
        for i in range(length):
            now += up[i]
            people[i] = now 
        maxPeople = max(people)
        for i in range(len(people)):
            if people[i] == maxPeople:
                return i + 1900
```

# 面试题 16.11. 跳水板

你正在使用一堆木板建造跳水板。有两种类型的木板，其中长度较短的木板长度为shorter，长度较长的木板长度为longer。你必须正好使用k块木板。编写一个方法，生成跳水板所有可能的长度。

返回的长度需要从小到大排列。

```python
class Solution:
    def divingBoard(self, shorter: int, longer: int, k: int) -> List[int]:
        # 可能会超时
        # 用set
        if k == 0:
            return []
        remain = k
        theSet = set()
        for i in range(k+1):
            e = shorter * remain + longer * (k-remain)
            theSet.add(e)
            remain -= 1
        ans = []
        for e in theSet:
            ans.append(e)
        ans.sort()
        return ans
```

```go
func divingBoard(shorter int, longer int, k int) []int {
    if k == 0 {
        ans := make([]int,0)
        return ans
    }
    if shorter == longer {
        ans := make([]int,0)
        ans = append(ans,shorter*k)
        return ans
    }
    a1 := k
    ans := make([]int,0)
    for i := 0; i < k+1; i += 1 {
        ans = append(ans,a1*shorter+(k-a1)*longer)
        a1 -= 1
    }
    return ans
}
```

# 面试题 16.13. 平分正方形

给定两个正方形及一个二维平面。请找出将这两个正方形分割成两半的一条直线。假设正方形顶边和底边与 x 轴平行。

每个正方形的数据square包含3个数值，正方形的左下顶点坐标[X,Y] = [square[0],square[1]]，以及正方形的边长square[2]。所求直线穿过两个正方形会形成4个交点，请返回4个交点形成线段的两端点坐标（两个端点即为4个交点中距离最远的2个点，这2个点所连成的线段一定会穿过另外2个交点）。2个端点坐标[X1,Y1]和[X2,Y2]的返回格式为{X1,Y1,X2,Y2}，要求若X1 != X2，需保证X1 < X2，否则需保证Y1 <= Y2。

若同时有多条直线满足要求，则选择斜率最大的一条计算并返回（与Y轴平行的直线视为斜率无穷大）。

```python
class Solution:
    def cutSquares(self, square1: List[int], square2: List[int]) -> List[float]:
        # 平分正方形一定通过中心
        x1,y1,width1 = square1
        x2,y2,width2 = square2
        center1 = [x1+width1/2,y1+width1/2]
        center2 = [x2+width2/2,y2+width2/2]
        # 如果两点重合，取斜率最大的，即x坐标,两点x坐标相同也这样处理
        if center1 == center2 or center1[0] == center2[0]:
            # 用它求四个y点,取最上面的和最下面的
            a,b,c,d = [y1,y2,y1+width1,y2+width2]
            lst = [a,b,c,d]
            lst.sort()
            ans = [center1[0],lst[0],center1[0],lst[-1]]

            return ans

        else:
            # 其他情况，两点式计算斜率。
            def calc_k_b(p1,p2): # 传入点坐标，返回k,b: y = kx + b
                x1,y1 = p1
                x2,y2 = p2 
                k = (y2-y1)/(x2-x1)
                b = -x1*k+y1 
                return k,b 

            k,b = calc_k_b(center1,center2)
            def XtoPoint(x,k,b): # 传入横坐标
                y = k*x + b 
                return [x,y]
            def YtoPoint(y,k,b): # 传入纵坐标
                x = (y-b)/k 
                return [x,y]
            x3 = x1 + width1
            x4 = x2 + width2
            y3 = y1 + width1
            y4 = y2 + width2
            # 如果角度大于45度，则在横边上才有交点
            if abs(k) > 1:
                tempList = []
                tempList.append(YtoPoint(y1,k,b))  
                tempList.append(YtoPoint(y2,k,b))  
                tempList.append(YtoPoint(y3,k,b))  
                tempList.append(YtoPoint(y4,k,b))  
            else:
                tempList = []
                tempList.append(XtoPoint(x1,k,b))  
                tempList.append(XtoPoint(x2,k,b))  
                tempList.append(XtoPoint(x3,k,b))  
                tempList.append(XtoPoint(x4,k,b)) 

            tempList.sort()
            ans = tempList[0]+tempList[-1]
            return ans

```



# 面试题 16.15. 珠玑妙算

珠玑妙算游戏（the game of master mind）的玩法如下。

计算机有4个槽，每个槽放一个球，颜色可能是红色（R）、黄色（Y）、绿色（G）或蓝色（B）。例如，计算机可能有RGGB 4种（槽1为红色，槽2、3为绿色，槽4为蓝色）。作为用户，你试图猜出颜色组合。打个比方，你可能会猜YRGB。要是猜对某个槽的颜色，则算一次“猜中”；要是只猜对颜色但槽位猜错了，则算一次“伪猜中”。注意，“猜中”不能算入“伪猜中”。

给定一种颜色组合solution和一个猜测guess，编写一个方法，返回猜中和伪猜中的次数answer，其中answer[0]为猜中的次数，answer[1]为伪猜中的次数。

```python
class Solution:
    def masterMind(self, solution: str, guess: str) -> List[int]:
        # 注意层级关系
        # 先计算猜中的数量,记录原始的solution和guess【如果有必要复原的话】
        origin_solution = solution
        origin_guess = guess
        solution = list(solution)
        guess = list(guess)
        true_guess = 0
        for i in range(len(solution)):
            if solution[i] == guess[i]:
                solution[i] = None # 防止之后重复算
                guess[i] = None
                true_guess += 1
        # 伪猜中的逻辑比较难算
        # 使用None标记辅助
        fake = 0
        for i in range(len(solution)):
            for j in range(len(guess)):
                if solution[i] != None:
                    if solution[i] == guess[j]:
                        solution[i] = None
                        guess[j] = None
                        fake += 1
        # 把solution和guess还原
        solution,guess = origin_solution,origin_guess # 可以不还原
        return [true_guess,fake]
```

# 面试题 16.16. 部分排序

给定一个整数数组，编写一个函数，找出索引m和n，只要将索引区间[m,n]的元素排好序，整个数组就是有序的。注意：n-m尽量最小，也就是说，找出符合条件的最短序列。函数返回值为[m,n]，若不存在这样的m和n（例如整个数组是有序的），请返回[-1,-1]。

```python
class Solution:
    def subSort(self, array: List[int]) -> List[int]:
        if len(array) == 0:
            return [-1,-1]
        leftMark,rightMark = -1,-1
        tempMax = array[0]
        tempMin = array[-1]
        n = len(array)
        i,j = 0,n-1
        while i < n:
            if array[i] >= tempMax:
                tempMax = array[i]
            elif array[i] < tempMax: # 乱序了，记录ii
                leftMark = i 
            i += 1
        while j >= 0:
            if array[j] <= tempMin:
                tempMin = array[j]
            elif array[j] > tempMin: # 乱序了，记录jj
                rightMark = j 
            j -= 1
        if (leftMark,rightMark) == (-1,-1): # 有序，从未更新过
            return [-1,-1]
        else:
            return [rightMark,leftMark]
```

# 面试题 16.17. 连续数列

给定一个整数数组，找出总和最大的连续数列，并返回总和。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 要求连续，则贪心
        temp_sum = 0 # 记录每次贪心的部分和
        ans = -0x7fffffff-1 # 记录备选项目,初始化为极小值
        # 注意处理全为负数的
        # 注意处理空输入
        if len(nums) == 0:
            return ans
        count = 0 # 标记负数个数
        p = 0
        while p < len(nums):
            if nums[p] < 0:
                count += 1
            temp_sum += nums[p]
            if temp_sum < 0: # 小于0则丢弃，从下一项开始重新考虑
                temp_sum = 0
                ans = max(ans,temp_sum)
            ans = max(temp_sum,ans)
            p += 1
        return ans if count != len(nums) else max(nums)
```

# 面试题 16.19. 水域大小

你有一个用于表示一片土地的整数矩阵land，该矩阵中每个点的值代表对应地点的海拔高度。若值为0则表示水域。由垂直、水平或对角连接的水域为池塘。池塘的大小是指相连接的水域的个数。编写一个方法来计算矩阵中所有池塘的大小，返回值需要从小到大排序。

```python
class Solution:
    def pondSizes(self, land: List[List[int]]) -> List[int]:
        # dfs
        m = len(land)
        n = len(land[0])
        visited = [[False for j in range(n)] for i in range(m)]
        direc = [(0,1),(0,-1),(1,0),(-1,0),(-1,-1),(-1,1),(1,-1),(1,1)]
        area = 0
        def dfs(i,j):
            nonlocal area
            if not (0<=i<m and 0<=j<n):
                return
            area += 1 
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and land[new_i][new_j] == 0 and visited[new_i][new_j] == False:
                    visited[new_i][new_j] = True
                    dfs(new_i,new_j)

        ans = []
        for i in range(m):
            for j in range(n):
                if land[i][j] == 0 and visited[i][j] == False:
                    area = 0 # 重置
                    visited[i][j] = True # 防止迂回调用，开始就设置True，其实visited数组可以用数值代替
                    dfs(i,j) # 调用
                    if area != 0:
                        ans.append(area)

        ans.sort()
        return ans
```

# 面试题 16.20. T9键盘

在老式手机上，用户通过数字键盘输入，手机将提供与这些数字相匹配的单词列表。每个数字映射到0至4个字母。给定一个数字序列，实现一个算法来返回匹配单词的列表。你会得到一张含有有效单词的列表。映射如下图所示：

```python
class Solution:
    def getValidT9Words(self, num: str, words: List[str]) -> List[str]:
        # 先做个字典映射，由字母到数字
        dict1 = {'a':'2','b':'2','c':'2','d':'3','e':'3','f':'3','g':'4','h':'4','i':'4','j':'5','k':'5','l':'5','m':'6','n':'6','o':'6','p':'7','q':'7','r':'7','s':'7','t':'8','u':'8','v':'8','w':'9','x':'9','y':'9','z':'9'}
        ans = []
        for word in words:
            temp_num = ''
            for char in word:
                temp_num += dict1[char]
            if temp_num == num:
                ans.append(word)
        return ans
```

# 面试题 16.24. 数对和

设计一个算法，找出数组中两数之和为指定值的所有整数对。一个数只能属于一个数对。

```python
class Solution:
    def pairSums(self, nums: List[int], target: int) -> List[List[int]]:
        # 疯狂两数之和,如果用闭区间的话注意检查最后一次
        nums.sort()
        left = 0
        right = len(nums) - 1
        ans = [] # 收集结果
        while left <= right:
            if nums[left] + nums[right] == target:
                if left != right: # 两个不能指向同一个数，使用完之后需要移动
                    ans.append([nums[left] , nums[right]])
                    left += 1
                    right -= 1
                else:
                    break
            elif nums[left] + nums[right] < target: # 小指针右边移
                left += 1
            elif nums[left] + nums[right] > target:# 大指针左移动
                right -= 1 
        return ans
```

# 面试题 16.25. LRU 缓存

设计和构建一个“最近最少使用”缓存，该缓存会删除最近最少使用的项目。缓存应该从键映射到值(允许你插入和检索特定键对应的值)，并在初始化时指定最大容量。当缓存被填满时，它应该删除最近最少使用的项目。

它应该支持以下操作： 获取数据 get 和 写入数据 put 。

获取数据 get(key) - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
写入数据 put(key, value) - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间。

```python
# 纯净版
class Node:
    def __init__(self,key = -1,val = -1):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class Dq:
    def __init__(self):
        self.header = Node()
        self.tailer = Node()
        self.header.next = self.tailer
        self.tailer.prev = self.header
        self.size = 0
    
    def appendleft(self,new_node):
        temp = self.header.next
        self.header.next = new_node
        new_node.prev = self.header
        new_node.next = temp
        temp.prev = new_node
        self.size += 1
    
    def popright(self): # 
        temp = self.tailer.prev
        temp.prev.next = self.tailer
        self.tailer.prev = temp.prev
        self.size -= 1
        return temp
    
    def remove(self,node):
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

class LRUCache:

    def __init__(self, capacity: int):
        self.hashmap = dict() # 储存逻辑为 key - Node(key,val)
        self.cache = Dq()
        self.cap = capacity

    def get(self, key: int) -> int:
        if key not in self.hashmap: return -1
        else:
            the_node = self.hashmap[key]
            self.cache.remove(the_node)
            self.cache.appendleft(the_node)
            return self.hashmap[key].val

    def put(self, key: int, value: int) -> None:
        # 键存在
        if key in self.hashmap:
            # 更新键值，并且提前
            the_node = self.hashmap[key]
            self.hashmap[key] = Node(key,value)
            self.cache.remove(the_node)
            self.cache.appendleft(self.hashmap[key])

        else: # 键不存在
            if self.cache.size < self.cap: # 容量未满
                new_node = Node(key,value)
                self.hashmap[key] = new_node
                self.cache.appendleft(new_node)                

            elif self.cache.size == self.cap: # 容量已满
                new_node = Node(key,value)
                self.hashmap[key] = new_node
                self.cache.appendleft(new_node)
                delete_node = self.cache.popright()
                del self.hashmap[delete_node.key]

```

# 面试题 16.26. 计算器

给定一个包含正整数、加(+)、减(-)、乘(*)、除(/)的算数表达式(括号除外)，计算其结果。

表达式仅包含非负整数，+， - ，*，/ 四种运算符和空格  。 整数除法仅保留整数部分。

```python
class Solution:
    def calculate(self, s: str) -> int:
        # 先格式化输入串，需要注意没有括号
        scan = []
        for i in s:
            if i != ' ':
                scan.append(i)
        # 执行的时候注意*或者/号左右两边的数可能是负数，并且不止一位数。

        symbol = "+"
        stack = []
        num = 0
        scan.append("+") # 防止扫完之后最后一个数没有被处理

        for ch in scan:
            if ch.isdigit():
                num = 10 * num + int(ch)            
            elif ch.isdigit() == False:
                if symbol == '+':
                    stack.append(num)
                elif symbol == "-":
                    stack.append(-num)
                elif symbol == "*":
                    stack[-1] *= num 
                elif symbol == "/":
                    stack[-1] = int(stack[-1]/num)  # 不能使用python的地板除,需要对它的除法进行处理             
                num = 0 # 清空计数
                symbol = ch  # 延迟更新符号
        return sum(stack)


```

# 面试题 17.01. 不用加号的加法

设计一个函数把两个数字相加。不得使用 + 或者其他算术运算符。

```java
class Solution {
    // 思路：位运算
    // 将进位和 ， 不进位和分离开
    public int add(int a, int b) {
        while (b != 0){
            int carry = (a&b) << 1;
            a = a ^ b;
            b = carry;
        }
        return a;
    }
}
```

# 面试题 17.05.  字母与数字

给定一个放有字母和数字的数组，找到最长的子数组，且包含的字母和数字的个数相同。

返回该子数组，若存在多个最长子数组，返回左端点下标值最小的子数组。若不存在这样的数组，返回一个空数组。

```python
class Solution:
    def findLongestSubarray(self, array: List[str]) -> List[str]:
        # 把数字和字母看作-1和1
        # 前缀和思想,用复制思想用了双倍的空间。先懒得优化了

        preSumDict = collections.defaultdict(int)
        cp = array.copy() # 复制一份
        for i in range(len(array)):
            if array[i].isdigit():
                array[i] = -1
            else:
                array[i] = 1

        # 扫当前和,包含当前位置 k-v对为包含当前位置的总和:索引
        preSumDict[0] = -1
        tempSum = 0
        tempLength = 0
        ans = [None,None]
        for i,n in enumerate(array):
            tempSum += n
            if preSumDict.get(tempSum) == None: # 这个值之前没有出现过
                preSumDict[tempSum] = i 
            elif preSumDict.get(tempSum) != None: # 只更新初次出现的
                if i - preSumDict[tempSum] > tempLength:
                    tempLength = i - preSumDict[tempSum]
                    ans = [preSumDict[tempSum],i]   # 记录的是索引，闭区间索引          
        if ans == [None,None]:
            return []
        return cp[ans[0]+1:ans[1]+1]
```

# 面试题 17.07. 婴儿名字

每年，政府都会公布一万个最常见的婴儿名字和它们出现的频率，也就是同名婴儿的数量。有些名字有多种拼法，例如，John 和 Jon 本质上是相同的名字，但被当成了两个名字公布出来。给定两个列表，一个是名字及对应的频率，另一个是本质相同的名字对。设计一个算法打印出每个真实名字的实际频率。注意，如果 John 和 Jon 是相同的，并且 Jon 和 Johnny 相同，则 John 与 Johnny 也相同，即它们有传递和对称性。

在结果列表中，选择 字典序最小 的名字作为真实名字。

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
    def trulyMostPopular(self, names: List[str], synonyms: List[str]) -> List[str]:
        # 并查集+计数
        nameDict = dict() # k == name, v = index
        mirror = dict() # k == index, v == name
        nameTimes = collections.defaultdict(int)
        for i,information in enumerate(names):
            temp = information.split("(")
            nameTimes[temp[0]] = int(temp[1][:-1])
            nameDict[temp[0]] = i
            mirror[i] = temp[0]

        for index,pair in enumerate(synonyms):
            p1,p2 = pair.split(",")
            i = nameDict.get(p1[1:])
            j = nameDict.get(p2[:-1])

        # 转换,注意有可能名字在重名集合却不在names里
        connect = []
        represent = collections.defaultdict(list)
        nowInd = len(names)
        for index,pair in enumerate(synonyms):
            p1,p2 = pair.split(",")
            i = nameDict.get(p1[1:])
            if i == None:
                nameDict[p1[1:]] = nowInd # 所以需要补上这样的类型
                mirror[nowInd] = p1[1:]
                i = nowInd
                nowInd += 1
            j = nameDict.get(p2[:-1])
            if j == None:
                nameDict[p2[:-1]] = nowInd
                mirror[nowInd] = p2[:-1]
                j = nowInd
                nowInd += 1  
                         
            connect.append([i,j])
        
        ufset = UF(nowInd)
        for i,j in connect: # 链接
            ufset.union(i,j)
        
        for i in range(nowInd):
            represent[ufset.find(i)].append(mirror[i]) # 找有几个不同的
        
        ans = []
        for key in represent:
            represent[key].sort()
            times = 0
            for element in represent[key]:
                times += nameTimes[element]
            n = represent[key][0]+"("+str(times)+")"
            ans.append(n)

        # 不需要考虑输出顺序
        return ans

```



# 面试题 17.09. 第 k 个数

有些数的素因子只有 3，5，7，请设计一个算法找出第 k 个数。注意，不是必须有这些素因子，而是必须不包含其他的素因子。例如，前几个数按顺序应该是 1，3，5，7，9，15，21。

```python
class Solution:
    def getKthMagicNumber(self, k: int) -> int:
        # 丑数题
        # dp+三指针
        if k <= 0:
            return 
        dp = [1 for i in range(k+1)]
        # 初始化三指针指向dp[1]
        p3,p5,p7 = 1,1,1
        for i in range(2,k+1):
            t3,t5,t7 = dp[p3]*3,dp[p5]*5,dp[p7]*7
            dp[i] = min(t3,t5,t7)
            if dp[i] == t3:
                p3 += 1
            if dp[i] == t5:
                p5 += 1
            if dp[i] == t7:
                p7 += 1
        return dp[-1]
```

# 面试题 17.10. 主要元素

数组中占比超过一半的元素称之为主要元素。给你一个 整数 数组，找出其中的主要元素。若没有，返回 -1 。请设计时间复杂度为 O(N) 、空间复杂度为 O(1) 的解决方案。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 摩尔投票法
        # 初始候选人为第一位，
        # 当票数为0时候，更换候选人
        # 如果遇到的元素值和候选人一样，则+1
        # 如果路过的元素值和候选人不一样，则-1
        # 当票数为0时候，更换候选人
        # 摩尔投票法要投两轮
        # 第一轮找出可能的候选人
        # 第二轮只对候选人记有效票数，得票大于一半，则为答案
        if len(nums) == 0: return -1
        count = 0
        select = nums[0]
        for i in nums:
            if count == 0: # 注意这个顺序
                select = i # 更换候选人
            if i == select:
                count += 1
            elif i != select:
                count -= 1
        # 第二轮
        supports = 0
        for i in nums:
            if i == select:
                supports += 1
        return select if supports > len(nums)/2 else -1
```

# 面试题 17.11. 单词距离

有个内含单词的超大文本文件，给定任意两个单词，找出在这个文件中这两个单词的最短距离(相隔单词数)。如果寻找过程在这个文件中会重复多次，而每次寻找的单词不同，你能对此优化吗?

```python
class Solution:
    def findClosest(self, words: List[str], word1: str, word2: str) -> int:
        index1 = []
        index2 = []
        for i,value in enumerate(words):
            if value == word1:
                index1.append(i)
            if value == word2:
                index2.append(i)
        # 然后转换成两分立数组中找最小绝对差
        # 假设单词在这里
        min_gap = len(words) # 初始化要求距离为表长度
        # 初始化指针
        p1 = 0
        p2 = 0
        # 指针移动逻辑是，移动数值小的那个，使它更接近于另一个
        while p1 < len(index1) and p2 < len(index2):
            gap = abs(index1[p1]-index2[p2])
            min_gap = min(gap,min_gap)
            if index1[p1] < index2[p2]:
                p1 += 1
            else:
                p2 += 1
        return min_gap
```

# 面试题 17.12. BiNode

二叉树数据结构TreeNode可用来表示单向链表（其中left置空，right为下一个链表节点）。实现一个方法，把二叉搜索树转换为单向链表，要求依然符合二叉搜索树的性质，转换操作应是原址的，也就是在原始的二叉搜索树上直接修改。

返回转换后的单向链表的头节点。

```python
class Solution:
    def convertBiNode(self, root: TreeNode) -> TreeNode:
        # 中序遍历的时候就直接处理
        # nonlocal可以用self.***代替，但是不如用nonlocal直观
        prev = None # 需要找到中序遍历的前一个节点,prev是运动着的指针
        cur = None # 扫树用到的头节点指针，固定指针

        def inorder_method(node):
            nonlocal prev
            nonlocal cur
            if node == None:
                return
            inorder_method(node.left)
            # 中序结果
            if cur == None: # 用根节点作为第一个节点
                prev = node
                cur = node # 给一个初值，之后不会再移动
            else:
                node.left = None
                prev.right = node
                prev = prev.right
            inorder_method(node.right)
            
        inorder_method(root)
        return cur # 头节点指针
```

# 面试题 17.14. 最小K个数

设计一个算法，找出数组中最小的k个数。以任意顺序返回这k个数均可。

```python
class Solution:
    def smallestK(self, arr: List[int], k: int) -> List[int]:
        # 大根堆，维护堆大小为k，如果比堆顶还大，则过滤掉
        # python原生支持的是小根堆，全部处理为-1
        if k == 0:
            return []
        arr = [-1 * i for i in arr]
        ans = [i for i in arr[:k]] 
        heapq.heapify(ans) # 堆化
        # 由于是负数，原来堆比堆顶还大的条件在这里改写成比堆顶还小
        for i in arr[k:]:
            if i <= ans[0]:
                pass
            elif i >  ans[0]: # 否则弹出堆顶，加入
                heapq.heappop(ans) # 弹出堆顶
                heapq.heappush(ans,i)
        # 还原堆，顺序任意
        ans = [-i for i in ans]
        return ans
```

```python
class Solution:
    def smallestK(self, arr: List[int], k: int) -> List[int]:
        # topK小问题，用大根堆
        if k == 0:
            return []
        maxHeap = [-i for i in arr[:k]]
        heapq.heapify(maxHeap) # 堆化
        for e in arr[k:]:# 过筛，比堆顶小【即真值比堆顶大，不要】
            if -e > maxHeap[0]:
                heapq.heappop(maxHeap)
                heapq.heappush(maxHeap,-e)
        ans = [-i for i in maxHeap]
        return ans
```

# 面试题 17.16. 按摩师

一个有名的按摩师会收到源源不断的预约请求，每个预约都可以选择接或不接。在每次预约服务之间要有休息时间，因此她不能接受相邻的预约。给定一个预约请求序列，替按摩师找到最优的预约集合（总预约时间最长），返回总的分钟数。

注意：本题相对原题稍作改动

```python
class Solution:
    def massage(self, nums: List[int]) -> int:
        # 打家劫舍换皮版本
        # 动态规划
        # 申请dp数组长度为n，dp[n]是到目前为止的最大值
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        dp = [0 for i in range(n)]
        dp[0] = nums[0]
        dp[1] = max(nums[0],nums[1])
        # 状态转移为 dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        for i in range(2,n):
            dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        return dp[-1] # 返回最后一个即可，它单调递增，可以状态压缩
```

# 面试题 17.20. 连续中值

随机产生数字并传递给一个方法。你能否完成这个方法，在每次产生新值时，寻找当前所有值的中间值（中位数）并保存。

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
        # 对顶堆
        # 内置堆是小根堆
        # 当前数量为偶时，往小根堆里加，当前数量为奇时，往大根堆里加
        # 往小根堆里加的时候先过一遍大根堆的筛，往大根堆里加的时候，过一遍小根堆的筛
        self.maxHeap = []
        self.minHeap = []
        self.size = 0

    def addNum(self, num: int) -> None:
        if self.size % 2 == 0:
            heapq.heappush(self.maxHeap,-num)
            e = -heapq.heappop(self.maxHeap)
            heapq.heappush(self.minHeap,e)
        else:
            heapq.heappush(self.minHeap,num)
            e = -heapq.heappop(self.minHeap)
            heapq.heappush(self.maxHeap,e)
        self.size += 1

    def findMedian(self) -> float:
        # 当前数目为偶数，返回两堆顶/2
        # 当前数目为奇数，返回小根堆顶
        if self.size % 2 == 1:
            return self.minHeap[0]
        elif self.size % 2 == 0: # 注意大堆顶要取负数
            return (self.minHeap[0]-self.maxHeap[0])/2
```

# 面试题 17.21. 直方图的水量

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水

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

```

```


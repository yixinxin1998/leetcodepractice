# 1. 两数之和

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
      # 未排序可以采用暴力搜索法
        n = len(nums)
        for i in range(n):
            p1 = i
            p2 = i + 1
            while p2 <= n-1:
                if nums[p1] + nums[p2] == target:
                    return [p1,p2]
                p2 += 1
        return 
```

# 2. 两数相加

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        result = ListNode() # 创建新链表作为答案
        head = result # 
        p1 = l1
        p2 = l2
        car = 0 #进位
        while p1 != None and p2 != None:
            if p1.val + p2.val + car <=9:
                node = ListNode(p1.val+p2.val+car,None) #创建新节点
                result.next = node # 结果链表延长，next指向新建节点
                result = result.next # 结果链表指针需要移动一位就绪下一次
                car = 0 # 下次不进位
            elif p1.val + p2.val + car >9:
                node = ListNode(p1.val+p2.val+car-10,None) #创建新节点，注意满10
                result.next = node # 结果链表延长，next指向新建节点
                result = result.next # 结果链表指针需要移动一位就绪下一次
                car = 1 # 下次要进位
            p1 = p1.next
            p2 = p2.next   

        cur = p1 if p1 != None else p2 # 把没有遍历完的链表补上,要记住还有之前可能保留下来的进位
        while cur != None:
            if cur.val + car <= 9:
                node = ListNode(cur.val+car,None)
                result.next = node
                result = result.next
                car = 0
            elif cur.val + car > 9:
                node = ListNode(cur.val+car-10,None)
                result.next = node
                result = result.next
                car = 1
            cur = cur.next
        # 要注意最后一次计算完毕时，是否有进位！如果有进位，需要把进位加上
        if car == 1:
            result.next = ListNode(car,None)
        return head.next
```

# 3. 无重复字符的最长子串

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 滑动窗口法
        window_dict = collections.defaultdict(int)
        left = 0
        right = 0
        ans = 0
        while right < len(s):
            temp_char = s[right] # 即将加入窗口的字符
            right += 1
            window_dict[temp_char] += 1 # 
            while window_dict[temp_char] > 1: #判断窗口是否需要收缩
                delete_char = s[left]
                left += 1
                window_dict[delete_char] -= 1
            ans = max(ans,right-left) # 注意这个ans收集缩进
        return ans
```

# 7. 整数反转

给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围 [−2^31,  2^31 − 1] ，就返回 0。

假设环境不允许存储 64 位整数（有符号或无符号）。

```python
class Solution:
    def reverse(self, x: int) -> int:
        x = str(x)
        lst = []
        for i in x:
            lst.append(i)
        lst[:] = lst[::-1]
        minus = False
        if lst[-1] == '-':
            lst.pop(-1)
            minus = True
        result = 0
        for i in lst:
            result = 10*(result) + int(i)
        if minus:
            result = -1 * result
        if result > ((2**31) -1) or result < (-1 * (2**31)) :
            return 0
        return result
```

# 8. 字符串转换整数 (atoi)

请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。

函数 myAtoi(string s) 的算法如下：

读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。
注意：

本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        if len(s) == 0:
            return 0
        i = 0
        valid = True
        # 去除前导0
        while s[0]==' ' and len(s) !=1:
            s = s[1:]
        if s[0]==' ' and len(s) == 1:
            return 0
         # 判断符号位
        sybol = 1
        if s[0] == "-":
            sybol = -1
            if len(s) != 1:
                s = s[1:]
            else:
                return 0
        elif s[0] == "+":
            if len(s) != 1:
                s = s[1:]
            else:
                return 0
        if not s[0].isdigit():
            return 0
          # 开始处理数字部分
        for i in s:
            if not i.isdigit():
                index = s.index(i)
                s = s[:index]
                break
        result = int(s) * sybol
        	# 处理越界情况
        if result < -2**31:
            result = -2**31
        elif result > 2**31 -1:
            result = 2**31 -1
        return result
            
                
```

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

# 9. 回文数

给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。例如，121 是回文，而 123 不是。

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
      # 字符串化判断
        x = str(x)
        y = x[::-1]
        if y == x:
            return True 
        else: 
            return False
```

# 13. 罗马数字转整数

罗马数字包含以下七种字符: `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        ans = 0
        dict1 = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        # 如果这一位的键值大于下一位键值，则ans直接加，p+=1
        # 如果这一位的键值小于下一位键值，则ans加上特殊处理，p+=2
        # 由于测试用例本身都是合法写法，所以少了很多合法性检验
        p = 0
        while p < len(s):
            if p+1 < len(s):
                if dict1[s[p]] >= dict1[s[p+1]]:
                    ans += dict1[s[p]]
                    p += 1
                elif dict1[s[p]] < dict1[s[p+1]]:
                    ans = ans - dict1[s[p]] + dict1[s[p+1]]
                    p += 2
            else: # 到达边界的情况
                ans += dict1[s[p]]
                p += 1
        return ans

```

# 14. 最长公共前缀

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
      # 暴力解法
        result = ''
        if len(strs) == 0:
            return result
        p = 0 # 词汇指针
        i = 0 # 字母指针
        strs.sort(key=len) # 排序一下，把最短的放前面比较好比较
        while i < len(strs[0]):
            check = strs[p][i] # 提取出待判别的字母
            while p < len(strs): # 检查每个字母
                if strs[p][i] != check:
                    return result
                elif p == len(strs)-1:
                    result += strs[p][i]
                p += 1
            p = 0
            i += 1
        return result
```

# 17. 电话号码的字母组合

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

```python
class Solution:
  	# 生长法
    def letterCombinations(self, digits: str) -> List[str]:
        if digits == "":
            return []
        dict1 = {"2":"abc","3":"def","4":"ghi",'5':"jkl","6":"mon","7":"pqrs","8":"tuv","9":"wxyz"}
        lst = list(digits)
        result = ['']
        while len(lst) != 0:
            elements = lst.pop(0)
            temp = []
            for i in result: # 把原来result中的排出
                for j in dict1[elements]: # 穷举
                    temp.append(i+j) # 做派生树
            result = temp 
        return result
```

# 19. 删除链表的倒数第 N 个结点

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

**进阶：**你能尝试使用一趟扫描实现吗？

```python
# 一种多轮扫描的朴素方法
# 进阶方法参照剑指offer22
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        sz = 0
        cur = head
        if head.next == None:
            head = None
            return head
        while cur.next != None:
            sz += 1
            cur = cur.next
        if n == sz+1 :
            head = head.next
            return head
        else:
            aim = sz  - n
            cur = head
            while aim > 0:
                aim -= 1
                cur = cur.next
            cur.next = cur.next.next
            return head

```

# 20. 有效的括号

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。

```python
class Solution:
		#一种较为复杂的栈方法，可以简化，只入栈左括号，当有匹配的右括号时候弹出，最终检查栈是否为空即可
    def isValid(self, s: str) -> bool:
        if len(s)%2 == 1:
            return False
        stack = []
        p = 0
        while p < len(s):
            if s[p] == '(':
                stack.append(s[p])
            elif s[p] == '{':
                stack.append(s[p])
            elif s[p] == '[':
                stack.append(s[p])
            elif s[p] == ')':
                if len(stack) == 0:
                    return False
                if not stack.pop(-1) == '(':
                    return False
            elif s[p] == '}':
                if len(stack) == 0:
                    return False                
                if not stack.pop(-1) == '{':
                    return False
            elif s[p] == ']':
                if len(stack) == 0:
                    return False
                if not stack.pop(-1) == '[':
                    return False
            p += 1
        if len(stack) == 0:
            return True
        else:
            return False
```

# 21. 合并两个有序链表

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:return l2
        if l2 is None:return l1
        if l1.val < l2.val: # 递归法
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2
```

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
      	#迭代法
        dummy = ListNode()
        move = dummy # 为了方便处理，添加哑节点
        while l1 != None and l2 != None:
            if l1.val < l2.val:
                move.next = l1
                l1 = l1.next
            else:
                move.next = l2
                l2 = l2.next
            move = move.next
        move.next = l1 if l1 else l2 # 剩下来的直接接上去
        return dummy.next
```

# 26. 删除有序数组中的重复项

给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

```python
class Solution:
  	# 双指针法，一个 扫描指针，一个填充指针
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        fast = slow = 1 # slow是填充指针，fast是扫描指针
        while fast < n:
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        
        return slow

```

# 27. 移除元素

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

```python
class Solution:
  	#同上题类似，双指针，填充指针+扫描指针
    def removeElement(self, nums: List[int], val: int) -> int:
        a = 0
        b = 0
        while a < len(nums):
            if nums[a] != val:
                nums[b] = nums[a]
                b += 1
            a += 1
        return b
```

# 34. 在排序数组中查找元素的第一个和最后一个位置

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

进阶：

你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 两次二分，分别找到起始位置和结束位置
        # 当找不到结果时候，返回-1
        # 先找左
        left = 0
        right = len(nums)-1
        while left <= right: # 闭区间找法
            mid = (left+right)//2
            if nums[mid] == target: # 不返回，收缩右边界，锁定左边界
                right = mid - 1
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
        # 判断是否是找到的结果,该算法left的结束的时候的意义是，小于target的数有多少个
        if left >= len(nums) or nums[left] != target : # 这一行很重要,短路运算符or，先后次序不能错
            left_bound = -1
        else:
            left_bound = left
        # 找右边界
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left+right)//2
            if nums[mid] == target: # 固定右边界，收缩左边界
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
        if right < 0 or nums[right] != target :
            right_bound = -1
        else:
            right_bound = right
        return [left_bound,right_bound]
```

# 35. 搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

```python
class Solution:
  	# 顺序扫描法，二分法更好
    def searchInsert(self, nums: List[int], target: int) -> int:
        p = 0
        while p < len(nums):
            if nums[p] >= target:
                return p
            p += 1
        return p
```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        # 越界处理
        if target > nums[-1]: return len(nums)
        if target < nums[0]: return 0
        # 二分过程
        while left <= right: # 闭区间搜索类型
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            # print(left,right) # 这种处理方式最终left一定不会等于right
        return left
```

# 36. 有效的数独

请你判断一个 9x9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
数独部分空格内已填入了数字，空白格用 '.' 表示。

注意：

一个有效的数独（部分已被填充）不一定是可解的。
只需要根据以上规则，验证已经填入的数字是否有效即可。

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        #行筛选
        for a in range (9):
            dict1 = {i:board[a].count(i) for i in board[a] if i != '.' }
            if len(dict1) != sum(dict1.values()):
                return(False)
        #列筛选
        for i in range (9):
            dict2 = {}
            for j in range (9):
                if board[j][i] not in dict2 and board[j][i] != '.':
                    dict2[board[j][i]] = 1
                elif board[j][i] in dict2:
                    return(False)
        # 九宫格筛选
        small_boards = [
                [
                    [board[i][j], board[i][j + 1], board[i][j + 2]],
                    [board[i + 1][j], board[i + 1][j + 1], board[i + 1][j + 2]],
                    [board[i + 2][j], board[i + 2][j + 1], board[i + 2][j + 2]]
                ] for i in range(0, 9, 3) for j in range(0, 9, 3)
            ]
        for i in range(9):
            dict3 = {}
            for j in range(3):
                for k in range(3):
                    if small_boards[i][j][k] != '.' and small_boards[i][j][k] not in dict3:
                        dict3[small_boards[i][j][k]] = 1
                    elif small_boards[i][j][k] in dict3:
                        return(False)
        return True
```

```
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # 行筛选，列筛选，九宫格筛选
        rowSet = [set() for i in range(9)]
        rowCount = [0 for i in range(9)]

        colSet = [set() for i in range(9)]
        colCount = [0 for i in range(9)]

        # 九宫格筛选
        girdSet = [set() for i in range(9)]
        girdCount = [0 for i in range(9)]

        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    e = board[i][j]
                    rowSet[i].add(e)
                    rowCount[i] += 1
                    colSet[j].add(e)
                    colCount[j] += 1
                    index = i // 3 * 3 + j // 3
                    girdSet[index].add(e)
                    girdCount[index] += 1
        
        for i in range(9):
            if len(rowSet[i]) != rowCount[i]:
                return False 
            if len(colSet[i]) != colCount[i]:
                return False 
            if len(girdSet[i]) != girdCount[i]:
                return False 
        return True 
```



# 38. 外观数列

给定一个正整数 n ，输出外观数列的第 n 项。

「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。

你可以将其视作是由递归公式定义的数字字符串序列：

countAndSay(1) = "1"
countAndSay(n) 是对 countAndSay(n-1) 的描述，然后转换成另一个数字字符串。
前五项如下：

1.     1
2.     11
3.     21
4.     1211
5.     111221
第一项是数字 1 
描述前一项，这个数是 1 即 “ 一 个 1 ”，记作 "11"
描述前一项，这个数是 11 即 “ 二 个 1 ” ，记作 "21"
描述前一项，这个数是 21 即 “ 一 个 2 + 一 个 1 ” ，记作 "1211"
描述前一项，这个数是 1211 即 “ 一 个 1 + 一 个 2 + 二 个 1 ” ，记作 "111221"
要 描述 一个数字字符串，首先要将字符串分割为 最小 数量的组，每个组都由连续的最多 相同字符 组成。然后对于每个组，先描述字符的数量，然后描述字符，形成一个描述组。要将描述转换为数字字符串，先将每组中的字符数量用数字替换，再将所有描述组连接起来。

```python
class Solution:
    def countAndSay(self, n: int) -> str:
      	# 生长
        s = '1' # 作为初始值开始生长
        count = 1
        while count < n :
            s = Solution.splited(s) # 调用类方法进行处理
            count += 1
        return s
    
    def splited(s:str): # 该方法把传入的字符串读取出来并且生成新的字符串
        lst = list(s)
        times = 0
        mark_element = None
        new_str = ''
        p1 = 0
        mark_element = lst[p1]
        while p1 < len(s) :
            if mark_element == lst[p1]:
                p1 += 1
                times += 1
            elif mark_element != lst[p1]:
                new_str += str(times)+str(mark_element)
                times = 0 # 重置统计次数
                mark_element = lst[p1]
        new_str += str(times) + str(mark_element)
        return new_str

            
```

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        # 迭代
        start = "1"
        for t in range(n-1):
            p = 0
            pivot = start[0]
            times = 0
            temp = ""
            while p < len(start):
                if start[p] == pivot:
                    times += 1
                    p += 1
                elif start[p] != pivot:
                    temp += str(times) + pivot
                    pivot = start[p]
                    times = 1
                    p += 1
            temp += str(times)+pivot 
            start = temp

        return start
```

```python
class Solution:
    def countAndSay(self, n: int) -> str:
				# 递归
        def helper(n):
            if n == 1:
                return "1"
            temp = helper(n-1)
            p = 0
            pivot = temp[0]
            times = 0
            ans = ""
            while p < len(temp):
                if temp[p] == pivot:
                    times += 1
                    p += 1
                elif temp[p] != pivot:
                    ans += str(times) + pivot
                    pivot = temp[p]
                    times = 1
                    p += 1
            ans += str(times) + pivot
            return ans

        return helper(n)
```



# 39. 组合总和

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：

所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # 回溯
        ans = [] # 存储答案
        stack = [] #存储每次选择的临时答案
        candidates.sort() # 预先排序
        
        def backtracking(candidates,stack,startindex,target): #参数选择列表和临时路径,目标数值
            if sum(stack) == target:
                ans.append(stack[:]) # 传值而不是传引用
                return 
            if sum(stack) > target:
                return # 过界则什么也不做，相当于剪枝
            p = startindex
            while p < len(candidates):
                stack.append(candidates[p])
                backtracking(candidates[p:],stack,startindex,target) # 只会从包含这一位且只在这一位的右边里面进行列表选择
                stack.pop()
                p += 1
        backtracking(candidates,stack,0,target)
        return ans
```

# 45. 跳跃游戏

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        #贪心思路为：跳到当前位置的最远位置，并且判断，下次跳跃时候会跳到更远位置。
        if len(nums) == 1:
            return 0
        step = 1
        p = 0
        cur_cover = nums[p]+p
        if p+nums[p]+1 < len(nums):
            next_cover_tuple = max((i+nums[i],i) for i in range(p,p+nums[p]+1)) 
            next_cover = next_cover_tuple[0]
            next_cover_index = next_cover_tuple[1] # p 需要更新，所以需要把index传递
        while cur_cover < len(nums) and p+nums[p]+1 < len(nums):
            step += 1
            cur_cover = next_cover
            p = next_cover_index
            if p+nums[p]+1 < len(nums):
                next_cover_tuple = max((i+nums[i],i) for i in range(p,p+nums[p]+1)) 
                next_cover = next_cover_tuple[0]
                next_cover_index = next_cover_tuple[1]
        return step
```

# 46. 全排列

给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
      	# 回溯法
        ans = [] #存最终答案
        stack = [] # 存每一步答案
        self.t = len(nums) # 终止条件
        def backtracking(lst,stack): # 参数，选择列表，路径
            if len(stack) == self.t:  # 终止条件
                ans.append(stack[:]) # 注意这里要用拷贝，因为append如果只是加入的stack，传入的只是stack的引用，而收集结果需要实实在在传入值
                return
            p = 0
            while p < len(lst):
                temp = lst.copy()
                e = temp.pop(p)
                stack.append(e)
                backtracking(temp,stack)
                stack.pop()
                p += 1
        backtracking(nums,stack)
        return ans
```

# 48. 旋转图像

给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        ##上下交换
        i = 0
        while i < len(matrix)//2 :
            matrix[i],matrix[len(matrix)-1-i] = matrix[len(matrix)-1-i],matrix[i]
            i += 1
        ##主对角线交换
        for i in range(len(matrix)):
            for j in range(i+1,len(matrix)):
                    matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]

```

# 49. 字母异位词分组

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
说明：

所有输入均为小写字母。
不考虑答案输出的顺序。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 把word排序整理
        # 为了方便，制造拷贝数组
        copy = []
        for index,value in enumerate(strs):
            temp = ''.join(sorted(value))
            copy.append([index,temp])
        copy.sort(key = lambda x:x[1]) # 按照排序后的字符作为key排序
        # 然后提取出序号构建答案数组
        ans = [] 
        mark = copy[0][1] # 初始化标记位
        index_lst = [] # 收集索引，作为构成最终答案使用
        templst = [] # 收集每层元素
        # 超暴力扫描
        for i in copy:
            if i[1] == mark:
                templst.append(strs[i[0]]) # 收集每个符合要求的元素
            elif i[1] != mark:
                index_lst.append(templst) # 将本层收集完毕
                mark = i[1] # 改变新标记位置
                templst = [strs[i[0]]] # 初始化下一层
        index_lst.append(templst) # 由于最后一次没有切换状态，需要主动收集
        return index_lst

```

# 50. Pow(x, n)

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，x^n）。

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        # 快速幂；先考虑n>=0；如果n小于0，取倒数
        # 递归法
        # 这一题对0的0次方定义为1
        # 注意坐标越界
        # 为了降低时间复杂度，必须要带备忘录的递归
        symbol = 1
        if n < 0:
            symbol = -1
        def submethod(x,n):
            if n == 0:
                return 1
            y = submethod(x,n//2)
            if n % 2 == 0:
                return y*y
            elif n % 2 == 1:
                return x * y * y
        if symbol < 0:
            return 1/(submethod(x,-n))
        else:
            return submethod(x,n)

```

# 53. 最大子序和

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 贪心解法 ，贪心的地方：当连续和小于0时候，放弃该和，从下一项重重新累计
        count = 0
        p = 0
        result = max(nums) #这一行针对全为负数时候
        while p < len(nums) :
            count += nums[p]
            if count < 0 :
                mark = p + 1
                count = 0
            elif count > result:
                result = count
            p += 1
        return result
```

# 54. 螺旋矩阵

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

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

# 55. 跳跃游戏

给定一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        #贪心，从第一个位置开始，针对每一个位置，看能达到的最远距离
        p = 0
        cover = 0
        while p < len(nums) and p <= cover:
            if nums[p]+p > cover:
                cover = nums[p]+p
            if cover >= len(nums)-1:
                return True
            p += 1
        return False
```

# 58. 最后一个单词的长度

给你一个字符串 s，由若干单词组成，单词之间用空格隔开。返回字符串中最后一个单词的长度。如果不存在最后一个单词，请返回 0 。

单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if len(s) == 0:
            return 0
        p = -1
        length = 0
        while p > -len(s) - 1:
            if s[p] == ' ' and length == 0:
                p -= 1
            elif s[p] != ' ':
                length += 1
                p -= 1
            elif s[p] == ' ' and length != 0:
                return length
        return length
```

# 59. 螺旋矩阵 II

给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        # 先创造一个 n*n的全0矩阵
        # 然后螺旋赋值
        matrix = [[0 for i in range(n)] for i in range(n)]
        row_start = 0 # 限制区
        col_start = -1 # 限制区
        row_end = len(matrix) # 限制区
        col_end = len(matrix[0]) # 限制
        size = len(matrix)*len(matrix[0])
        # 起点
        i = 0
        j = -1 # 注意这个起点赋值是为了统一语法
        k = 1
        while k <= n**2:
            ########################################
            j += 1
            while j < col_end:
                matrix[i][j] = k
                k += 1
                j += 1
            # 回退到没超过的边界，并且记录下一次边界
            j -= 1 
            col_end -= 1
            ########################################
            i += 1
            while i < row_end:
                matrix[i][j] = k
                k += 1
                i += 1
            # 回退到没超过的边界，并且记录下一次边界
            i -= 1
            row_end -= 1
            ########################################
            j -= 1
            while j > col_start:
                matrix[i][j] = k
                k += 1
                j -= 1
            # 回退到没超过的边界，并且记录下一次边界
            j += 1
            col_start += 1
            ########################################
            i -= 1
            while i > row_start:
                matrix[i][j] = k
                k += 1
                i -= 1
            i += 1 
            row_start += 1
        return matrix
            
```

# 61. 旋转链表

给你一个链表的头节点 `head` ，旋转链表，将链表每个节点向右移动 `k` 个位置。

```python
# 思路：先成环，再断开链接
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head == None:
            return 
        # 先成环 再断开
        cur = head
        size = 0
        while cur.next != None:
            size += 1
            cur = cur.next
        size += 1
        tail = cur
        tail.next = head
        # 成环结束
        k = size -( k % size)
        p = 0
        fast = head
        slow = tail
        while p != k:
            p += 1
            fast = fast.next
            slow = slow.next
        slow.next = None
        head = fast
        return head
```

# 66. 加一

给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        digits[-1] += 1
        for i in range(-1,-len(digits),-1): # 检查是否进位
            if digits[i] == 10:
                digits[i] = 0
                digits[i-1] = digits[i-1]+1 
        if digits[0] == 10: # 检查最高位
            digits[0] = 0
            digits[:] = [1] + digits[:]
        return digits
```

# 67. 二进制求和

给你两个二进制字符串，返回它们的和（用二进制表示）。

输入为 **非空** 字符串且只包含数字 `1` 和 `0`。

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        carry = 0
        a = list(a)
        b = list(b)
        # a 长于 b,否则交换,方便处理
        if len(a) < len(b):
            a,b = b,a
        ans = ''
        while len(b) != 0 and len(a) != 0:
            t = int(b.pop(-1)) + int(a.pop(-1)) + carry
            carry = t//2
            result = t%2 
            ans = str(result) + ans

        while len(a) != 0: # 考虑剩下了的较长串
            t = int(a.pop(-1)) + carry
            carry = t//2
            result = t%2   
            ans = str(result) + ans

        if carry == 1: # 考虑是否有遗留的进位
            ans = '1'+ans
        return ans
```

# 69. x 的平方根

实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        # 二分法
        # 计算并返回 x 的平方根，其中 x 是非负整数。
        # 最开始的两边起点是1和xx
        # 对于x=0,1 直接返回
        if x <= 1:
            return x
        l = 1
        r = x
        ans = -1 # 初始化ans
        while l <= r: # 注意这里是小于等于
            mid = (l+r)//2
            if mid * mid <= x:
                l = mid+1 # 需要增大mid，所以l改变
                ans = mid # 注意 ans赋值丢在了小于这一行里，因为中断循环的时候既有可能从<=出来，也有可能从>出来，
                # 但是走>出来时候，返回结果比实际结果大1，为了避免这一点，所以从<=出来
            elif mid * mid > x:
                r = mid-1 # 需要减小mid，所以r改变

        return ans
 
```

# 70. 爬楼梯

假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**注意：**给定 *n* 是一个正整数。

```python
class Solution:
    def climbStairs(self, n: int) -> int:       
        return Solution.fib(n)

    def fib(n,a=1,b=1):
            if n == 0 :
                return a
            else:
                return Solution.fib(n-1,b,a+b)
```

# 73. 矩阵置零

给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

进阶：

一个直观的解决方案是使用  O(mn) 的额外空间，但这并不是一个好的解决方案。
一个简单的改进方案是使用 O(m + n) 的额外空间，但这仍然不是最好的解决方案。
你能想出一个仅使用常量空间的解决方案吗？

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
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

# 74. 搜索二维矩阵

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 用二叉树的思想看，起点为右上角
        m = len(matrix)
        n = len(matrix[0])
        i = 0
        j = n - 1
        while i < m and j >= 0:
            print(matrix[i][j])
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
        return False
        
        # 该方法比两次二分差劲
```

# 75. 颜色分类

给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 交换位置排序法。扫一轮。且双指针扫
        # 第一轮，先把0扫出来。
        p = 0 #全局位置指针
        p0 = 0 #用于指示0的指针
        while p0 < len(nums):
            if nums[p0] == 0:
                temp = p #需要从首位开始指，需要被调整的位置
                nums[p0],nums[p] = nums[p],nums[p0]
                p = temp + 1 #交换之后全局指针也需要移动
            p0 += 1 #无论是否交换，p0指针都需要移动
        # 之后
        p1 = p #用于指示1的指针，它可以直接从p之后开始扫
        while p1 < len(nums):
            if nums[p1] == 1:
                temp = p
                nums[p1],nums[p] = nums[p],nums[p1]
                p = temp + 1 #交换完之后
            p1 += 1
```

# 76. 最小覆盖子串

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # 滑动窗口
        # 先设计两个字典，利用python的collections.defaultdict
        target_dict = collections.defaultdict(int)
        window_dict = collections.defaultdict(int)
        # 先把t加入到target_dict中
        for char in t:
            target_dict[char] += 1
        # 初始化窗口索引
        left = 0
        right = 0
        # 初始化标记，该标记指示窗口是否已经包含了全部的子字符串,vaild表示的是字符种类个数，只有当该种类的字符数量已经大于等于要求的值的时候，vaild才+1
        valid = 0
        # 初始化符合条件的索引起始和覆盖长度，由于不确定字符串s是否有符合条件的t，所以初始化时候要默认不合法
        start_index = 0
        length = len(s) + 1 # 超越了s的全长，所以是不合法的，把它作为默认值
        while right < len(s): # 当右指针还没有到右端点之外的时候
            # 先预先看好即将要加到窗口内的字符
            temp_char = s[right]
            right += 1
            if temp_char in target_dict: #如果它在目标字典中
                window_dict[temp_char] += 1 #则它就被窗口字典记录
                if window_dict[temp_char] == target_dict[temp_char]: # 不用>=的原因是超过之后如果加入相同字符，valid直接就+1了
                    valid += 1
                while valid == len(target_dict): #只有当valid的字符种类数和对应字符的数量全部达标时候，考虑开始收缩窗口
                    if right - left < length: # 当这个窗口的值小于已经记录的最小窗口时候，才更新start_indext 和 length
                        start_index = left
                        length = right - left
                    
                    delete_char = s[left] # 记录即将被移除的那个字符
                    left += 1
                    if delete_char in target_dict: # 如果它在目标字典中
                        window_dict[delete_char] -= 1 # 它被移除则减少它的数量计数
                        if window_dict[delete_char] < target_dict[delete_char]:
                            valid -= 1 
                        
                    # 由于它是一个循环体，那么它的终止条件 在while valid == len(target_dict): 不被满足时候，停止收缩
            # 注意缩进，和最开始的while对齐
            # 如果滑动窗口滑动完毕之后 length还是原来的不合法值，则return ''
        if length > len(s):
            return ''
        # 否则 根据start_index 和长度返回结果
        else:
            return s[start_index:start_index+length]
```

# 77. 组合

给定两个整数 *n* 和 *k*，返回 1 ... *n* 中所有可能的 *k* 个数的组合。

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
    		# 回溯
        ans = [] # 收集最终结果
        stack = [] # 收集每一条路径下的子结果
        def backtracking(n,stack,startindex):
            if len(stack) == k:
                ans.append(stack[:]) #传入的是值而不是引用
            for i in range(startindex,n+1):
                stack.append(i) #做选择
                backtracking(n,stack,i+1) # 注意回溯的时候的startindex是从i+1位开始！！！
                stack.pop() # 取消选择
        backtracking(n,stack,1)
        return ans
```

# 78. 子集

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # 实际上是对长度在全长之内进行n次取结果。那么每个单次用回溯，从n递减到0用while或者for
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
        for i in range(n+1): # 相当于把全部可能的长度穷举
            backtracking(nums,stack,0,i)
        return ans
```

# 80. 删除有序数组中的重复项 II

给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 最多出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # 扫描指针+填充指针+允许填充次数
        # 注意返回值是长度,而且需要修改数组
        scan = 0 # 扫描
        p = 0 # 填充
        time = 0 # 允许填充次数不得大于2
        mark = nums[0]
        while scan < len(nums): # 这个循环对最后一次收集数据需要再判断
            if nums[scan] == mark and time < 2:
                time += 1
                nums[p] = nums[scan]
                p += 1
                scan += 1
            elif nums[scan] == mark and time >= 2: # 直接跨过去
                time += 1
                scan += 1
            elif nums[scan] != mark: # 如果标志不一样
                time = 1 # 重制次数
                nums[p] = nums[scan]
                mark = nums[scan] # 重置标记位
                scan += 1
                p += 1
        
        return p # 返回的是长度，
                
```

# 83. 删除排序链表中的重复元素

存在一个按升序排列的链表，给你这个链表的头节点 `head` ，请你删除所有重复的元素，使每个元素 **只出现一次** 。

返回同样按升序排列的结果链表。

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
      # 双指针，扫描指针和填充指针
        if head == None:
            return head
        fast = head.next
        slow = head
        while fast != None:
            if fast.val == slow.val:
                slow.next = fast.next
            else:
                slow = slow.next
            fast = fast.next
        return head
```

# 88. 合并两个有序数组

给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # 利用nums1进行排序，由于不希望移动nums1大量元素，所以从尾部开始插入
        p1 = m - 1
        p2 = n - 1
        p = m + n - 1
        while p1 >= 0 or p2 >= 0 :
            if p2 == -1:
                nums1[p] = nums1[p1]
                p1 -= 1
                p -= 1
            elif p1 == -1:
                nums1[p] = nums2[p2]
                p2 -= 1
                p -= 1
            elif nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
                p -= 1
            elif nums1[p1] <= nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
                p -= 1
```

# 92. 反转链表 II

给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        # 需要找到left的前一个节点，right的后一个节点
        # 创建哑节点统一语法
        # 如果为单节点。返回
        # left和right是位置不是值。
        # 把left和right先转化成节点
        cur = head
        count = 1
        while count != left:
            cur = cur.next
            count += 1
        left = cur
        while count != right:
            cur = cur.next
            count += 1
        right = cur
        dummy = ListNode(-1,head)
        cur = dummy
        cur2 = head
        while cur2 != left:
            cur = cur.next
            cur2 = cur2.next
        # 循环结束后，cur指的是left前一个节点
        before_left_node = cur
        left_node = cur2
        # 需要翻转left到right，把right找到先
        while cur2 != right:
            cur2 = cur2.next
        right_node = cur2
        next_right_node = right_node.next
        before_left_node.next = None # 断开链接
        right_node.next = None # 断开right之后的链接
        # 然后反转需要反转的链表
        # left_node之后的链表由于断开了right，所以不需要cur2的终止条件为rihgt_node
        cur1 = None
        cur2 = left_node
        while cur2 != None:
            temp = cur2.next
            cur2.next = cur1
            cur1 = cur2
            cur2 = temp   
        # 翻转完毕之后 cur1指着反转后的头节点
        # 原来的left_node要指向 next_right_node
        before_left_node.next = cur1
        left_node.next = next_right_node
        return dummy.next
```

# 94. 二叉树的中序遍历

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        lst = []
        def submethod(node):
            if node != None:
                submethod(node.left)
                lst.append(node.val)
                submethod(node.right)
        submethod(root)
        return lst  
```

# 98. 验证二叉搜索树

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        # 利用中序遍历,中序遍历有序和二叉搜索树是充要条件
        ans = []
        def submethod(node):
            if node != None:
                submethod(node.left)
                ans.append(node.val)
                submethod(node.right)
        submethod(root)
        return ans == sorted(ans) and len(ans) == len(set(ans))
```

# 99. 恢复二叉搜索树

给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
  	# 交换节点值而不是改变节点连接方式
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        inOrder_lst = []
        def inOrder(node):
            if node != None:
                inOrder(node.left)
                inOrder_lst.append(node) # 加入的是节点
                inOrder(node.right)
        inOrder(root)
        # 发现其中错误排序时候，交换俩错误排序值
        left = 0
        right = len(inOrder_lst)-1
        error1 = 0
        error2 = 0
        # 找错误排序时候注意越界问题
        while left < len(inOrder_lst)-1:
            if inOrder_lst[left].val<inOrder_lst[left+1].val:
                left += 1
            else:
                error1 = left # 记录的是索引
                break
        while right > 0:
            if inOrder_lst[right].val>inOrder_lst[right-1].val:
                right -= 1
            else :
                error2 = right # 记录的是索引
                break
        inOrder_lst[left].val,inOrder_lst[right].val = inOrder_lst[right].val,inOrder_lst[left].val

```

# 100. 相同的树

```python
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
      	# 递归
        # 树相同的条件，值相同，左右子树进一步判断，注意False条件
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

# 101. 对称二叉树

给定一个二叉树，检查它是否是镜像对称的。

```python
# 进阶方法见剑指offer27
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        # 利用先序遍历和变体【根左右和根右左遍历一遍，答案如果一样则对称】
        lst1 = []
        lst2 = []
        def submethod1(node):
            if node != None:
                lst1.append(node.val)
                submethod1(node.left)
                submethod1(node.right)
            else: #需要用None占位，因为常规先序遍历会自动忽略None值
                lst1.append(None)
        def submethod2(node):
            if node != None:
                lst2.append(node.val)
                submethod2(node.right)
                submethod2(node.left)
            else: #需要用None占位，因为常规先序遍历会自动忽略None值
                lst2.append(None)
        submethod1(root)
        submethod2(root)
        return lst1 == lst2
```

# 102. 二叉树的层序遍历

给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。 （即逐层地，从左到右访问所有节点）。

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
    		# BFS
        if root == None:
            return []
        queue = [root]
        result = []
        while len(queue) != 0:
            level = [] #存值，给result
            newqueue =[] #传递给下一次的queue的暂存
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left != None:
                    newqueue.append(i.left)
                if i.right != None:
                    newqueue.append(i.right)
            queue = newqueue
            result.append(level)
        return result

```

# 103. 二叉树的锯齿形层序遍历

给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if root == None:
            return []
        queue = [root] #借助队列管理
        ans = [] # 最终返回的结果
        count = 0 # 用于激活每层是否倒序，作为切片参数
        while len(queue) != 0:
            level = []
            new_queue = []
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            level = level[::(-1)**count] # 用于决定每层是否倒序
            count += 1
            queue = new_queue
            ans.append(level)
        return ans
```

# 104. 二叉树的最大深度

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

**说明:** 叶子节点是指没有子节点的节点。

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root != None:
            return 1+max(self.maxDepth(root.left),self.maxDepth(root.right))
        else:
            return 0
```

# 106. 从中序与后序遍历序列构造二叉树

根据一棵树的中序遍历与后序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
返回如下的二叉树：

​	3

   / \
  9  20
    /  \
   15   7

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # 递归构造
        # 后序的最后一个作为gap分隔，然后用它来建树
        # 树的递归终止条件为postrder == []
        if len(postorder) == 0:
            return None
        else:
            # 以中序来切片
            gap = inorder.index(postorder[-1])
            root = TreeNode(postorder[-1])
            root.left = self.buildTree(inorder[:gap],postorder[:gap])
            root.right = self.buildTree(inorder[gap+1:],postorder[gap:len(postorder)-1])
            return root

```

# 107. 二叉树的层序遍历 II

给定一个二叉树，返回其节点值自底向上的层序遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

```python
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if root == None:
            return []
        # 层序遍历，倒序输出ans
        ans = [] # 最终结果
        queue = [root] # 借助队列管理
        while len(queue) != 0:
            level = [] # 用来加入每层结果
            new_queue = [] # 暂存下一层需要遍历的元素，用来更新queue
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            queue = new_queue
            ans.append(level)
        return ans[::-1] #倒序输出
```

# 108. 将有序数组转换为二叉搜索树

给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        # 对称递归
        if nums == []:
            return
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
```

# 109. 有序链表转换二叉搜索树

给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

```python
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        def buildTree(left,right): # 根据传入的链表递归建造子树
            if left == right: # 递归终止条件
                return None
            mid = self.getMid(left,right)
            node = TreeNode(mid.val)
            node.left = buildTree(left,mid) # 注意左闭右开，右端点取mid【因为mid已经用过了】
            node.right = buildTree(mid.next,right)# 注意左闭右开，左端点取mid.next【因为mid已经用过了】
            return node # 最终返回树节点
        root = buildTree(head,None) # 开始建树
        return root
        

    def getMid(self,left,right): # 由于是单向链表，所以链表取数值使用左闭右开
        # 这个取中点的方法无需断开原链表，只需要传入参数的时候控制边界
        # 利用快慢指针找到中点
        fast = left
        slow = left
        while fast != right and fast.next != right:
            fast = fast.next.next
            slow = slow.next
        return slow

```

# 110. 平衡二叉树

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1 。

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        # 递归，并且需要构建一个height方法
        if root == None:
            return True
        return abs(self.height(root.left)-self.height(root.right)) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)
                      
    def height(self,node):
        if node == None:
            return 0
        else:
            return max(self.height(node.left),self.height(node.right)) + 1
```

# 111. 二叉树的最小深度

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

**说明：**叶子节点是指没有子节点的节点。

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0
        queue = [root] # 队列管理BFS
        depth = 1
        while len(queue) != 0:
            newqueue = []
            for i in queue:
                if (i.left == None and i.right == None):
                    return depth # 有叶子就直接返回了
                if i.left != None:
                    newqueue.append(i.left)
                if i.right != None:
                    newqueue.append(i.right)
            depth += 1 # 
            queue = newqueue
```

# 112. 路径总和

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。

叶子节点 是指没有子节点的节点。

```python
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if root == None:
            return False
        if root.val == targetSum and root.left == root.right == None:
            return True
        elif root.val != targetSum and root.left == root.right == None:
            return False
        else:
            return self.hasPathSum(root.left,targetSum-root.val) or self.hasPathSum(root.right,targetSum-root.val)
```

# 114. 二叉树展开成链表

给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root == None:
            return []
        # 借助列表完成，之后使用java时候再巩固成morris遍历
        lst = [] # 这个lst存储的是节点
        def submethod(node):
            if node != None:
                lst.append(node)
                if node.left != None:
                    submethod(node.left)
                if node.right != None:
                    submethod(node.right)
        submethod(root)
        p = 0 
        while p < len(lst) - 1 :# 最后一个节点需要额外处理
            lst[p].right = lst[p+1]
            lst[p].left = None
            p += 1
        lst[-1].left = None
        lst[-1].right = None
        return root # lst第一个节点正好是需要返回的，所以直接返回root不需要和中序遍历一样以最左节点为根
```

# 116. 填充每个节点的下一个右侧节点指针

给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        # BFS得到所有节点的层
        if root == None:
            return 
        ans = []
        queue = [root]
        while len(queue) != 0:
            level = [] #记录该层的节点
            new_queue = [] #收集下一次要记录的节点
            for i in queue:
                level.append(i)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            # 填充每个节点的next
            p = 0
            if len(level) < 2:
                pass
            elif len(level) >= 2:
                while p < len(level) - 1:
                    level[p].next = level[p+1]
                    p += 1
            queue = new_queue
        return root
```

 # 117. 填充每个节点的下一个右侧节点指针 II

给定一个二叉树

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        # BFS得到所有节点的层
        if root == None:
            return 
        ans = []
        queue = [root]
        while len(queue) != 0:
            level = [] #记录该层的节点
            new_queue = [] #收集下一次要记录的节点
            for i in queue:
                level.append(i)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            # 填充每个节点的next
            p = 0
            if len(level) < 2:
                pass
            elif len(level) >= 2:
                while p < len(level) - 1:
                    level[p].next = level[p+1]
                    p += 1
            queue = new_queue
        return root
```

# 118. 杨辉三角

给定一个非负整数 *numRows，*生成杨辉三角的前 *numRows* 行。

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        n = numRows
        rusult = []
        lst = [1]
        p = 0
        temp = []
        while len(rusult) < n:
            rusult.append(lst)
            p = 0
            while p < len(lst)-1:
                temp.append((lst[p] + lst[p+1]))
                p += 1
            lst = [1] + temp + [1]
            temp = []
        return rusult
```

# 119. 杨辉三角 II

给定一个非负索引 *k*，其中 *k* ≤ 33，返回杨辉三角的第 *k* 行。

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        n = rowIndex+1
        rusult = []
        lst = [1]
        p = 0
        temp = []
        while len(rusult) < n:
            rusult.append(lst)
            p = 0
            while p < len(lst)-1:
                temp.append((lst[p] + lst[p+1]))
                p += 1
            lst = [1] + temp + [1]
            temp = []
        return rusult[n-1]
```

# 121. 买卖股票的最佳时机

给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 扫第一轮构建列表找到到今天为止，历史最低点的价格
        # 然后计算，如果是今天卖出，用今日股价减去今日为止历史最低点的差值
        min_price = prices[0]
        profits_today = []
        p = 0
        while p < len(prices):
            if prices[p] < min_price:
                min_price = prices[p]
            profits_today.append(prices[p]-min_price)
            p += 1
        max_profits = max(profits_today)
        if max_profits > 0:
            return max_profits
        else:
            return 0
```

# 122. 买卖股票的最佳时机 II

给定一个数组 prices ，其中 prices[i] 是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        i = 1
        while i <= len(prices)-1:
            if prices[i-1]<prices[i]:
                profit += (prices[i]-prices[i-1])
                i += 1
            else:
                i += 1
        return profit
```

# 125. 验证回文串

给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

**说明：**本题中，我们将空字符串定义为有效的回文串。

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s=s.lower()
        new = ''
        for i in s:
            if i.isalpha() or i.isdigit():
                new += ''.join(i)
        if new[:] == new[::-1]:
            return True
        else:
            return False
```

# 134. 加油站

在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明: 

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):  # 总提供都小于总需求肯定不能环游
            return -1
        #以下是必定有结果的情况
        rest = [gas[p] - cost[p] for p in range(len(gas))]  # 对每个站单独计算剩余油量
        # 从0开始遍历rest，求rest的sum，当sum<0时候，说明这个点以及这个点之前不可能为起点
        rest = rest + rest  # 直接复制，方便环形计算
        p = 0 # 下标指针
        times = 0 #用来考虑本轮计数次数是否已满
        all_sum = 0
        while times < len(gas):
            all_sum += rest[p]
            times += 1
            if all_sum >= 0 and times == len(gas):
                return p-(times-1) #到p这里完成了len次计数，回退times-1个，说明是从那个开始计算的
            elif all_sum >= 0: #不重置
                p += 1
            elif all_sum < 0:#重置
                p += 1
                times = 0
                all_sum = 0
```

# 136. 只出现一次的数字

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        u = 0
        for i in range(len(nums)):
            u ^= nums[i]            
        return u 
```

# 137. 只出现一次的数字 II

给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次 。**请你找出并返回那个只出现了一次的元素。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 对每一位来说进行二进制数字格式相加，然后二count位要模3，然后最终结果转化成int
        ans = 0
        for i in range(0,32):
            total = sum((val>>i & 1) for val in nums)
            total = total % 3
            ans += total * 2 ** (i)
            if i == 31: # 符号位判断
                if total != 0:
                    ans -= 2**32

        return ans
```

# 138. 复制带随机指针的链表

给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：

val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。

 

示例 1：



输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
示例 2：



输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
示例 3：



输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
示例 4：

输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。


提示：

0 <= n <= 1000
-10000 <= Node.val <= 10000
Node.random 为空（null）或指向链表中的节点。

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
        # 空间复杂度为n的解法
        # 创建字典，把原始节点全部丢进去
        # 注意考虑空链表
        if head == None:
            return None
        dict1 = defaultdict(None) 
        dict1[None] = None # 由于有random可能指向空指针
        cur1 = head
        while cur1 != None:
            dict1[cur1] = Node(cur1.val)
            cur1 = cur1.next
        # 然后第二次遍历字典，把所有next和random填充上
        cur1 = head
        while cur1 != None:
            dict1[cur1].next = dict1[cur1.next]
            dict1[cur1].random = dict1[cur1.random]
            cur1 = cur1.next
        return dict1[head]

```

# 141. 环形链表

给定一个链表，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 true 。 否则，返回 false 。

 

进阶：

你能用 O(1)（即，常量）内存解决此问题吗？

```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
      # 双指针，快慢指针。
        if head == None:
            return False
        quick = head
        slow = head
        while quick != None and quick.next != None:
            quick = quick.next.next
            slow = slow.next
            if slow == quick:
                return True
        return False
```

# 142. 环形链表 II

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

说明：不允许修改给定的链表。

进阶：

你是否可以使用 O(1) 空间解决此题？

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        # 快慢指针，快指针每次走两步，慢指针每次走一步。
        # 情况1，快指针发现无环，则返回null
        # 情况2，快慢指针相遇，说明有环。但是需要找到环的位置还需要
        # 再定义一个指针从头节点开始，新指针和慢都以一步的速度走，相遇点即为入环点。
        # 数学证明易得a = c
        fast = head
        slow = head
        while fast and fast.next != None: # 此循环如果中断则无环
            fast = fast.next.next
            slow = slow.next
            if slow == fast: # 相遇之后进入内层循环
                new = head
                while new != slow:
                    new = new.next
                    slow = slow.next
                return new
        return None
```

# 144. 二叉树的前序遍历

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        def submethod(node):
            if node != None:
                lst.append(node.val)
                submethod(node.left)
                submethod(node.right)
        lst = []
        submethod(root)
        return lst
```

# 145. 二叉树的后序遍历

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        lst = []
        def submethod(node):
            if node != None:
                submethod(node.left)
                submethod(node.right)
                lst.append(node.val)
        submethod(root)
        return lst
```

# 150. 逆波兰表达式求值

根据 逆波兰表示法，求表达式的值。

有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

说明：

整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        while len(tokens) != 0:
            if  tokens[0] != '+' and tokens[0] != '-' and tokens[0] != '*' and tokens[0] != '/':
                stack.append(tokens.pop(0))
            else:
                a = int(stack.pop(-1))
                b = int(stack.pop(-1))
                if tokens[0] == '+':
                    tokens.pop(0)
                    stack.append(b+a)
                elif tokens[0] == '-':
                    tokens.pop(0)
                    stack.append(b-a)
                elif tokens[0] == '*':
                    tokens.pop(0)
                    stack.append(b*a)
                elif tokens[0] == '/':
                    tokens.pop(0)
                    stack.append(b/a)
        return int(stack[0])
```

# 151. 翻转字符串里的单词

给你一个字符串 s ，逐个翻转字符串中的所有 单词 。

单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。

请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。

说明：

输入字符串 s 可以在前面、后面或者单词间包含多余的空格。
翻转后单词间应当仅用一个空格分隔。
翻转后的字符串中不应包含额外的空格。

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.split(' ')
        while '' in s:
            s.remove('')
        ans = ''
        for i in s:
            ans = ' '+ i + ans
        return ans[1:]
```

# 153. 寻找旋转排序数组中的最小值

已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # 先考虑未发生旋转的情况
        # 此时最左边小于最右边【且只有在初始状态满足这个条件的时候它是未旋转，直接返回首项】
        # 这是所有数字各不相同的情况
        left = 0
        right = len(nums)-1
        if nums[left] < nums[right]:
            return nums[left]
        # 否则，一定发生了旋转，且最左边一定小于最右边
        # 画图辅助，分为左排序数组和右排序数组，最小值是右排序的第一个值
        # 开始二分查找，闭区间查找
        # situation1 :中值要么比有效范围内的最左边小，中值处于右排序，收缩右排序
        # situation2 :要么比有效范围内的最左边大，中值处于左排序，收缩左排序
        while left  < right : # left和right循环终止条件为指向同一位置
            mid = (left+right)//2
            # 循环中，left会指向最小值
            if nums[mid] < nums[left]: # s1
                right = mid
            elif nums[mid] > nums[left]: # s2
                left = mid
            else:  # 这一条很关键，退出循环
                left += 1
        return nums[left]
```

# 154. 寻找旋转排序数组中的最小值 II

已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # 先考虑未发生旋转的情况
        # 此时最左边小于最右边【且只有在初始状态满足这个条件的时候它是未旋转，直接返回首项】
        # 相同的情况下见while循环中的else
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

# 155. 最小栈

设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

push(x) —— 将元素 x 推入栈中。
pop() —— 删除栈顶的元素。
top() —— 获取栈顶元素。
getMin() —— 检索栈中的最小元素。

```python
class MinStack:
		# 指针辅助法
    def __init__(self):
        """
        initialize your data structure here.
        """
        self._data = []
        self._min_index = 0
    
    def _min(self):
        p1 = 0
        min1 = 0
        while p1 < len(self._data) :
            if self._data[p1] < self._data[min1]:
                min1 = p1
            p1 += 1
        self._min_index = min1

    def push(self, val: int) -> None:
        self._data.append(val)
        MinStack._min(self)

    def pop(self) -> None:
        self._data.pop(-1)
        MinStack._min(self)


    def top(self) -> int:
        return self._data[-1]

    def getMin(self) -> int:
        return self._data[self._min_index]

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

```python
class MinStack:
		# 双栈法
    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]

    def push(self, x: int) -> None:
        self.stack.append(x)
        self.min_stack.append(min(x, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


```

# 160. 相交链表

给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。

图示两个链表在节点 c1 开始相交：

题目数据 保证 整个链式结构中不存在环。

注意，函数返回结果后，链表必须 保持其原始结构 。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        dict1 = {}
        cur1 = headA
        while cur1 != None:
            dict1[cur1] = cur1.val
            cur1 = cur1.next
        cur2 = headB
        while cur2 != None:
            if dict1.get(cur2) != None:
                return cur2
            elif dict1.get(cur2) == None:
                cur2 = cur2.next
        return None
```

# 167. 两数之和 II - 输入有序数组

给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。

函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 1 开始计数 ，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。

你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # 创建俩指针从两端开始
        # 且这里的索引是从1开始,在返回结果时候要注意
        # 而且这里默认答案一定存在
        left = 0
        right = len(numbers)-1
        ans = []
        while left < right:
            if numbers[left]+numbers[right] == target:
                ans.append(left+1)
                ans.append(right+1)
                return ans
            elif numbers[left]+numbers[right] > target: #值大了，所以右指针左移
                right -= 1
            elif numbers[left]+numbers[right] < target: #值小了，所以左指针右移
                left += 1
```

# 168. Excel表列名称

给定一个正整数，返回它在 Excel 表中相对应的列名称。

```
class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        str1 = ''
        while columnNumber > 26:
            if columnNumber%26 != 0:
                t = columnNumber%26
                columnNumber = columnNumber//26
 
            elif columnNumber%26 == 0:
                t = columnNumber%26 + 26
                columnNumber = columnNumber//26 - 1
            str1 = chr(t+64) +str1

        str1 = chr(columnNumber+64) +str1
        return str1
```

# 169. 多数元素

给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```python
class Solution:
  	#	排序后中点一定是
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]
```

# 171. Excel表列序号

给定一个Excel表格中的列名称，返回其相应的列序号。

```python
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        p = -1
        result = 0
        while p > -len(columnTitle)-1:
            result += (ord(columnTitle[p])-64)*(26**(-p-1))
            p -= 1
        return result
```

# 173. 二叉搜索树迭代器

实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：
BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 的根节点 root 会作为构造函数的一部分给出。指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。
boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则返回 false 。
int next()将指针向右移动，然后返回指针处的数字。
注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返回 BST 中的最小元素。

你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序遍历中至少存在一个下一个数字。

```python
class BSTIterator:
		#	初始化的时候就把next数组做好
    def __init__(self, root: TreeNode):
        self.point = None
        self.inorder = []
        self.p = -1
        def submethod(node):
            if node != None:
                if node.left:
                    submethod(node.left)
                self.inorder.append(node)
                if node.right:
                    submethod(node.right)
        submethod(root)

    def next(self) -> int:
        if self.point == None:
            self.p += 1
            return self.inorder[self.p].val

    def hasNext(self) -> bool:
        if self.p + 1 < len(self.inorder):
            return True
        return False
 
```

# 175. 组合两个表

表1: Person

+-------------+---------+
| 列名         | 类型     |
+-------------+---------+
| PersonId    | int     |
| FirstName   | varchar |
| LastName    | varchar |
+-------------+---------+
PersonId 是上表主键
表2: Address

+-------------+---------+
| 列名         | 类型    |
+-------------+---------+
| AddressId   | int     |
| PersonId    | int     |
| City        | varchar |
| State       | varchar |
+-------------+---------+
AddressId 是上表主键


编写一个 SQL 查询，满足条件：无论 person 是否有地址信息，都需要基于上述两表提供 person 的以下信息：

```sql
# Write your MySQL query statement below
select FirstName,LastName,City,State
from Person
left outer join Address on Person.PersonId = Address.PersonId;
```

# 183. 从不订购的客户

某网站包含两个表，Customers 表和 Orders 表。编写一个 SQL 查询，找出所有从不订购任何东西的客户。

Customers 表：

+----+-------+
| Id | Name  |
+----+-------+
| 1  | Joe   |
| 2  | Henry |
| 3  | Sam   |
| 4  | Max   |
+----+-------+
Orders 表：

+----+------------+
| Id | CustomerId |
+----+------------+
| 1  | 3          |
| 2  | 1          |
+----+------------+
例如给定上述表格，你的查询应返回：

+-----------+
| Customers |
+-----------+
| Henry     |
| Max       |
+-----------+

```sql
# Write your MySQL query statement below
select Customers.Name as Customers
from Customers
where Customers.Id not in (select Orders.CustomerId from Orders);
# 效率很低啊
```

```SQL
# Write your MySQL query statement below
select Customers.Name as Customers
from Customers
left outer join Orders on Orders.CustomerId = Customers.id
where CustomerId is null; # 这里不能写成CustomerId = null。

```



# 189. 旋转数组

给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

进阶：

尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
你可以使用空间复杂度为 O(1) 的 原地 算法解决这个问题吗？

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums[:] = nums[len(nums)-k::] + nums[:len(nums)-k]

```

# 190. 颠倒二进制位

颠倒给定的 32 位无符号整数的二进制位。

提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 2 中，输入表示有符号整数 -3，输出表示有符号整数 -1073741825。


进阶:
如果多次调用这个函数，你将如何优化你的算法？

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        # 位运算
        ans = 0
        for i in range(32):
            bit = (n>>i)&1
            ans = (ans+bit<<1)
        ans = ans >> 1
        return ans
```

# 191. 位1的个数

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。

提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        for i in bin(n):
            if i == '1':
                count += 1
        return count
```

# 198. 打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

 

示例 1：

输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
示例 2：

输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。


提示：

1 <= nums.length <= 100
0 <= nums[i] <= 400

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 动态规划
        # 带dp数组的动态规划，确定状态转移
        # 利用空间复杂度为O(n)的dp数组
        # 对于每个格子，要么选，要么不选之后考虑相邻格子
        # 其状态转移方程为
        # fn = max(fn-1 , fn-2 + nums[i])
        # 当n=2时候，选max(f0,f1)
        # 当n=1的时候，选f0
        dp = [-1 for i in range(len(nums))] # 初始化dp数组
        for i in range(len(nums)): # 
            if i == 1:
                dp[i] = max(nums[0],nums[1])
            elif i == 0:
                dp[i] = nums[0]
            else:
                dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        return dp[-1] 
```

# 199. 二叉树的右视图

给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

```python
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        # 广度优先遍历的每层的最后一个值
        # 借助队列管理
        if root == None:
            return []
        ans = [] # 最终结果
        queue = [root] #BFS的队列管理
        while len(queue) != 0:
            level = []
            new_queue = []
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            queue = new_queue
            ans.append(level[-1]) # 只把每一层的最后一个值加入
        return ans
```

# 202. 快乐数

编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」定义为：

对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
如果 可以变为  1，那么这个数就是快乐数。
如果 n 是快乐数就返回 true ；不是，则返回 false 。

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        ban = [89,145,42,20,4,16,37,58]
        while Solution.change(n) != 1 :
            if Solution.change(n) in ban:
                return False
            n = Solution.change(n)
        if Solution.change(n) == 1:
            return True

    def change(n:int):
        n = str(n)
        result = 0
        for i in n:
            result += int(i)**2
        return result
```

# 203. 移除链表元素

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

```python
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        if head == None:
            return head
        fast = head.next
        slow = head
        while fast != None:
            if fast.val == val:
                slow.next = fast.next            
                fast = fast.next
            else:
                fast = fast.next
                slow = slow.next
        if head.val == val:
            head = head.next
        return head
```

# 205. 同构字符串

给定两个字符串 s 和 t，判断它们是否是同构的。

如果 s 中的字符可以按某种映射关系替换得到 t ，那么这两个字符串是同构的。

每个出现的字符都应当映射到另一个字符，同时不改变字符的顺序。不同字符不能映射到同一个字符上，相同字符只能映射到同一个字符上，字符可以映射到自己本身。

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        gap_dict1 = {}
        p1 = 0
        state1 = True
        while p1 < len(s):
            if s[p1] not in gap_dict1:
                gap_dict1[s[p1]] = t[p1]
            if s[p1] in gap_dict1:
                if gap_dict1[s[p1]] != t[p1]:
                    state1 = False
            p1 += 1

        gap_dict2 = {}
        p2 = 0
        state2 = True
        while p2 < len(t):
            if t[p2] not in gap_dict2:
                gap_dict2[t[p2]] = s[p2]
            if t[p2] in gap_dict2:
                if gap_dict2[t[p2]] != s[p2]:
                    state2 = False
            p2 += 1
        
        return (state1 and state2)
```

# 206. 反转链表

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None:
            return None
        New = ListNode(head.val,None)
        cur = head.next
        while cur != None:
            node = ListNode(cur.val,New)
            New = node
            cur = cur.next
        return New
            
```

# 209. 长度最小的子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # 滑动窗口法
        window = 0
        left = 0
        right = 0
        ans = len(nums)+1 # 初始化默认为不含有符合条件的子数组，标记长于全场
        while right < len(nums):
            window += nums[right] # window加入
            right += 1
            while left<=right and window >= target: # 收缩条件
                ans = min(ans,right-left) # 先收集最小值，再调整窗口，注意此时的right已经在while前+1了
                window -= nums[left]
                left += 1
        return 0 if ans == len(nums)+1 else ans # 如果滑动完毕ans都没有被更新过，则返回0

            
```

# 216. 组合总和 III

找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。

说明：

所有数字都是正整数。
解集不能包含重复的组合。 

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        nums = [x for x in range(1,10)]
        ans = [] # 收集最终答案
        stack = [] # 收集路径答案
        # 回溯
        def backtracking(nums,stack,startindex):
            # 参数为：选择列表，选择路径，起始序号【该序号也用于分隔防止重复选择】
            if sum(stack) == n and len(stack) == k : #限定条件在外部结构体已经定义了，就不写在本轮参数
                ans.append(stack[:]) # 传值而不是传引用
                return 
            if sum(stack) > n or len(stack) > k: # 简易剪枝
                return 
            p = startindex
            while p < len(nums):
                stack.append(nums[p]) # 做选择
                backtracking(nums,stack,p+1) # 注意用的参数是p + 1
                stack.pop() # 撤销选择
                p += 1
        backtracking(nums,stack,0)
        return ans
```

# 217. 存在重复元素

给定一个整数数组，判断是否存在重复元素。

如果存在一值在数组中出现至少两次，函数返回 `true` 。如果数组中每个元素都不相同，则返回 `false` 。

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if len(nums) == len(set(nums)):
            return False
        else:
            return True
```

# 219. 存在重复元素 II

给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 nums [i] = nums [j]，并且 i 和 j 的差的 绝对值 至多为 k。

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # 特别巧妙的哈希滑动窗口
        # 维护一个窗口大小最多为k的哈希表
        # 每次要加入窗口元素时候
        # 1.先查找哈希表内是否有这个要加入的元素
        # 2.把元素加入
        # 3.如果窗口大小大于k，remove掉最左边的元素，利用下标来算得值，再remove
        window = set()
        n = len(nums)
        for i in range(n):
            if nums[i] in window:
                return True
            window.add(nums[i])
            if len(window) > k:
                window.remove(nums[i-k])
        return False

```

# 222. 完全二叉树的节点个数

给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。

完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。

```python
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        # 递归查找
        # 如果左深度等于右深度，则左边一定是满二叉树，右边不一定是满二叉树，对右边递归
        # 如果左深度不等于右深度，左边一定大于右边，则对右边一定是满二叉树，左边不一定是满二叉树，对左边递归
        if root == None:
            return 0
        height_left = self.height(root.left)
        height_right = self.height(root.right)
        # 注意运算时候是左子树数量+根（1）+右子树数量
        if height_left == height_right:
            return (2**height_left-1) + 1+ self.countNodes(root.right) 
        else:
            return self.countNodes(root.left) +1+ (2**height_right-1)
        # 利用高度来求节点数
    def height(self,node): #根节点深度为1的计算
        if node == None:
            return 0
        else:
            return 1+max(self.height(node.left),self.height(node.right))
# 这一个解法的瑕疵是，height的计算在递归的时候都重复计算了，可以优化一下变成递归一次+1。
```

# 225. 用队列实现栈

请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通队列的全部四种操作（push、top、pop 和 empty）。

实现 MyStack 类：

void push(int x) 将元素 x 压入栈顶。
int pop() 移除并返回栈顶元素。
int top() 返回栈顶元素。
boolean empty() 如果栈是空的，返回 true ；否则，返回 false 。


注意：

你只能使用队列的基本操作 —— 也就是 push to back、peek/pop from front、size 和 is empty 这些操作。
你所使用的语言也许不支持队列。 你可以使用 list （列表）或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。

```python
class MyStack:
    from collections import deque
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._data = collections.deque()


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self._data.appendleft(x)

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        if not MyStack.empty(self):
            e = self._data.popleft()
        return e

    def top(self) -> int:
        """
        Get the top element.
        """
        if not MyStack.empty(self):
            e = self._data[0]
        return e

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self._data) == 0



# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```

# 226. 翻转二叉树

翻转一棵二叉树。

示例：

输入：

​	 4

   /   \
  2     7
 / \   / \
1   3 6   9
输出：

​	 4

   /   \
  7     2
 / \   / \
9   6 3   1

```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root == None:
            return None
        def swap(node):
            if node != None:
                node.left,node.right = node.right,node.left
            if node.left != None:
                swap(node.left)
            if node.right != None:
                swap(node.right)
        swap(root)
        return root
```

# 227. 基本计算器 II

给你一个字符串表达式 `s` ，请你实现一个基本计算器来计算并返回它的值。

整数除法仅保留整数部分。

```python
class Solution:
    def calculate(self, s: str) -> int:
        # 思路 先处理所有的乘除法部分，包括连乘和连除，使用栈记录。
        # 先处理乘除法，遇见加号先直接入栈数字，遇见减号入栈数字的相反数，遇见乘除号将数字弹出一位与下一位进行运算后进行入栈
        # 字符处理去除全部的空格,并且把所有大于一位数的数字合并
        s = list(s)
        new_lst = [] #暂存格式
        p = 0 # 指向全部的位置
        while p < len(s):
            if s[p] != ' ' and s[p].isdigit() == False: #符号处理
                new_lst.append(s[p])  
                p += 1
            elif s[p].isdigit():
                mark = p + 1
                number = int(s[p])
                while mark < len(s) and s[mark].isdigit():
                    number = int(number)*10 + int(s[mark])
                    mark += 1
                new_lst.append(str(number))
                p = mark
            elif s[p] == ' ':
                p += 1
        s = new_lst
        stack = []
        p = 0
        while p < len(s):
            if s[p].isdigit():
                stack.append(int(s[p]))
                p += 1
            elif s[p] == '-':
                stack.append(-int(s[p+1]))
                p += 2
            elif s[p] == '*':
                prev = stack.pop()
                stack.append(prev*int(s[p+1]))
                p += 2
            elif s[p] == '/':
                prev = stack.pop()
                stack.append(int(prev/int(s[p+1])))
                p += 2
            else :
                p += 1
        return sum(stack)
```

# 228. 汇总区间

给定一个无重复元素的有序整数数组 nums 。

返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。

列表中的每个区间范围 [a,b] 应该按如下格式输出：

"a->b" ，如果 a != b
"a" ，如果 a == b

```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        if len(nums) == 0:
            return []
        elif len(nums) == 1:
            return [str(nums[0])]
        p = 0
        result = []
        nums += [nums[-1]] #由于p的比较只会比较到num的倒数第二个
        #那么把nums改造，复制它的最后一个元素
        while p < len(nums)-1: # p只会走到nums的倒数第二个 
            level = ''
            start = p
            while nums[p]+1 == nums[p+1] and p < len(nums)-1:
                p += 1
            end = p
            if start != end:
                level = str(nums[start])+'->'+str(nums[end])
                p += 1
            elif nums[start] == nums[end]:
                level = str(nums[start])
                p += 1
            result.append(level)
        return result
```

# 230. 二叉搜索树中第K小的元素

给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k` 个最小元素（从 1 开始计数）。

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # 中序遍历 取第k个
        # 递归法
        lst = []
        def submethod(node):
            if node != None:
                submethod(node.left)
                lst.append(node.val)
                submethod(node.right)
        submethod(root)
        return lst[k-1]
```

# 231. 2 的幂

给你一个整数 n，请你判断该整数是否是 2 的幂次方。如果是，返回 true ；否则，返回 false 。

如果存在一个整数 x 使得 n == 2x ，则认为 n 是 2 的幂次方。

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        return n&(n-1) == 0
```

# 232. 用栈实现队列

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

实现 MyQueue 类：

void push(int x) 将元素 x 推到队列的末尾
int pop() 从队列的开头移除并返回元素
int peek() 返回队列开头的元素
boolean empty() 如果队列为空，返回 true ；否则，返回 false


说明：

你只能使用标准的栈操作 —— 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。


进阶：

你能否实现每个操作均摊时间复杂度为 O(1) 的队列？换句话说，执行 n 个操作的总时间复杂度为 O(n) ，即使其中一个操作可能花费较长时间。

```python
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []
        self.temp_stack = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.stack.append(x)


    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if len(self.stack) != 0:
            while len(self.stack) != 1:
                self.temp_stack.append(self.stack.pop(-1))
        e = self.stack.pop(-1)
        while len(self.temp_stack) != 0:
            self.stack.append(self.temp_stack.pop(-1))
        return e


    def peek(self) -> int:
        """
        Get the front element.
        """
        if not self.empty():
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

# 234. 回文链表

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        lst = []
        cur = head
        while cur != None:
            lst.append(cur.val)
            cur = cur.next                   
        cur = head
        while cur.next != None:
            if cur.val == lst.pop(-1):
                cur = cur.next
            else:
                return False
        return True
```

# 235. 二叉搜索树的最近公共祖先

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 如果俩节点的值都小于根节点，那么递归到根节点的左子树
        # 如果俩节点的值都大于根节点，那么递归到根节点的右子树
        # 如果俩节点的值包夹了根节点【闭区间】，那么就是这个根节点,直接else就行
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left,p,q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right,p,q)
        else:
            return root
```

# 237. 删除链表中的节点

请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点。传入函数的唯一参数为 **要被删除的节点** 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        # 这一题很有意思，把下一个值填充到这个节点上，
        node.val = node.next.val
        node.next = node.next.next
```

# 238. 除自身以外数组的乘积

给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 节约空间复杂度
        # 假设原数组为 a b c d
        # 第一轮左扫得到L数组 1 a ab abc ，利用临时变量K
        # 第二轮倒序右扫利用临时变量K从1开始直接对R数组进行操作
        L = [1]
        p = 1
        while p < len(nums):
            L.append(L[p-1]*nums[p-1])
            p += 1
        temp = 1
        p = -1
        while p > -len(nums) -1:
            L[p] = temp * L[p]
            temp *= nums[p]
            p -= 1
        return L
```

# 240. 搜索二维矩阵 II

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 转45度来看，【即看右上角】
        # 如果target和目标相等，返回true
        # 如果target小于matrix，则左找
        # 如果target大于matrix，则下找
        # 如果越界，则返回false
        m = len(matrix[0]) # 行限制
        n = len(matrix) # 列限制
        i = 0
        j = len(matrix[0])-1 
        # i，j为起始查找点
        while i >=0 and i < n and j<m and j>=0:
            print(matrix[i][j])
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] >= target:
                j -= 1
            elif matrix[i][j] <= target:
                i += 1
        return False
```

# 242. 有效的字母异位词

给定两个字符串 *s* 和 *t* ，编写一个函数来判断 *t* 是否是 *s* 的字母异位词。

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s = sorted(s)
        t = sorted(t)
        if s == t:
            return True
        else:
            return False
```

# 257. 二叉树的所有路径

给定一个二叉树，返回所有从根节点到叶子节点的路径。

**说明:** 叶子节点是指没有子节点的节点。

```python
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        # 利用回溯法做
        ans = [] # 收集最终答案的列表形式
        stack = [root.val] # 收集路径答案
        def backtracking(node,stack): #选择列表，做出的选择
            if node == None:
                return 
            if node.left == None and node.right == None:
                ans.append(stack[:]) # 收集结果
            lst = []
            if node.left != None:
                lst.append(node.left)
            if node.right != None:
                lst.append(node.right)
            for i in lst:
                stack.append(i.val)
                backtracking(i,stack)
                stack.pop()
        backtracking(root,stack)
        new_ans = []
        for i in ans:
            string = ''
            for j in i:
                string += str(j)+'->'
            new_ans.append(string[:-2])
        
        return new_ans
```

# 258. 各位相加

给定一个非负整数 `num`，反复将各个位上的数字相加，直到结果为一位数。

```python
class Solution:
    def addDigits(self, num: int) -> int:
        lst = [int(i) for i in str(num)]
        while len(lst) != 1:
            k = sum(lst)
            lst = [int(i) for i in str(k)]
        return lst[0]
```

# 260. 只出现一次的数字 III

给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。你可以按 任意顺序 返回答案。

进阶：你的算法应该具有线性时间复杂度。你能否仅使用常数空间复杂度来实现？

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        # 分治法
        # 首先第一轮直接全异或。然后得到的结果相当于最终a和b的异或，然后根据a和b异或结果中任意一位为1的位
        # 将数组分成两小组，两小组再全组异或输出结果
        a = 0
        for i in nums:
            a ^= i
        # 此时a至少有一位为1，从最低位数起，如果对上了则停止
        mark = 0
        for i in range(32): # 找到a为1的那一位
            if (a>>(i))&1 == 1:
                mark = i # 标记好是第几位,最低位置记作第0位
                break
        group1 = []
        group2 = []
        for i in nums:
            if (i>>mark)&1 == 1:
                group1.append(i)
            elif (i>>mark)&1 == 0:
                group2.append(i)
        ans1 = 0
        ans2 = 0
        for i in group1:
            ans1 ^= i
        for i in group2:
            ans2 ^= i
        return [ans1,ans2]
```

# 263. 丑数

给你一个整数 n ，请你判断 n 是否为 丑数 。如果是，返回 true ；否则，返回 false 。

丑数 就是只包含质因数 2、3 和/或 5 的正整数。

```python
class Solution:
    def isUgly(self, n: int) -> bool:
        if n < 1:
            return False
        while n%2 == 0:
            n = n/2
        while n%3 == 0:
            n = n/3
        while n%5 == 0:
            n = n/5
        if n == 1:
            return True
        else:
            return False
```

# 268. 丢失的数字

给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。

进阶：

你能否实现线性时间复杂度、仅使用额外常数空间的算法解决此问题?

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        sum1 = sum(nums)
        sum_all = len(nums)*(len(nums)+1)//2
        result = sum_all - sum1
        return result
```

# 274. H 指数

给定一位研究者论文被引用次数的数组（被引用次数是非负整数）。编写一个方法，计算出研究者的 h 指数。

h 指数的定义：h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。且其余的 N - h 篇论文每篇被引用次数 不超过 h 次。

例如：某人的 h 指数是 20，这表示他已发表的论文中，每篇被引用了至少 20 次的论文总共有 20 篇。

```python
class Solution:
#排序之后倒序指针
    def hIndex(self, citations: List[int]) -> int:
        citations.sort()
        p = -1
        h = 0
        while p > - len(citations) - 1:
            if citations[p] > h:
                h += 1
                p -= 1
            else:
                break 
        return h
```

# 275. H 指数 II

给定一位研究者论文被引用次数的数组（被引用次数是非负整数），数组已经按照 升序排列 。编写一个方法，计算出研究者的 h 指数。

h 指数的定义: “h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。（其余的 N - h 篇论文每篇被引用次数不多于 h 次。）"

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        h = 0
        p = -1 # index
        while p > -len(citations) - 1:
            if citations[p] > h:
                h += 1
                p -= 1
            else:
                break
        return h
```

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        # 利用二分法去找
        # 实际上相当于倒着找，靠左的边界
        # 注意边界条件处理
        n = len(citations)
        left = 0
        right = n-1 # 全闭区间找法
        while left<=right:
            mid = (left+right)//2
            if citations[mid] >= n - mid: # 中点元素值数量偏多，中点坐标需要左移动，改变right值
                right = mid-1
            elif citations[mid] < n - mid: # 中点元素值偏少，中点坐标需要右边移动，改变left值
                left = mid + 1
        # 循环结束条件为 left + 1 == right
        # 需要返回的值为 n-left
        return n - left
```

# 278. 第一个错误的版本

你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。

假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。

你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

```python
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return an integer
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 统计目标“True"的最靠左索引 [F,F,F,F,TTT]
        left = 1
        right = n
        while left < right: # 这里不能是等号，会陷入死循环
            mid = (left+right)//2
            if isBadVersion(mid) == True:
                right = mid
            elif isBadVersion(mid) == False:
                left = mid + 1
        return left
```

# 283. 移动零

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        count = 0
        for i in range(len(nums)-1,-1,-1):
            if nums[i] == 0:
                nums.pop(i)
                count += 1
        step = 1
        while step <= count:
            nums.append(0)
            step += 1
        return nums
```

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 双指针，一个扫描指针一个填充指针
        # 从左扫
        p1 = 0 # 扫描指针
        p2 = 0 # 填充指针
        count = 0 # 计算有几个0
        while p1 < len(nums):
            if nums[p1] == 0:
                count += 1
                p1 += 1
            elif nums[p1] != 0:
                nums[p2] = nums[p1]
                p2 += 1
                p1 += 1
        p = - 1
        while count > 0: # 把数出来0全部填充/其实可以接着p2直接填充
            nums[p] = 0
            p -= 1
            count -= 1
```

# 285. 二叉搜索树中的中序后继

给定一棵二叉搜索树和其中的一个节点 p ，找到该节点在树中的中序后继。如果节点没有中序后继，请返回 null 。

节点 p 的后继是值比 p.val 大的节点中键值最小的节点。

 

示例 1：

输入：root = [2,1,3], p = 1
输出：2
解释：这里 1 的中序后继是 2。请注意 p 和返回值都应是 TreeNode 类型。
示例 2：

输入：root = [5,3,6,2,4,null,null,1], p = 6
输出：null
解释：因为给出的节点没有中序后继，所以答案就返回 null 了。

```python
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        inorder_lst = []
        # 注意比较的是节点
        # p已经确定在树中了
        # 直接中序遍历法查找即可
        def inorder(node):
            if node == None:
                return
            inorder(node.left)
            inorder_lst.append(node)
            inorder(node.right)
        inorder(root) # 把节点存在数组中
        for i in range(len(inorder_lst)): # 遍历数组
            if inorder_lst[i] == p:
                if i <= len(inorder_lst)-2: 
                    return inorder_lst[i+1]
                else:
                    return None

```

# 287. 寻找重复数

给定一个包含 n + 1 个整数的数组 nums ，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。

假设 nums 只有 一个重复的整数 ，找出 这个重复的数 。

你设计的解决方案必须不修改数组 nums 且只用常量级 O(1) 的额外空间。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # 这一题的陷阱，数字虽然只有一个是重复的，但是不知道这个数重复多少次
        # 所以不能使用位运算
        # 由于要求不能修改数组，且只能使用常量级的额外空间
        # 思路：由于一共有n+1个数字，利用二分查找的思路
        # 每次查找子数组的一半，例如是8个数字的话，[2,3,5,4,3,2,6,7]
        # 按照闭区间分组[1,4],[5,8]
        # 对区间计数,取4为标志，如果count<=4的数量为4，则说明1，2，3，4占齐了，那么重复数字一定在另一边
        # 如果count>4 说明重复的数字在[1,4]组里，再对这一组进行拆分[1,2],[3,4]
        # 如果对[1,2]的count > 2，则在这里
        # 如果对[2]的count > 2且 这个列表长度为1了，返回
        
        left = 1 # 注意这个left是1
        right = len(nums)-1 # 这个right的值是n
        while left < right: # 采用小于号，由于left和right不是同步靠拢式更新
            mid = (left+right)//2
            count = 0
            for i in nums:
                if i <= mid:
                    count += 1
            if count > mid: # 如果记录个数大于查找长度，说明在[left,mid]区间内
                right = mid 
            else: # 其实只会记录数小于长度数量，说明在另一半[mid+1,right]内
                left = mid + 1
        return left

                
```

# 289. 生命游戏

根据 百度百科 ，生命游戏，简称为生命，是英国数学家约翰·何顿·康威在 1970 年发明的细胞自动机。

给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态：1 即为活细胞（live），或 0 即为死细胞（dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：

如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。给你 m x n 网格面板 board 的当前状态，返回下一个状态。

```python
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 四种定律，暴力解
        # 创建一个状态判定函数,传入参数为:i,j
        def judge(i,j):  # 注意不能算自己
            state = 0
            for x in range(j-1,j+2):
                for y in range(i-1,i+2):
                    if 0<=x<len(board[0]) and 0<=y<len(board):
                        if board[y][x] == 1:
                            state += 1
            if board[i][j] == 1: # 剔除掉自己
                state -= 1
            return state
            
        # 开始遍历
        # 为了防止在搜索时发生篡改
        # 使用列表记录需要改变的值
        alive_lst = []
        death_lst = []
        for i in range(len(board[0])): # 横坐标
            for j in range(len(board)): # 纵坐标
                state = judge(j,i)
                # print([j,i],[state])
                if board[j][i] == 1:
                    if state < 2 or state > 3: # 本身要是活细胞
                        death_lst.append([j,i]) # 这些细胞会死亡
                elif board[j][i] == 0: # 本身是死细胞
                    if state == 3:
                        alive_lst.append([j,i]) # 这些细胞会复活
        # print("alive",alive_lst)
        # print("death",death_lst)
        for coord in death_lst:
            board[coord[0]][coord[1]] = 0
        for coord in alive_lst:
            board[coord[0]][coord[1]] = 1
        return board
                

```

# 290. 单词规律

给定一种规律 pattern 和一个字符串 str ，判断 str 是否遵循相同的规律。

这里的 遵循 指完全匹配，例如， pattern 里的每个字母和字符串 str 中的每个非空单词之间存在着双向连接的对应规律。

示例1:

输入: pattern = "abba", str = "dog cat cat dog"
输出: true

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        s = s.split(' ') 
        if len(s) != len(pattern):
            return False
        dict1 = {}
        p = 0
        while p < len(pattern):
            if not pattern[p] in dict1:
                dict1[pattern[p]] = s[p]
            else:
                if dict1[pattern[p]] != s[p]:
                    return False
            p += 1       
        dict2 = {}
        p = 0
        while p < len(pattern):
            if not s[p] in dict2:
                dict2[s[p]] = pattern[p]
            else:
                if dict2[s[p]] != pattern[p]:
                    return False
            p += 1

        return True
```

# 292. Nim 游戏

你和你的朋友，两个人一起玩 Nim 游戏：

桌子上有一堆石头。
你们轮流进行自己的回合，你作为先手。
每一回合，轮到的人拿掉 1 - 3 块石头。
拿掉最后一块石头的人就是获胜者。
假设你们每一步都是最优解。请编写一个函数，来判断你是否可以在给定石头数量为 n 的情况下赢得游戏。如果可以赢，返回 true；否则，返回 false 。

```python
class Solution:
    def canWinNim(self, n: int) -> bool:
        return n%4 != 0
```

# 300. 最长递增子序列

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 动态规划
        dp = [1 for i in range(len(nums))] # 先全部初始化为1
        # dp[i] 的意思是，以nums[i]结尾的最长递增子序列的长度
        for i in range(len(dp)): # i是索引
            p = 0 # p是辅助扫描索引
            while p < i:
                if nums[p] < nums[i]:
                    dp[i] = max(dp[p]+1,dp[i])
                p += 1
        print(dp)
        return max(dp)
```

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 动态规划,比上面效率高
        n = len(nums)
        dp = [1 for i in range(len(nums))] # 先全部初始化为1
        # dp[i] 的意思是，以nums[i]结尾的最长递增子序列的长度
        for i in range(0,n):
            group = []
            for j in range(0,i):
                if nums[i] > nums[j]: # 只有当nums[i] 大于nums[j]才进组
                    group.append(dp[j])
            if group != []: # 组内非空才变更
                dp[i] = max(group) + 1
        # print(dp)
        return max(dp)
```

# 319. 灯泡开关

初始时有 n 个灯泡处于关闭状态。

对某个灯泡切换开关意味着：如果灯泡状态为关闭，那该灯泡就会被开启；而灯泡状态为开启，那该灯泡就会被关闭。

第 1 轮，每个灯泡切换一次开关。即，打开所有的灯泡。

第 2 轮，每两个灯泡切换一次开关。 即，每两个灯泡关闭一个。

第 3 轮，每三个灯泡切换一次开关。

第 i 轮，每 i 个灯泡切换一次开关。 而第 n 轮，你只切换最后一个灯泡的开关。

找出 n 轮后有多少个亮着的灯泡。

```python
class Solution:
    def bulbSwitch(self, n: int) -> int:
        # 一个找规律数学问题
        # 完全平方数的因数是奇数个，奇数次操作使得灯泡为亮
        return int(math.sqrt(n))
```

# 322. 零钱兑换

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 动态规划,带备忘录剪枝
        book = dict() # 创建备忘录
        def dp(n):
            if n in book:return book[n]
            if n == 0:
                return 0
            elif n < 0:
                return -1
            res = float('INF') #初始化为无穷大的值
            for coin in coins:
                subproblem = dp(n-coin)
                if subproblem == -1:continue #开启下一轮循环
                res = min(res,1+subproblem)           
            book[n] = res if res != float("INF") else -1 #每个结果都要作为备忘录
            return book[n]
        return dp(amount)
```

# 326. 3的幂

给定一个整数，写一个函数来判断它是否是 3 的幂次方。如果是，返回 true ；否则，返回 false 。

整数 n 是 3 的幂次方需满足：存在整数 x 使得 n == 3^x

```python
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        while n >= 3 :
            n = n/3
        if n ==1 :
            return True
        else:
            return False
```

# 338. 比特位计数

给定一个非负整数 **num**。对于 **0 ≤ i ≤ num** 范围中的每个数字 **i** ，计算其二进制数中的 1 的数目并将它们作为数组返回。

进阶:

给出时间复杂度为O(n*sizeof(integer))的解答非常容易。但你可以在线性时间O(n)内用一趟扫描做到吗？
要求算法的空间复杂度为O(n)。
你能进一步完善解法吗？

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = []
        lst = [x for x in range(n+1)]
        for i in lst:
            temp = bin(i&0xffffff)
            s = temp.count('1')
            ans.append(s)
        return ans
```

# 342. 4的幂

给定一个整数，写一个函数来判断它是否是 4 的幂次方。如果是，返回 true ；否则，返回 false 。

整数 n 是 4 的幂次方需满足：存在整数 x 使得 n == 4^x

```python
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and n&(n-1) == 0 and n & 0xaaaaaaaa == 0
```

# 344. 反转字符串

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s) - 1
        while left <= len(s)//2 -1 :
            s[left],s[right] = s[right],s[left]
            left += 1
            right -= 1
           
```

# 345. 反转字符串中的元音字母

编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        #转化成字符数组，左右指针交换。
        need = [] #记录需要交换的坐标
        set1 = {'a','e','i','o','u','A','E','I','O','U'}
        p = 0
        while p < len(s):
            if s[p] in set1:
                need.append(p)
            p += 1
        left = 0 
        right = len(need)-1
        s = list(s)
        while left < right:
            s[need[left]],s[need[right]] = s[need[right]],s[need[left]]
            left += 1
            right -= 1
        result = ''.join(i for i in s)
        return result
```

```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        elementSet = set("aeiouAEIOU")
        s = list(s)
        left = 0
        right = len(s) - 1
        while left < right:
            while left < right and s[left] not in elementSet:
                left += 1
            while left < right and s[right] not in elementSet:
                right -= 1
            s[left],s[right] = s[right],s[left]
            left += 1
            right -= 1
        return ''.join(s)
```

# 349. 两个数组的交集

给定两个数组，编写一个函数来计算它们的交集。

**说明：**

- 输出结果中的每个元素一定是唯一的。
- 我们可以不考虑输出结果的顺序。

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        c = set(nums1)&set(nums2)
        return list(c)
```

# 350. 两个数组的交集 II

给定两个数组，编写一个函数来计算它们的交集。

说明：

输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
我们可以不考虑输出结果的顺序。
进阶：

如果给定的数组已经排好序呢？你将如何优化你的算法？
如果 nums1 的大小比 nums2 小很多，哪种方法更优？
如果 nums2 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dict1 = {i:nums1.count(i) for i in nums1}
        lst = []
        for i in nums2:
            if i in dict1:
                lst.append(i)
                dict1[i] -= 1
                if dict1.get(i) == 0:
                    dict1.pop(i)
        return lst

```

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 排序后，双指针
        nums1.sort()
        nums2.sort()
        p1 = 0
        p2 = 0 
        ans = [] # 收集答案用
        while p1 < len(nums1) and p2 < len(nums2):
            if nums1[p1] < nums2[p2]:
                p1 += 1
            elif nums1[p1] > nums2[p2]:
                p2 += 1
            elif nums1[p1] == nums2[p2]:
                ans.append(nums1[p1]) # 收集结果再改变指针坐标
                p1 += 1
                p2 += 1
         
        return ans
```

# 367. 有效的完全平方数

给定一个 正整数 num ，编写一个函数，如果 num 是一个完全平方数，则返回 true ，否则返回 false 。

进阶：不要 使用任何内置的库函数，如  sqrt 。

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        # 左闭右闭二分查找
        left = 1
        right = num
        while left <= right:
            mid = (left+right)//2
            if mid*mid == num:
                return True
            elif mid*mid > num: # 中间数值过大，缩小右边界
                right = mid - 1
            elif mid*mid < num: # 中间数值过小，缩小左边界
                left = mid + 1
        return False
```

# 374. 猜数字大小

猜数字游戏的规则如下：

每轮游戏，我都会从 1 到 n 随机选择一个数字。 请你猜选出的是哪个数字。
如果你猜错了，我会告诉你，你猜测的数字比我选出的数字是大了还是小了。
你可以通过调用一个预先定义好的接口 int guess(int num) 来获取猜测结果，返回值一共有 3 种可能的情况（-1，1 或 0）：

-1：我选出的数字比你猜的数字小 pick < num
1：我选出的数字比你猜的数字大 pick > num
0：我选出的数字和你猜的数字一样。恭喜！你猜对了！pick == num
返回我选出的数字。

```python
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num: int) -> int:

class Solution:
    def guessNumber(self, n: int) -> int:
        left = 1
        right = n
        mid = (left+right)//2
        while guess(mid) != 0:
            if guess(mid) == -1:
                right = mid - 1
                mid = (left+right)//2
            elif guess(mid) == 1:
                left = mid + 1
                mid = (left+right)//2
        return mid
```

# 376. 摆动序列

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。

相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        # 此题难点在于处理最左边和最右边峰值
        # 这一题的实际是求序列中有多少个局部峰值，设计峰值算法
        if len(nums) <= 1:
            return len(nums)
        curDiff = 0 #当前这一对的差值
        preDiff = 0 #先前那一对的差值
        result = 1 # 记录峰值个数，默认最右边有一个峰值
        p = 1
        while p < len(nums):
            curDiff = nums[p]-nums[p-1]
            if (curDiff > 0 and preDiff <= 0) or (curDiff<0 and preDiff>=0): #这里的等号是为了让初始能被记录
                result += 1
                preDiff = curDiff #注意这一行的层级在if下面
            p += 1
        return result
```

# 382. 链表随机节点

给定一个单链表，随机选择链表的一个节点，并返回相应的节点值。保证每个节点被选的概率一样。

进阶:
如果链表十分大且长度未知，如何解决这个问题？你能否使用常数级空间复杂度实现？

示例:

// 初始化一个单链表 [1,2,3].
ListNode head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);
Solution solution = new Solution(head);

// getRandom()方法应随机返回1,2,3中的一个，保证每个元素被返回的概率相等。
solution.getRandom();

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:

    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.lst = []
        cur = head
        while cur != None:
            self.lst.append(cur.val)
            cur = cur.next


    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        index = random.randint(0,len(self.lst)-1)
        return self.lst[index]


# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()
```

# 383. 赎金信

给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。

(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        # 计数排序
        magazine_dict = collections.Counter(magazine)
        ransomNote_dict = collections.Counter(ransomNote)
        for element in ransomNote_dict:
            if ransomNote_dict[element] > magazine_dict[element]:return False
        return True
```

# 384. 打乱数组

给你一个整数数组 nums ，设计算法来打乱一个没有重复元素的数组。

实现 Solution class:

Solution(int[] nums) 使用整数数组 nums 初始化对象
int[] reset() 重设数组到它的初始状态并返回
int[] shuffle() 返回数组随机打乱后的结果

```python
class Solution:

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.origin = nums.copy()


    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        self.nums = self.origin
        return self.nums

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        temp = self.nums.copy()
        new_lst = []
        while len(temp) != 0:
            new_lst.append(temp.pop(random.randint(0,len(temp)-1)))
        self.nums = new_lst
        return self.nums



# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()
```

# 387. 字符串中的第一个唯一字符

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        if len(s) == 1:
            return 0
        p1 = 0
        p2 = 0
        while p1 < len(s):
            if p1 == p2:
                p2 += 1
            elif p2 > len(s)-1:
                return len(s)-1
            elif s[p1] == s[p2] :
                p1 += 1
                p2 = 0
            elif s[p1] != s[p2] and p2 != len(s)-1:
                p2 += 1
            elif s[p1] != s[p2] and p2 == len(s)-1:
                return p1
        return -1
```

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        # 哈希表
        dict1 = defaultdict(int)
        for i in s:
            dict1[i] += 1
        for p in range(len(s)):
            if dict1[s[p]] == 1:
                return p
        return -1

```

# 389. 找不同

给定两个字符串 s 和 t，它们只包含小写字母。

字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。

请找出在 t 中被添加的字母。

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        result = 0
        for i in s:
            result ^= ord(i)
        for i in t:
            result ^= ord(i)
        return chr(result)
```

# 397. 整数替换

给定一个正整数 n ，你可以做如下操作：

如果 n 是偶数，则用 n / 2替换 n 。
如果 n 是奇数，则可以用 n + 1或n - 1替换 n 。
n 变为 1 所需的最小替换次数是多少？

```python
class Solution:
    def integerReplacement(self, n: int) -> int:
        count = 0 # 记录操作次数
        # 贪心，偶数操作应该尽量的多
        # 如果末尾数是01的情况，直接减1
        # 如果末尾是11的情况，选择加1 
        # 但是 对于数字3 而言：其2进制b11:
        # 走加法路线 3+1 -》 4； 4/2 -〉2 ； 2/2 -》 1；
        # 走减法路线 3 - 1 -〉2； 2/2 -》 1； 只需要两步
        while n != 1:
            if n%2 == 0:
                n = n//2
            else:
                if n!=3 and (n>>1)&1 == 1:
                    n = n + 1
                else:
                    n -= 1
            count += 1
        return count
```

# 400. 第 N 位数字

在无限的整数序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...中找到第 n 位数字。

 

注意：n 是正数且在 32 位整数范围内（n < 231）。

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

# 404. 左叶子之和

计算给定二叉树的所有左叶子之和。

```python
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        # 检查每个节点的左孩子节点，如果它是叶子节点，则加入计算
        ans = []
        def submethod(node):
            if node != None:
                # 收集
                if node.left != None:
                    if node.left.left == None and node.left.right == None:
                        ans.append(node.left.val)
                submethod(node.left)
                submethod(node.right)
        submethod(root)
        return sum(ans)
```

# 409. 最长回文串

给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。

在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。

注意:
假设字符串的长度不会超过 1010。

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        # 取最大的奇数和全部的偶数
        dict1 = collections.Counter(s)
        length = 0
        odd_max = 0 # 标记是否存在奇数
        # 偶数先直接加入，奇数减去1，如果扫描过奇数，最终值可以把这个length值加一个
        for i in dict1:
            if dict1[i]%2 == 0:
                length += dict1[i]
            elif dict1[i]%2 == 1:
                odd_max = 1
                length += dict1[i]-1
        if odd_max == 1:
            length += odd_max
        return length
```



# 412. Fizz Buzz

写一个程序，输出从 1 到 n 数字的字符串表示。

1. 如果 n 是3的倍数，输出“Fizz”；

2. 如果 n 是5的倍数，输出“Buzz”；

3.如果 n 同时是3和5的倍数，输出 “FizzBuzz”。

```python
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        lst = [""]*n
        for i in range(0,n):
            if (i+1)%3 != 0 and (i+1)%5 != 0:
                lst[i] = lst[i].join(str(i+1))
            if (i+1)%3 == 0 and (i+1)%15 != 0:
                str1 = 'Fizz'
                lst[i] = lst[i].join(str1)
            if (i+1)%5 == 0 and (i+1)%15 != 0:
                str2 = 'Buzz'
                lst[i] = lst[i].join(str2)
            if (i+1)%15 == 0:
                lst[i] = lst[i].join('FizzBuzz')
        return lst
```

```python
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        ans = []
        for i in range(1,n+1):
            temp = ""
            if i % 3 == 0:
                temp += "Fizz"
            if i % 5 == 0:
                temp += "Buzz"
            if temp == "":
                temp += str(i)
            ans.append(temp)
        return ans
```

# 413. 等差数列划分

如果一个数列至少有三个元素，并且任意两个相邻元素之差相同，则称该数列为等差数列。

例如，以下数列为等差数列:

1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
以下数列不是等差数列。

1, 1, 2, 5, 7


数组 A 包含 N 个数，且索引从0开始。数组 A 的一个子数组划分为数组 (P, Q)，P 与 Q 是整数且满足 0<=P<Q<N 。

如果满足以下条件，则称子数组(P, Q)为等差数组：

元素 A[P], A[p + 1], ..., A[Q - 1], A[Q] 是等差的。并且 P + 1 < Q 。

函数要返回数组 A 中所有为等差数组的子数组个数。

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # 首先找到所有元素的相邻gap，利用最大gap算长度至少为3的子序列的数量
        if len(nums) < 3:
            return 0
        p = 0
        gap_lst = []
        while p < len(nums) - 1:
            gap_lst.append(nums[p+1]-nums[p])
            p += 1
        # 从头到尾找符合要求的gap内的元素，至少两个，n个差值代表这一组数列长度为n+1
        # 存在一个列表中，这个列表只会存大于等于2的数，并且进一步计算出符合题目要求的子数组数目
        # 不可以使用hashmap，因为要连续的
        temp_lst = []
        p = 0
        mark = gap_lst[p]
        temp_lenth = 0
        while p < len(gap_lst):
            if gap_lst[p] == mark:
                temp_lenth += 1
                p += 1
            elif gap_lst[p] != mark:
                if temp_lenth >= 2:
                    temp_lst.append(temp_lenth)
                mark = gap_lst[p] #重置标记
                temp_lenth = 0  #重置长度,这里p不变
            
        #最后一组也需要收集
        if temp_lenth >= 2:
            temp_lst.append(temp_lenth)
        # gap为k-1，长度为k的数组中具有多少个大于等于3的子数组求法为
        #  比如是gap是4，对应5个数，那么长度为5，4，3的数数目分别为1，2，3
        # 考虑求和公式的方便 那么是从1～n求和。其中n为 gap+1+1-3
        sum1 = 0
        for i in temp_lst:
            sum1 += (i+1+1-3+1)*(i+1+1-3)/2
        return int(sum1)
```

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # dp 使用动态规划
        # dp[i] 为 nums[i]为最后一个元素的等差数组的个数
        # 状态转移方程为： 如果差值相等nums[i] - nums[i-1] == gap，dp[i] = dp[i-1] + 1 意思是：dp[i]的数量是既包括了前一个的所有延长，还+了一个新的1。
        # 否则，dp[i] = 0 ,重新计算gap
        # 初始化gap = nums[1] - nums[0]
        if len(nums) < 3:
            return 0
        ans = 0 # 计算答案
        n = len(nums)
        dp = [0 for i in range(n)]
        gap = nums[1] - nums[0]
        for i in range(2,n):
            if nums[i] - nums[i-1] == gap:
                dp[i] = dp[i-1] + 1
            else:
                gap = nums[i] - nums[i-1] # 重置gap
                dp[i] = 0 # 重置dp
        # 有两种处理方式，一种是处理dp总和，一种是在计算dp的过程中就把每个dp值给计算进去
        # 状态压缩可以压到O1
        return sum(dp)
```

# 415. 字符串相加

给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和。

提示：

num1 和num2 的长度都小于 5100
num1 和num2 都只包含数字 0-9
num1 和num2 都不包含任何前导零
你不能使用任何內建 BigInteger 库， 也不能直接将输入的字符串转换为整数形式

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        # 倒序计算，倒序弹出
        car = 0 #初始进位值为0
        # 转化成列表进行计算
        if len(num2) > len(num1):
            num1,num2 = num2,num1
        num1 = list(num1)
        num2 = list(num2)
        ans = [] # 存结果的列表
        while num1 and num2:
            temp1 = int(num1.pop(-1))
            temp2 = int(num2.pop(-1))
            if (temp1) + (temp2) + car >= 10:
                ans.append(str(temp1+temp2+car-10))
                car = 1
            elif temp1 + temp2 +car <10:
                ans.append(str(temp1+temp2+car))
                car = 0
        remain = num1  # 剩余的一个需要继续加
        while remain:
            num = int(remain.pop(-1))
            if num + car >= 10:
                ans.append(str(num+car-10))
                car = 1
            elif num + car < 10:
                ans.append(str(num+car))
                car = 0
        # 最后一次是否有进位
        if car == 1:
            ans.append(str(car))
        
        # 最终的ans需要倒序组合成结果
        ans = ans[::-1]
        result = ''.join(ans)
        return result
```

# 426. 将二叉搜索树转化为排序的双向链表

将一个 二叉搜索树 就地转化为一个 已排序的双向循环链表 。

对于双向循环列表，你可以将左右孩子指针作为双向循环链表的前驱和后继指针，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

特别地，我们希望可以 就地 完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中最小元素的指针。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""

class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        # 中序遍历后存储在数组中
        # 利用取模统一语法进行计算
        # 注意节点数可以为0
        inorder_lst = []
        def inorder_method(node):
            if node == None:
                return 
            inorder_method(node.left)
            inorder_lst.append(node)
            inorder_method(node.right)
        inorder_method(root) # 执行方案
        p = 0
        length = len(inorder_lst) # 存储列表长度
        if length == 0: # 空树处理
            return
        while p < length:
            inorder_lst[p].right = inorder_lst[(p+1)%length]
            inorder_lst[p].left = inorder_lst[(p-1)%length]
            p += 1
        return inorder_lst[0]

```

# 429. N 叉树的层序遍历

给定一个 N 叉树，返回其节点值的*层序遍历*。（即从左到右，逐层遍历）。

树的序列化输入是用层序遍历，每组子节点都由 null 值分隔（参见示例）。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        # 借助队列管理，ans是最终返回值
        if root == None:
            return []
        ans = []
        queue = [root]
        level = []
        while len(queue) != 0:
            level = [] 
            new_queue = [] 
            for i in queue:
                if i.val != None:
                    level.append(i.val)
            for i in queue:   # 注意，N叉树的children是一个列表，所以要对列表再进行for循环
                for children in i.children:
                    new_queue.append(children)
            queue = new_queue
            ans.append(level)
        return ans
```

# 434. 字符串中的单词数

统计字符串中的单词个数，这里的单词指的是连续的不是空格的字符。

请注意，你可以假定字符串里不包括任何不可打印的字符。

示例:

输入: "Hello, my name is John"
输出: 5
解释: 这里的单词是指连续的不是空格的字符，所以 "Hello," 算作 1 个单词。

```python
class Solution:
    def countSegments(self, s: str) -> int:
        # 借助栈思路，
        # 给一个默认栈顶“ ‘
        # 当如果栈顶是’ ‘，即将入栈元素不为’ ‘时候入栈，否则正常入栈
        stack = [' ']
        p = 0
        while p < len(s):
            if stack[-1] != ' ':
                stack.append(s[p])
            elif stack[-1] == ' ':
                if s[p] != ' ':
                    stack.append(s[p])
            p += 1
        stack = stack[1:] # 去除第一个前导‘ ’
        # 记录分割‘ ’的数量
        if stack == []:
            return 0
        count = 0
        for i in stack:
            if i == ' ':
                count += 1
        if stack[-1] == ' ':
            return count
        
        return count+1
```

# 438. 找到字符串中所有字母异位词

给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。

字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。

说明：

字母异位词指字母相同，但排列不同的字符串。
不考虑答案输出的顺序。
示例 1:

输入:
s: "cbaebabacd" p: "abc"

输出:
[0, 6]

解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的字母异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的字母异位词。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 滑动窗口法
        # 建立俩字典
        target_dict = collections.defaultdict(int)
        window_dict = collections.defaultdict(int)
        ans = [] # 记录最终返回答案
        for i in p:
            target_dict[i] += 1 #初始化目标字典
        left = 0
        right = 0 # 初始化窗口大小
        valid = 0 # 合法字符个数
        while right < len(s):
            temp_char = s[right]
            right += 1
            if temp_char in target_dict:
                window_dict[temp_char] += 1
                if window_dict[temp_char] == target_dict[temp_char]:
                    valid += 1
            while (right-left) >= len(p):
                if valid == len(target_dict): # 收集结果
                    ans.append(left)
                delete_char = s[left]
                left += 1
                if delete_char in target_dict:
                    if target_dict[delete_char] == window_dict[delete_char]:
                        valid -= 1
                    window_dict[delete_char] -= 1
        return ans
```

# 442. 数组中重复的数据

给定一个整数数组 a，其中1 ≤ a[i] ≤ n （n为数组长度）, 其中有些元素出现两次而其他元素出现一次。

找到所有出现两次的元素。

你可以不用到任何额外空间并在O(n)时间复杂度内解决这个问题吗？

示例：

输入:
[4,3,2,7,8,2,3,1]

输出:
[2,3]

```python
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        # 计数排序
        template = [[i,0] for i in range(0,len(nums)+1)] # 为了方便思考，加了[0,0]进去
        for number in nums:
            template[number][1] += 1
        ans = []
        for lst in template:
            if lst[1] == 2:
                ans.append(lst[0])
        return ans
```

# 445. 两数相加 II

给你两个 非空 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

你可以假设除了数字 0 之外，这两个数字都不会以零开头。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 将两个链表的数值存储在栈中
        # 存完之后栈顶为最低位
        # 每次相加俩栈顶
        # 当有一个栈为空时，把进位和其他位组合成新链表 相加
        stack1 = []
        cur1 = l1
        while cur1 != None:
            stack1.append(cur1.val)
            cur1 = cur1.next
        stack2 = []
        cur2 = l2
        while cur2 != None:
            stack2.append(cur2.val)
            cur2 = cur2.next
        # 默认进位为0，然后开始计算
        carry = 0
        temp_cur = ListNode()
        head = temp_cur
        while stack1 != [] and stack2 != []: # 两者都不为空时
            val = stack1.pop()+stack2.pop()
            real_val = (val + carry) % 10
            carry = (val + carry) // 10
            temp_node = ListNode(real_val)
            temp_cur.next = temp_node
            temp_cur = temp_cur.next

        # 还剩下一个栈，将所有值弹出
        stack = stack1 if stack1 != [] else stack2
        while stack != []:
            val = stack.pop()
            real_val = (val + carry) % 10
            carry = (val + carry) // 10
            temp_node = ListNode(real_val)
            temp_cur.next = temp_node
            temp_cur = temp_cur.next
        # 如果还有进位 要把进位加进去
        if carry > 0:
            temp_node = ListNode(carry)
            temp_cur.next = temp_node
            temp_cur = temp_cur.next
        # 此时的结果是倒序的
        # 翻转链表
        cur1 = None
        cur2 = head.next
        while cur2 != None:
            temp = cur2.next
            cur2.next = cur1
            cur1 = cur2
            cur2 = temp
        return cur1 #
```

# 448. 找到所有数组中消失的数字

给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，并以数组的形式返回结果。

 

示例 1：

输入：nums = [4,3,2,7,8,2,3,1]
输出：[5,6]
示例 2：

输入：nums = [1,1]
输出：[2]


提示：

n == nums.length
1 <= n <= 10^5
1 <= nums[i] <= n
进阶：你能在不使用额外空间且时间复杂度为 O(n) 的情况下解决这个问题吗? 你可以假定返回的数组不算在额外空间内。

```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        # 基数排序
        count_lst = [0 for x in range(len(nums))]
        for i in nums:
            count_lst[i-1] += 1
        ans = []
        p = 0
        while p < len(count_lst):
            if count_lst[p]==0:
                ans.append(p+1)
            p += 1
        return ans
```

# 451. 根据字符出现频率排序

给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        from collections import Counter
        freq = Counter(s)
        freq_list = list(freq.items())
        freq_list.sort(key=lambda x:x[1], reverse = True)
        return ''.join([c[0]*c[1] for c in freq_list])
```

# 453. 最小操作次数使数组元素相等

给定一个长度为 *n* 的 **非空** 整数数组，每次操作将会使 *n* - 1 个元素增加 1。找出让数组所有元素相等的最小操作次数。

```python
class Solution:
    def minMoves(self, nums: List[int]) -> int:
        # 逆向思维，即最大减一
        # 找出最小值，其余每个值和最小值做差即可
        count = 0
        min_num = min(nums)
        for i in nums:
            count += (i-min_num)
        return count
```

# 455. 分发饼干

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort() # 孩子升序排序
        s.sort() # 饼干升序排序
        #大饼干优先给大孩子
        count = 0
        sp = -1
        gp = -1
        while  ( sp >= -len(s) and gp >= -len(g)):
            if s[sp] >= g[gp]:
                count += 1
                sp -= 1
                gp -= 1
            elif s[sp] < g[gp]:
                gp -= 1  #饼干是慢指针，饼干没有被消耗只移动人指针
        return count
```

# 461. 汉明距离

两个整数之间的 [汉明距离] 指的是这两个数字对应二进制位不同的位置的数目。

给你两个整数 `x` 和 `y`，计算并返回它们之间的汉明距离。

```python
pythonclass Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return str(bin(x^y)).count('1')
```

# 462. 最少移动次数使数组元素相等 II

给定一个非空整数数组，找到使所有数组元素相等所需的最小移动数，其中每次移动可将选定的一个元素加1或减1。 您可以假设数组的长度最多为10000。

```python
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        # 数学思想，找中位数，下面证明为什么是中位数
        # 为了方便，我们先假设一共有2n+1个数，它们从小到大排序之后如下：

        #. . . a m b . . .
        #其中m是中位数。此时，m左边有n个数，m右边也有n个数。我们假设把m左边所有数变成m需要的代价是x，把m右边所有数变成m的代价是y，此时的总代价就是t = x+y

        #好，如果你觉得中位数不是最优解，我们来看看把所有数都变成a的总代价是多少。 由于之前m右边n个数变成m的代价是y，现在让右边的数全变成a，此时右边的数的代价是y+(m-a)*n；m左边的n个数全变成a，它们的代价会减少到x-(m-a)*n。所以两边相加，结果还是 x-(m-a)*n + y+(m-a)*n == x+y。 但是，别忘了，m也要变成a，所以总代价是x+y+m-a，大于x+y。同理，如果让所有数都变成比m大的b，总代价则变为x+y+b-m（你可以自己算一下），依然比x+y大。并且越往左移或者往右移，这个值都会越来越大。 因此，在有2n+1个数的时候，选择中位数就是最优解。
        nums.sort()
        pivot = nums[len(nums)//2]
        gap = 0
        for i in nums:
            gap += abs(i-pivot)
        return gap
```

# 476. 数字的补数

给你一个 **正** 整数 `num` ，输出它的补数。补数是对该数的二进制表示取反。

```python
class Solution:
    def findComplement(self, num: int) -> int:
        num = list(bin(num))
        p = 1
        while p < len(num):
            if num[p] == '0':
                num[p] = '1'
            elif num[p] == '1':
                num[p] = '0'
            p += 1
        temp = ''.join(num)[2:]
        temp = temp[::-1]
        ans = 0
        p = 0
        while p < len(temp):
            ans += int(temp[p])*2**(p)
            p += 1
        return ans
```

# 477. 汉明距离总和

两个整数的 [汉明距离](https://baike.baidu.com/item/汉明距离/475174?fr=aladdin) 指的是这两个数字的二进制数对应位不同的数量。

给你一个整数数组 `nums`，请你计算并返回 `nums` 中任意两个数之间汉明距离的总和。

```python
class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        # 思路：统计每一位有多少个1，多少个非1.然后乘法计数原理
        # 最终对每一位是Cn2的挑选，而有效挑选为 i*j 其中i是1的个数，j是0的个数，累加求和
        # 还需要把所有数全部转换成二进制【python中2进制不自动补齐32位】
        ans = 0
        # 而10的9次方小于2的30次方，枚举即可
        for k in range(30):
            i = sum(((val>>k)&1) for val in nums)
            j = len(nums) - i
            ans += i * j
        return ans
```

# 478. 在圆内随机生成点

给定圆的半径和圆心的 x、y 坐标，写一个在圆中产生均匀随机点的函数 randPoint 。

说明:

输入值和输出值都将是浮点数。
圆的半径和圆心的 x、y 坐标将作为参数传递给类的构造函数。
圆周上的点也认为是在圆中。
randPoint 返回一个包含随机点的x坐标和y坐标的大小为2的数组。

```python
import math
import random

# 极坐标法，比笛卡尔坐标的随机要效率快很多

class Solution:

    def __init__(self, radius: float, x_center: float, y_center: float):
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center

    def randPoint(self) -> List[float]:
        l = self.radius * math.sqrt(random.random())
        deg = random.random() * math.pi * 2
        x = l * math.cos(deg) + self.x_center
        y = l * math.sin(deg) + self.y_center
        return [x, y]


```

# 485. 最大连续 1 的个数

给定一个二进制数组， 计算其中最大连续 1 的个数。

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        s = ''
        for i in nums:
            s += str(i)
        s = s.split('0')
        maxlen = 0
        for i in s:
            if len(i) > maxlen:
                maxlen = len(i)
        return maxlen
```

# 495. 提莫攻击

在《英雄联盟》的世界中，有一个叫 “提莫” 的英雄，他的攻击可以让敌方英雄艾希（编者注：寒冰射手）进入中毒状态。现在，给出提莫对艾希的攻击时间序列和提莫攻击的中毒持续时间，你需要输出艾希的中毒状态总时长。

你可以认为提莫在给定的时间点进行攻击，并立即使艾希处于中毒状态。

```python
class Solution:
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        # 一轮扫描
        p = 1
        ans = 0
        while p < len(timeSeries):
            if timeSeries[p]-timeSeries[p-1] >= duration: # 如果间隔大于持续时间，则ans上升持续时间
                ans += duration
            elif timeSeries[p]-timeSeries[p-1] < duration: # 如果间隔小于持续时间，则ans上升间隔
                ans += (timeSeries[p]-timeSeries[p-1])
            p += 1
        # 最后一次加上持续时间
        return ans + duration

```

# 500. 键盘行

给你一个字符串数组 words ，只返回可以使用在 美式键盘 同一行的字母打印出来的单词。键盘如下图所示。

美式键盘 中：

第一行由字符 "qwertyuiop" 组成。
第二行由字符 "asdfghjkl" 组成。
第三行由字符 "zxcvbnm" 组成。

```python
class Solution:
    def findWords(self, words: List[str]) -> List[str]: 
        origin = words.copy()     
        ## 规范化words
        p = 0
        while p < len(words):
            words[p] = words[p].lower()
            p += 1
        p = 0
        ## 把words中的每个单词check
        lst = []
        while p < len(words):
            if self.check(words[p]):
                lst.append(origin[p]) #添加时候注意要添加成原词
            p += 1
        return lst

    
    def check(self,s):
        dict1 = {i:"l1" for i in "qwertyuiop"}
        dict2 = {i:"l2" for i in "asdfghjkl"}
        dict3 = {i:"l3" for i in "zxcvbnm"}
        dict1.update(dict2)
        dict1.update(dict3)
        for i in s:
            if dict1[i] != dict1[s[0]]:
                return False
        return True
```

# 501. 二叉搜索树中的众数

给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。

假定 BST 有如下定义：

结点左子树中所含结点的值小于等于当前结点的值
结点右子树中所含结点的值大于等于当前结点的值
左子树和右子树都是二叉搜索树
例如：
给定 BST [1,null,2,2],

   1
    \
     2
    /
   2
返回[2].

提示：如果众数超过1个，不需考虑输出顺序

进阶：你可以不使用额外的空间吗？（假设由递归产生的隐式调用栈的开销不被计算在内）

```python
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        # 中序遍历
        lst = [] # 存储中序遍历的结果
        def inOrder(node):
            if node == None:
                return 
            inOrder(node.left)
            lst.append(node.val)
            inOrder(node.right)
        inOrder(root)
        dict1 = collections.Counter(lst)
        # 利用collections中的Counter找到最大频次，然后把最大频次的数据收集
        times = 0 # 默认值为0
        ans = [] # 收集结果
        for i in dict1:
            if dict1[i] < times:
                pass
            elif dict1[i] == times:
                ans.append(i)
            elif dict1[i] > times:
                ans = [i]
                times = dict1[i]
        return ans
```

# 504. 七进制数

给定一个整数，将其转化为7进制，并以字符串形式输出。

```python
class Solution:
    def convertToBase7(self, num: int) -> str:
        origin = num
        if num == 0:
            return '0'
        result = ''
        num = abs(num)
        while num > 0:
            tail = num % 7
            result = str(tail) + result
            num = num // 7
        if origin < 0:
            result = '-'+result
            return result
        else:
            return result
```

# 506. 相对名次

给出 N 名运动员的成绩，找出他们的相对名次并授予前三名对应的奖牌。前三名运动员将会被分别授予 “金牌”，“银牌” 和“ 铜牌”（"Gold Medal", "Silver Medal", "Bronze Medal"）。

(注：分数越高的选手，排名越靠前。)

示例 1:

输入: [5, 4, 3, 2, 1]
输出: ["Gold Medal", "Silver Medal", "Bronze Medal", "4", "5"]
解释: 前三名运动员的成绩为前三高的，因此将会分别被授予 “金牌”，“银牌”和“铜牌” ("Gold Medal", "Silver Medal" and "Bronze Medal").
余下的两名运动员，我们只需要通过他们的成绩计算将其相对名次即可。

```python
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        # 先将数列处理成自带索引的数量,格式为 index,value
        p = 0
        while p < len(score):
            score[p] = [p,score[p]]
            p += 1
        # 再根据数值value排序
        score.sort(key=lambda x:x[1])
        # 排序完成后，x[0]为原来的索引位置
        # 创建一个长度为N的空表，作为答案赋值用:
        ans = [None for i in range(len(score))]
        # 取出最后三个颁发奖牌
        index,medal = score.pop()
        ans[index] = 'Gold Medal'
        if len(score) != 0:
            index,medal = score.pop()
            ans[index] = 'Silver Medal'
        if len(score) != 0:
            index,medal = score.pop()
            ans[index] = 'Bronze Medal'
        rank = 4
        for i in score[::-1]:
            ans[i[0]] = str(rank)
            rank += 1
        return ans
```

# 507. 完美数

对于一个 正整数，如果它和除了它自身以外的所有 正因子 之和相等，我们称它为 「完美数」。

给定一个 整数 n， 如果是完美数，返回 true，否则返回 false

```python
class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        if num == 1:
            return False
        lst = []
        t = math.sqrt(num)
        p = 1
        while p <= t:
            if num%p == 0:
                lst.append(p)
                lst.append(num//p)
            p += 1
        result = sum(lst)
        if result / num == 2:
            return True
        else:
            return False
```

# 509. 斐波那契数

```python
class Solution:
    def fib(self, n: int) -> int:
        return self.fib1(n)

    def fib1(self,n,a=0,b=1):
        if n == 0:
            return a
        else:
            return self.fib1(n-1,b,a+b)
```

```python
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        dp = [0 for i in range(n+1)] # 初始化dp数组，为了索引方便，申请n+1长度的数组
        dp[0],dp[1] = 0,1 # 迭代法dp，给数组赋值,没有状态压缩
        i = 2
        while i <= n:
            dp[i] = dp[i-1]+dp[i-2]
            i += 1
        return dp[n]
```

# 513. 找树左下角的值

```python
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        # BFS找最后一行最左边的值
        ans = []
        queue = [root]
        while len(queue) != 0:
            level = []
            new_queue = []
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            ans.append(level)
            queue = new_queue
        return ans[-1][0]
```

# 515. 在每个树行中找最大值

```python
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        # BFS
        if root == None:
            return []
        queue = [root]
        ans = []
        while len(queue) != 0:
            level = []
            new_queue = []
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left:
                    new_queue.append(i.left)
                if i.right:
                    new_queue.append(i.right)
            max1 = max(level)
            ans.append(max1)
            queue = new_queue
        return ans
```

# 520. 检测大写字母

给定一个单词，你需要判断单词的大写使用是否正确。

我们定义，在以下情况时，单词的大写用法是正确的：

全部字母都是大写，比如"USA"。
单词中所有字母都不是大写，比如"leetcode"。
如果单词不只含有一个字母，只有首字母大写， 比如 "Google"。
否则，我们定义这个单词没有正确使用大写字母。

```python
# 要么全大写，要么全小写，要么第一个字母大写其余小写。利用isupper和islower
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        return word.islower() or word.isupper() or (word[0].isupper() and word[1:].islower())
```

# 530. 二叉搜索树的最小绝对差

给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。

```python
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        # 利用中序遍历求相邻元素的gap ，由于所有元素非负，那么一定是相邻两个差最小
        inorder_lst = [] #存中序遍历
        def submethod(node):
            if node != None:
                submethod(node.left)
                inorder_lst.append(node.val)
                submethod(node.right)
        submethod(root)
        gap_lst = []
        p = 0
        while p < len(inorder_lst) - 1:
            gap_lst.append(inorder_lst[p+1]-inorder_lst[p])
            p += 1
        return min(gap_lst)
```

# 537. 复数乘法

复数 可以用字符串表示，遵循 "实部+虚部i" 的形式，并满足下述条件：

实部 是一个整数，取值范围是 [-100, 100]
虚部 也是一个整数，取值范围是 [-100, 100]
i^2 == -1
给你两个字符串表示的复数 num1 和 num2 ，请你遵循复数表示形式，返回表示它们乘积的字符串。

```python
class Solution:
    def complexNumberMultiply(self, num1: str, num2: str) -> str:
        n1 = num1.split("+")
        n2 = num2.split('+')
        # int转化 且去除i
        n1[0],n1[1],n2[0],n2[1] = int(n1[0]),int(n1[1][:-1]),int(n2[0]),int(n2[1][:-1])
        real = n1[0]*n2[0] - n1[1]*n2[1]
        imag = n1[0]*n2[1] + n2[0]*n1[1]
        result = str(real)+'+'+str(imag)+'i'
        return result
```

# 538. 把二叉搜索树转换为累加树

给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树。

```python
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        # 一轮中序遍历获取总和
        sum_num = []
        node_lst = []
        def inOrder1(root):
            if root != None:
                inOrder1(root.left)
                sum_num.append(root.val)
                inOrder1(root.right)
        inOrder1(root)
        k = [sum(sum_num)] # 利用传引用的方式
        def inOrder2(root):
            if root != None:
                inOrder2(root.left)
                temp = root.val # 注意这一行的运用，先存下当前值
                root.val = k[0] # 将k值赋予节点
                k[0] = k[0]-temp # 更新k值
                inOrder2(root.right)
        inOrder2(root)
        return root
```

# 541. 反转字符串 II

给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。

如果剩余字符少于 k 个，则将剩余字符全部反转。
如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。

```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        # 扫描指针+双翻转指针
        # 转化成列表处理
        s = list(s)
        p = 0
        # 先扫描条件:隔2k个，前k个全部反转
        while p+2*k-1 < len(s):
            pl = p
            pr = p+k-1
            while pl < pr:
                s[pl],s[pr] = s[pr],s[pl]
                pl += 1
                pr -= 1
            p += 2*k
        # 对尾巴条件进行处理
        if p+k > len(s):
            pl = p
            pr = len(s)-1
            while pl < pr:
                s[pl],s[pr] = s[pr],s[pl]
                pl += 1
                pr -= 1
        elif p+2*k > len(s) and p + k <= len(s):
            pl = p
            pr = p + k - 1
            while pl < pr:
                s[pl],s[pr] = s[pr],s[pl]
                pl += 1
                pr -= 1
        ans = ''.join(s)
        return ans        
            
```

```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        s = list(s)
        for i in range(0, len(s), 2 * k):
            s[i:i+k] = reversed(s[i:i+k])
        return "".join(s)
```

# 557. 反转字符串中的单词 III

给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        result = ''
        return result.join(i[::-1]+' ' for i in s.split(' '))[:-1]
```

# 559. N 叉树的最大深度

给定一个 N 叉树，找到其最大深度。

最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。

N 叉树输入按层序遍历序列化表示，每组子节点由空值分隔（请参见示例）。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if root == None:
            return 0
```

# 561. 数组拆分 I

给定长度为 2n 的整数数组 nums ，你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从 1 到 n 的 min(ai, bi) 总和最大。

返回该 最大总和 。

```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        # 数学，实质上是排序后取每组的第一个数
        nums.sort()
        p = 0
        ans = 0
        while p < len(nums):
            ans += nums[p]
            p += 2
        return ans
```

# 566. 重塑矩阵

在MATLAB中，有一个非常有用的函数 reshape，它可以将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。

给出一个由二维数组表示的矩阵，以及两个正整数r和c，分别表示想要的重构的矩阵的行数和列数。

重构后的矩阵需要将原始矩阵的所有元素以相同的行遍历顺序填充。

如果具有给定参数的reshape操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。

```python
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        sz = 0
        lst = []
        result = []
        for i in mat:
            for j in i:
                sz += 1
                lst.append(j)
        if sz != r * c:
            return mat
        while len(lst) != 0:
            temp = []
            for i in range(c):
                temp.append(lst.pop(0))
            result.append(temp)
        return result
```

# 567. 字符串的排列

给定两个字符串 `s1` 和 `s2`，写一个函数来判断 `s2` 是否包含 `s1` 的排列。

换句话说，第一个字符串的排列之一是第二个字符串的 **子串** 。

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # 滑动窗口法
        # 这里的子串是连续的，s2是待选取区间，而且窗口大小是固定的
        # 建立字典
        if len(s2) < len(s1):
            return False
        target_dict = collections.defaultdict(int)
        window_dict = collections.defaultdict(int)
        for i in s1:
            target_dict[i] += 1
        left = 0 # 同侧双指针
        right = 0 + len(s1) # 固定大小窗口
        for k in range(left,right): # 初始化窗口内容
            if s2[k] in target_dict:
                window_dict[s2[k]] += 1
        if window_dict == target_dict: #  如果窗口内容已经符合要求，直接返回正确
            return True
        valid = 0 # 用以标记符合条件的字符种类和数目是否符合要求
        while right < len(s2): # 初始化不同时，开始滑动
            temp_char = s2[right] #记录即将加入的字符
            delete_char = s2[left] #记录即将被删除的字符
            right += 1
            left += 1
            if temp_char in target_dict:
                window_dict[temp_char] += 1
            if delete_char in target_dict:
                window_dict[delete_char] -= 1
            if window_dict == target_dict:
                return True
        return False
```

# 575. 分糖果

给定一个偶数长度的数组，其中不同的数字代表着不同种类的糖果，每一个数字代表一个糖果。你需要把这些糖果平均分给一个弟弟和一个妹妹。返回妹妹可以获得的最大糖果的种类数。

```python
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        # 利用字典,类型超过一半则封顶一半，不超过一半则所有特别的糖全给妹妹
        dict1 = collections.defaultdict(int)
        for i in candyType:
            dict1[i] += 1
        if len(dict1) >= len(candyType)//2:
            return len(candyType)//2
        else:
            return len(dict1)
```

# 589. N 叉树的前序遍历

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        lst = []
        def submethod(node):
            if node :
                lst.append(node.val)
                for i in node.children:
                    submethod(i)
        submethod(root)
        return lst
```

# 590. N 叉树的后序遍历

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        lst = []
        def submethod(node):
            if node != None:
                for i in node.children:
                    submethod(i)
                lst.append(node.val)
        submethod(root)
        return lst
```

# 595. 大的国家

这里有张 World 表

+-----------------+------------+------------+--------------+---------------+
| name            | continent  | area       | population   | gdp           |
+-----------------+------------+------------+--------------+---------------+
| Afghanistan     | Asia       | 652230     | 25500100     | 20343000      |
| Albania         | Europe     | 28748      | 2831741      | 12960000      |
| Algeria         | Africa     | 2381741    | 37100000     | 188681000     |
| Andorra         | Europe     | 468        | 78115        | 3712000       |
| Angola          | Africa     | 1246700    | 20609294     | 100990000     |
+-----------------+------------+------------+--------------+---------------+
如果一个国家的面积超过 300 万平方公里，或者人口超过 2500 万，那么这个国家就是大国家。

编写一个 SQL 查询，输出表中所有大国家的名称、人口和面积。

例如，根据上表，我们应该输出:

+--------------+-------------+--------------+
| name         | population  | area         |
+--------------+-------------+--------------+
| Afghanistan  | 25500100    | 652230       |
| Algeria      | 37100000    | 2381741      |
+--------------+-------------+--------------+

```sql
# Write your MySQL query statement below
select name,population,area
from World
where area > 3000000 or population > 25000000;
```

# 598. 范围求和 II

给定一个初始元素全部为 0，大小为 m*n 的矩阵 M 以及在 M 上的一系列更新操作。

操作用二维数组表示，其中的每个操作用一个含有两个正整数 a 和 b 的数组表示，含义是将所有符合 0 <= i < a 以及 0 <= j < b 的元素 M[i][j] 的值都增加 1。

在执行给定的一系列操作后，你需要返回矩阵中含有最大整数的元素个数。

```python
class Solution:
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        # 贪心，排序操作数
        # 找到横向的最小值
        # 找到纵向的最小值
        # 返回两者乘积
        # 如果操作次数为空，返回原来矩阵大小
        if len(ops) == 0:
            return m*n
        row = []
        for i in ops:
            row.append(i[0])
        col = []
        for i in ops:
            col.append(i[1])
        min_row = min(row)
        min_col = min(col)
        return min_col*min_row
```

# 599. 两个列表的最小索引总和

假设Andy和Doris想在晚餐时选择一家餐厅，并且他们都有一个表示最喜爱餐厅的列表，每个餐厅的名字用字符串表示。

你需要帮助他们用最少的索引和找出他们共同喜爱的餐厅。 如果答案不止一个，则输出所有答案并且不考虑顺序。 你可以假设总是存在一个答案。

```python
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        # 做两个字典，
        # 找两个的交集
        # 根据交集找索引

        dict1 = {list1[i]:i for i in range(len(list1))}
        dict2 = {list2[i]:i for i in range(len(list2))}
        set1 = set(list1)
        set2 = set(list2)
        intersection = set1&set2
        index_lst = []
        for i in intersection:
            index_lst.append([i,(dict1[i]+dict2[i])])
        min_sum = index_lst[0][1]
        for i in index_lst:
            if min_sum > i[1]:
                min_sum = i[1]
        ans = []
        for i in index_lst:
            if i[1] == min_sum:
                ans.append(i[0])
        return ans
```

# 606. 根据二叉树创建字符串

你需要采用前序遍历的方式，将一个二叉树转换成一个由括号和整数组成的字符串。

空节点则用一对空括号 "()" 表示。而且你需要省略所有不影响字符串与原始二叉树之间的一对一映射关系的空括号对。

示例 1:

输入: 二叉树: [1,2,3,4]
       1
     /   \
    2     3
   /    
  4     

输出: "1(2(4))(3)"

解释: 原本将是“1(2(4)())(3())”，
在你省略所有不必要的空括号对之后，
它将是“1(2(4))(3)”。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def tree2str(self, root: TreeNode) -> str:
        # 欧拉遍历带hock的先序遍历
        # 由于题目限制，所以需要对子方法进行分类讨论而不是直接hock
        # 情况1 ： 有左右孩子
        # 情况2 ： 只有左孩子没有右孩子
        # 情况3 ： 只有右孩子没有左孩子
        # 情况4 ： 没有孩子 
        # 情况5 ： 是None
        self.ans = ''
        def submethod(node):
            if node != None and node.left != None and node.right != None:
                self.ans += "("
                self.ans += str(node.val)
                submethod(node.left)
                submethod(node.right)
                self.ans += ")"
            elif node != None and node.left != None and node.right == None:
                self.ans += "("
                self.ans += str(node.val)
                submethod(node.left)
                self.ans += ")"
            elif node != None and node.left == None and node.right != None:
                self.ans += "("
                self.ans += str(node.val)
                self.ans += "()"
                submethod(node.right)
                self.ans += ")"
            elif node != None and node.left == None and node.right == None:
                self.ans += "("
                self.ans += str(node.val)
                self.ans += ")"
            else:
                pass

             
        submethod(root)
        self.ans = self.ans[1:len(self.ans)-1]
        return self.ans
```

# 617. 合并二叉树

给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

示例 1:

输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7

```python
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        # 递归
        if root1 == None: return root2
        if root2 == None: return root1
        root1.val = root1.val + root2.val
        root1.left = self.mergeTrees(root1.left,root2.left)
        root1.right = self.mergeTrees(root1.right,root2.right)
        return root1
```

# 622. 设计循环队列

设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。

循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。

你的实现应该支持如下操作：

MyCircularQueue(k): 构造器，设置队列长度为 k 。
Front: 从队首获取元素。如果队列为空，返回 -1 。
Rear: 获取队尾元素。如果队列为空，返回 -1 。
enQueue(value): 向循环队列插入一个元素。如果成功插入则返回真。
deQueue(): 从循环队列中删除一个元素。如果成功删除则返回真。
isEmpty(): 检查循环队列是否为空。
isFull(): 检查循环队列是否已满。

```python
class MyCircularQueue:

    def __init__(self, k: int):
        self.p = 0 # 用以表示首指针，以及推倒尾指针，插入位置
        self.queue = [[None] for i in range(k)]
        self.size = 0
        self.maxsize = k

    def enQueue(self, value: int) -> bool:
        # enqueue本身无需更新头部索引
        if self.isFull():
            return False
        else:
            self.queue[(self.p+self.size)%self.maxsize] = value
            self.size += 1
            return True
        


    def deQueue(self) -> bool:
        # dequeue需要更新头部索引
        if self.isEmpty():
            return False
        else:
            self.queue[self.p] == None
            self.p = (self.p + 1) % self.maxsize
            self.size -= 1
            return True



    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.p]


    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[(self.p+self.size-1)%self.maxsize]

    def isEmpty(self) -> bool:
        return self.size == 0


    def isFull(self) -> bool:
        return self.size == self.maxsize

```

# 623. 在二叉树中增加一行

给定一个二叉树，根节点为第1层，深度为 1。在其第 d 层追加一行值为 v 的节点。

添加规则：给定一个深度值 d （正整数），针对深度为 d-1 层的每一非空节点 N，为 N 创建两个值为 v 的左子树和右子树。

将 N 原先的左子树，连接为新节点 v 的左子树；将 N 原先的右子树，连接为新节点 v 的右子树。

如果 d 的值为 1，深度 d - 1 不存在，则创建一个新的根节点 v，原先的整棵树将作为 v 的左子树。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def addOneRow(self, root: TreeNode, val: int, depth: int) -> TreeNode:
        # BFS
        # 当搜索深度到达指定深度时，把v加入，
        # 加入的逻辑是，如果该节点有孩子，则暂存孩子节点，记录该孩子是左孩子还是右孩子，然后新节点加入，
        if depth == 1: # 特殊情况处理
            v = TreeNode(val,left=root,right=None)
            return v
        every_level = []
        queue = [root]
        while len(every_level) != depth:
            level = []
            new_queue = []
            for i in queue:
                level.append(i) # 注意这里加的是节点
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            every_level.append(level)
            queue = new_queue
        # print(every_level) 测试用
        # 此时 every_level的最后一层为需要加爹妈的节点 
        # every_level的倒数第二层为需要加孩子的节点,这里采取加孩子的方式
        for i in every_level[-2]:
                # 这个题不管i是否有俩孩子，这一层都要加满
                temp = i.left
                i.left = TreeNode(val,temp,None)
                temp = i.right
                i.right = TreeNode(val,None,temp)
        return roo
```

# 637. 二叉树的层平均值

给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。

```python
class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        # BFS遍历求层平均值
        ans = [] # 答案存在ans里
        queue = [root]
        while len(queue) != 0:
            level = []
            new_queue = []
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            avg = sum(level)/len(level) # 取层的平均值
            ans.append(avg)
            queue = new_queue
        return ans
```

# 643. 子数组最大平均数 I

给定 `n` 个整数，找出平均数最大且长度为 `k` 的连续子数组，并输出该最大平均数。

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # 固定窗口大小的滑动窗口
        # 开始先计算最大值就行，结果再/k
        left = 0
        right = 0+k
        max_sum = sum(nums[0:k]) # 初始化窗口内的平均值,0~k是左闭右开是k个数
        temp_sum = sum(nums[0:k])
        while right < len(nums):
            temp_sum = temp_sum+nums[right]-nums[left]
            max_sum = max(max_sum,temp_sum) # 更新max值,千万要注意，max值和temp值要分开算
            left += 1
            right += 1
        return max_sum/k
```

# 645. 错误的集合

集合 s 包含从 1 到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个数字复制了成了集合里面的另外一个数字的值，导致集合 丢失了一个数字 并且 有一个数字重复 。

给定一个数组 nums 代表了集合 S 发生错误后的结果。

请你找出重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。

```python
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        # 找重复用哈希，利用sum找缺失
        k = 0
        n = len(nums)
        dict1 = {}
        p = 0
        while p < len(nums):
            if dict1.get(nums[p]) == None:
                dict1[nums[p]] = 'A'
            else:
                k = nums[p]
                break 
            p += 1
        sum1 = sum(nums)-k #去掉重复的那个数
        ans = (n+1)*n//2 - sum1
        return [k,ans]
```

# 653. 两数之和 IV - 输入 BST

给定一个二叉搜索树和一个目标结果，如果 BST 中存在两个元素且它们的和等于给定的目标结果，则返回 true。

```python
class Solution:
    def findTarget(self, root: TreeNode, k: int) -> bool:
        # 中序遍历得到数组，然后双指针对撞
        lst = []
        def submethod(node):
            if node != None:
                submethod(node.left)
                lst.append(node.val)
                submethod(node.right)
        submethod(root)
        if len(lst) < 2:
            return False
        left = 0
        right = len(lst) - 1
        while left < right:
            if lst[left] + lst[right] == k:
                return True
            elif lst[left] + lst[right] < k:
                left += 1
            elif lst[left] + lst[right] > k:
                right -= 1
        return False
```

# 654. 最大二叉树

给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：

二叉树的根是数组 nums 中的最大元素。
左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
返回有给定数组 nums 构建的 最大二叉树 。

```python
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if nums == []:
            return
        max_num = max(nums)
        root = TreeNode(max_num)
        index = nums.index(max_num)
        nums_left = nums[:index]
        nums_right = nums[index+1:]
        root.left = self.constructMaximumBinaryTree(nums_left)
        root.right = self.constructMaximumBinaryTree(nums_right)
        return root
```

# 657. 机器人能否返回原点

在二维平面上，有一个机器人从原点 (0, 0) 开始。给出它的移动顺序，判断这个机器人在完成移动后是否在 (0, 0) 处结束。

移动顺序由字符串表示。字符 move[i] 表示其第 i 次移动。机器人的有效动作有 R（右），L（左），U（上）和 D（下）。如果机器人在完成所有动作后返回原点，则返回 true。否则，返回 false。

注意：机器人“面朝”的方向无关紧要。 “R” 将始终使机器人向右移动一次，“L” 将始终向左移动等。此外，假设每次移动机器人的移动幅度相同。

```python
class Solution:
    def judgeCircle(self, moves: str) -> bool:
        #起始坐标
        x = 0
        y = 0
        for i in moves:
            if i == 'U': y += 1
            elif i == 'D': y -= 1
            elif i == 'L': x -= 1
            elif i == 'R': x += 1
        return x==0 and y==0
```

# 673. 最长递增子序列的个数

给定一个未排序的整数数组，找到最长递增子序列的个数。

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        # 动态规划
        # dp数组是 dp[i]为以nums[i]结尾的最长的递增子序列的长度
        dp = [1 for i in range(len(nums))]
        # count数组是 count[i]为以nums[i]结尾的最长递增子序列的长度
        count = [1 for i in range(len(nums))]
        for i in range(len(nums)):
            for j in range(i):
                if nums[i]>nums[j]:
                    # dp[i] 的更新方法很简单
                    # 注意count[i]的更新方法
                    if dp[i] < dp[j] + 1: # 当i因为j更新时候，那么i'继承了'j的最长递增子序列的个数
                        dp[i] = dp[j] + 1 # 更新dp[i]，即更新最大长度
                        count[i] = count[j] # 继承新的递增子序列的个数，会覆盖掉长度比他短的原值
                        # 尤其注意，即使在某个子区间执行了下面的elif，之后执行这条if的时候，原来的elif赋值过的count值被刷新
                    elif dp[i] == dp[j] + 1:
                        count[i] += count[j] # 之后有相同长度的值需要继承时，直接加上继承的递增子序列的个数
        # 此时的dp数组和count数组已经准备完成，由于dp中有一样长度的子序列，所以要找出全部的索引。以这些索引求和
        max_length = max(dp)
        index = []
        for i in range(len(dp)):
            if dp[i] == max_length:
                index.append(i)
        result = 0
        for i in index:
            result += count[i]
        return result
```

# 674. 最长连续递增序列

给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。

连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        max_length = 0
        p = 1
        temp_length = 1
        while p < len(nums):
            if nums[p] - nums[p-1] > 0:
                temp_length += 1
                p += 1 #
            elif nums[p] - nums[p-1] <= 0:
                max_length = max(max_length,temp_length)
                p += 1
                temp_length = 1
        # 若到结尾都还没有收集结果，执行收集指令
        max_length = max(max_length,temp_length)
        return max_length
```

# 682. 棒球比赛

你现在是一场采用特殊赛制棒球比赛的记录员。这场比赛由若干回合组成，过去几回合的得分可能会影响以后几回合的得分。

比赛开始时，记录是空白的。你会得到一个记录操作的字符串列表 ops，其中 ops[i] 是你需要记录的第 i 项操作，ops 遵循下述规则：

整数 x - 表示本回合新获得分数 x
"+" - 表示本回合新获得的得分是前两次得分的总和。题目数据保证记录此操作时前面总是存在两个有效的分数。
"D" - 表示本回合新获得的得分是前一次得分的两倍。题目数据保证记录此操作时前面总是存在一个有效的分数。
"C" - 表示前一次得分无效，将其从记录中移除。题目数据保证记录此操作时前面总是存在一个有效的分数。
请你返回记录中所有得分的总和。

```python
class Solution:
    def calPoints(self, ops: List[str]) -> int:
        point = []
        for i in ops:
            if i == 'C':
                point.pop(-1)
            elif i == 'D':
                point.append(point[-1]*2)
            elif i == '+':
                point.append(point[-1]+point[-2])
            else:
                point.append(int(i))
        return sum(point)
```

# 693. 交替位二进制数

给定一个正整数，检查它的二进制表示是否总是 0、1 交替出现：换句话说，就是二进制表示中相邻两位的数字永不相同。

```python
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        # 位运算
        # 例如 10101
        # 向右移位之后得到 01010
        # 两者 异或之后得到 11111 
        # 11111 + 1 得到 100000
        # 100000 和 100000-1 进行and运算
        # 得到全0
        
        # 如果不是交替类型，比如 11100
        # 右移位为 1110
        # 两者异或 10001
        # 其 +1 的都 10010
        # 10010 和 10010-1 进行and运算
        # 无法得到全0

        n2 = (n>>1)^n
        return n2&(n2+1)== 0
```

# 700. 二叉搜索树中的搜索

给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。

```python
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        def submethod(node):
            if node == None:
                return node
            if node.val == val:
                return node
            elif node.val > val:
                return submethod(node.left)
            elif node.val < val:
                return submethod(node.right)
        return submethod(root)
```

# 701. 二叉搜索树中的插入操作

给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。

注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。

```python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        # 迭代法
        if root == None:
            node = TreeNode(val)
            return node
        node = root # node作为指针开始遍历
        while node != None:
            if val < node.val:
                if node.left != None:
                    node = node.left
                else:
                    node.left = TreeNode(val) # 创建需要插入的树节点
                    break
            elif val > node.val:
                if node.right != None:
                    node = node.right
                else:
                    node.right = TreeNode(val)
                    break
        return root
```

```python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            node = TreeNode(val)
            return node
        if root.val > val: # 递归
            root.left = self.insertIntoBST(root.left, val)
        if root.val < val: # 递归
            root.right = self.insertIntoBST(root.right, val)
        return root
```

# 704. 二分查找

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start = 0
        end = len(nums)-1
        while start <= end:
            mid = (end+start)//2 #写成 （start+end)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid - 1
            else:
                start = mid + 1        
        return -1
```

# 705. 设计哈希集合

不使用任何内建的哈希表库设计一个哈希集合（HashSet）。

实现 MyHashSet 类：

void add(key) 向哈希集合中插入值 key 。
bool contains(key) 返回哈希集合中是否存在这个值 key 。
void remove(key) 将给定值 key 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。

```python
class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashset = [[None] for x in range(100000)] # 这里取巧了，直接申请了很大的空间，之后重写

    def add(self, key: int) -> None:
        index = (key + 1)% len(self.hashset)
        self.hashset[index] = key


    def remove(self, key: int) -> None:
        index = (key + 1)% len(self.hashset)
        if self.hashset[index] != None:
            self.hashset[index] = None
    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        index = (key + 1)% len(self.hashset)
        if self.hashset[index] == key:
            return True
        else:
            return False


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
```

# 706. 设计哈希映射

不使用任何内建的哈希表库设计一个哈希映射（HashMap）。

实现 MyHashMap 类：

MyHashMap() 用空映射初始化对象
void put(int key, int value) 向 HashMap 插入一个键值对 (key, value) 。如果 key 已经存在于映射中，则更新其对应的值 value 。
int get(int key) 返回特定的 key 所映射的 value ；如果映射中不包含 key 的映射，返回 -1 。
void remove(key) 如果映射中存在 key 的映射，则移除 key 和它所对应的 value 。

```python
class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        ## 利用子数组索引0和1记录k，v对
        self.hashmap = [[None,-1] for x in range(100000)] # 申请了足够大的空间，取巧了


    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        index = (key+1)%len(self.hashmap)
        self.hashmap[index][0] = key
        self.hashmap[index][1] = value


    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        index = (key+1)%len(self.hashmap)
        return self.hashmap[index][1]


    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        index = (key+1)%len(self.hashmap)
        if self.hashmap[index][0] != None:
            self.hashmap[index][0] = None
            self.hashmap[index][1] = -1


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```

# 707. 设计链表

设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。

在链表类中实现这些功能：

get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。

```python
class Node:
    def __init__(self,val=None,next=None):
        self.val = val
        self.next = next

class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 0
        # 创建两个哨兵节点,其值为0【不在题目要求的数据内就行】
        self.head = Node(0,None)
        self.tail = Node(0,None)
        self.head.next = self.tail # 初始化

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index >= self.size:
            return -1
        p = 0
        cur = self.head.next
        while p != index:
            cur = cur.next
            p += 1
        return cur.val

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.addAtIndex(0,val)


    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self.addAtIndex(self.size,val)


    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index < 0:
            index = 0
        if index > self.size:
            return
        if index >= 0 :
            self.size += 1
            node = Node(val,None)
            p = 0
            cur = self.head.next
            prev = self.head
            while p != index:
                cur = cur.next
                prev = prev.next
                p += 1
            prev.next = node
            node.next = cur
        
        # 打印检查用
        # ans_lst = []
        # cur = self.head.next
        # while cur != self.tail:
        #     ans_lst.append(cur.val)
        #     cur = cur.next
        # print(ans_lst)

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < 0 or index >= self.size:
            return
        else:
            self.size -= 1
            p = 0
            cur = self.head.next
            prev = self.head
            while p != index:
                p += 1
                cur = cur.next
                prev = prev.next
            prev.next = cur.next

# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```

# 709. 转换成小写字母

给你一个字符串 `s` ，将该字符串中的大写字母转换成相同的小写字母，返回新的字符串。

```python
class Solution:
    def toLowerCase(self, s: str) -> str:
        k = ''
        for i in s:
            k += i.lower()
        return k
```

# 728. 自除数

自除数 是指可以被它包含的每一位数除尽的数。

例如，128 是一个自除数，因为 128 % 1 == 0，128 % 2 == 0，128 % 8 == 0。

还有，自除数不允许包含 0 。

给定上边界和下边界数字，输出一个列表，列表的元素是边界（含边界）内所有的自除数。

```python
class Solution:
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        ans = []
        for i in range(left,right+1):
            temp = str(i)
            p = 0
            while p < len(temp):
                if int(temp[p]) == 0:
                    break
                if i % int(temp[p]) != 0:
                    break
                elif i % int(temp[p]) == 0 and p != len(temp)-1:
                    pass
                elif i % int(temp[p]) == 0 and p == len(temp)-1:
                    ans.append(i)
                p += 1
        return ans
```

# 739. 每日温度

请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # 利用单调栈来解决
        # 单调栈元素更新的时候对结果列表进行更新
        ans = [0 for i in range(len(temperatures))] # 初始化每一个结果为0
        stack = [] # 初始化构建单调栈，栈的右边为栈头，push操作和pop操作在右边执行
        # 栈内元素应该按照从底到上是递减排序的。
        # 因为我们需要做到的是，每次要入栈的时候，如果入栈元素是大于当前栈内的栈头元素时，则需要进行结果填充处理
        # 如果入栈元素小于等于当前栈内的栈头元素，则直接入栈无需对ans进行处理
        # 栈内记录的是索引
        p = 0
        while p < len(temperatures):
            if stack == []: #栈空直接加入
                stack.append(p)
            # 栈非空时，每次加入需要对比。即将入栈元素小于等于栈顶，则直接入栈
            if temperatures[stack[-1]] >= temperatures[p]:
                stack.append(p)
            # 如果大于栈顶，进行while循环，对每个可能符合条件的进行赋值，直到新栈顶大于即将入栈元素
            if temperatures[stack[-1]] < temperatures[p]:
                while stack != [] and temperatures[stack[-1]] < temperatures[p]:
                    ans[stack[-1]] = p - stack[-1]
                    stack.pop()
                # 处理完之后，这个元素要入栈
                stack.append(p)
            p += 1
        return ans
```

# 747. 至少是其他数字两倍的最大数

给你一个整数数组 nums ，其中总是存在 唯一的 一个最大整数 。

请你找出数组中的最大元素并检查它是否 至少是数组中每个其他数字的两倍 。如果是，则返回 最大元素的下标 ，否则返回 -1 。

```python
class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        max_num = nums[0]
        index = 0
        #找到最大值
        p = 0
        while p < len(nums):
            if nums[p] > max_num:
                max_num = nums[p]
                index = p
            p += 1
        #对每个元素判断
        p = 0
        while p < len(nums):
            if nums[p] > max_num/2 and p != index:
                return -1
            p += 1
        return index
```

# 762. 二进制表示中质数个计算置位

给定两个整数 L 和 R ，找到闭区间 [L, R] 范围内，计算置位位数为质数的整数个数。

（注意，计算置位代表二进制表示中1的个数。例如 21 的二进制表示 10101 有 3 个计算置位。还有，1 不是质数。）

```python
class Solution:
    def countPrimeSetBits(self, left: int, right: int) -> int:
        prime = {2,3,5,7,11,13,17,19,23,29,31}
        ans = 0
        for i in range(left,right+1):
            num = bin(i)[2:]
            count = 0
            for i in num:
                if i == '1':
                    count += 1
            if count in prime:
                ans += 1
        return ans
```

# 766. 托普利茨矩阵

给你一个 m x n 的矩阵 matrix 。如果这个矩阵是托普利茨矩阵，返回 true ；否则，返回 false 。

如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是 托普利茨矩阵 。

```python
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        # 一种找规律的简单方法，
        # 按照行数来判断
        # 前一行丢弃最后一个，与后一行丢弃前一个作比较
        temp = matrix[0][:-1]
        for i in range(1,len(matrix)):
            if temp == matrix[i][1:]: 
                temp = matrix[i][:-1]
            elif temp != matrix[i][1:]:
                return False
        return True
```

# 783. 二叉搜索树节点最小距离

给你一个二叉搜索树的根节点 `root` ，返回 **树中任意两不同节点值之间的最小差值** 。

```python
class Solution:
    def minDiffInBST(self, root: TreeNode) -> int:
        # 先中序遍历存储结果
        # 然后去中序遍历中的最小gap
        temp_lst = []
        def inOrder(node):
            if node == None:
                return 
            inOrder(node.left)
            temp_lst.append(node.val)
            inOrder(node.right)
        inOrder(root)
        min_gap = 0xffffffff
        p = 1
        while p < len(temp_lst):
            if min_gap > temp_lst[p]-temp_lst[p-1]:
                min_gap = temp_lst[p]-temp_lst[p-1]
            p += 1
        return min_gap
```

# 771. 宝石与石头

给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。

J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。

```python
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        # 哈希
        jew_set = set(jewels)
        count = 0
        for i in stones:
            if i in jew_set:
                count += 1
        return count
```

# 788. 旋转数字

我们称一个数 X 为好数, 如果它的每位数字逐个地被旋转 180 度后，我们仍可以得到一个有效的，且和 X 不同的数。要求每位数字都要被旋转。

如果一个数的每位数字被旋转以后仍然还是一个数字， 则这个数是有效的。0, 1, 和 8 被旋转后仍然是它们自己；2 和 5 可以互相旋转成对方（在这种情况下，它们以不同的方向旋转，换句话说，2 和 5 互为镜像）；6 和 9 同理，除了这些以外其他的数字旋转以后都不再是有效的数字。

现在我们有一个正整数 N, 计算从 1 到 N 中有多少个数 X 是好数？

```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        # 旋转后还需要和自身不同才是好数
        ban_set = {'3','4','7',} # 不可旋转的数
        rotate_dict = {'1':'1','2':'5','5':'2','6':'9','8':'8','9':'6','0':'0'}
        count = 0
        for i in range(1,n+1):
            p = 0
            i = str(i)
            rotate = ''
            while p < len(i):
                if i[p] in ban_set:
                    break
                elif i[p] not in ban_set:
                    rotate += rotate_dict[i[p]]
                p += 1
            if rotate != i and p == len(i):
                count +=1
        return count
```

# 796. 旋转字符串

给定两个字符串, A 和 B。

A 的旋转操作就是将 A 最左边的字符移动到最右边。 例如, 若 A = 'abcde'，在移动一次之后结果就是'bcdea' 。如果在若干次旋转操作之后，A 能变成B，那么返回True。

```python
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False
        template = s + s
        return (goal in template)
```

# 804. 唯一摩尔斯密码词

国际摩尔斯密码定义一种标准编码方式，将每个字母对应于一个由一系列点和短线组成的字符串， 比如: "a" 对应 ".-", "b" 对应 "-...", "c" 对应 "-.-.", 等等。

为了方便，所有26个英文字母对应摩尔斯密码表如下：

[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
给定一个单词列表，每个单词可以写成每个字母对应摩尔斯密码的组合。例如，"cab" 可以写成 "-.-..--..."，(即 "-.-." + ".-" + "-..." 字符串的结合)。我们将这样一个连接过程称作单词翻译。

返回我们可以获得所有词不同单词翻译的数量。

```python
class Solution:
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        alphabet = [chr(i) for i in range(97,97+26)]
        morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        dict1 = dict(zip(alphabet,morse))
        def trans(s:str):
            result = ''
            for i in s:
                result += dict1[i]
            return result
        lst = []
        for i in words:
            lst.append(trans(i))
        lst = set(lst)
        return len(lst)
```

# 806. 写字符串需要的行数

我们要把给定的字符串 S 从左到右写到每一行上，每一行的最大宽度为100个单位，如果我们在写某个字母的时候会使这行超过了100 个单位，那么我们应该把这个字母写到下一行。我们给定了一个数组 widths ，这个数组 widths[0] 代表 'a' 需要的单位， widths[1] 代表 'b' 需要的单位，...， widths[25] 代表 'z' 需要的单位。

现在回答两个问题：至少多少行能放下S，以及最后一行使用的宽度是多少个单位？将你的答案作为长度为2的整数列表返回。

```python
class Solution:
    def numberOfLines(self, widths: List[int], s: str) -> List[int]:
        alphabet = list("abcdefghijklmnopqrstuvwxyz")
        dict1 = dict(zip(alphabet,widths))
        p = 0
        count_line = 0
        store = 0 # 记录这一行存了多少个值
        while p < len(s):
            if dict1[s[p]] + store <= 100:
                store += dict1[s[p]]
            elif dict1[s[p]] + store > 100:
                store = dict1[s[p]] # 换行重置
                count_line += 1
            p += 1

        return [count_line+1,store]
```

# 824. 山羊拉丁文

给定一个由空格分割单词的句子 S。每个单词只包含大写或小写字母。

我们要将句子转换为 “Goat Latin”（一种类似于 猪拉丁文 - Pig Latin 的虚构语言）。

山羊拉丁文的规则如下：

如果单词以元音开头（a, e, i, o, u），在单词后添加"ma"。
例如，单词"apple"变为"applema"。

如果单词以辅音字母开头（即非元音字母），移除第一个字符并将它放到末尾，之后再添加"ma"。
例如，单词"goat"变为"oatgma"。

根据单词在句子中的索引，在单词最后添加与索引相同数量的字母'a'，索引从1开始。
例如，在第一个单词后添加"a"，在第二个单词后添加"aa"，以此类推。
返回将 S 转换为山羊拉丁文后的句子。

```python
class Solution:
    def toGoatLatin(self, sentence: str) -> str:
        # 先转换成列表，再对列表进行加工
        element_dict = {"A","a","E","e","I","i","O","o","U","u"}
        lst = sentence.split(' ')
        for i in range(len(lst)):
            if lst[i][0] not in element_dict:
                lst[i] = lst[i][1:]+lst[i][0] + "ma"
            else:
                lst[i] += "ma"
            lst[i] += (i+1)*"a"
        s = ' '.join(lst)
        return s
```

# 832. 翻转图像

给定一个二进制矩阵 A，我们想先水平翻转图像，然后反转图像并返回结果。

水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 [1, 1, 0] 的结果是 [0, 1, 1]。

反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。例如，反转 [0, 1, 1] 的结果是 [1, 0, 0]。

```python
class Solution:
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        # 模拟
        i = 0
        while i < len(image): #水平翻转
            image[i] = image[i][::-1]
            i += 1
        # 翻转图片
        i = 0
        while i < len(image):
            j = 0
            while j < len(image[0]):
                image[i][j] = 1 if image[i][j] == 0 else 0
                j += 1
            i += 1
        return image
```

# 846. 一手顺子

爱丽丝有一手（hand）由整数数组给定的牌。 

现在她想把牌重新排列成组，使得每个组的大小都是 W，且由 W 张连续的牌组成。

如果她可以完成分组就返回 true，否则返回 false。

```python
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        # 当手牌数量不能被除以尽时，之间返回False
        if len(hand)%groupSize != 0:
            return False
        # 再进一步考虑 用暴力解法过
        hand.sort()
        dict1 = collections.Counter(hand) # 处理完之后进行模拟
        # 每次找到剩余元素中的最小值
        while dict1: # 当dict1非空时
            min_num = min(dict1) # 找到最小键
            for i in range(min_num,min_num+groupSize):
                value = dict1[i]
                if not value: return False # Counter的特点是，用不存在的键查询时，返回0
                if value == 1: del dict1[i]
                else: dict1[i] -= 1
        return True

        
```

# 852. 山脉数组的峰顶索引

符合下列属性的数组 arr 称为 山脉数组 ：
arr.length >= 3
存在 i（0 < i < arr.length - 1）使得：
arr[0] < arr[1] < ... arr[i-1] < arr[i]
arr[i] > arr[i+1] > ... > arr[arr.length - 1]
给你由整数组成的山脉数组 arr ，返回任何满足 arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1] 的下标 i 。

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        #即找上极值
        prev_diff = 0
        cur_diff = 0
        p = 1
        while p < len(arr):
            prev_diff = cur_diff
            cur_diff = arr[p] - arr[p-1]
            if prev_diff > 0 and cur_diff < 0:
                return p - 1
            p += 1
        return
```

# 860. 柠檬水找零

在柠檬水摊上，每一杯柠檬水的售价为 5 美元。

顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。

每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。

注意，一开始你手头没有任何零钱。

如果你能给每位顾客正确找零，返回 true ，否则返回 false 。

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        have = {5:0,10:0,20:0}
        while have[5] >=0 and have[10] >=0 and have[20] >= 0 and len(bills) != 0:
            e = bills.pop(0)
            have[e] += 1
            if e == 10:
                have[5] -= 1
            elif e == 20:
                if have[10] > 0:
                    have[10] -= 1
                    have[5] -= 1
                else:
                    have[5] -= 3
        if len(bills) == 0 and have[5] >=0 and have[10] >=0 and have[20] >= 0:
            return True
        else:
            return False
```

# 867. 转置矩阵

给你一个二维整数数组 `matrix`， 返回 `matrix` 的 **转置矩阵** 。

矩阵的 **转置** 是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。

```python
class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        # 记录原来的长和宽，
        # 竖向遍历之后，重新组合成矩阵
        hang = len(matrix[0])
        lie = len(matrix)
        lst = deque() #存纵向遍历的
        p = 0 
        q = 0
        while p < hang:
            while q < lie:
                lst.append(matrix[q][p])
                q += 1
            p += 1
            q = 0
        # 开始切割
        hang,lie = lie,hang
        ans = []
        for j in range(lie):
            temp = []
            for i in range(hang):
                temp.append(lst.popleft())
            ans.append(temp)
        return ans
```

# 868. 二进制间距

给定一个正整数 n，找到并返回 n 的二进制表示中两个 相邻 1 之间的 最长距离 。如果不存在两个相邻的 1，返回 0 。

如果只有 0 将两个 1 分隔开（可能不存在 0 ），则认为这两个 1 彼此 相邻 。两个 1 之间的距离是它们的二进制表示中位置的绝对差。例如，"1001" 中的两个 1 的距离为 3 。

```python
class Solution:
    def binaryGap(self, n: int) -> int:
        # 转化成为二进制之后 索引相减，取最大值
        n = list(bin(n)[2:])
        count_1 = [] #用于记录1的索引
        p = 0
        while p < len(n):
            if n[p] == '1':
                count_1.append(p)
            p += 1
        if len(count_1) == 1:
            return 0
        gap_lst = []
        p = 1
        while p < len(count_1):
            gap_lst.append(count_1[p]-count_1[p-1])
            p+= 1
        return max(gap_lst)
```

# 869. 重新排序得到 2 的幂

给定正整数 N ，我们按任何顺序（包括原始顺序）将数字重新排序，注意其前导数字不能为零。

如果我们可以通过上述方式得到 2 的幂，返回 true；否则，返回 false。

```python
class Solution:
    def reorderedPowerOf2(self, n: int) -> bool:
        # 2**0 ~ 2**29次方再范围内
        # 获取每一个数的各个位数的统计字典
        num_lst = [str(2**i) for i in range(0,30)]
        dict1 = [self.trans(i) for i in num_lst]
        
        dict2 = {i:str(n).count(str(i)) for i in range(0,10)}
        return dict2 in dict1

    def trans(self,s:str):
        # 返回一个字典值
        dict_ans = {i:s.count(str(i)) for i in range(0,10)}
        return dict_ans
```

# 872. 叶子相似的树

请考虑一棵二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 叶值序列 。



举个例子，如上图所示，给定一棵叶值序列为 (6, 7, 4, 9, 8) 的树。

如果有两棵二叉树的叶值序列是相同，那么我们就认为它们是 叶相似 的。

如果给定的两个根结点分别为 root1 和 root2 的树是叶相似的，则返回 true；否则返回 false 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        # 前序遍历收集叶子,
        leaves1 = []
        leaves2 = []
        def preorder(root,lst):
            if root == None:
                return 
            if root.left == None and root.right == None:
                lst.append(root.val)
            preorder(root.left,lst)
            preorder(root.right,lst)
        preorder(root1,leaves1)
        preorder(root2,leaves2)
        return leaves1 == leaves2
```

# 876. 链表的中间结点

给定一个头结点为 `head` 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

```python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        while fast != None and slow != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next
        return slow
```

# 877. 石子游戏

亚历克斯和李用几堆石子在做游戏。偶数堆石子排成一行，每堆都有正整数颗石子 piles[i] 。

游戏以谁手中的石子最多来决出胜负。石子的总数是奇数，所以没有平局。

亚历克斯和李轮流进行，亚历克斯先开始。 每回合，玩家从行的开始或结束处取走整堆石头。 这种情况一直持续到没有更多的石子堆为止，此时手中石子最多的玩家获胜。

假设亚历克斯和李都发挥出最佳水平，当亚历克斯赢得比赛时返回 true ，当李赢得比赛时返回 false 。

```python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        # 这题有个陷阱，并不是贪心就能拿到最多的！
        # 比如：1，2，100， 4。你先拿了4，则把100放给了对手！
        # 数学思路为：假设偶数格子为红色，奇数格子为蓝色，那么你可以控制你拿到的所有棋子均为红色或者均为蓝色。
        # 你一定可以选取红色和 或者 蓝色和 更多的那一种方法,即必胜
        return True
```

# 888. 公平的糖果棒交换

爱丽丝和鲍勃有不同大小的糖果棒：A[i] 是爱丽丝拥有的第 i 根糖果棒的大小，B[j] 是鲍勃拥有的第 j 根糖果棒的大小。

因为他们是朋友，所以他们想交换一根糖果棒，这样交换后，他们都有相同的糖果总量。（一个人拥有的糖果总量是他们拥有的糖果棒大小的总和。）

返回一个整数数组 ans，其中 ans[0] 是爱丽丝必须交换的糖果棒的大小，ans[1] 是 Bob 必须交换的糖果棒的大小。

如果有多个答案，你可以返回其中任何一个。保证答案存在。

```python
class Solution:
    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        sum1 = sum(aliceSizes)
        sum2 = sum(bobSizes)
        half_gap = abs(sum1-sum2)//2
        # 需要知道他俩差多少
        # 一人需要减少差值的一半，另一人需要增大差值的一半
        if sum2>sum1:
            for i in aliceSizes:
                if i+half_gap in bobSizes:
                    return [i,i+half_gap]
        elif sum1>sum2:
            for i in bobSizes:
                if i+half_gap in aliceSizes:
                    return [i+half_gap,i]
        
```

# 893. 特殊等价字符串组

你将得到一个字符串数组 A。

每次移动都可以交换 S 的任意两个偶数下标的字符或任意两个奇数下标的字符。

如果经过任意次数的移动，S == T，那么两个字符串 S 和 T 是 特殊等价 的。

例如，S = "zzxy" 和 T = "xyzz" 是一对特殊等价字符串，因为可以先交换 S[0] 和 S[2]，然后交换 S[1] 和 S[3]，使得 "zzxy" -> "xzzy" -> "xyzz" 。

现在规定，A 的 一组特殊等价字符串 就是 A 的一个同时满足下述条件的非空子集：

该组中的每一对字符串都是 特殊等价 的
该组字符串已经涵盖了该类别中的所有特殊等价字符串，容量达到理论上的最大值（也就是说，如果一个字符串不在该组中，那么这个字符串就 不会 与该组内任何字符串特殊等价）
返回 A 中特殊等价字符串组的数量。

```python
class Solution:
    def numSpecialEquivGroups(self, words: List[str]) -> int:
        # 先提取出偶数下标子数组，排序；再提取奇数下标子数组，排序。拼接之后替换原字符串。
        # 再对words利用set计数
        temp = []
        for i in words:
            p = 0
            even_lst = []
            odd_lst = []
            while p < len(i):
                if p%2 == 0:
                    even_lst.append(i[p])
                else:
                    odd_lst.append(i[p])
                p += 1
            even_lst.sort()
            odd_lst.sort()
            ans = ''.join(i for i in even_lst) + ''.join(j for j in odd_lst)
            temp.append(ans)
        return len(set(temp))
```

# 896. 单调数列

如果数组是单调递增或单调递减的，那么它是单调的。

如果对于所有 i <= j，A[i] <= A[j]，那么数组 A 是单调递增的。 如果对于所有 i <= j，A[i]> = A[j]，那么数组 A 是单调递减的。

当给定的数组 A 是单调数组时返回 true，否则返回 false。

```python
class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        if len(nums) <= 2:
            return True
        p = 0
        while p < len(nums)-1:
            if nums[p] <= nums[p+1] and p == len(nums)-2:
                return True        
            elif nums[p] <= nums[p+1]:
               p += 1               
            else:
                break
        p = 0 
        while p < len(nums)-1:
            if nums[p] >= nums[p+1] and p == len(nums)-2:
                return True        
            elif nums[p] >= nums[p+1]:
               p += 1               
            else:
                break 
        return False
```

# 897. 递增顺序搜索树

给你一棵二叉搜索树，请你 **按中序遍历** 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。

```python
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        # 先中序遍历，将【节点】存入列表，
        # 再将列表进行处理，将列表中的每个节点left节点置空，right节点置为下一个节点
        node_lst = []
        def submethod(node):
            if node != None:
                submethod(node.left) # 左
                node_lst.append(node) # 中
                submethod(node.right) # 右
        submethod(root)
        p = 0
        while p < len(node_lst) - 1: #最后一个先不处理
            node_lst[p].left = None
            node_lst[p].right = node_lst[p+1]
            p += 1
        # 处理最后一个节点
        node_lst[-1].left = None
        node_lst[-1].right = None
        return node_lst[0] # 注意返回值是取最左边值作为根
```

# 905. 按奇偶排序数组

给定一个非负整数数组 `A`，返回一个数组，在该数组中， `A` 的所有偶数元素之后跟着所有奇数元素。

你可以返回满足此条件的任何数组作为答案。

```python
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        lst = []
        result = []
        p = 0
        while p < len(nums):
            if nums[p]%2 == 0:
                lst.insert(0,p) #偶数加在左边
            elif nums[p]%2 == 1:
                lst.append(p) #奇数加在右边
            p += 1
        for i in lst:
            result.append(nums[i])
        return result
```

# 908. 最小差值 I

给你一个整数数组 A，请你给数组中的每个元素 A[i] 都加上一个任意数字 x （-K <= x <= K），从而得到一个新数组 B 。

返回数组 B 的最大值和最小值之间可能存在的最小差值。

```python
class Solution:
    def smallestRangeI(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        min_num = min(nums)
        if max_num - min_num <= 2*k:
            return 0
        else:
            return max_num - min_num - 2*k
```

# 912. 排序数组

给你一个整数数组 `nums`，请你将该数组升序排列。

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # 手写快排,pivot必须随机不然时间超了
        self.fast_sort(nums)
        return nums
    
    def fast_sort(self,L):
        if len(L) < 2:
            return 
        pivot = L[random.randint(0,len(L)-1)]
        Less = []
        Equal = []
        Greater = []
        while L:
            if L[-1] == pivot:
                Equal.append(L.pop(-1)) 
            elif L[-1] > pivot:
                Greater.append(L.pop(-1))
            elif L[-1] < pivot:
                Less.append(L.pop(-1))
        self.fast_sort(Less)
        self.fast_sort(Greater)
        while Less:
            L.append(Less.pop(0))
        while Equal:
            L.append(Equal.pop(0))
        while Greater:
            L.append(Greater.pop(0))
```

# 917. 仅仅反转字母

给定一个字符串 `S`，返回 “反转后的” 字符串，其中不是字母的字符都保留在原地，而所有字母的位置发生反转。

```python
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        #记录字母的index，对撞指针
        alphabet_set=set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        s = list(s)
        p = 0
        index_lst = []
        while p < len(s):
            if s[p] in alphabet_set:
                index_lst.append(p)
            p += 1
        print(index_lst)
        left = 0
        right = len(index_lst)-1
        while left < right:
            s[index_lst[left]],s[index_lst[right]] = s[index_lst[right]], s[index_lst[left]]
            left += 1
            right -= 1
        ans = ''.join(s)
        return ans
```

# 922. 按奇偶排序数组 II

给定一个非负整数数组 A， A 中一半整数是奇数，一半整数是偶数。

对数组进行排序，以便当 A[i] 为奇数时，i 也是奇数；当 A[i] 为偶数时， i 也是偶数。

你可以返回任何满足上述条件的数组作为答案。

```python
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        # 用两个列表辅助
        # 前偶后奇，双指针交换
        if len(nums) == 2:
            if nums[0] % 2 == 0:
                return nums
            else:
                nums = nums[::-1]
                return nums
        odd_list = [x for x in nums if x%2 == 1]
        even_list = [x for x in nums if x%2 == 0]
        nums = even_list + odd_list
        left = 1
        right = len(nums)-2
        while left < right:
            nums[left],nums[right] = nums[right],nums[left]
            left += 2
            right -= 2
        return nums
```

# 930. 和相同的二元子数组

```python
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        # 前缀和法,初始化前缀和表，注意这个表要包含全部的nums，也就是长度为n+1
        prefix = [0]
        p = 0
        temp_sum = 0
        ans = 0 # 统计有多少个有效值
        n = len(nums)
        while p < len(nums):
            temp_sum += nums[p]
            prefix.append(temp_sum)
            p += 1

        dict1 = collections.defaultdict(int)
        dict1[0] += 1 # 
        # 往右边移动时候，此时的前缀和已知，需要找到在这之前另一个前缀和，两者之差为goal
        for i in range(n):
            r = prefix[i+1] # 右边界，右开
            l = r - goal # 
            ans += dict1[l] # 统计在这个数之前找到了多少个符合条件的值
            dict1[r] += 1 # 
        return ans
           
```

```python
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        # 滑动窗口法，基于前缀和
        left1 = 0
        left2 = 0
        # 由于需要考虑收缩时候处理一堆0的问题
        # 那么需要设置两个左指针
        # 例如 1 000000 1 0 1 1target = 3
        # left1～right 是第一个1到第三个1
        # left2～right 是跨过了那一堆0
        # 左闭右闭区间
        right = 0
        ans = 0
        sum1 = 0 # 用以记录数字总和,l1~r,它为t+1
        sum2 = 0 # 用以记录总和l2～r，他为t
        while right < len(nums):
            sum1 += nums[right] # window的数值更新
            while left1<=right and sum1 > goal: # left1可以=right，循环条件终止时，sum1 == goal
                sum1 -= nums[left1]
                left1 += 1
            sum2 += nums[right] # 另一个window数值更新
            while left2<=right and sum2 >= goal:# left2可以=right，循环条件终止时，sum2 == goal - 1
                sum2 -= nums[left2]
                left2 += 1
            ans += left2-left1 # 收集结果
            right += 1           
        return ans
```

```
解题思路

思路一：滑动窗口

按照常规的滑动窗口思路，只要设置一个左指针和一个右指针

滑动窗口元素和小于goal：右指针右移
滑动窗口元素和大于goal：左指针右移
但是这里要注意一个点：那就是子数组前面为0怎么办

很明显，仅仅是上面两个指针，左指针必然会跳过0，然后右移

那么我们可以考虑设置两个左指针，一个指向那一堆0前面的1，一个指向那一堆0后的1

比如：

1 0 0 0 0 1 1 0 1

L1 L2 R

那么L2与L1之间有几个0，那这个部分就有多少种子数组

同时本题只含有0与1，那么肯定可以保证滑动窗口内的元素和要么等于goal，要么等于goal+1

那么我们可以定义：

L1与R直接的窗口内元素为goal+1
L2与R直接的窗口内元素为goal

```

# 933. 最近的请求次数

写一个 RecentCounter 类来计算特定时间范围内最近的请求。

请你实现 RecentCounter 类：

RecentCounter() 初始化计数器，请求数为 0 。
int ping(int t) 在时间 t 添加一个新请求，其中 t 表示以毫秒为单位的某个时间，并返回过去 3000 毫秒内发生的所有请求数（包括新请求）。确切地说，返回在 [t-3000, t] 内发生的请求数。
保证 每次对 ping 的调用都使用比之前更大的 t 值。

```python
class RecentCounter:

    def __init__(self):
        self.container = collections.deque()


    def ping(self, t: int) -> int:
        # 添加维护方法，当self.container[p]< t-3000时候，截断并且仅仅保留在数据范围内的部分
        self.container.append(t)
        while self.container[0] < t-3000:
            self.container.popleft()
        return len(self.container)
```

# 938. 二叉搜索树的范围和

给定二叉搜索树的根结点 `root`，返回值位于范围 *`[low, high]`* 之间的所有结点的值的和。

```python
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        # 利用中序遍历
        ans = []
        def inorder_submethod(node,ans):
            if node != None:
                inorder_submethod(node.left,ans)
                ans.append(node.val)
                inorder_submethod(node.right,ans)
        inorder_submethod(root,ans)
        p = 0
        index_low = 0
        index_high = 0
        result = 0
        while p < len(ans):
            if ans[p] >= low:
                index_low = p
                break
            p += 1
        while p < len(ans):
            if ans[p] == high:
                index_high = p
                break
            elif ans[p] > high:
                index_high = p -1
                break
            p += 1
        return sum(ans[index_low:index_high+1])
```

# 944. 删列造序

给你由 n 个小写字母字符串组成的数组 strs，其中每个字符串长度相等。

这些字符串可以每个一行，排成一个网格。例如，strs = ["abc", "bce", "cae"] 可以排列为：

abc
bce
cae
你需要找出并删除 不是按字典序升序排列的 列。在上面的例子（下标从 0 开始）中，列 0（'a', 'b', 'c'）和列 2（'c', 'e', 'e'）都是按升序排列的，而列 1（'b', 'c', 'a'）不是，所以要删除列 1 。

返回你需要删除的列数。

```python
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        # 模拟即可
        # 用ans记录需要删除的列的数量
        ans = 0
        for i in range(len(strs[0])) :# i表示横向宽度
            for j in range(1,len(strs)): # j表示纵向宽度
                if strs[j][i] >= strs[j-1][i]:
                    pass
                else:
                    ans += 1
                    break
        return ans
```

# 961. 重复 N 次的元素

在大小为 `2N` 的数组 `A` 中有 `N+1` 个不同的元素，其中有一个元素重复了 `N`次。

返回重复了 `N` 次的那个元素。

```python
class Solution:
    def repeatedNTimes(self, nums: List[int]) -> int:
        # 先排序
        # 最中间有两个元素，只有两种极端情况。一种是全在左半边，一种是全在右半边
        nums.sort()
        # 这四个元素记为 ABCD，如果 AB相等则返回B值，否则返回C值
        A = nums[len(nums)//2-2]
        B = nums[len(nums)//2-1]
        C = nums[len(nums)//2]
        return A if A==B else C
```

# 965. 单值二叉树

如果二叉树每个节点都具有相同的值，那么该二叉树就是*单值*二叉树。

只有给定的树是单值二叉树时，才返回 `true`；否则返回 `false`。

```python
class Solution:
    def isUnivalTree(self, root: TreeNode) -> bool:
        lst = []
        def submethod(node):
            if node != None:
                lst.append(node.val)
                submethod(node.left)
                submethod(node.right)
        submethod(root)
        for i in lst:
            if i != lst[0]:
                return False
        return True
```

# 973. 最接近原点的 K 个点

我们有一个由平面上的点组成的列表 points。需要从中找出 K 个距离原点 (0, 0) 最近的点。

（这里，平面上两点之间的距离是欧几里德距离。）

你可以按任何顺序返回答案。除了点坐标的顺序之外，答案确保是唯一的。

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # 无须开根号
        # 借助了内置排序
        # 其实可以手写一个快排
        points.sort(key = lambda x: x[0]**2 + x[1]**2)
        return points[:k]
```

# 977. 有序数组的平方

给你一个按 **非递减顺序** 排序的整数数组 `nums`，返回 **每个数字的平方** 组成的新数组，要求也按 **非递减顺序** 排序。

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        # 最大值一定来源于两端！
        nums = [i**2 for i in nums]
        ans = []
        while len(nums) != 0:
            if nums[0] > nums[-1]:
                ans.append(nums.pop(0))
            else:
                ans.append(nums.pop(-1))
        # 然后倒序切片输出
        return ans[::-1]
```

# 978. 最长湍流子数组

当 A 的子数组 A[i], A[i+1], ..., A[j] 满足下列条件时，我们称其为湍流子数组：

若 i <= k < j，当 k 为奇数时， A[k] > A[k+1]，且当 k 为偶数时，A[k] < A[k+1]；
或 若 i <= k < j，当 k 为偶数时，A[k] > A[k+1] ，且当 k 为奇数时， A[k] < A[k+1]。
也就是说，如果比较符号在子数组中的每个相邻元素对之间翻转，则该子数组是湍流子数组。

返回 A 的最大湍流子数组的长度。

```python
class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        # 当数组等于1时候先判断
        if len(arr) <= 1:            
            return 1
        # 先得到它的gap数组
        p = 1
        gap_lst = []
        while p < len(arr):
            gap_lst.append(arr[p]-arr[p-1])
            p += 1
        # 处理[1,1] 和 [1,2] 类型
        if len(gap_lst) == 1:
            if gap_lst[0] == 0:
                return 1
            else:
                return 2
        if len(gap_lst)==2: # 如果gap数组只有两位长，符合条件直接返回3【方便right定为1】
            if gap_lst[0]*gap_lst[1] < 0:
                return 3
        # 打补丁，如果数组中数字全都一样
        mark = gap_lst[0]
        a = 0
        for i in gap_lst:
            if i == mark:
                a += 1
        if a == len(gap_lst) and mark == 0:
            return 1
        elif a == len(gap_lst) and mark != 0:
            return 2

        # 再在gap数组中利用元素>0,=0,<0 开始滑动窗口
        left = 0
        right = 1
        max_length = 1
        need_symbol = gap_lst[0] # 取第一位为符号位
        while right < len(gap_lst):
            target_num = gap_lst[right] # 记录即将加入数字
            if need_symbol * target_num < 0: #说明符合条件 right可以继续扩大
                max_length = max(max_length,right-left+1)
                right += 1
                need_symbol = target_num # 下次的符号位置卷入
            elif need_symbol * target_num > 0 : #说明不符合条件，left的值直接等于此次right的值
                left = right
                right += 1
                need_symbol = target_num
            elif target_num == 0 or need_symbol == 0: # 有0加入时候要再跨越，且记得判断索引是否越界,注意有连续0,和头部0
                while gap_lst[right] == 0 :
                    right += 1
                    if right == len(gap_lst):
                        break
                left = right
                if right < len(gap_lst):
                    need_symbol = gap_lst[right]        
        return max_length+1
```

# 989. 数组形式的整数加法

对于非负整数 X 而言，X 的数组形式是每位数字按从左到右的顺序形成的数组。例如，如果 X = 1231，那么其数组形式为 [1,2,3,1]。

给定非负整数 X 的数组形式 A，返回整数 X+K 的数组形式。

```python
class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        # 把k数组化，转化成大数相加数组类表示
        temp = list(str(k))
        k = [int(i) for i in temp]
        car = 0 # 默认进位为0
        ans = [] # 存结果的列表
        num1 = num
        num2 = k
        while num1 and num2:
            temp1 = (num1.pop(-1))
            temp2 = (num2.pop(-1))
            if (temp1) + (temp2) + car >= 10:
                ans.append((temp1+temp2+car-10))
                car = 1
            elif temp1 + temp2 +car <10:
                ans.append((temp1+temp2+car))
                car = 0
        remain = num1 if num1 else num2# 剩余的一个需要继续加
        while remain:
            num = (remain.pop(-1))
            if num + car >= 10:
                ans.append((num+car-10))
                car = 1
            elif num + car < 10:
                ans.append((num+car))
                car = 0
        # 最后一次是否有进位
        if car == 1:
            ans.append((car)) 
        return ans[::-1] #结果需要倒序
```

# 1005. K 次取反后最大化的数组和

给定一个整数数组 A，我们只能用以下方法修改该数组：我们选择某个索引 i 并将 A[i] 替换为 -A[i]，然后总共重复这个过程 K 次。（我们可以多次选择同一个索引 i。）

以这种方式修改数组后，返回数组可能的最大和。

```python
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int: 
        #贪心，有负数，优先把所有负数转正。越小的负数越优先。 
        #最后如果k有盈余且全正，k只针对再次排序后的最小值进行符号变换【首项为0则直接求和】,直接模2简化
        minus = 0
        if_zero = None
        nums.sort() #排序，把负数先搞出来
        p = 0
        while p < len(nums) and nums[p] < 0:
            minus += 1
            p += 1
        if minus >= k: #负数过多
            p = 0
            while p < k:
                nums[p] = -nums[p]
                p += 1
            return sum(nums)
        else: #负数不足
            over = k-minus
            p = 0
            while nums[p]<0:
                nums[p] = -nums[p]
                p += 1
            nums.sort() #再排序
            if over % 2 == 1:
                nums[0] = -nums[0]
            return sum(nums)
```

# 1009. 十进制整数的反码

每个非负整数 N 都有其二进制表示。例如， 5 可以被表示为二进制 "101"，11 可以用二进制 "1011" 表示，依此类推。注意，除 N = 0 外，任何二进制表示中都不含前导零。

二进制的反码表示是将每个 1 改为 0 且每个 0 变为 1。例如，二进制数 "101" 的二进制反码为 "010"。

给你一个十进制数 N，请你返回其二进制表示的反码所对应的十进制整数。

```python
class Solution:
    def bitwiseComplement(self, n: int) -> int:
        num = n
        num = list(bin(num))
        p = 1
        while p < len(num):
            if num[p] == '0':
                num[p] = '1'
            elif num[p] == '1':
                num[p] = '0'
            p += 1
        temp = ''.join(num)[2:]
        temp = temp[::-1]
        ans = 0
        p = 0
        while p < len(temp):
            ans += int(temp[p])*2**(p)
            p += 1
        return ans
```

# 1018. 可被 5 整除的二进制前缀

给定由若干 0 和 1 组成的数组 A。我们定义 N_i：从 A[0] 到 A[i] 的第 i 个子数组被解释为一个二进制数（从最高有效位到最低有效位）。

返回布尔值列表 answer，只有当 N_i 可以被 5 整除时，答案 answer[i] 为 true，否则为 false。

```python
class Solution:
    def prefixesDivBy5(self, nums: List[int]) -> List[bool]:
        ans = []
        prefix = 0
        for num in nums:
            prefix = ((prefix*2)+num) % 5
            ans.append(prefix == 0)
        return ans
```

# 1019. 链表中的下一个更大节点

给出一个以头节点 head 作为第一个节点的链表。链表中的节点分别编号为：node_1, node_2, node_3, ... 。

每个节点都可能有下一个更大值（next larger value）：对于 node_i，如果其 next_larger(node_i) 是 node_j.val，那么就有 j > i 且  node_j.val > node_i.val，而 j 是可能的选项中最小的那个。如果不存在这样的 j，那么下一个更大值为 0 。

返回整数答案数组 answer，其中 answer[i] = next_larger(node_{i+1}) 。

注意：在下面的示例中，诸如 [2,1,5] 这样的输入（不是输出）是链表的序列化表示，其头节点的值为 2，第二个节点值为 1，第三个节点值为 5 。

```python
class Solution:
    def nextLargerNodes(self, head: ListNode) -> List[int]:
        # 利用单调栈进行解决，因为需要找到下一个更大的节点，那么栈就应该是从底到头为递减的栈
        stack = [] # 栈记录节点值和当前节点索引,因为这一题需要返回的是节点值。节点索引是为了给ans赋值用
        # 先搜一轮得到链表长度
        sz = 0
        cur = head
        while cur:
            cur = cur.next
            sz += 1

        ans = [0 for i in range(sz)] # 初始化所有值为0,ans要存的是节点值
        cur = head
        index = 0
        while cur:
            if len(stack) == 0:
                stack.append([index,cur]) 
            if cur.val <= stack[-1][1].val: # 当值小于等于栈顶时，直接加进去就行
                stack.append([index,cur])
            if cur.val > stack[-1][1].val: # 当值大于栈顶时，需要进行ans收集处理,需要进行while循环
                while len(stack) != 0 and cur.val > stack[-1][1].val:
                    ans[stack[-1][0]] = cur.val
                    stack.pop(-1)
                stack.append([index,cur])
            cur = cur.next
            index += 1
        return ans
```

# 1021. 删除最外层的括号

有效括号字符串为空 ("")、"(" + A + ")" 或 A + B，其中 A 和 B 都是有效的括号字符串，+ 代表字符串的连接。例如，""，"()"，"(())()" 和 "(()(()))" 都是有效的括号字符串。

如果有效字符串 S 非空，且不存在将其拆分为 S = A+B 的方法，我们称其为原语（primitive），其中 A 和 B 都是非空有效括号字符串。

给出一个非空有效字符串 S，考虑将其进行原语化分解，使得：S = P_1 + P_2 + ... + P_k，其中 P_i 是有效括号字符串原语。

对 S 进行原语化分解，删除分解中每个原语字符串的最外层括号，返回 S 。

示例 1：

输入："(()())(())"
输出："()()()"
解释：
输入字符串为 "(()())(())"，原语化分解得到 "(()())" + "(())"，
删除每个部分中的最外层括号后得到 "()()" + "()" = "()()()"。

```python
class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        # 利用栈思想，一轮扫描
        # 将准备删去的括号索引进行标记,被标记的索引括号删去
        bracket_stack = []
        mark_lst = []
        p = 0
        while p < len(s):
            if s[p] == '(':
                if bracket_stack == []:
                    mark_lst.append(p)
                bracket_stack.append('(')
                p += 1
            elif s[p] == ')':
                bracket_stack.pop(-1)
                if bracket_stack == []:
                    mark_lst.append(p)
                p += 1
        p = 0
        ans = ''
        while p < len(s):
            if p != mark_lst[0]:
                ans += s[p]
            elif p == mark_lst[0]:
                mark_lst.pop(0)
            p += 1
        return ans
        # 将准备删去的括号索引进行标记,被标记的索引括号删去
```

# 1037. 有效的回旋镖

回旋镖定义为一组三个点，这些点各不相同且**不**在一条直线上。

给出平面上三个点组成的列表，判断这些点是否可以构成回旋镖。

```python
class Solution:
    def isBoomerang(self, points: List[List[int]]) -> bool:
        # 数学问题，俩向量是否共线
        # 三个点坐标分别为
        # points[0][0],points[0][1]
        # points[1][0],points[1][1]
        # points[2][0],points[2][1]
        x1,y1 = points[0][0],points[0][1]
        x2,y2 = points[1][0],points[1][1]
        x3,y3 = points[2][0],points[2][1]
        if (y1-y2)*(x2-x3) == (y2-y3)*(x1-x2):
            return False
        else:
            return True
```

# 1038. 把二叉搜索树转换为累加树

给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        # 一轮中序遍历获取总和
        sum_num = []
        node_lst = []
        def inOrder1(root):
            if root != None:
                inOrder1(root.left)
                sum_num.append(root.val)
                inOrder1(root.right)
        inOrder1(root)
        k = [sum(sum_num)] # 利用传引用的方式
        def inOrder2(root):
            if root != None:
                inOrder2(root.left)
                temp = root.val # 注意这一行的运用，先存下当前值
                root.val = k[0] # 将k值赋予节点
                k[0] = k[0]-temp # 更新k值
                inOrder2(root.right)
        inOrder2(root)
        return root
```

# 1041. 困于环中的机器人

在无限的平面上，机器人最初位于 (0, 0) 处，面朝北方。机器人可以接受下列三条指令之一：

"G"：直走 1 个单位
"L"：左转 90 度
"R"：右转 90 度
机器人按顺序执行指令 instructions，并一直重复它们。

只有在平面中存在环使得机器人永远无法离开时，返回 true。否则，返回 false。

```python
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        # 情况1执行完指令之后不朝着北方，则有回环
        # 情况2或者是执行完指令后朝着北方但是在原地
        # 情况1仅仅需要计算LR是否不相等
        # 情况2是在LR相等的情况下是否在原地 直接模拟
        count = 0 #记录LR数量
        p = 0
        # 情况1
        while p < len(instructions):
            if instructions[p] == 'L':
                count += 1
            elif instructions[p] == 'R':
                count -= 1
            p += 1
        if count%4 != 0:
            return True
        # 情况2
        p = 0
        start = [0,0]
        mark =  1 # 初始化记录哪个坐标需要变换,代表start[1]
        symbol = 0 #记录是坐标是进行加还是减
        Lcount = 0
        Rcount = 0
        while p < len(instructions):
            if instructions[p] == 'L':
                Lcount += 1
                if mark == 1: #交换指向坐标
                    mark = 0
                elif mark == 0:
                    mark = 1
            elif instructions[p] == 'R':
                Rcount += 1
                if mark == 1: #交换指向坐标
                    mark = 0
                elif mark == 0:
                    mark = 1
            elif instructions[p] == 'G':
                if symbol >= 0:
                    start[mark] = start[mark]+1
                elif symbol < 0:
                    start[mark] = start[mark]-1
            if (Lcount-Rcount)%4 == 3 or (Lcount-Rcount)%4 == 0:
                symbol = 1
            elif  (Lcount-Rcount)%4 == 1 or (Lcount-Rcount)%4 == 2:
                symbol = -1
            p += 1
        if start == [0,0]:
            return True
        else:
            return False
```

# 1047. 删除字符串中的所有相邻重复项

给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

```python
class Solution:
    def removeDuplicates(self, s: str) -> str:
        # 利用栈
        stack = []
        for i in s:   # 判断即将入栈的元素的特点         
            if stack != [] and i == stack[-1]:
                stack.pop()
            else:
                stack.append(i)
        return ''.join(stack)
```

# 1089. 复写零

给你一个长度固定的整数数组 arr，请你将该数组中出现的每个零都复写一遍，并将其余的元素向右平移。

注意：请不要在超过该数组长度的位置写入元素。

要求：请对输入的数组 就地 进行上述修改，不要从函数返回任何东西。

```python
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        p = 0
        while p < len(arr):
            if arr[p] == 0:
                arr.pop(-1)
                arr.insert(p,0)
                p += 1 # 要跨过你添加的这个0
            p += 1
```

# 1104. 二叉树寻路

在一棵无限的二叉树上，每个节点都有两个子节点，树中的节点 逐行 依次按 “之” 字形进行标记。

如下图所示，在奇数行（即，第一行、第三行、第五行……）中，按从左到右的顺序进行标记；

而偶数行（即，第二行、第四行、第六行……）中，按从右到左的顺序进行标记。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/06/28/tree.png)

给你树上某一个节点的标号 `label`，请你返回从根节点到该标号为 `label` 节点的路径，该路径是由途经的节点标号所组成的。

**示例 1：**

```
输入：label = 14
输出：[1,3,4,14]
```

**示例 2：**

```
输入：label = 26
输出：[1,2,6,10,26]
```

```python
class Solution:
    def pathInZigZagTree(self, label: int) -> List[int]:
        # 以根节点为第一行,h为行号
        # 则奇数行遵循普通有序二叉树，其值和序号的关系为val = index
        # 偶数行是逆转的普通有序二叉树,其值和序号的关系为:val + index = 2**(h) + 2**(h-1) - 1
        # 先确定label在第几层，再找label的父母
        # label如果是在奇数层，则用他的值直接用作找父母的值
        # label如果是在偶数层，把他的值转化成序号，作为下一次的找父母的值
        ans = []
        def calc(label):  # 这个计算最好是在本子上画出来辅助理解
            # 递归终止条件为：label == 1
            if label == 1:
                return ans.append(1)
            h = math.ceil(math.log(label+1,2)) # 判断在第几层
            ans.append(label)
            if h%2 == 0:
                index = 2**h + 2**(h-1) -1-label
                return calc(index//2) # 找父母
            elif h%2 == 1:
                return calc(2**(h-1)+2**(h-2)-1-label//2) #找父母
        calc(label)
        return ans[::-1] #由于是倒着加的，要逆序一下
               
```

# 1108. IP 地址无效化

给你一个有效的 [IPv4](https://baike.baidu.com/item/IPv4) 地址 `address`，返回这个 IP 地址的无效化版本。

所谓无效化 IP 地址，其实就是用 `"[.]"` 代替了每个 `"."`。

```python
class Solution:
    def defangIPaddr(self, address: str) -> str:
        p = 0
        s = ''
        while p < len(address):
            if address[p] == '.':
                s += '[.]'
            else:
                s += address[p]
            p += 1
        return s
```

# 1128. 等价多米诺骨牌对的数量

给你一个由一些多米诺骨牌组成的列表 dominoes。

如果其中某一张多米诺骨牌可以通过旋转 0 度或 180 度得到另一张多米诺骨牌，我们就认为这两张牌是等价的。

形式上，dominoes[i] = [a, b] 和 dominoes[j] = [c, d] 等价的前提是 a==c 且 b==d，或是 a==d 且 b==c。

在 0 <= i < j < dominoes.length 的前提下，找出满足 dominoes[i] 和 dominoes[j] 等价的骨牌对 (i, j) 的数量。

```python
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        # 内层先排序
        for i in dominoes:
            if i[1] < i[0]:
                i.reverse()
        # 再根据第一个排序排序
        dominoes.sort(key=lambda x:x[0])
        # 再根据第一个保持不变的情况下，以第二个排序，用到了sum
        dominoes.sort(key=sum)
        # 然后需要根据数学知识进行Cn2的组合
        lst = [] # 记录有多少组骨牌
        while dominoes:
            mark = dominoes[0]
            count = 0
            while len(dominoes) != 0 and mark == dominoes[0] :
                dominoes.pop(0)
                count += 1
            lst.append(count)
        ans = 0
        for i in lst:
            if i>=2:
                ans += i*(i-1)/2
        return int(ans)
```

# 1137. 第 N 个泰波那契数

泰波那契序列 Tn 定义如下： 

T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下 Tn+3 = Tn + Tn+1 + Tn+2

给你整数 n，请返回第 n 个泰波那契数 Tn 的值。

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        return self.ti(n)

    def ti(self,n,a=0,b=1,c=1):
        if n == 0:
            return a
        else:
            return self.ti(n-1,b,c,a+b+c)
```

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        # 动态规划法，不压缩状态
        if n <= 1:
            return n
        dp = [0 for i in range(n+1)] # 申请n+1长度的辅助数组
        # 迭代法，方向为1 -》 n
        dp[0],dp[1],dp[2] = 0,1,1
        i = 3
        while i <= n:
            dp[i] = dp[i-1] + dp[i-2] + dp[i-3]
            i += 1
        return dp[n]
```

# 1161. 最大层内元素和

给你一个二叉树的根节点 root。设根节点位于二叉树的第 1 层，而根节点的子节点位于第 2 层，依此类推。

请你找出层内元素之和 最大 的那几层（可能只有一层）的层号，并返回其中 最小 的那个。

```python
class Solution:
    def maxLevelSum(self, root: TreeNode) -> int:
        # BFS
        ans = [] #存储层序遍历的值,这个值已经是求和过的
        queue = [root]
        while len(queue) != 0:
            level = []
            new_queue = []
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            ans.append(sum(level))
            queue = new_queue
        # 自己手写一个符合要求的max函数检查ans
        p = 0
        result = 0 # 需要最终返回的层序号,最后return的时候要加一
        max1 = ans[0]
        while p < len(ans):
            if ans[p] > max1:
                result = p
                max1 = ans[p]
            p += 1
        return result+1
```

# 1184. 公交站间的距离

环形公交路线上有 n 个站，按次序从 0 到 n - 1 进行编号。我们已知每一对相邻公交站之间的距离，distance[i] 表示编号为 i 的车站和编号为 (i + 1) % n 的车站之间的距离。

环线上的公交车都可以按顺时针和逆时针的方向行驶。

返回乘客从出发点 start 到目的地 destination 之间的最短距离。

```python
class Solution:
    def distanceBetweenBusStops(self, distance: List[int], start: int, destination: int) -> int:
        # 发挥python切片特性替代指针求和
        all_distance = sum(distance)
        if start > destination:
            destination,start = start,destination
        # 使得start < destination: 好切片讨论
        clockwise = sum(distance[start:destination])
        other = all_distance - clockwise
        if clockwise <= other:
            return clockwise
        else:
            return other
```

# 1189. “气球” 的最大数量

给你一个字符串 text，你需要使用 text 中的字母来拼凑尽可能多的单词 "balloon"（气球）。

字符串 text 中的每个字母最多只能被使用一次。请你返回最多可以拼凑出多少个单词 "balloon"。

```python
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        # 建立字典，木桶原理
        set1 = set("balon") # 最终l和o的数量还要除以2
        dict1 = collections.defaultdict(int)
        for i in set1:
            dict1[i] = 0 # 初始化字典，保证每个字母都被加入
        for i in text:
            if i in set1:
                dict1[i] += 1
        dict1['l'] //= 2
        dict1['o'] //= 2
        min_num = 0xffffffff
        for i in dict1:
            if dict1[i]<min_num:
                min_num = dict1[i]
        return min_num
```

# 1200. 最小绝对差

给你个整数数组 `arr`，其中每个元素都 **不相同**。

请你找到所有具有最小绝对差的元素对，并且按升序的顺序返回。

```python
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        arr.sort()
        min_gap = 0xffffffff
        p = 1
        ans_lst = []
        while p < len(arr):
            if arr[p]-arr[p-1] < min_gap:
                min_gap = arr[p]-arr[p-1]
            p += 1
        p = 1
        while p < len(arr):
            if arr[p]-arr[p-1] == min_gap:
                ans_lst.append([arr[p-1],arr[p]])
            p += 1
        return ans_lst
```

# 1207. 独一无二的出现次数

给你一个整数数组 `arr`，请你帮忙统计数组中每个数的出现次数。

如果每个数的出现次数都是独一无二的，就返回 `true`；否则返回 `false`。

```python
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        dict1 = {i:arr.count(i) for i in arr}
        lst = []
        for i in dict1.values():
            lst.append(i)
        if len(lst) == len(set(lst)):
            return True
        else:
            return False
```

# 1217. 玩筹码

数轴上放置了一些筹码，每个筹码的位置存在数组 chips 当中。

你可以对 任何筹码 执行下面两种操作之一（不限操作次数，0 次也可以）：

将第 i 个筹码向左或者右移动 2 个单位，代价为 0。
将第 i 个筹码向左或者右移动 1 个单位，代价为 1。
最开始的时候，同一位置上也可能放着两个或者更多的筹码。

返回将所有筹码移动到同一位置（任意位置）上所需要的最小代价。

```python
class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        # 数学思想
        # 理解操作的效果，即可以把所有筹码都移动到0和1上，而不需要花费任何代价
        # 然后看0上的筹码多还是1上的筹码多，取少的那一堆
        count_0 = 0
        count_1 = 0
        for i in position:
            if i%2 == 0:
                count_0 += 1
            else:
                count_1 += 1
        return min(count_0,count_1)
```

# 1221. 分割平衡字符串

在一个 平衡字符串 中，'L' 和 'R' 字符的数量是相同的。

给你一个平衡字符串 s，请你将它分割成尽可能多的平衡字符串。

注意：分割得到的每个字符串都必须是平衡字符串。

返回可以通过分割得到的平衡字符串的 最大数量 。

```python
class Solution:
    def balancedStringSplit(self, s: str) -> int:
        # 典型贪心，借助栈思想
        count = 0
        R_stack = 0
        L_stack = 0
        p = 0
        while p < len(s):
            if s[p]=='R':
                R_stack += 1
                if R_stack == L_stack:
                    count += 1
                    R_stack = 0
                    L_stack = 0
            elif s[p]=='L':
                L_stack += 1
                if R_stack == L_stack:
                    count += 1
                    R_stack = 0
                    L_stack = 0
            p += 1
        return count
```

```python
class Solution:
    def balancedStringSplit(self, s: str) -> int:
        # 字符串已知平衡
        left = 0
        count = 0 # 统计平衡数量
        # 括号栈匹配思想，只要关成了0，就记录
        for ch in s:
            if ch == "R":
                left += 1
            elif ch == "L":
                left -= 1
            if left == 0:
                count += 1
        return count
```

# 1227. 飞机座位分配概率

有 n 位乘客即将登机，飞机正好有 n 个座位。第一位乘客的票丢了，他随便选了一个座位坐下。

剩下的乘客将会：

如果他们自己的座位还空着，就坐到自己的座位上，

当他们自己的座位被占用时，随机选择其他座位
第 n 位乘客坐在自己的座位上的概率是多少？

```python
class Solution:
    def nthPersonGetsNthSeat(self, n: int) -> float:
        # 数学思路
        # 从最少的开始想起，当第n个人进入的时候，n-1个座位已经满了，只剩下一个座位，那个座位是自己座位的概率就是1/2
        # 注意条件：如果他们自己的座位还空着，就坐到自己的座位上
        # 第一个人是1
        # 这个思路有问题
        
        return 1 if n == 1 else 1/2
```

# 1239. 串联字符串的最大长度

给定一个字符串数组 arr，字符串 s 是将 arr 某一子序列字符串连接所得的字符串，如果 s 中的每一个字符都只出现过一次，那么它就是一个可行解。

请返回所有可行解 s 中最长长度。

```python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        # 回溯
        ans = [] # 存储所有可行解，结果返回最大长度
        stack = [] # 用栈存储路径
        dict1 = collections.defaultdict(int)
        def backtracking(arr,stack):#选择列表，选择路径
            if sum(dict1.values()) == len(dict1):
                str1 = ''.join(stack)                
                ans.append(str1)
            p = 0
            while p < len(arr):
                temp = arr.copy()
                stack.append(temp[p]) # 做选择
                for i in arr[p]:
                    dict1[i] += 1
                backtracking(temp[p+1:],stack)
                stack.pop()
                for i in arr[p]:
                    dict1[i] -= 1
                    if dict1[i] == 0:
                        del dict1[i]
                p += 1
        backtracking(arr,stack)
        # 此时ans保存的是所有不包含重复字母的组合
        # 需要进一步处理
        maxlength = 0
        for i in ans:
            if len(i) > maxlength:
                maxlength = len(i)
        return maxlength
```

# 1262. 可被三整除的最大和

给你一个整数数组 nums，请你找出并返回能被三整除的元素最大和。

示例 1：

输入：nums = [3,6,5,1,8]
输出：18
解释：选出数字 3, 6, 1 和 8，它们的和是 18（可被 3 整除的最大和）。

```python
class Solution:
    def maxSumDivThree(self, nums: List[int]) -> int:
        # 数学思路 先排序
        # 再用索引记录最小的两个除以3 余1 的数
        # 用索引记录两个 除以3 余2 的数
        nums.sort()
        sum1 = sum(nums)
        if sum1 % 3 == 0:
            return sum1
        p = 0
        remain_1 = []
        remain_2 = []
        while p < len(nums): # 收集余2和余1的数
            if len(remain_1) < 2:
                if nums[p] % 3 == 1:
                    remain_1.append(nums[p])
            if len(remain_2) < 2:
                if nums[p] % 3 == 2:
                    remain_2.append(nums[p])
            if len(remain_1) == 2 and len(remain_2) == 2:
                break
            p += 1
        print(remain_1,remain_2)
        if sum1 % 3 == 1: # 余1的解决方式为丢一个1或者丢两个2
            if len(remain_1) < 1 and len(remain_2) < 2:
                return 0 # 无法找到满足条件的
            if len(remain_1) >= 1 and len(remain_2) <2: #只能丢1
                return sum1 - remain_1[0]
            if len(remain_1) < 1 and len(remain_2) == 2: #只能丢两个2
                return sum1 - sum(remain_2)
            if len(remain_1) >= 1 and len(remain_2) == 2: # 都可以丢，丢较小的
                return sum1 - min(remain_1[0],sum(remain_2))
            
        elif sum1 % 3 == 2: #余2的解决方式为丢两个1或者丢一个2
            if len(remain_1) < 2 and len(remain_2) < 1:
                return 0 #无满足条件的
            if len(remain_1) == 2 and len(remain_2) < 1:
                return sum1 - sum(remain_1)
            if len(remain_1) < 2 and len(remain_2) >= 1:
                return sum1 - remain_2[0]
            if len(remain_1) == 2 and len(remain_2) >= 1:
                return sum1 - min(sum(remain_1),remain_2[0])
```

# 1266. 访问所有点的最小时间

平面上有 n 个点，点的位置用整数坐标表示 points[i] = [xi, yi] 。请你计算访问所有这些点需要的 最小时间（以秒为单位）。

你需要按照下面的规则在平面上移动：

每一秒内，你可以：
沿水平方向移动一个单位长度，或者
沿竖直方向移动一个单位长度，或者
跨过对角线移动 sqrt(2) 个单位长度（可以看作在一秒内向水平和竖直方向各移动一个单位长度）。
必须按照数组中出现的顺序来访问这些点。
在访问某个点时，可以经过该点后面出现的点，但经过的那些点不算作有效访问。

```python
class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        #典型的贪心
        # 只要后者坐标其中之一不等于前者坐标，则爬对角线
        # 只有一个不相同的时候，加上这个时间。
        # 可以简化为作差得到向量坐标，时间 += min(abs(x),abs(y))+abs((abs(x)-abs(y)))
        t = 0
        p = 1
        while p < len(points):
            x,y = points[p][0]-points[p-1][0] ,points[p][1]-points[p-1][1]
            t += min(abs(x),abs(y))+abs((abs(x)-abs(y)))
            p += 1
        return t
```

# 1281. 整数的各位积和之差

给你一个整数 `n`，请你帮忙计算并返回该整数「各位数字之积」与「各位数字之和」的差。

```python
class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        n = list(str(n))
        mul = 1
        sum1 = 0
        for i in n:
            mul *= int(i)
            sum1 += int(i) 
        return mul-sum1
```

# 1287. 有序数组中出现次数超过25%的元素

给你一个非递减的 **有序** 整数数组，已知这个数组中恰好有一个整数，它的出现次数超过数组元素总数的 25%。

请你找到并返回这个整数

```python
class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        # 直接哈希暴力。没有用到有序的性质
        dict1 = collections.defaultdict(int)
        for i in arr:
            dict1[i] += 1
        for i in dict1:
            if dict1[i] > len(arr)/4:
                return i
```

# 1290. 二进制链表转整数

给你一个单链表的引用结点 head。链表中每个结点的值不是 0 就是 1。已知此链表是一个整数数字的二进制表示形式。

请你返回该链表所表示数字的 十进制值 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        result = 0
        cur = head
        sz = 0
        while cur != None:
            sz += 1
            cur = cur.next
        mul = 0
        cur = head
        while cur != None:
            result += cur.val*(2**(sz-mul-1))
            mul += 1
            cur = cur.next
        return result
```

# 1291. 顺次数

我们定义「顺次数」为：每一位上的数字都比前一位上的数字大 1 的整数。

请你返回由 [low, high] 范围内所有顺次数组成的 有序 列表（从小到大排序）。

```python
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        # 以原始字符串的截断技术
        # 生长
        origin = '123456789'
        lenth_low = len(str(low))
        lenth_high = len(str(high))
        p = 0
        temp = [] # 先存储所有长度符合要求的，之后再剔除
        lenth = lenth_low
        while lenth <= lenth_high:
            while p < len(origin)-lenth+1:
                temp.append(int(origin[p:p+lenth]))
                p += 1
            lenth += 1
            p = 0
        # 利用pop去除不符合要求的
        p = 0
        while p < len(temp):
            if temp[p] < low:
                temp.pop(0)
            else:
                break
        p = -1
        while p > -len(temp) - 1:
            if temp[p] > high:
                temp.pop()
            else:
                break
        return temp
```

# 1295. 统计位数为偶数的数字

给你一个整数数组 `nums`，请你返回其中位数为 **偶数** 的数字的个数。

```python
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        for i in nums:
            if len(str(i))%2 == 0:
                count += 1
        return count
```

# 1296. 划分数组为连续数字的集合

给你一个整数数组 nums 和一个正整数 k，请你判断是否可以把这个数组划分成一些由 k 个连续数字组成的集合。
如果可以，请返回 True；否则，返回 False。

```python
class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:        
        # 当手牌数量不能被除以尽时，之间返回False
        if len(nums)%k != 0:
            return False
        # 再进一步考虑 用暴力解法过
        nums.sort()
        dict1 = collections.Counter(nums) # 处理完之后进行模拟
        # 每次找到剩余元素中的最小值
        while dict1: # 当dict1非空时
            min_num = min(dict1) # 找到最小键
            for i in range(min_num,min_num+k):
                value = dict1[i]
                if not value: return False # Counter的特点是，用不存在的键查询时，返回0
                if value == 1: del dict1[i]
                else: dict1[i] -= 1
        return True

```

# 1299. 将每个元素替换为右侧最大元素

给你一个数组 `arr` ，请你将每个元素用它右边最大的元素替换，如果是最后一个元素，用 `-1` 替换。

完成所有替换操作后，请你返回这个数组。

```python
class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        # 定义一个max,初始为-1,从右边开始扫
        max_num = -1
        p = -1
        while p > -len(arr) - 1:
            temp = arr[p] # 暂存这个数值
            arr[p] = max_num
            if temp > max_num:
                max_num = temp
            p -= 1
        return arr
```

# 1302. 层数最深叶子节点的和

给你一棵二叉树的根节点 `root` ，请你返回 **层数最深的叶子节点的和** 。

```python
class Solution:
    def deepestLeavesSum(self, root: TreeNode) -> int:
        # BFS返回最后一层的和
        ans = [] # 初始化ans
        queue = [root]
        while len(queue) != 0:
            level = []
            new_queue = []
            for i in queue:
                level.append(i.val)
            for i in queue:
                if i.left:
                    new_queue.append(i.left)
                if i.right:
                    new_queue.append(i.right)
            queue = new_queue
            ans.append(level)
        # 最终结果返回最后一层的和即可
        return sum(ans[-1])
```

# 1304. 和为零的N个唯一整数

给你一个整数 `n`，请你返回 **任意** 一个由 `n` 个 **各不相同** 的整数组成的数组，并且这 `n` 个数相加和为 `0` 。

```python
class Solution:
    def sumZero(self, n: int) -> List[int]:
        # 分为n为奇数和n为偶数考虑
        # 如果n是偶数，直接lst.append(mid//2)
        # n 是奇数之间补0
        if n % 2 == 0:
            lst = []
            for i in range(1,n//2+1):
                lst.append(i)
                lst.append(-i)
        if n % 2 == 1:
            lst = [0]
            for i in range(1,n//2+1):
                lst.append(i)
                lst.append(-i)
        return lst
```

# 1305. 两棵二叉搜索树中的所有元素

给你 `root1` 和 `root2` 这两棵二叉搜索树。

请你返回一个列表，其中包含 **两棵树** 中的所有整数并按 **升序** 排序。

```python
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        # 中序遍历两棵树，得到两个有序的列表。然后两个列表归并排序
        # 利用双端队列deque增强时间效率
        sorted_lst1 = deque()
        sorted_lst2 = deque()
        def submethod(node,lst): # 递归带备忘录
            if node == None:
                return
            submethod(node.left,lst)
            lst.append(node.val)
            submethod(node.right,lst)
        submethod(root1,sorted_lst1)
        submethod(root2,sorted_lst2)
        # 归并排序
        ans = deque()
        while sorted_lst1 and sorted_lst2:
            if sorted_lst1[0] < sorted_lst2[0]:
                ans.append(sorted_lst1.popleft())
            else:
                ans.append(sorted_lst2.popleft())
        ans = ans + sorted_lst1 if sorted_lst1 else ans + sorted_lst2
        return ans
```

# 1309. 解码字母到整数映射

给你一个字符串 s，它由数字（'0' - '9'）和 '#' 组成。我们希望按下述规则将 s 映射为一些小写英文字符：

字符（'a' - 'i'）分别用（'1' - '9'）表示。
字符（'j' - 'z'）分别用（'10#' - '26#'）表示。 
返回映射之后形成的新字符串。

题目数据保证映射始终唯一。

```python
class Solution:
    def freqAlphabets(self, s: str) -> str:
        #倒过来看,借助#是否激活来判断弹出几个数
        alphabet1 = [chr(x) for x in range(97, 97 + 9)]
        num1 = [str(x) for x in range(1,10)]
        alphabet2 = [chr(x) for x in range(97 + 9, 97 + 26)]
        num2 = [str(x)  for x in range(10, 27)]
        dict1 = dict(zip(num1, alphabet1))
        dict2 = dict(zip(num2, alphabet2))
        dict1.update(dict2)

        num_set = set(num1)
        ans = ''
        p = -1
        while p > -len(s) - 1:
            if s[p] != '#':
                ans = dict1[s[p]] + ans
                p -= 1
            else:
                p -= 1
                ans = dict1[s[p-1]+s[p]] + ans
                p -= 2
        return ans
```

# 1313. 解压缩编码列表

给你一个以行程长度编码压缩的整数列表 nums 。

考虑每对相邻的两个元素 [freq, val] = [nums[2*i], nums[2*i+1]] （其中 i >= 0 ），每一对都表示解压后子列表中有 freq 个值为 val 的元素，你需要从左到右连接所有子列表以生成解压后的列表。

请你返回解压后的列表。

```python
class Solution:
    def decompressRLElist(self, nums: List[int]) -> List[int]:
        p = 1
        result = []
        while p < len(nums):
            temp = []
            for i in range(nums[p-1]): # p-1为指定次数
                temp.append(nums[p]) # p为制定元素
            result += temp
            p += 2
        return result
```

# 1323. 6 和 9 组成的最大数字

给你一个仅由数字 6 和 9 组成的正整数 `num`。

你最多只能翻转一位数字，将 6 变成 9，或者把 9 变成 6 。

请返回你可以得到的最大数字。

```python
class Solution:
    def maximum69Number (self, num: int) -> int:
        num = list(str(num))
        p = 0
        while p < len(num):
            if num[p] == '6':
                num[p] = '9'
                break
            p += 1
        str1 = ''.join(i for i in num)
        return int(str1)
```

# 1329. 将矩阵按对角线排序

矩阵对角线 是一条从矩阵最上面行或者最左侧列中的某个元素开始的对角线，沿右下方向一直到矩阵末尾的元素。例如，矩阵 mat 有 6 行 3 列，从 mat2,0 开始的 矩阵对角线 将会经过 mat2,0、mat3,1 和 mat4,2 。

给你一个 m * n 的整数矩阵 mat ，请你将同一条 矩阵对角线 上的元素按升序排序后，返回排好序的矩阵。

```python
class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        # 思路：先收集再填充，收集用临时列表
        # 一条对角线的处理方式
        # 先从主对角线处理,处理上半区
        # 这一题特别绕，在草稿纸上画出坐标关系
        for j in range(len(mat[0])):
            temp_lst = []
            for i in range(len(mat)): # i是第i行
                if i+j < len(mat[0]):
                    temp_lst.append(mat[i][i+j])                    
            temp_lst.sort()
            # print(temp_lst)
            # 填充
            p = 0
            for i in range(len(mat)): # i是第i行
                if i+j < len(mat[0]):
                    mat[i][i+j] = temp_lst[p]
                    p += 1
        # 处理下半区
        for j in range(len(mat)):
            temp_lst = []
            for i in range(len(mat)):
                if i >= j and i-j < len(mat[0]): # 注意这个条件
                    temp_lst.append(mat[i][i-j])
            temp_lst.sort()
            # print(temp_lst)
            p = 0
            for i in range(len(mat)):
                if i >= j and i-j < len(mat[0]):
                    mat[i][i-j] = temp_lst[p]
                    p += 1
        return mat

```

# 1331. 数组序号转换

给你一个整数数组 arr ，请你将数组中的每个元素替换为它们排序后的序号。

序号代表了一个元素有多大。序号编号的规则如下：

序号从 1 开始编号。
一个元素越大，那么序号越大。如果两个元素相等，那么它们的序号相同。
每个数字的序号都应该尽可能地小。

```python
class Solution:
    def arrayRankTransform(self, arr: List[int]) -> List[int]:
        # 处理arr为[]
        if len(arr) == 0:
            return []
        # 将arr带上它的索引
        arr_index = [[index,value,0] for index,value in enumerate(arr)] # 其中0为备用，之后赋予rank
        arr_index.sort(key = lambda x: x[1])
        rank = 1 # 排名
        p = 0
        while p < len(arr_index)-1:
            if arr_index[p][1] < arr_index[p+1][1]:
                arr_index[p][2] = rank
                rank += 1
                p += 1
            elif arr_index[p][1] == arr_index[p+1][1]:
                arr_index[p][2] = rank
                p += 1
        # 还需要检查最后一个
        if arr_index[p][1] == arr_index[p-1][1]:
            arr_index[p][2] = rank
        else:
            arr_index[p][2] = rank
        ans = [[] for i in range(len(arr))] # 填充答案
        for i in arr_index:
            ans[i[0]] = i[2]
        return ans
```

# 1337. 矩阵中战斗力最弱的 K 行

给你一个大小为 m * n 的矩阵 mat，矩阵由若干军人和平民组成，分别用 1 和 0 表示。

请你返回矩阵中战斗力最弱的 k 行的索引，按从最弱到最强排序。

如果第 i 行的军人数量少于第 j 行，或者两行军人数量相同但 i 小于 j，那么我们认为第 i 行的战斗力比第 j 行弱。

军人 总是 排在一行中的靠前位置，也就是说 1 总是出现在 0 之前。

```python
class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        force_lst = [sum(i) for i in mat]
        dict1 = []
        for index,force in enumerate(force_lst): # 统计完之后保存
            dict1.append([force,index])
        dict1.sort(key = lambda x:x[0])
        ans = []
        p = 0
        while p < k:
            ans.append(dict1[p][1])
            p += 1
        return ans
```

```python
class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        # 二分法 + 堆 解决
        # 寻找的是最弱的k行，即数值最小的k个数，【用大根堆维护会比较效率高】
        # 二分法找到每一行第一个非1
        index_lst = []
        for i,value in enumerate(mat):
            index_lst.append((-self.binary_search(value),-i))
        # 此时index_lst 中为 【-战斗力，-索引】
        # 注意键的选取选择
        max_heap = [] # 大顶堆
        for i in index_lst:
            heapq.heappush(max_heap,i)
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        ans = []
        while max_heap:
            ans.append(-heapq.heappop(max_heap)[1])
        return ans[::-1]
    
    def binary_search(self,lst): # 找到最左边的0
        left = 0
        right = len(lst) - 1
        while left <= right: 
            mid = (left+right)//2
            if lst[mid] == 1: # 缩小范围，只会在mid及之后
                left = mid + 1
            elif lst[mid] == 0: # 缩小范围，在左半边查找
                right = mid - 1
        return left


```

# 1338. 数组大小减半

给你一个整数数组 `arr`。你可以从中选出一个整数集合，并删除这些整数在数组中的每次出现。

返回 **至少** 能删除数组中的一半整数的整数集合的最小大小。

```python
class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        origin = len(arr) # 记录原长度
        now_length = 0
        count = 0
        sort_lst = [] # 准备排序的数组
        # 贪心，每次删除频次最多的，一旦小于原长度一半则返回
        dict1 = collections.Counter(arr) # k为元素，v为键值
        for i in dict1: # 加入数组中方便贪心排序
            sort_lst.append(dict1[i]) # 仅仅记录频次即可
        sort_lst.sort(reverse = True)
        for i in sort_lst:
            now_length += i
            count += 1
            if now_length >= origin//2:
                return count
```

# 1342. 将数字变成 0 的操作次数

给你一个非负整数 `num` ，请你返回将它变成 0 所需要的步数。 如果当前数字是偶数，你需要把它除以 2 ；否则，减去 1 。

```python
class Solution:
    def numberOfSteps(self, num: int) -> int:
        step = 0
        while num != 0:
            if num%2 == 0:
                num = num//2
                step += 1
            else:
                num -= 1
                step += 1
        return step
```

# 1343. 大小为 K 且平均值大于等于阈值的子数组数目 

给你一个整数数组 `arr` 和两个整数 `k` 和 `threshold` 。

请你返回长度为 `k` 且平均值大于等于 `threshold` 的子数组数目。

```python
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        # 固定宽度的滑动窗口
        # 直接算总和
        limit = k * threshold
        temp_sum = sum(arr[0:k]) # 初始化总和
        left = 0
        right = 0+k
        ans = 0 # 记录答案用
        if temp_sum >= limit: # 先看初始化是否符合情况
            ans += 1

        while right < len(arr):
            # 窗口移动
            temp_sum = temp_sum + arr[right] - arr[left]
            right += 1
            left += 1
            if temp_sum >= limit: # 筛选
                ans += 1
        return ans


```

# 1344. 时钟指针的夹角

给你两个数 `hour` 和 `minutes` 。请你返回在时钟上，由给定时间的时针和分针组成的较小角的角度（60 单位制）。

```python
class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        # 时针角度由hour和minutes同时决定
        # 分钟角度只由minutes决定
        minute_angle = minutes * 6
        hour_angele = hour%12 * 30 + 0.5 * minutes
        ans = min(abs(minute_angle-hour_angele),360-abs(hour_angele-minute_angle))
        return ans
```

# 1346. 检查整数及其两倍数是否存在

给你一个整数数组 arr，请你检查是否存在两个整数 N 和 M，满足 N 是 M 的两倍（即，N = 2 * M）。

更正式地，检查是否存在两个下标 i 和 j 满足：

i != j
0 <= i, j < arr.length
arr[i] == 2 * arr[j]

```python
class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
    # 双指针法,先排序数组
        arr.sort()
        # 如果它大于等于0，则往后扫
        # 如果它小于0，则往前扫
        p = 0
        while p < len(arr) and p >= 0:
            if arr[p] >= 0: #往后扫
                j = p + 1
                while  j < len(arr):
                    if arr[j] == 2 * arr[p]:
                        return True
                    j += 1
            elif arr[p] < 0: #往前扫
                j = p - 1
                while j >= 0:
                    if arr[j] == 2 * arr[p]:
                        return True
                    j -= 1
            p += 1
        return False
```

# 1356. 根据数字二进制下 1 的数目排序

给你一个整数数组 arr 。请你将数组中的元素按照其二进制表示中数字 1 的数目升序排序。

如果存在多个数字二进制中 1 的数目相同，则必须将它们按照数值大小升序排列。

请你返回排序后的数组。

```python
class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:
        def count_num(l:str):
            p = 0
            count = 0
            while p < len(l):
                if l[p] == '1':
                    count += 1
                p += 1
            return count 
        arr.sort() # 如果
        arr.sort(key=lambda x:count_num(bin(x)[2:]))
        return arr
```

# 1365. 有多少小于当前数字的数字

给你一个数组 nums，对于其中每个元素 nums[i]，请你统计数组中比它小的所有数字的数目。

换而言之，对于每个 nums[i] 你必须计算出有效的 j 的数量，其中 j 满足 j != i 且 nums[j] < nums[i] 。

以数组形式返回答案。

```python
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        ans = []
        for i in nums:
            count = 0
            for j in nums:                
                if j < i:
                    count += 1
            ans.append(count)
        return ans
```

# 1374. 生成每种字符都是奇数个的字符串

给你一个整数 n，请你返回一个含 n 个字符的字符串，其中每种字符在该字符串中都恰好出现 奇数次 。

返回的字符串必须只含小写英文字母。如果存在多个满足题目要求的字符串，则返回其中任意一个即可。

```python
class Solution:
    def generateTheString(self, n: int) -> str:
        if n%2 == 1:
            return 'a'*n
        else:
            return 'a'*(n-1)+'b'
```

# 1379. 找出克隆二叉树中的相同节点

给你两棵二叉树，原始树 original 和克隆树 cloned，以及一个位于原始树 original 中的目标节点 target。

其中，克隆树 cloned 是原始树 original 的一个 副本 。

请找出在树 cloned 中，与 target 相同 的节点，并返回对该节点的引用（在 C/C++ 等有指针的语言中返回 节点指针，其他语言返回节点本身）。

注意：

你 不能 对两棵二叉树，以及 target 节点进行更改。
只能 返回对克隆树 cloned 中已有的节点的引用。
进阶：如果树中允许出现值相同的节点，你将如何解答？

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        ans = []
        def submethod_find(node,target):
            if node == None:
                return 
            if target.val == node.val:
                ans.append(node)
            submethod_find(node.left,target)
            submethod_find(node.right,target)
        submethod_find(cloned,target)
        return ans[0]
```

# 1380. 矩阵中的幸运数

给你一个 m * n 的矩阵，矩阵中的数字 各不相同 。请你按 任意 顺序返回矩阵中的所有幸运数。

幸运数是指矩阵中满足同时下列两个条件的元素：

在同一行的所有元素中最小
在同一列的所有元素中最大

```python
class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        # 记录每一行的最小值
        row_min = []
        for i in matrix:
            row_min.append(min(i))
        # 记录每一列的最大值
        col_max = []
        for i in range(len(matrix[0])): # i是列标号
            temp_max = matrix[0][i] # 初始化最大值为每一列的第一个
            for j in range(len(matrix)): # j是行标号
            # 扫法是定i动j，初始化最大值为
                if temp_max < matrix[j][i]:
                    temp_max = matrix[j][i]
            col_max.append(temp_max)
        ans = [] # 收集结果
        # 这样暴力扫完之后，直接再次比对每个值，是否满足 幸运条件
        for i in range(len(matrix)): # i是行标号
            for j in range(len(matrix[0])): # j是列标号
                if matrix[i][j] == row_min[i] and matrix[i][j] == col_max[j]:
                    ans.append(matrix[i][j])
        return ans
        # 这样的题逻辑结构容易混淆的，不要吝啬于写注释
```

# 1381. 设计一个支持增量操作的栈

请你设计一个支持下述操作的栈。

实现自定义栈类 CustomStack ：

CustomStack(int maxSize)：用 maxSize 初始化对象，maxSize 是栈中最多能容纳的元素数量，栈在增长到 maxSize 之后则不支持 push 操作。
void push(int x)：如果栈还未增长到 maxSize ，就将 x 添加到栈顶。
int pop()：弹出栈顶元素，并返回栈顶的值，或栈为空时返回 -1 。
void inc(int k, int val)：栈底的 k 个元素的值都增加 val 。如果栈中元素总数小于 k ，则栈中的所有元素都增加 val 。

```python
class CustomStack:

    def __init__(self, maxSize: int):
        self.maxSize = maxSize
        self.Stack = []


    def push(self, x: int) -> None:
        if len(self.Stack) < self.maxSize:
            self.Stack.append(x)
        else:
            return


    def pop(self) -> int:
        if len(self.Stack) != 0:
            return self.Stack.pop(-1)
        return -1


    def increment(self, k: int, val: int) -> None:
        p = 0
        if len(self.Stack) < k:
            while p < len(self.Stack):
                self.Stack[p] += val
                p += 1
        else:
            while p < k:
                self.Stack[p] += val
                p += 1

```

# 1382. 将二叉搜索树变平衡

给你一棵二叉搜索树，请你返回一棵 平衡后 的二叉搜索树，新生成的树应该与原来的树有着相同的节点值。

如果一棵二叉搜索树中，每个节点的两棵子树高度差不超过 1 ，我们就称这棵二叉搜索树是 平衡的 。

如果有多种构造方法，请你返回任意一种。

```python
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        # 先中序遍历二叉树
        # 然后从列表中取节点获取二叉树【这个二叉树不是绝对平衡二叉树，没有avl那种性能好，但是构建速度快】
        lst = [] # 存储中序遍历节点
        def inOrder(node):
            if node != None:
                inOrder(node.left)
                lst.append(node) # 注意这里存的是节点而不是节点值
                inOrder(node.right)
        inOrder(root) # 调用一次之后返回的是中序列表
        def bulidTree(in_lst:list): #参数为列表
            if len(in_lst) == 0:
                return 
            tnode = TreeNode()
            mid = len(in_lst)//2
            tnode.val = in_lst[mid].val
            tnode.left = bulidTree(in_lst[:mid]) # 注意左闭右开，递归
            tnode.right = bulidTree(in_lst[mid+1:])
            return tnode
        return bulidTree(lst)

```

# 1387. 将整数按权重排序

我们将整数 x 的 权重 定义为按照下述规则将 x 变成 1 所需要的步数：

如果 x 是偶数，那么 x = x / 2
如果 x 是奇数，那么 x = 3 * x + 1
比方说，x=3 的权重为 7 。因为 3 需要 7 步变成 1 （3 --> 10 --> 5 --> 16 --> 8 --> 4 --> 2 --> 1）。

给你三个整数 lo， hi 和 k 。你的任务是将区间 [lo, hi] 之间的整数按照它们的权重 升序排序 ，如果大于等于 2 个整数有 相同 的权重，那么按照数字自身的数值 升序排序 。

请你返回区间 [lo, hi] 之间的整数按权重排序后的第 k 个数。

注意，题目保证对于任意整数 x （lo <= x <= hi） ，它变成 1 所需要的步数是一个 32 位有符号整数。

```python
class Solution:
    def getKth(self, lo: int, hi: int, k: int) -> int:
        lst = [x for x in range(lo,hi+1)]
        lst.sort(key=weight)
        return lst[k-1]

    
def weight(num:int):
    count = 0
    while num != 1:
        if num % 2 == 0:
            num = num/2
            count += 1
        else:
            num = 3*num + 1
            count += 1
    return count
```

# 1389. 按既定顺序创建目标数组

给你两个整数数组 nums 和 index。你需要按照以下规则创建目标数组：

目标数组 target 最初为空。
按从左到右的顺序依次读取 nums[i] 和 index[i]，在 target 数组中的下标 index[i] 处插入值 nums[i] 。
重复上一步，直到在 nums 和 index 中都没有要读取的元素。
请你返回目标数组。

题目保证数字插入位置总是存在。

```python
class Solution:
    def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
        #利用python列表的insert语法
        lst = []
        for i in nums:
            lst.insert(index.pop(0),i)
        return lst
```

# 1394. 找出数组中的幸运数

在整数数组中，如果一个整数的出现频次和它的数值大小相等，我们就称这个整数为「幸运数」。

给你一个整数数组 arr，请你从中找出并返回一个幸运数。

如果数组中存在多个幸运数，只需返回 最大 的那个。
如果数组中不含幸运数，则返回 -1 。

```python
class Solution:
    def findLucky(self, arr: List[int]) -> int:
        # 利用collecions.Counter()
        hashmap = collections.Counter(i for i in arr)
        lst = []
        for i in hashmap.items():
            if i[0] == i[1]:
                lst.append(i[0])
        if len(lst) == 0:
            return -1
        else:
            return max(lst)
```

# 1399. 统计最大组的数目

给你一个整数 n 。请你先求出从 1 到 n 的每个整数 10 进制表示下的数位和（每一位上的数字相加），然后把数位和相等的数字放到同一个组中。

请你统计每个组中的数字数目，并返回数字数目并列最多的组有多少个。

```python
class Solution:
    def countLargestGroup(self, n: int) -> int:
        hashMap = collections.Counter()
        for i in list(str(s) for s in range(1,n+1)):
            key = sum(int(a) for a in i)
            hashMap[key] += 1
        max1 = max(hashMap.values()) 
        ans = sum(1 for v in hashMap.values() if v==max1)
        return ans
```

# 1413. 逐步求和得到正数的最小值

给你一个整数数组 nums 。你可以选定任意的 正数 startValue 作为初始值。

你需要从左到右遍历 nums 数组，并将 startValue 依次累加上 nums 数组中的值。

请你在确保累加和始终大于等于 1 的前提下，选出一个最小的 正数 作为 startValue 。

```python
class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        sum_result = 0
        min_num = 0xffffffff
        for i in nums: # 找到过程中的最小值
            sum_result += i
            if sum_result < min_num:
                min_num = sum_result
        if min_num >= 1: # 如果最小值大于1，则返回1就行
            return 1
        elif min_num < 1: # 如果最小值为0或者负数，则返回-k+1
            return -min_num + 1
```

# 1414. 和为 K 的最少斐波那契数字数目

给你数字 k ，请你返回和为 k 的斐波那契数字的最少数目，其中，每个斐波那契数字都可以被使用多次。

斐波那契数字定义为：

F1 = 1
F2 = 1
Fn = Fn-1 + Fn-2 ， 其中 n > 2 。
数据保证对于给定的 k ，一定能找到可行解。

```python
class Solution:
    def findMinFibonacciNumbers(self, k: int) -> int:
        # 贪心
        # 先生成fib列表
        def yield_fib_lst(maxium,anslst,a=1,b=1):
            if maxium >= a + b :
                anslst.append(a+b)
                return yield_fib_lst(maxium,anslst,b,a+b)
            else:
                return anslst
        fib_list = [1,1]
        yield_fib_lst(k,fib_list)
        count = 1 # 记录需要多少次,题目说了一定有一次以上
        # 下面是贪心过程，每次尽量选大的，因为fib短，所以懒得用二分查找了
        res = k - fib_list.pop()
        while res != 0:
            if res < fib_list[-1]:
                fib_list.pop()
            else:
                res -= fib_list.pop()
                count += 1
        return count
```

# 1417. 重新格式化字符串

给你一个混合了数字和字母的字符串 s，其中的字母均为小写英文字母。

请你将该字符串重新格式化，使得任意两个相邻字符的类型都不同。也就是说，字母后面应该跟着数字，而数字后面应该跟着字母。

请你返回 重新格式化后 的字符串；如果无法按要求重新格式化，则返回一个 空字符串 。

```python
class Solution:
    def reformat(self, s: str) -> str:
        ans = ''
        s = list(s)
        alpha_lst = []
        digit_lst = []
        # 筛选出两个列表,存储字母和数字
        for i in s:
            if i.isalpha():
                alpha_lst.append(i)
            else:
                digit_lst.append(i)
        # 两长度的绝对值差小于等于1才可以
        if abs(len(alpha_lst)-len(digit_lst)) > 1:
            return ans
        else:
            while alpha_lst and digit_lst:
                ans += alpha_lst.pop()
                ans += digit_lst.pop()
            # 考虑字母多一个或者字母少一个。
            #字母多一个显然直接字母加到结尾
            if alpha_lst:
                ans += alpha_lst[0]
            # 数字多一个【即字母少一个】，数字加到开头
            if digit_lst:
                ans = digit_lst[0] + ans
        return ans
```

# 1418. 点菜展示表

给你一个数组 orders，表示客户在餐厅中完成的订单，确切地说， orders[i]=[customerNamei,tableNumberi,foodItemi] ，其中 customerNamei 是客户的姓名，tableNumberi 是客户所在餐桌的桌号，而 foodItemi 是客户点的餐品名称。

请你返回该餐厅的 点菜展示表 。在这张表中，表中第一行为标题，其第一列为餐桌桌号 “Table” ，后面每一列都是按字母顺序排列的餐品名称。接下来每一行中的项则表示每张餐桌订购的相应餐品数量，第一列应当填对应的桌号，后面依次填写下单的餐品数量。

注意：客户姓名不是点菜展示表的一部分。此外，表中的数据行应该按餐桌桌号升序排列。

```python
class Solution:
    def displayTable(self, orders: List[List[str]]) -> List[List[str]]:
        # 需要选取合适的数据结构进行处理
        # 本质是创建多对多的映射
        # 桌号 - 餐点 的映射
        # 餐点 - 数量 的映射
        # 先处理所有的菜,得到每桌有的菜
        food = set()
        table = set()
        for i in orders:
            food.add(i[2])
            table.add(i[1])
        food = sorted(food) # 集合转化成排序列表
        table = sorted(table)
        table.sort(key = lambda x:int(x)) # 这里要进一步按照数值大小排序
        orders.sort(key = lambda x:int(x[1])) # orders排序一下，方便下面收集

        # 以下逻辑比较麻烦
        p = 0 # p是全局指针
        every_line = [] # 收集每一行
        for i in table:
            temp_dict = defaultdict() # 对于每一桌，先创建临时字典
            for name in food: # 初始化每一种菜品为0
                temp_dict[name] = 0
            
            while p < len(orders): # 注意p是全局指针，当p没有换路时候的逻辑比较简单
                if orders[p][1] == i: 
                    temp_dict[orders[p][2]] += 1
                    p += 1
                elif orders[p][1] != i: # 当p换路时，p不能+1，而是收集本轮数据
                    temp_lst = [i]
                    for values in temp_dict.values():
                        temp_lst.append(str(values)) #注意字符化
                    every_line.append(temp_lst)
                    break
        temp_lst = [i] # 还要收集最后一次
        for values in temp_dict.values():
            temp_lst.append(str(values))
        every_line.append(temp_lst)

        ans = [] # 最终答案
        ans.append(["Table"]+food)
        for i in every_line:
            ans.append(i)
        return ans
```

# 1431. 拥有最多糖果的孩子

给你一个数组 candies 和一个整数 extraCandies ，其中 candies[i] 代表第 i 个孩子拥有的糖果数目。

对每一个孩子，检查是否存在一种方案，将额外的 extraCandies 个糖果分配给孩子们之后，此孩子有 最多 的糖果。注意，允许有多个孩子同时拥有 最多 的糖果数目。

```python
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        pivot = max(candies)
        p = 0
        ans_lst = []
        while p < len(candies):
            if candies[p]+extraCandies >= pivot:
                ans_lst.append(True)
            else:
                ans_lst.append(False)
            p += 1
        return ans_lst
```

# 1433. 检查一个字符串是否可以打破另一个字符串

给你两个字符串 s1 和 s2 ，它们长度相等，请你检查是否存在一个 s1  的排列可以打破 s2 的一个排列，或者是否存在一个 s2 的排列可以打破 s1 的一个排列。

字符串 x 可以打破字符串 y （两者长度都为 n ）需满足对于所有 i（在 0 到 n - 1 之间）都有 x[i] >= y[i]（字典序意义下的顺序）。

```python
class Solution:
    def checkIfCanBreak(self, s1: str, s2: str) -> bool:
        # 对s1,s2从小到大排序
        # 如果gap始终>=0，则可以打破，或者始终<=0，也可以打破
        # 很冗杂
        s1 = sorted(s1)
        s2 = sorted(s2)
        gap_lst = []
        for p in range(len(s1)):
            gap_lst.append(ord(s1[p])-ord(s2[p]))
        mark1 = True
        for i in gap_lst:
            if i >= 0:
                pass
            else:
                mark1 = False
                break
        if mark1 == True : return True
        mark2 = True
        for i in gap_lst:
            if i <= 0:
                pass
            else:
                mark2 = False
                break
        if mark2 == True : return True
        return False


```

# 1436. 旅行终点站

给你一份旅游线路图，该线路图中的旅行线路用数组 paths 表示，其中 paths[i] = [cityAi, cityBi] 表示该线路将会从 cityAi 直接前往 cityBi 。请你找出这次旅行的终点站，即没有任何可以通往其他城市的线路的城市。

题目数据保证线路图会形成一条不存在循环的线路，因此只会有一个旅行终点站。

```python
class Solution:
    def destCity(self, paths: List[List[str]]) -> str:
        # 这一题翻译不清醒，题干给的条件相当强,题中数组一定是旅行中的城市
        start_set = {i[0] for i in paths}
        end_set = {i[1] for i in paths}
        element = (end_set - start_set)
        ans = ''
        for i in element:
            ans += str(i)
        return ans
```

# 1437. 是否所有 1 都至少相隔 k 个元素

给你一个由若干 `0` 和 `1` 组成的数组 `nums` 以及整数 `k`。如果所有 `1` 都至少相隔 `k` 个元素，则返回 `True` ；否则，返回 `False` 。

```python
class Solution:
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        # 一轮扫记录所有1的index
        p = 0
        index_lst = []
        while p < len(nums):
            if nums[p] == 1:
                index_lst.append(p)
            p += 1
        # 只要相邻的隔了两个以上元素就行
        p = 0
        while p < len(index_lst) - 1:
            if index_lst[p+1] - index_lst[p] - 1< k :
                return False
            p += 1
        return True
```

# 1441. 用栈操作构建数组

给你一个目标数组 target 和一个整数 n。每次迭代，需要从  list = {1,2,3..., n} 中依序读取一个数字。

请使用下述操作来构建目标数组 target ：

Push：从 list 中读取一个新元素， 并将其推入数组中。
Pop：删除数组中的最后一个元素。
如果目标数组构建完成，就停止读取更多元素。
题目数据保证目标数组严格递增，并且只包含 1 到 n 之间的数字。

请返回构建目标数组所用的操作序列。

题目数据保证答案是唯一的。

```python
class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        # 下一项是要求的，则push
        # 下一项是不要求的，则push + pop
        operator_lst = []
        number = 1
        p = 0
        while p < len(target) and target[p]<=n:
            if target[p] == number and target[p]<=n:
                operator_lst.append('Push')
                number += 1
                p += 1
            elif target[p] != number and target[p]<=n:
                operator_lst.append('Push')
                operator_lst.append('Pop')
                number += 1
            
        return operator_lst
```

# 1446. 连续字符

给你一个字符串 `s` ，字符串的「能量」定义为：只包含一种字符的最长非空子字符串的长度。

请你返回字符串的能量。

```python
class Solution:
    def maxPower(self, s: str) -> int:
        energy = 1
        i = 0
        j = i + 1
        while i < len(s) and j < len(s):
            while j < len(s):
                if s[i] == s[j] and j != len(s)-1:
                    j += 1
                elif s[i] != s[j] :
                    if j-i > energy:
                        energy = j-i
                    break
                elif (s[i] == s[j] and j == len(s)-1):
                    if j-i >= energy:
                        energy = j-i + 1
                    break
            i = j
            j = i + 1
        return energy
```

# 1450. 在既定时间做作业的学生人数

给你两个整数数组 startTime（开始时间）和 endTime（结束时间），并指定一个整数 queryTime 作为查询时间。

已知，第 i 名学生在 startTime[i] 时开始写作业并于 endTime[i] 时完成作业。

请返回在查询时间 queryTime 时正在做作业的学生人数。形式上，返回能够使 queryTime 处于区间 [startTime[i], endTime[i]]（含）的学生人数。

```python
class Solution:
    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        p = 0
        count = 0
        while p < len(startTime):
            if startTime[p] <= queryTime and endTime[p] >= queryTime:
                count += 1
            p += 1
        return count
```

# 1456. 定长子串中元音的最大数目

给你字符串 s 和整数 k 。

请返回字符串 s 中长度为 k 的单个子字符串中可能包含的最大元音字母数。

英文中的 元音字母 为（a, e, i, o, u）。

```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        # 滑动窗口
        target_set = {'a','e','i','o','u'}
        # 初始化左右边界
        left = 0
        right = k 
        max_num = 0 
        # 初始化最大值
        for i in s[left:right]:
            if i in target_set:
                max_num += 1
        now_num = max_num
        while right < len(s):
            temp_char = s[right] #记录下来即将加入窗口的字符
            delete_char = s[left]
            right += 1
            left += 1
            if temp_char in target_set and delete_char in target_set:
                pass
            elif temp_char in target_set and delete_char not in target_set:
                now_num +=1
            elif temp_char not in target_set and delete_char in target_set:
                now_num -= 1
            max_num = max(max_num,now_num)
        return max_num
```

# 1460. 通过翻转子数组使两个数组相等

给你两个长度相同的整数数组 target 和 arr 。

每一步中，你可以选择 arr 的任意 非空子数组 并将它翻转。你可以执行此过程任意次。

如果你能让 arr 变得与 target 相同，返回 True；否则，返回 False 。

```python
class Solution:
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        # 实质上是完成冒泡排序，如果俩数组排序后相等即相等。
        if len(target) != len(arr):
            return False
        target.sort()
        arr.sort()
        return target == arr
```

# 1464. 数组中两元素的最大乘积

给你一个整数数组 nums，请你选择数组的两个不同下标 i 和 j，使 (nums[i]-1)*(nums[j]-1) 取得最大值。

请你计算并返回该式的最大值。

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if len(nums) == 2:
            return (nums[0]-1)*(nums[1]-1)
        nums.sort()
        return (nums[-1]-1)*(nums[-2]-1)
```

# 1470. 重新排列数组

给你一个数组 nums ，数组中有 2n 个元素，按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。

请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，返回重排后的数组。

```python
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        #节约性能直接切片nums,防止动态数组调整大小大量消耗
        x = nums[:len(nums)//2]
        y = nums[len(nums)//2:]
        ans = []
        while len(x) != 0:
            ans.append(x.pop(0))
            ans.append(y.pop(0))
        return ans
```

# 1480. 一维数组的动态和

给你一个数组 nums 。数组「动态和」的计算公式为：runningSum[i] = sum(nums[0]…nums[i]) 。

请返回 nums 的动态和。

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        #带备忘录的运算防止O(n2)
        p = 0
        lst = [0]
        while p < len(nums):
            lst.append(nums[p]+lst[-1])
            p += 1
        return lst[1:]
```

# 1486. 数组异或操作

给你两个整数，n 和 start 。

数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）且 n == nums.length 。

请返回 nums 中所有元素按位异或（XOR）后得到的结果。

```python
class Solution:
    def xorOperation(self, n: int, start: int) -> int:
        # 找规律用数学解法时间效率很高，这里直接用模拟
        ans = 0
        for i in range(n):
            ans ^= start+2*i
        return ans
```

# 1491. 去掉最低工资和最高工资后的工资平均值

给你一个整数数组 salary ，数组里每个数都是 唯一 的，其中 salary[i] 是第 i 个员工的工资。

请你返回去掉最低工资和最高工资以后，剩下员工工资的平均值。

```python
class Solution:
    def average(self, salary: List[int]) -> float:
        max_salary = max(salary)
        min_salary = min(salary)
        avg = (sum(salary)-max_salary-min_salary)/(len(salary)-2)
        return avg
```

# 1492. n 的第 k 个因子

给你两个正整数 n 和 k 。

如果正整数 i 满足 n % i == 0 ，那么我们就说正整数 i 是整数 n 的因子。

考虑整数 n 的所有因子，将它们 升序排列 。请你返回第 k 个因子。如果 n 的因子数少于 k ，请你返回 -1 。

```python
class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        # 为了提升性能，防止长度频繁调整，按大因子和小因子进行列表合并
        small = []
        big = []
        step = 1
        sqrt_value = math.sqrt(n)
        while step <= sqrt_value:
            if n%step == 0:
                small.append(step)
                big.append(n//step)
            step += 1
        big = big[::-1] # 倒序处理big
        # 检查small的最后一个和big的第一个是否相等
        if small[-1] == big[0]:
            merge = small+big[1:]
        else:
            merge = small+big
        if k > len(merge):
            return -1
        else:
            return merge[k-1]
```

# 1496. 判断路径是否相交

给你一个字符串 path，其中 path[i] 的值可以是 'N'、'S'、'E' 或者 'W'，分别表示向北、向南、向东、向西移动一个单位。

机器人从二维平面上的原点 (0, 0) 处开始出发，按 path 所指示的路径行走。

如果路径在任何位置上出现相交的情况，也就是走到之前已经走过的位置，请返回 True ；否则，返回 False 。

```python
class Solution:
    def isPathCrossing(self, path: str) -> bool:
        dict1 = {}
        coordination = [0, 0]
        p = 0
        if dict1.get(tuple(coordination)) == None:
            dict1[tuple(coordination)] = 'Mark'
        while p < len(path):
            if path[p] == 'N':
                coordination[1] += 1
            elif path[p] == 'S':
                coordination[1] -= 1
            elif path[p] == 'E':
                coordination[0] -= 1
            elif path[p] == 'W':
                coordination[0] += 1
            p += 1
            coordination = list(coordination)
            if dict1.get(tuple(coordination)) == None:
                dict1[tuple(coordination)] = 'Mark'
            else:
                return True
        return False
```

# 1502. 判断能否形成等差数列

给你一个数字数组 arr 。

如果一个数列中，任意相邻两项的差总等于同一个常数，那么这个数列就称为 等差数列 。

如果可以重新排列数组形成等差数列，请返回 true ；否则，返回 false 。

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        arr.sort()
        gap = arr[1]-arr[0]
        i = 1
        while i < len(arr):
            if arr[i] - arr[i-1] != gap:
                return False
            i += 1
        return True
```

# 1507. 转变日期格式

给你一个字符串 date ，它的格式为 Day Month Year ，其中：

Day 是集合 {"1st", "2nd", "3rd", "4th", ..., "30th", "31st"} 中的一个元素。
Month 是集合 {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"} 中的一个元素。
Year 的范围在 [1900, 2100] 之间。
请你将字符串转变为 YYYY-MM-DD 的格式，其中：

YYYY 表示 4 位的年份。
MM 表示 2 位的月份。
DD 表示 2 位的天数。

```python
class Solution:
    def reformatDate(self, date: str) -> str:
        day,month,year = date.split(' ')
        day = day[:-2]
        day = day if len(day)>1 else '0'+day
        month_dict = {"Jan":"01", "Feb":'02', "Mar":'03', "Apr":'04', "May":'05', "Jun":'06', "Jul":'07', "Aug":'08', "Sep":'09', "Oct":'10', "Nov":'11', "Dec":'12'}
        return (year+'-'+month_dict[month]+'-'+day)
```

# 1512. 好数对的数目

给你一个整数数组 nums 。

如果一组数字 (i,j) 满足 nums[i] == nums[j] 且 i < j ，就可以认为这是一组 好数对 。

返回好数对的数目。

```python
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        i = 0
        j = i + 1
        count = 0
        while i < len(nums) and j < len(nums):
            while j < len(nums):
                if nums[i] == nums[j]:
                    count += 1
                j += 1
            i += 1
            j = i + 1
        return count
```

# 1518. 换酒问题

小区便利店正在促销，用 numExchange 个空酒瓶可以兑换一瓶新酒。你购入了 numBottles 瓶酒。

如果喝掉了酒瓶中的酒，那么酒瓶就会变成空的。

请你计算 最多 能喝到多少瓶酒。

```python
class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        can_drink = numBottles
        while numBottles >= numExchange:
            get = numBottles//numExchange
            numBottles = numBottles%numExchange + get
            can_drink += get
        return can_drink
```

# 1523. 在区间范围内统计奇数数目

给你两个非负整数 `low` 和 `high` 。请你返回 `low` 和 `high` 之间（包括二者）奇数的数目。

```python
class Solution:
    def countOdds(self, low: int, high: int) -> int:
        if high%2 == 0 and low%2 == 0:
            return (high-low)//2
        else:
            return (high-low)//2 + 1
```

# 1528. 重新排列字符串

给你一个字符串 s 和一个 长度相同 的整数数组 indices 。

请你重新排列字符串 s ，其中第 i 个字符需要移动到 indices[i] 指示的位置。

返回重新排列后的字符串。

```python
class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        s = list(s)
        dict1 = dict(zip(indices,s)) #字典，利用序号找字母
        ans = ''
        for i in range(0,len(s)):
            ans += dict1[i]
        return ans
```

# 1550. 存在连续三个奇数的数组

给你一个整数数组 `arr`，请你判断数组中是否存在连续三个元素都是奇数的情况：如果存在，请返回 `true` ；否则，返回 `false` 。

```python
class Solution:
    def threeConsecutiveOdds(self, arr: List[int]) -> bool:
        if len(arr) <= 2:
            return False
        p = 0
        count = 0
        while p < len(arr) :
            if arr[p] % 2 == 1:
                count += 1
            elif arr[p] % 2 == 0:
                count = 0
            if count == 3:
                return True
            p += 1
        return False
```

# 1561. 你可以获得的最大硬币数目

有 3n 堆数目不一的硬币，你和你的朋友们打算按以下方式分硬币：

每一轮中，你将会选出 任意 3 堆硬币（不一定连续）。
Alice 将会取走硬币数量最多的那一堆。
你将会取走硬币数量第二多的那一堆。
Bob 将会取走最后一堆。
重复这个过程，直到没有更多硬币。
给你一个整数数组 piles ，其中 piles[i] 是第 i 堆中硬币的数目。

返回你可以获得的最大硬币数目。

```python
class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        # 贪心，bob给最少的，alice给最多的，自己拿第二的
        piles.sort()
        ans = 0
        while len(piles) != 0:
            piles.pop(0)
            piles.pop(-1)
            ans += piles.pop(-1)
        return ans
```

# 1572. 矩阵对角线元素的和

给你一个正方形矩阵 `mat`，请你返回矩阵对角线元素的和。

请你返回在矩阵主对角线上的元素和副对角线上且不在主对角线上元素的和。

```python
class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        p = 0
        ans = 0
        lenth_is_odd = len(mat)%2 
        while p < len(mat):
            ans += mat[p][p]
            p += 1
        p = 0
        while p < len(mat):
            ans += mat[len(mat)-1-p][p]
            p += 1
        if lenth_is_odd:
            mid = len(mat)//2
            ans -= mat[mid][mid]
        return ans
```

# 1582. 二进制矩阵中的特殊位置

给你一个大小为 rows x cols 的矩阵 mat，其中 mat[i][j] 是 0 或 1，请返回 矩阵 mat 中特殊位置的数目 。

特殊位置 定义：如果 mat[i][j] == 1 并且第 i 行和第 j 列中的所有其他元素均为 0（行和列的下标均 从 0 开始 ），则位置 (i, j) 被称为特殊位置。

```python
class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        # 暴力搜索
        # 加入集合，如果i，j已经在集合中就不进一步搜索
        visited_rows = set() # 横向扫描
        visited_cols = set() # 纵向扫描
        count = 0 # 记录合法位置数目
        def submethod_check(i,j): # 传入坐标进行搜索
            p = 0
            while p < len(mat[0]):
                if mat[i][p] == 1 and p != j: # 在横行中扫描，如果不对劲
                    visited_rows.add(i) # 则此行不对劲
                    return False
                p += 1
            p = 0
            while p < len(mat) : # 在纵列中扫描，如果不对劲
                if mat[p][j] == 1 and p != i:
                    visited_cols.add(j) # 则此列不对劲
                    return False
                p += 1
            return True # 如果没有过滤出来，则返回True
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if (i not in visited_rows) and (j not in visited_cols) and mat[i][j] == 1:
                    count = count+ 1 if submethod_check(i,j) else count
        return count
```

# 1588. 所有奇数长度子数组的和

给你一个正整数数组 arr ，请你计算所有可能的奇数长度子数组的和。

子数组 定义为原数组中的一个连续子序列。

请你返回 arr 中 所有奇数长度子数组的和 。

```python
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        # 滑动窗口
        ans = 0
        i = 0
        while i < len(arr): # 注意滑动窗口的方法，先计算以i开头的全部可能，制造出与j有关的窗口
            j = i
            while j < len(arr):
                ans += sum(arr[i:j+1])
                j += 2
            i += 1
        return ans
```

# 1603. 设计停车系统

请你给一个停车场设计一个停车系统。停车场总共有三种不同大小的车位：大，中和小，每种尺寸分别有固定数目的车位。

请你实现 ParkingSystem 类：

ParkingSystem(int big, int medium, int small) 初始化 ParkingSystem 类，三个参数分别对应每种停车位的数目。
bool addCar(int carType) 检查是否有 carType 对应的停车位。 carType 有三种类型：大，中，小，分别用数字 1， 2 和 3 表示。一辆车只能停在  carType 对应尺寸的停车位中。如果没有空车位，请返回 false ，否则将该车停入车位并返回 true 。

```python
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.big = ['A' for i in range(big)]
        self.mid = ['A' for i in range(medium)]
        self.small = ['A' for i in range(small)]


    def addCar(self, carType: int) -> bool:
        if carType == 1:
            if len(self.big)>0:
                self.big.pop(0)
                return True
            else:
                return False
        if carType == 2:
            if len(self.mid)>0:
                self.mid.pop(0)
                return True
            else:
                return False
        if carType == 3:
            if len(self.small)>0:
                self.small.pop(0)
                return True
            else:
                return False

```

# 1609. 奇偶树

如果一棵二叉树满足下述几个条件，则可以称为 奇偶树 ：

二叉树根节点所在层下标为 0 ，根的子节点所在层下标为 1 ，根的孙节点所在层下标为 2 ，依此类推。
偶数下标 层上的所有节点的值都是 奇 整数，从左到右按顺序 严格递增
奇数下标 层上的所有节点的值都是 偶 整数，从左到右按顺序 严格递减
给你二叉树的根节点，如果二叉树为 奇偶树 ，则返回 true ，否则返回 false 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isEvenOddTree(self, root: TreeNode) -> bool:
        # BFS
        queue = [root]
        ans = []
        mark = 0 # 作为层数标记，判断递增/递减和奇偶性
        while len(queue) != 0:
            level = []
            newqueue = []
            if mark % 2 == 0:
                for i in queue:
                    if i.val % 2 != 1:
                        return False
                    else:
                        level.append(i.val)
                if sorted(level) != level or len(level) != len(set(level)):
                    return False
            elif mark % 2 == 1:
                for i in queue:
                    if i.val % 2 != 0:
                        return False
                    else:
                        level.append(i.val)
                if sorted(level)[::-1] != level or len(level) != len(set(level)):
                    return False
            for i in queue:
                if i.left != None:
                    newqueue.append(i.left)
                if i.right != None:
                    newqueue.append(i.right)
            queue = newqueue
            mark += 1
        return True
```

# 1614. 括号的最大嵌套深度

如果字符串满足以下条件之一，则可以称之为 有效括号字符串（valid parentheses string，可以简写为 VPS）：

字符串是一个空字符串 ""，或者是一个不为 "(" 或 ")" 的单字符。
字符串可以写为 AB（A 与 B 字符串连接），其中 A 和 B 都是 有效括号字符串 。
字符串可以写为 (A)，其中 A 是一个 有效括号字符串 。
类似地，可以定义任何有效括号字符串 S 的 嵌套深度 depth(S)：

depth("") = 0
depth(C) = 0，其中 C 是单个字符的字符串，且该字符不是 "(" 或者 ")"
depth(A + B) = max(depth(A), depth(B))，其中 A 和 B 都是 有效括号字符串
depth("(" + A + ")") = 1 + depth(A)，其中 A 是一个 有效括号字符串
例如：""、"()()"、"()(()())" 都是 有效括号字符串（嵌套深度分别为 0、1、2），而 ")(" 、"(()" 都不是 有效括号字符串 。

给你一个 有效括号字符串 s，返回该字符串的 s 嵌套深度 。

```python
class Solution:
    def maxDepth(self, s: str) -> int:
        #利用栈思想，记录扫描过程中最大的栈深度
        lst = [0] # 给无括号的加上标记
        p = 0
        count = 0
        while p < len(s):
            if s[p] == '(':
                count += 1
            elif s[p] == ')':
                lst.append(count)
                count -= 1
            p += 1
        return max(lst)
```

# 1619. 删除某些元素后的数组均值

给你一个整数数组 `arr` ，请你删除最小 `5%` 的数字和最大 `5%` 的数字后，剩余数字的平均值。

与 **标准答案** 误差在 `10^-5` 的结果都被视为正确结果。

```python
class Solution:
    def trimMean(self, arr: List[int]) -> float:
        # 排序后切片
        n = len(arr)//20
        arr.sort()
        new_arr = arr[n:len(arr)-n]
        ans = sum(new_arr)/(len(new_arr))
        return ans
```

# 1630. 等差子数组

如果一个数列由至少两个元素组成，且每两个连续元素之间的差值都相同，那么这个序列就是 等差数列 。更正式地，数列 s 是等差数列，只需要满足：对于每个有效的 i ， s[i+1] - s[i] == s[1] - s[0] 都成立。

例如，下面这些都是 等差数列 ：

1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
下面的数列 不是等差数列 ：

1, 1, 2, 5, 7
给你一个由 n 个整数组成的数组 nums，和两个由 m 个整数组成的数组 l 和 r，后两个数组表示 m 组范围查询，其中第 i 个查询对应范围 [l[i], r[i]] 。所有数组的下标都是 从 0 开始 的。

返回 boolean 元素构成的答案列表 answer 。如果子数组 nums[l[i]], nums[l[i]+1], ... , nums[r[i]] 可以 重新排列 形成 等差数列 ，answer[i] 的值就是 true；否则answer[i] 的值就是 false 。

```python
class Solution:
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        p = 0
        ans = []
        while p < len(l):
            judge = nums[l[p]:r[p]+1] #切片判断 judge是队列
            ans.append(Solution.is_true(self,judge))
            p += 1
        return ans
    
    def is_true(self,lst):
        # 判断子序列是否是等差数列。
        lst.sort()
        if len(lst) < 2:
            return False
        p = 0
        gap = []
        while p < len(lst) - 1:
            gap.append(lst[p+1]-lst[p])
            p += 1
        mark = gap[0]
        for i in gap:
            if i != mark:
                return False
        return True
```

# 1636. 按照频率将数组升序排序

给你一个整数数组 `nums` ，请你将数组按照每个值的频率 **升序** 排序。如果有多个值的频率相同，请你按照数值本身将它们 **降序** 排序。 

请你返回排序后的数组。

```python
class Solution:
    def frequencySort(self, nums: List[int]) -> List[int]:
        # 由于给的数组长度较短，直接利用python建立字典的count语法
        dict1 = {(i,nums.count(i)) for i in nums} #key是数值，value是频次
        lst = list(dict1)
        # 两次排序，先按照数值降序
        # 再按照value排序
        lst.sort(reverse = True)
        lst.sort(key = lambda x:x[1])
        # 再解包
        ans = []
        for i in lst:
            for u in range(i[1]):
                ans.append(i[0])
        return ans
```

# 1637. 两点之间不包含任何点的最宽垂直面积

给你 n 个二维平面上的点 points ，其中 points[i] = [xi, yi] ，请你返回两点之间内部不包含任何点的 最宽垂直面积 的宽度。

垂直面积 的定义是固定宽度，而 y 轴上无限延伸的一块区域（也就是高度为无穷大）。 最宽垂直面积 为宽度最大的一个垂直面积。

请注意，垂直区域 边上 的点 不在 区域内。

```python
class Solution:
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        # 根据横坐标排序，取相邻的最大gap
        points.sort(key = lambda x:x[0])
        max_gap = 0 # 初始化一个最小值
        p = 1
        while p < len(points):
            max_gap = max(max_gap,points[p][0]-points[p-1][0])
            p += 1
        return max_gap
```

# 1652. 拆炸弹

你有一个炸弹需要拆除，时间紧迫！你的情报员会给你一个长度为 n 的 循环 数组 code 以及一个密钥 k 。

为了获得正确的密码，你需要替换掉每一个数字。所有数字会 同时 被替换。

如果 k > 0 ，将第 i 个数字用 接下来 k 个数字之和替换。
如果 k < 0 ，将第 i 个数字用 之前 k 个数字之和替换。
如果 k == 0 ，将第 i 个数字用 0 替换。
由于 code 是循环的， code[n-1] 下一个元素是 code[0] ，且 code[0] 前一个元素是 code[n-1] 。

给你 循环 数组 code 和整数密钥 k ，请你返回解密后的结果来拆除炸弹！

```python
class Solution:
    def decrypt(self, code: List[int], k: int) -> List[int]:
        # 这一题需要改变数组,需要一个复制件
        if k == 0:
            return [0 for i in range(len(code))]
        cp = code.copy()
        p = 0
        while p < len(code):
            temp = 0           
            if k > 0:
                u = p+1 #记录下标
                for i in range(k): #先考虑k大于0
                    temp += cp[u%len(code)] #只对复制件进行计算
                    u += 1
                code[p] = temp
                p += 1
            if k < 0:
                u = p-1 #记录下标
                for i in range(-k):
                    temp += cp[u%len(code)] #只对复制件进行计算
                    u -= 1
                code[p] = temp
                p += 1

        return code
```

# 1662. 检查两个字符串数组是否相等

给你两个字符串数组 word1 和 word2 。如果两个数组表示的字符串相同，返回 true ；否则，返回 false 。

数组表示的字符串 是由数组中的所有元素 按顺序 连接形成的字符串。

```python
class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        return ''.join(i for i in word1)==''.join(i for i in word2)
```

# 1669. 合并两个链表

给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。

请你将 list1 中第 a 个节点到第 b 个节点删除，并将list2 接在被删除节点的位置。

下图中蓝色边和节点展示了操作后的结果：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/28/fig1.png)

请你返回结果链表的头指针。

```python
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        # 根据图示需要找到四个节点
        # list1的a-1，b+1
        # list2的头节点和尾节点
        cur_a_minus_1 = list1
        cur_b_plus_1 = list1
        count = 0
        while count < a-1:
            cur_a_minus_1 = cur_a_minus_1.next
            count += 1
        count = 0
        while count < b+1:
            cur_b_plus_1 = cur_b_plus_1.next
            count += 1
        list2_tail = list2
        while list2_tail.next != None:
            list2_tail = list2_tail.next
        cur_a_minus_1.next = list2
        list2_tail.next = cur_b_plus_1
        return list1
```

# 1670. 设计前中后队列

请你设计一个队列，支持在前，中，后三个位置的 push 和 pop 操作。

请你完成 FrontMiddleBack 类：

FrontMiddleBack() 初始化队列。
void pushFront(int val) 将 val 添加到队列的 最前面 。
void pushMiddle(int val) 将 val 添加到队列的 正中间 。
void pushBack(int val) 将 val 添加到队里的 最后面 。
int popFront() 将 最前面 的元素从队列中删除并返回值，如果删除之前队列为空，那么返回 -1 。
int popMiddle() 将 正中间 的元素从队列中删除并返回值，如果删除之前队列为空，那么返回 -1 。
int popBack() 将 最后面 的元素从队列中删除并返回值，如果删除之前队列为空，那么返回 -1 。
请注意当有 两个 中间位置的时候，选择靠前面的位置进行操作。比方说：

将 6 添加到 [1, 2, 3, 4, 5] 的中间位置，结果数组为 [1, 2, 6, 3, 4, 5] 。
从 [1, 2, 3, 4, 5, 6] 的中间位置弹出元素，返回 3 ，数组变为 [1, 2, 4, 5, 6] 。

```python
class FrontMiddleBackQueue:
    # 类似双端队列的构建
    def __init__(self):
        self.queue = []

    def pushFront(self, val: int) -> None:
        self.queue = [val]+self.queue


    def pushMiddle(self, val: int) -> None:
        mid = len(self.queue)//2
        self.queue.insert(mid,val)


    def pushBack(self, val: int) -> None:
        self.queue = self.queue + [val]


    def popFront(self) -> int:
        if len(self.queue) != 0:
            return self.queue.pop(0)
        return -1


    def popMiddle(self) -> int:
        if len(self.queue) != 0:
            mid = (len(self.queue)-1)//2
            return self.queue.pop(mid)
        return -1


    def popBack(self) -> int:
        if len(self.queue) != 0:
            return self.queue.pop(-1)
        return -1



# Your FrontMiddleBackQueue object will be instantiated and called as such:
# obj = FrontMiddleBackQueue()
# obj.pushFront(val)
# obj.pushMiddle(val)
# obj.pushBack(val)
# param_4 = obj.popFront()
# param_5 = obj.popMiddle()
# param_6 = obj.popBack()
```

# 1672. 最富有客户的资产总量

给你一个 m x n 的整数网格 accounts ，其中 accounts[i][j] 是第 i 位客户在第 j 家银行托管的资产数量。返回最富有客户所拥有的 资产总量 。

客户的 资产总量 就是他们在各家银行托管的资产数量之和。最富有客户就是 资产总量 最大的客户。

```python
class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        return max(sum(i) for i in accounts)
```

# 1678. 设计 Goal 解析器

请你设计一个可以解释字符串 command 的 Goal 解析器 。command 由 "G"、"()" 和/或 "(al)" 按某种顺序组成。Goal 解析器会将 "G" 解释为字符串 "G"、"()" 解释为字符串 "o" ，"(al)" 解释为字符串 "al" 。然后，按原顺序将经解释得到的字符串连接成一个字符串。

给你字符串 command ，返回 Goal 解析器 对 command 的解释结果。

```python
class Solution:
    def interpret(self, command: str) -> str:
        p = 0
        ans = ''
        while p < len(command):
            if command[p] == 'G':
                ans = ans + 'G'
                p += 1
            elif command[p] == '(':
                if command[p+1] == ')':
                    ans += 'o'
                    p += 2
                else:
                    ans += 'al'
                    p += 4
        return ans
```

# 1679. K 和数对的最大数目

给你一个整数数组 nums 和一个整数 k 。

每一步操作中，你需要从数组中选出和为 k 的两个整数，并将它们移出数组。

返回你可以对数组执行的最大操作数。

```python
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort() # 先排序
        # 然后双指针，如果不满足，则往内走一步，如果满足，两指针都要走
        left = 0
        right = len(nums)-1
        count = 0 # 计数
        while left < right:
            if nums[left]+nums[right] == k:
                left += 1
                right -= 1
                count += 1
            elif nums[left]+nums[right] > k:
                right -= 1
            elif nums[left]+nums[right] < k:
                left += 1
        return count
```

# 1684. 统计一致字符串的数目

给你一个由不同字符组成的字符串 allowed 和一个字符串数组 words 。如果一个字符串的每一个字符都在 allowed 中，就称这个字符串是 一致字符串 。

请你返回 words 数组中 一致字符串 的数目。

```python
class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        allowed_dict = collections.defaultdict(int)
        count = 0
        for i in allowed:
            allowed_dict[i] = 1
        for i in words:
            temp_dict = collections.defaultdict(int)
            set1 = set(i)
            for char in set1:
                temp_dict[char] = 1
            ALLOWED = True
            for k in temp_dict:
                if k not in allowed_dict:
                    ALLOWED = False
                    break
            if ALLOWED:
                count += 1
        return count
```

# 1688. 比赛中的配对次数

给你一个整数 n ，表示比赛中的队伍数。比赛遵循一种独特的赛制：

如果当前队伍数是 偶数 ，那么每支队伍都会与另一支队伍配对。总共进行 n / 2 场比赛，且产生 n / 2 支队伍进入下一轮。
如果当前队伍数为 奇数 ，那么将会随机轮空并晋级一支队伍，其余的队伍配对。总共进行 (n - 1) / 2 场比赛，且产生 (n - 1) / 2 + 1 支队伍进入下一轮。
返回在比赛中进行的配对次数，直到决出获胜队伍为止。

```python
class Solution:
		# 模拟， 数学解法直接返回n-1
    def numberOfMatches(self, n: int) -> int:
        ans = 0
        while n != 1:
            if n%2 == 1:
                ans += (n-1)//2
                n = (n-1)//2 + 1
            else:
                ans += n/2
                n = n//2
        return int(ans)
```

# 1689. 十-二进制数的最少数目

如果一个十进制数字不含任何前导零，且每一位上的数字不是 0 就是 1 ，那么该数字就是一个 十-二进制数 。例如，101 和 1100 都是 十-二进制数，而 112 和 3001 不是。

给你一个表示十进制整数的字符串 n ，返回和为 n 的 十-二进制数 的最少数目。

```python
class Solution:
		# n是一个普通十进制数
    def minPartitions(self, n: str) -> int:
        # 本质是找字符串中最大位数的数值
        return int(max(list(n)))
```

# 1694. 重新格式化电话号码

给你一个字符串形式的电话号码 number 。number 由数字、空格 ' '、和破折号 '-' 组成。

请你按下述方式重新格式化电话号码。

首先，删除 所有的空格和破折号。
其次，将数组从左到右 每 3 个一组 分块，直到 剩下 4 个或更少数字。剩下的数字将按下述规定再分块：
2 个数字：单个含 2 个数字的块。
3 个数字：单个含 3 个数字的块。
4 个数字：两个分别含 2 个数字的块。
最后用破折号将这些块连接起来。注意，重新格式化过程中 不应该 生成仅含 1 个数字的块，并且 最多 生成两个含 2 个数字的块。

返回格式化后的电话号码。

```python
class Solution:
    def reformatNumber(self, number: str) -> str:
        number = list(number)
        while '-' in number: # 这里用while偷懒了，过长的字符串效率很低
            number.remove('-')
        while ' ' in number:
            number.remove(' ')
        result = ''
        while len(number) > 4:
            for i in number[0:3]:
                result += str(i) 
            result += '-'
            number = number[3:]
        if len(number) == 4:
            for i in number[0:2]:
                result += str(i) 
            result += '-'
            for i in number[2:4]:
                result += str(i)
        else:
            for i in number:
                result += str(i) 
        return result
```

# 1700. 无法吃午餐的学生数量

学校的自助午餐提供圆形和方形的三明治，分别用数字 0 和 1 表示。所有学生站在一个队列里，每个学生要么喜欢圆形的要么喜欢方形的。
餐厅里三明治的数量与学生的数量相同。所有三明治都放在一个 栈 里，每一轮：

如果队列最前面的学生 喜欢 栈顶的三明治，那么会 拿走它 并离开队列。
否则，这名学生会 放弃这个三明治 并回到队列的尾部。
这个过程会一直持续到队列里所有学生都不喜欢栈顶的三明治为止。

给你两个整数数组 students 和 sandwiches ，其中 sandwiches[i] 是栈里面第 i 个三明治的类型（i = 0 是栈的顶部）， students[j] 是初始队列里第 j 名学生对三明治的喜好（j = 0 是队列的最开始位置）。请你返回无法吃午餐的学生数量。

```python
class Solution:
    def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
        lenth = len(students)
        times = 0
        while times != len(students): #终止条件，是否学生已经循环一轮
            if students[0] == sandwiches[0]:
                students.pop(0)
                sandwiches.pop(0)
                times = 0
            elif students != sandwiches[0]:
                students.append(students.pop(0))
                times += 1
        return len(students)
```

# 1704. 判断字符串的两半是否相似

给你一个偶数长度的字符串 s 。将其拆分成长度相同的两半，前一半为 a ，后一半为 b 。

两个字符串 相似 的前提是它们都含有相同数目的元音（'a'，'e'，'i'，'o'，'u'，'A'，'E'，'I'，'O'，'U'）。注意，s 可能同时含有大写和小写字母。

如果 a 和 b 相似，返回 true ；否则，返回 false 。

```python
class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        set1 = {'a','e','i','o','u','A','E','I','O','U'}
        x = s[:len(s)//2]
        y = s[len(s)//2:]
        count1 = 0
        count2 = 0
        for i in x:
            if i in set1:
                count1 += 1
        for i in y:
            if i in set1:
                count2 += 1

        return count1==count2
```

# 1710. 卡车上的最大单元数

请你将一些箱子装在 一辆卡车 上。给你一个二维数组 boxTypes ，其中 boxTypes[i] = [numberOfBoxesi, numberOfUnitsPerBoxi] ：

numberOfBoxesi 是类型 i 的箱子的数量。
numberOfUnitsPerBoxi 是类型 i 每个箱子可以装载的单元数量。
整数 truckSize 表示卡车上可以装载 箱子 的 最大数量 。只要箱子数量不超过 truckSize ，你就可以选择任意箱子装到卡车上。

返回卡车可以装载 单元 的 最大 总数。

```python
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        # 排序贪心
        boxTypes.sort(key = lambda x:x[1],reverse = True)
        v = 0 # 初始化可以装载的数量
        p = 0 # 初始化指针
        while truckSize >= 0 and p < len(boxTypes):
            if truckSize - boxTypes[p][0] >= 0:
                truckSize -= boxTypes[p][0]
                v += boxTypes[p][0]*boxTypes[p][1]
            else:
                break
            p += 1
        if truckSize > 0 and p <len(boxTypes): #没有分配完
            v += boxTypes[p][1] * truckSize
        return v
```

# 1711. 大餐计数

大餐 是指 恰好包含两道不同餐品 的一餐，其美味程度之和等于 2 的幂。

你可以搭配 任意 两道餐品做一顿大餐。

给你一个整数数组 deliciousness ，其中 deliciousness[i] 是第 i 道餐品的美味程度，返回你可以用数组中的餐品做出的不同 大餐 的数量。结果需要对 10^9 + 7 取余。

注意，只要餐品下标不同，就可以认为是不同的餐品，即便它们的美味程度相同。

```python
class Solution:
    def countPairs(self, deliciousness: List[int]) -> int:
        # 一种进行21次方次的两数之和非排序解法，
        dict1 = Counter(deliciousness) # 建立 数字-频率 字典
        # 先字典中不相等的数字计数，次数为 频率*频率，
        max_num = max(deliciousness)
        target = [2**i for i in range(0,22)] # 因为题目给出的范围是单个菜在2**20次方内，注意包括（1，0）这样的数对
        count = 0 # 计数   
        # 如果选中的数字本身就是2的次方，那么次数为组合数cn2 
        # 注意数据中有0
        for t in target: # 遍历目标值
            for i in dict1:
                if t-i in dict1: # 两数之和的hash
                    if i == t-i: # 同值
                        count = count + (dict1[i]*(dict1[t-i]-1))/2
                    elif i != t-i: # 不同值
                        count = count + (dict1[i]*dict1[t-i])/2
        return int(count)%(10**9+7)

```

# 1716. 计算力扣银行的钱

Hercy 想要为购买第一辆车存钱。他 每天 都往力扣银行里存钱。

最开始，他在周一的时候存入 1 块钱。从周二到周日，他每天都比前一天多存入 1 块钱。在接下来每一个周一，他都会比 前一个周一 多存入 1 块钱。

给你 n ，请你返回在第 n 天结束的时候他在力扣银行总共存了多少块钱。

```python
class Solution:
    def totalMoney(self, n: int) -> int:
        # 处理成 已经xx周 xx 天形式
        weeks = n//7
        days = n%7
        weeks_money = (28+(28+7*weeks-7))*weeks//2
        days_money = days*weeks+(1+days)*days//2
        return weeks_money+days_money
```

# 1721. 交换链表中的节点

给你链表的头节点 `head` 和一个整数 `k` 。

**交换** 链表正数第 `k` 个节点和倒数第 `k` 个节点的值后，返回链表的头节点（链表 **从 1 开始索引**）。

```python
class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        # 遍历一遍记录size 
        # 本方法只交换节点值而不是交换节点,只需要找到两个节点
        # 交换节点的方法太复杂了需要找六个相关节点
        size = 0
        cur = head
        while cur != None:
            size += 1
            cur = cur.next
        if size == 1:
            return head
        aim = k
        p = 1
        cur = head
        while p != aim:
            cur = cur.next
            p += 1
        cur1 = cur
        aim = size -k + 1
        p = 1
        cur = head
        while p != aim:
            cur = cur.next
            p += 1
        cur2 = cur
        # 交换节点值
        cur1.val,cur2.val = cur2.val,cur1.val
        return head
```

# 1725. 可以形成最大正方形的矩形数目

给你一个数组 rectangles ，其中 rectangles[i] = [li, wi] 表示第 i 个矩形的长度为 li 、宽度为 wi 。

如果存在 k 同时满足 k <= li 和 k <= wi ，就可以将第 i 个矩形切成边长为 k 的正方形。例如，矩形 [4,6] 可以切成边长最大为 4 的正方形。

设 maxLen 为可以从矩形数组 rectangles 切分得到的 最大正方形 的边长。

请你统计有多少个矩形能够切出边长为 maxLen 的正方形，并返回矩形 数目 。

```python
class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        lenth_lst = [min(i) for i in rectangles]
        return lenth_lst.count(max(lenth_lst))
```

# 1732. 找到最高海拔

有一个自行车手打算进行一场公路骑行，这条路线总共由 n + 1 个不同海拔的点组成。自行车手从海拔为 0 的点 0 开始骑行。

给你一个长度为 n 的整数数组 gain ，其中 gain[i] 是点 i 和点 i + 1 的 净海拔高度差（0 <= i < n）。请你返回 最高点的海拔 。

```python
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        lst = [0]
        high = 0
        while len(gain) != 0:
            high += gain.pop(0)
            lst.append(high)
        max1 = lst[0]
        return max(lst)
```

# 1748. 唯一元素的和

给你一个整数数组 `nums` 。数组中唯一元素是那些只出现 **恰好一次** 的元素。

请你返回 `nums` 中唯一元素的 **和** 。

```python
class Solution:
    def sumOfUnique(self, nums: List[int]) -> int:
        dict1 = {i:nums.count(i) for i in nums}
        sum1 = 0
        for i in dict1:
            if dict1[i] == 1:
                sum1 += i
        return sum1
```

# 1752. 检查数组是否经排序和轮转得到

给你一个数组 nums 。nums 的源数组中，所有元素与 nums 相同，但按非递减顺序排列。

如果 nums 能够由源数组轮转若干位置（包括 0 个位置）得到，则返回 true ；否则，返回 false 。

源数组中可能存在 重复项 。

注意：我们称数组 A 在轮转 x 个位置后得到长度相同的数组 B ，当它们满足 A[i] == B[(i+x) % A.length] ，其中 % 为取余运算。

```python
class Solution:
    def check(self, nums: List[int]) -> bool:
        template = sorted(nums)
        nums = nums + nums
        p = 0
        i = 0
        count = 0
        while p < len(nums):
            if nums[p] == template[i]:
                p += 1
                i += 1
                count += 1
                if count == len(template):
                    return True
            elif nums[p] != template[i]:
                p += 1
                count = 0
                i = 0
        return False
```

# 1758. 生成交替二进制字符串的最少操作数

给你一个仅由字符 '0' 和 '1' 组成的字符串 s 。一步操作中，你可以将任一 '0' 变成 '1' ，或者将 '1' 变成 '0' 。

交替字符串 定义为：如果字符串中不存在相邻两个字符相等的情况，那么该字符串就是交替字符串。例如，字符串 "010" 是交替字符串，而字符串 "0100" 不是。

返回使 s 变成 交替字符串 所需的 最少 操作数。

```python
class Solution:
    def minOperations(self, s: str) -> int:
        # 字符串最终要么奇数位全为1，偶数位全为0。情况1
        # 要么奇数位全为0，偶数位全为1。情况2
        # 统计s变成情况1需要的次数，统计s变成情况2需要的次数，取较小的那一个
        p = 0
        count1 = 0
        while p < len(s): # 情况1,偶数位为0，奇数位位1.
            if p%2 == 0:
                if s[p] == '0': 
                    pass
                else:
                    count1 += 1
            elif p%2 == 1:
                if s[p] == '1':
                    pass
                else:
                    count1 += 1
            p += 1
        p = 0
        count2 = 0
        while p < len(s): # 情况2:偶数位为1，奇数位位0
            if p%2 == 0:
                if s[p] == '1':
                    pass
                else:
                    count2 += 1
            elif p%2 == 1:
                if s[p] == '0':
                    pass
                else:
                    count2 += 1
            p += 1
        return min(count1,count2)
```

# 1768. 交替合并字符串

给你两个字符串 word1 和 word2 。请你从 word1 开始，通过交替添加字母来合并字符串。如果一个字符串比另一个字符串长，就将多出来的字母追加到合并后字符串的末尾。

返回 合并后的字符串 。

```python
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        p = 0
        result = ''
        while p < len(word1) and p < len(word2):
            result += word1[p]
            result += word2[p]
            p += 1
        if len(word1) == len(word2):
            pass
        elif len(word1) > len(word2):
            result += word1[p:]
        else:
            result += word2[p:]
        return result
```

# 1773. 统计匹配检索规则的物品数量

给你一个数组 items ，其中 items[i] = [typei, colori, namei] ，描述第 i 件物品的类型、颜色以及名称。

另给你一条由两个字符串 ruleKey 和 ruleValue 表示的检索规则。

如果第 i 件物品能满足下述条件之一，则认为该物品与给定的检索规则 匹配 ：

ruleKey == "type" 且 ruleValue == typei 。
ruleKey == "color" 且 ruleValue == colori 。
ruleKey == "name" 且 ruleValue == namei 。
统计并返回 匹配检索规则的物品数量 。

```python
class Solution:
    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        count = 0
        if ruleKey == 'type':
            for i in items:
                if i[0] == ruleValue:
                    count += 1
        elif ruleKey == 'color':
            for i in items:
                if i[1] == ruleValue:
                    count += 1
        elif ruleKey == 'name':
            for i in items:
                if i[2] == ruleValue:
                    count += 1
        return count
```

# 1780. 判断一个数字是否可以表示成三的幂的和

给你一个整数 n ，如果你可以将 n 表示成若干个不同的三的幂之和，请你返回 true ，否则请返回 false 。

对于一个整数 y ，如果存在整数 x 满足 y == 3x ，我们称这个整数 y 是三的幂。

```python
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        # 将数字转化成标准3进制之后，检查每一位是否为0或者1 【不能大于1】
        def change(n:int):
            ans = ''
            while n != 0:
                ans += str(n%3)
                n = n//3
            return ans # 这个结果是倒序的三进制表示数，对这一题来说不需要转换
        ans = change(n)
        for i in ans:
            if int(i) > 1:
                return False # 筛出来其中有大于1的就返回
        return True
```

# 1784. 检查二进制字符串字段

给你一个二进制字符串 s ，该字符串 不含前导零 。

如果 s 最多包含 一个由连续的 '1' 组成的字段 ，返回 true 。否则，返回 false 。

```python
class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        index_lst = [] # 记录下1的索引
        for i in range(len(s)):
            if s[i] == '1':
                index_lst.append(i)
        gap_lst = []
        if len(index_lst) == 1: return True # 只有一个1,肯定是True
        p = 1
        while p < len(index_lst): #给索引作差
            gap_lst.append(index_lst[p]-index_lst[p-1])
            p += 1 
        for i in gap_lst: # 如果所有差都是1，则True
            if i != 1:
                return False
        return True
```

# 1790. 仅执行一次字符串交换能否使两个字符串相等

给你长度相等的两个字符串 s1 和 s2 。一次 字符串交换 操作的步骤如下：选出某个字符串中的两个下标（不必不同），并交换这两个下标所对应的字符。

如果对 其中一个字符串 执行 最多一次字符串交换 就可以使两个字符串相等，返回 true ；否则，返回 false 。

```python
class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        # 建立俩字典，看是否字符数量相等
        dict1 = defaultdict(int)
        for i in s1:
            dict1[i] += 1
        dict2 = defaultdict(int)
        for i in s2:
            dict2[i] += 1
        if dict1 != dict2:
            return False
        # 然后对应位置对比
        p = 0
        count = 0
        while p < len(s1):
            if s1[p] != s2[p]:
                count += 1
            p += 1
        return count <= 2

```

# 1791. 找出星型图的中心节点

有一个无向的 星型 图，由 n 个编号从 1 到 n 的节点组成。星型图有一个 中心 节点，并且恰有 n - 1 条边将中心节点与其他每个节点连接起来。

给你一个二维整数数组 edges ，其中 edges[i] = [ui, vi] 表示在节点 ui 和 vi 之间存在一条边。请你找出并返回 edges 所表示星型图的中心节点。

```python
class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        # 找出前两条边的公共节点
        lst = edges[0]+edges[1]
        dict1 = collections.defaultdict(int)
        for i in lst:
            dict1[i] += 1
        for i in dict1:
            if dict1[i] == 2:
                return i
            
```

# 1800. 最大升序子数组和

给你一个正整数组成的数组 nums ，返回 nums 中一个 升序 子数组的最大可能元素和。

子数组是数组中的一个连续数字序列。

已知子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，若对所有 i（l <= i < r），numsi < numsi+1 都成立，则称这一子数组为 升序 子数组。注意，大小为 1 的子数组也视作 升序 子数组。

```python
class Solution:
    def maxAscendingSum(self, nums: List[int]) -> int:
        # 一轮扫描+贪心
        # 如果前面这个数小于后面这个数，则sum继续累加
        # 如果前面这个数大于等于后面这个数，则sum重置
        # 为了同一语法，给nums最后加上一个-1值
        nums = nums + [-1]
        max_sum = 0
        temp_sum = 0
        p = 0
        while p < len(nums)-1:
            if nums[p] < nums[p+1]:
                temp_sum += nums[p]
                p += 1
            elif nums[p] >= nums[p+1]:
                temp_sum += nums[p]
                max_sum = max(max_sum,temp_sum)
                temp_sum = 0 # 重置
                p += 1
        return max_sum
```

# 1805. 字符串中不同整数的数目

给你一个字符串 word ，该字符串由数字和小写英文字母组成。

请你用空格替换每个不是数字的字符。例如，"a123bc34d8ef34" 将会变成 " 123  34 8  34" 。注意，剩下的这些整数为（相邻彼此至少有一个空格隔开）："123"、"34"、"8" 和 "34" 。

返回对 word 完成替换后形成的 不同 整数的数目。

只有当两个整数的 不含前导零 的十进制表示不同， 才认为这两个整数也不同。

```python
class Solution:
    def numDifferentIntegers(self, word: str) -> int:
        word = list(word)
        p = 0
        while p < len(word):
            if word[p].isalpha():
                word[p] = ' '
            p += 1
        # 此时要把单个数字字符拼接起来
        p = 0
        new_word_lst = []
        temp  = ''
        for i in word:
            if i != '':
                temp += i
            elif i == '':
                if temp != '':
                    new_word_lst.append(temp)
                temp = ''
        word = temp.split(' ')
        hashmap = collections.Counter(int(i) for i in word if i != '')
        value = 0
        for i in hashmap.items():
            value += 1
        return value
```

# 1812. 判断国际象棋棋盘中一个格子的颜色

给你一个坐标 `coordinates` ，它是一个字符串，表示国际象棋棋盘中一个格子的坐标。下图是国际象棋棋盘示意图。

如果所给格子的颜色是白色，请你返回 true，如果是黑色，请返回 false 。

给定坐标一定代表国际象棋棋盘上一个存在的格子。坐标第一个字符是字母，第二个字符是数字。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/04/03/chessboard.png)

```python
class Solution:
    def squareIsWhite(self, coordinates: str) -> bool:
        dict1 = {'a':1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8}
        return (dict1[coordinates[0]]+int(coordinates[1]))%2 == 1
```

# 1816. 截断句子

句子 是一个单词列表，列表中的单词之间用单个空格隔开，且不存在前导或尾随空格。每个单词仅由大小写英文字母组成（不含标点符号）。

例如，"Hello World"、"HELLO" 和 "hello world hello world" 都是句子。
给你一个句子 s 和一个整数 k ，请你将 s 截断 ，使截断后的句子仅含 前 k 个单词。返回 截断 s 后得到的句子。

```python
class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        s = s.split(' ')
        result = ''
        p = 0
        while p < k:
            result = result + s[p] + ' '
            p += 1
        return result[:-1]
```

# 1822. 数组元素积的符号

已知函数 signFunc(x) 将会根据 x 的正负返回特定值：

如果 x 是正数，返回 1 。
如果 x 是负数，返回 -1 。
如果 x 是等于 0 ，返回 0 。
给你一个整数数组 nums 。令 product 为数组 nums 中所有元素值的乘积。

返回 signFunc(product) 。

```python
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        ans = 1
        for i in nums:
            if i < 0:
                ans *= -1
            elif i ==0 :
                return 0
        return ans
```

# 1827. 最少操作使数组递增

给你一个整数数组 nums （下标从 0 开始）。每一次操作中，你可以选择数组中一个元素，并将它增加 1 。

比方说，如果 nums = [1,2,3] ，你可以选择增加 nums[1] 得到 nums = [1,3,3] 。
请你返回使 nums 严格递增 的 最少 操作次数。

我们称数组 nums 是 严格递增的 ，当它满足对于所有的 0 <= i < nums.length - 1 都有 nums[i] < nums[i+1] 。一个长度为 1 的数组是严格递增的一种特殊情况。

```python
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        p = 1
        count = 0
        while p < len(nums):
            if nums[p] <= nums[p-1]:
                gap = nums[p-1]-nums[p]+1
                nums[p] = nums[p-1]+1
                count += gap
            p += 1
        return count
```

# 1828. 统计一个圆中点的数目

给你一个数组 points ，其中 points[i] = [xi, yi] ，表示第 i 个点在二维平面上的坐标。多个点可能会有 相同 的坐标。

同时给你一个数组 queries ，其中 queries[j] = [xj, yj, rj] ，表示一个圆心在 (xj, yj) 且半径为 rj 的圆。

对于每一个查询 queries[j] ，计算在第 j 个圆 内 点的数目。如果一个点在圆的 边界上 ，我们同样认为它在圆 内 。

请你返回一个数组 answer ，其中 answer[j]是第 j 个查询的答案。

```python
class Solution:
    def countPoints(self, points: List[List[int]], queries: List[List[int]]) -> List[int]:
        # 利用点到圆形的距离是否小于半价来判断
        ans = []
        while len(queries) != 0:
            count = 0
            for i in points:
                if sqrt((i[0]-queries[0][0])**2+(i[1]-queries[0][1])**2) <= queries[0][2]:
                    count += 1
            ans.append(count)
            queries.pop(0)
        return ans
```

# 1832. 判断句子是否为全字母句

全字母句 指包含英语字母表中每个字母至少一次的句子。

给你一个仅由小写英文字母组成的字符串 sentence ，请你判断 sentence 是否为 全字母句 。

如果是，返回 true ；否则，返回 false 。

```python
class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        return len(set(sentence)) == 26
```

# 1833. 雪糕的最大数量

夏日炎炎，小男孩 Tony 想买一些雪糕消消暑。

商店中新到 n 支雪糕，用长度为 n 的数组 costs 表示雪糕的定价，其中 costs[i] 表示第 i 支雪糕的现金价格。Tony 一共有 coins 现金可以用于消费，他想要买尽可能多的雪糕。

给你价格数组 costs 和现金量 coins ，请你计算并返回 Tony 用 coins 现金能够买到的雪糕的 最大数量 。

注意：Tony 可以按任意顺序购买雪糕。

```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        # 这个并不是背包问题，不需要动态规划
        # 一个简单的排序之后贪心即可
        # 难度在于不动用函数内置api排序的情况下手写一个快排 这里手写快排如果pivot不随机则超时
        count = 0
        costs.sort()
        while len(costs) != 0:
            if coins >= costs[0]:
                count += 1
                coins -= costs[0]
                costs.pop(0)
            else:
                break 
        return count   

    def quick_sort(self,lst:List):
        if len(lst) < 2:
            return
        num = random.randint(0,len(lst)-1)
        pivot = lst[num]
        Less = []
        Equal = []
        Greater = []
        while len(lst) != 0:
            if lst[0] < pivot:
                Less.append(lst.pop(0))
            elif lst[0] > pivot:
                Greater.append(lst.pop(0))
            else:
                Equal.append(lst.pop(0))
        self.quick_sort(Less)
        self.quick_sort(Greater)
        while len(Less) != 0:
            lst.append(Less.pop(0))
        while len(Equal) != 0:
            lst.append(Equal.pop(0))
        while len(Greater) != 0:
            lst.append(Greater.pop(0))
```

# 1837. K 进制表示下的各位数字总和

给你一个整数 n（10 进制）和一个基数 k ，请你将 n 从 10 进制表示转换为 k 进制表示，计算并返回转换后各位数字的 总和 。

转换后，各位数字应当视作是 10 进制数字，且它们的总和也应当按 10 进制表示返回

```python
class Solution:
    def sumBase(self, n: int, k: int) -> int:
        #先写个进制转换
        #再求各位和即可
        sum1 = 0
        while n // k > 0:
            sum1 += n % k
            n //= k
        sum1 += n % k
        return sum1
```

# 1838. 最高频元素的频数

元素的 频数 是该元素在一个数组中出现的次数。

给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。

执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数 。

```python
class Solution(object):
    def maxFrequency(self, nums, k):
        # 滑动窗口法（280ms，92%）
        # 如果k >= nums[q]*(q+1-p)-sum(nums[p:q+1])
        # 则maxfreq = max(maxfreq,q+1-p)，q+1
        # 如果k < nums[q]*(q+1-p)-sum(nums[p:q+1])
        # 则p+1，q+1
        if len(nums) == 1: return 1
        nums.sort()
        l, r, maxfreq = 0, 0, 1
        cursum = 0 # 初始化
        while r < len(nums):
            cursum += nums[r] # 要加入的元素，作为基数
            if k >= nums[r]*(r+1-l)-cursum: # 计算最右边的数在窗口大小内是否耗尽了k
                maxfreq = max(maxfreq, r+1-l) 
            else: # 如果不满足，则把基数丢弃一个，左移窗口
                cursum -= nums[l]
                l += 1
            r += 1
        return maxfreq


```

# 1844. 将所有数字用字符替换

给你一个下标从 0 开始的字符串 s ，它的 偶数 下标处为小写英文字母，奇数 下标处为数字。

定义一个函数 shift(c, x) ，其中 c 是一个字符且 x 是一个数字，函数返回字母表中 c 后面第 x 个字符。

比方说，shift('a', 5) = 'f' 和 shift('x', 0) = 'x' 。
对于每个 奇数 下标 i ，你需要将数字 s[i] 用 shift(s[i-1], s[i]) 替换。

请你替换所有数字以后，将字符串 s 返回。题目 保证 shift(s[i-1], s[i]) 不会超过 'z' 。

```python
class Solution:
    def replaceDigits(self, s: str) -> str:
        i = 1
        s = list(s)
        while i < len(s):
            s[i] = self.shift(s[i-1], s[i])
            i += 2
        ans = ''.join(s)
        return ans

    
    def shift(self,element:str,num):
        element = chr(ord(element)+int(num))
        return element
```

# 1846. 减小和重新排列数组后的最大元素

给你一个正整数数组 arr 。请你对 arr 执行一些操作（也可以不进行任何操作），使得数组满足以下条件：

arr 中 第一个 元素必须为 1 。
任意相邻两个元素的差的绝对值 小于等于 1 ，也就是说，对于任意的 1 <= i < arr.length （数组下标从 0 开始），都满足 abs(arr[i] - arr[i - 1]) <= 1 。abs(x) 为 x 的绝对值。
你可以执行以下 2 种操作任意次：

减小 arr 中任意元素的值，使其变为一个 更小的正整数 。
重新排列 arr 中的元素，你可以以任意顺序重新排列。
请你返回执行以上操作后，在满足前文所述的条件下，arr 中可能的 最大值 。

```python
class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        # 排序，扫描，贪心
        # 检查第一个值是否为1
        # 循环：检查下一个值，只允许下一个值和该值相等或者比该值大1
        # 若比该值大的值大于等于2，则赋值为该值+1，继续检查
        arr.sort()
        if len(arr) == 1:
            return 1
        arr[0] = 1 # 将1赋值给第一位
        p = 1
        while p < len(arr):
            if arr[p] - arr[p-1] <= 1:
                pass
            elif arr[p] - arr[p-1] >= 2:
                arr[p] = arr[p-1] + 1
            p += 1
        return arr[-1]

```

# 1859. 将句子排序

一个 句子 指的是一个序列的单词用单个空格连接起来，且开头和结尾没有任何空格。每个单词都只包含小写或大写英文字母。

我们可以给一个句子添加 从 1 开始的单词位置索引 ，并且将句子中所有单词 打乱顺序 。

比方说，句子 "This is a sentence" 可以被打乱顺序得到 "sentence4 a3 is2 This1" 或者 "is2 sentence4 This1 a3" 。
给你一个 打乱顺序 的句子 s ，它包含的单词不超过 9 个，请你重新构造并得到原本顺序的句子。

```python
class Solution:
    def sortSentence(self, s: str) -> str:
        s = s.split(' ') # 先分割
        s.sort(key=lambda x:x[-1])
        result = ''
        for i in s:
            result += i[:-1]+' '
        return result[:-1]
```

# 1860. 增长的内存泄露

给你两个整数 memory1 和 memory2 分别表示两个内存条剩余可用内存的位数。现在有一个程序每秒递增的速度消耗着内存。

在第 i 秒（秒数从 1 开始），有 i 位内存被分配到 剩余内存较多 的内存条（如果两者一样多，则分配到第一个内存条）。如果两者剩余内存都不足 i 位，那么程序将 意外退出 。

请你返回一个数组，包含 [crashTime, memory1crash, memory2crash] ，其中 crashTime是程序意外退出的时间（单位为秒）， memory1crash 和 memory2crash 分别是两个内存条最后剩余内存的位数。

```python
class Solution:
    def memLeak(self, memory1: int, memory2: int) -> List[int]:
        i = 1
        while i <= max(memory1,memory2):
            if memory1 >= memory2:
                memory1 -= i
            else:
                memory2 -= i
            i += 1
        return [i,memory1,memory2]
```

# 1869. 哪种连续子字符串更长

给你一个二进制字符串 s 。如果字符串中由 1 组成的 最长 连续子字符串 严格长于 由 0 组成的 最长 连续子字符串，返回 true ；否则，返回 false 。

例如，s = "110100010" 中，由 1 组成的最长连续子字符串的长度是 2 ，由 0 组成的最长连续子字符串的长度是 3 。
注意，如果字符串中不存在 0 ，此时认为由 0 组成的最长连续子字符串的长度是 0 。字符串中不存在 1 的情况也适用此规则。

```python
class Solution:
    def checkZeroOnes(self, s: str) -> bool:
        p = 0
        mark_0 = 0
        mark_1 = 0
        lenth_0 = []
        lenth_1 = []
        while p < len(s):
            if s[p] == '0':
                mark_0 += 1
                lenth_1.append(mark_1)
                mark_1 = 0
            else:
                mark_1 += 1
                lenth_0.append(mark_0)
                mark_0 = 0
            p += 1
        lenth_1.append(mark_1) #防止直到最后也没有切换而补充加入列表
        lenth_0.append(mark_0)
        return max(lenth_1) > max(lenth_0)
```

# 1876. 长度为三且各字符不同的子字符串

如果一个字符串不含有任何重复字符，我们称这个字符串为 好 字符串。

给你一个字符串 s ，请你返回 s 中长度为 3 的 好子字符串 的数量。

注意，如果相同的好子字符串出现多次，每一次都应该被记入答案之中。

子字符串 是一个字符串中连续的字符序列。

```python
class Solution:
    def countGoodSubstrings(self, s: str) -> int:
        # 这里的子字符串 是一个字符串中连续的字符序列。
        if len(s) <3:
            return 0
        count = 0
        p = 0
        while p < len(s)-2:
            temp = []
            temp.append(s[p])
            temp.append(s[p+1])
            temp.append(s[p+2])
            if len(set(temp)) == 3:
                count += 1
            p += 1
        return count
```

# 1877. 数组中最大数对和的最小值

一个数对 (a,b) 的 数对和 等于 a + b 。最大数对和 是一个数对数组中最大的 数对和 。

比方说，如果我们有数对 (1,5) ，(2,3) 和 (4,4)，最大数对和 为 max(1+5, 2+3, 4+4) = max(6, 5, 8) = 8 。
给你一个长度为 偶数 n 的数组 nums ，请你将 nums 中的元素分成 n / 2 个数对，使得：

nums 中每个元素 恰好 在 一个 数对中，且
最大数对和 的值 最小 。
请你在最优数对划分的方案下，返回最小的 最大数对和 。

```python
class Solution:
    def minPairSum(self, nums: List[int]) -> int:
        # 数学思路，分组方法为第k大配第k小
        nums.sort()
        # 对撞指针得到分配组
        group = []
        left = 0
        right = len(nums) - 1
        while left < right:
            group.append(nums[left]+nums[right])
            left += 1
            right -= 1
        return max(group)
```

# 1880. 检查某单词是否等于两单词之和

字母的 字母值 取决于字母在字母表中的位置，从 0 开始 计数。即，'a' -> 0、'b' -> 1、'c' -> 2，以此类推。

对某个由小写字母组成的字符串 s 而言，其 数值 就等于将 s 中每个字母的 字母值 按顺序 连接 并 转换 成对应整数。

例如，s = "acb" ，依次连接每个字母的字母值可以得到 "021" ，转换为整数得到 21 。
给你三个字符串 firstWord、secondWord 和 targetWord ，每个字符串都由从 'a' 到 'j' （含 'a' 和 'j' ）的小写英文字母组成。

如果 firstWord 和 secondWord 的 数值之和 等于 targetWord 的数值，返回 true ；否则，返回 false 。

```python
class Solution:
    def isSumEqual(self, firstWord: str, secondWord: str, targetWord: str) -> bool:
        alphabet = [chr(x) for x in range(97,97+10)]
        num = [str(x) for x in range(10)]
        dict1 = dict(zip(alphabet,num))

        def change(s:str):
            result = ''
            for i in s:
                result += dict1[i]
            return int(result)
        
        return change(firstWord)+change(secondWord) == change(targetWord)
```

# 1886. 判断矩阵经轮转后是否一致

给你两个大小为 n x n 的二进制矩阵 mat 和 target 。现 以 90 度顺时针轮转 矩阵 mat 中的元素 若干次 ，如果能够使 mat 与 target 一致，返回 true ；否则，返回 false 。

```python
class Solution:
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        # 先写出轮转函数，得到三个轮转值，判断轮转值是否和目标相同,为了偷懒转4次，可以统一语法，无需mat和target提前判断
        times = 4
        def rotate(mat):
            # 上下对调
            # 对角线对调
            n = len(mat)
            for i in range(n//2):
                mat[i],mat[n-i-1] = mat[n-i-1],mat[i]
            for i in range(n):
                for j in range(i,n):
                    mat[i][j],mat[j][i] = mat[j][i],mat[i][j]
            return mat
        for i in range(times):
            if target == rotate(mat):
                return True
        return False
```

# 1893. 检查是否区域内所有整数都被覆盖

给你一个二维整数数组 ranges 和两个整数 left 和 right 。每个 ranges[i] = [starti, endi] 表示一个从 starti 到 endi 的 闭区间 。

如果闭区间 [left, right] 内每个整数都被 ranges 中 至少一个 区间覆盖，那么请你返回 true ，否则返回 false 。

已知区间 ranges[i] = [starti, endi] ，如果整数 x 满足 starti <= x <= endi ，那么我们称整数x 被覆盖了。

```python
class Solution:
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        # 超级暴搜法
        rails = [] # 直接模拟其中有多少整数
        ranges.sort(key = lambda x:x[0])
        for i in ranges: 
            for num in range(i[0],i[1]+1):
                if rails == []:
                    rails.append(num)
                elif rails[-1] < num:
                    rails.append(num)
        # 然后看left在其中的索引，right在其中的索引，有一个越界了则gg
        if left < rails[0] or right > rails[-1]:
            return False
        # 在其中时，找索引
        mark1 = -1 # 标记索引l
        mark2 = -1 # 标记索引r
        for i in range(len(rails)):
            if rails[i] == left:
                mark1 = i
            if rails[i] == right:
                mark2 = i
        if mark1 == -1 or mark2 == -1: # 有一个数没找到
            return False
        if mark2 - mark1 + 1 == right - left + 1:
            return True
        return False # 筛完了都找不到

```

# 1894. 找到需要补充粉笔的学生编号

一个班级里有 n 个学生，编号为 0 到 n - 1 。每个学生会依次回答问题，编号为 0 的学生先回答，然后是编号为 1 的学生，以此类推，直到编号为 n - 1 的学生，然后老师会重复这个过程，重新从编号为 0 的学生开始回答问题。

给你一个长度为 n 且下标从 0 开始的整数数组 chalk 和一个整数 k 。一开始粉笔盒里总共有 k 支粉笔。当编号为 i 的学生回答问题时，他会消耗 chalk[i] 支粉笔。如果剩余粉笔数量 严格小于 chalk[i] ，那么学生 i 需要 补充 粉笔。

请你返回需要 补充 粉笔的学生 编号 。

```python
class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        # 带优化的直接模拟
        # 先求一轮sum，然后k模上这个sum
        # 再k递减求序号，终止条件为k<0
        sum_num = sum(chalk)
        k = k%sum_num
        p = 0
        while p < len(chalk):
            if k-chalk[p]<0:
                return p
            else:
                k -= chalk[p]
                p += 1
```

# 1897. 重新分配字符使所有字符串都相等

给你一个字符串数组 words（下标 从 0 开始 计数）。

在一步操作中，需先选出两个 不同 下标 i 和 j，其中 words[i] 是一个非空字符串，接着将 words[i] 中的 任一 字符移动到 words[j] 中的 任一 位置上。

如果执行任意步操作可以使 words 中的每个字符串都相等，返回 true ；否则，返回 false 。

```python
class Solution:
    def makeEqual(self, words: List[str]) -> bool:
        # 只需要确定是否可行
        # 遍历所有字符，加入计数字典，如果字典中所有值都可以整除组数，则True
        dict1 = collections.defaultdict(int)
        for i in words:
            for j in i:
                dict1[j] += 1
        for i in dict1:
            if dict1[i]%len(words)!= 0:
                return False
        return True
                
```

# 1903. 字符串中的最大奇数

给你一个字符串 num ，表示一个大整数。请你在字符串 num 的所有 非空子字符串 中找出 值最大的奇数 ，并以字符串形式返回。如果不存在奇数，则返回一个空字符串 "" 。

子字符串 是字符串中的一个连续的字符序列。

```python
class Solution:
    def largestOddNumber(self, num: str) -> str:
        # 数学思想，找到从右边往左数的第一个奇数，然后直接截取首位到此位即可
        p = -1
        while p > -len(num) - 1:
            if int(num[p])%2 == 0:
                p -= 1
            elif int(num[p])%2 == 1:
                break
        return num[:len(num)+p+1]
```

# 1909. 删除一个元素使数组严格递增

给你一个下标从 0 开始的整数数组 nums ，如果 恰好 删除 一个 元素后，数组 严格递增 ，那么请你返回 true ，否则返回 false 。如果数组本身已经是严格递增的，请你也返回 true 。

数组 nums 是 严格递增 的定义为：对于任意下标的 1 <= i < nums.length 都满足 nums[i - 1] < nums[i] 。

```python
class Solution:
    def canBeIncreasing(self, nums: List[int]) -> bool:
        p = 1
        # 利用差值来判断
        gap_lst = []
        while p < len(nums):
            gap_lst.append(nums[p]-nums[p-1])
            p += 1
        # 如果差值全部大于0，返回True
        # 如果差值有两个及以上的负数，返回False
        # 如果差值有一个负数，分类讨论
        p = 0
        times = 0 
        mark = -1
        while p < len(gap_lst):
            if gap_lst[p] > 0:
                p += 1
            elif gap_lst[p] <= 0:
                times += 1
                mark = p
                p += 1
        if times == 0:
            return True
        elif times >= 2:
            return False
        elif mark == len(gap_lst)-1: return True # 如果是最后一位为负数，直接True
        elif mark == 0 : return True # 第一位为负数，直接True
        elif times == 1: # 有两种可能，一种是删除负数位置前的，一种是删除负数位置后的，
            # 回归nums，要删除的是mark，如果nums[mark-1] < nums[mark+1]: return True
            # 回归nums，要删除的是mark+1，如果nums[mark] < nums[mark+2]: return True
            if nums[mark-1] < nums[mark+1]: return True
            elif nums[mark] < nums[mark+2]: return True
            else:return False 

```

# 1920. 基于排列构建数组

给你一个 从 0 开始的排列 nums（下标也从 0 开始）。请你构建一个 同样长度 的数组 ans ，其中，对于每个 i（0 <= i < nums.length），都满足 ans[i] = nums[nums[i]] 。返回构建好的数组 ans 。

从 0 开始的排列 nums 是一个由 0 到 nums.length - 1（0 和 nums.length - 1 也包含在内）的不同整数组成的数组。

```python
class Solution:
    def buildArray(self, nums: List[int]) -> List[int]:
        ans = [[] for i in range(len(nums))]
        for i in range(len(ans)):
            ans[i] = nums[nums[i]]
        return ans #尬模拟就行
```

# 1929. 数组串联

给你一个长度为 n 的整数数组 nums 。请你构建一个长度为 2n 的答案数组 ans ，数组下标 从 0 开始计数 ，对于所有 0 <= i < n 的 i ，满足下述所有要求：

ans[i] == nums[i]
ans[i + n] == nums[i]
具体而言，ans 由两个 nums 数组 串联 形成。

返回数组 ans 。

```python
class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        # 送分题
        return nums+nums
```


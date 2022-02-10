# 《新剑指offer》

# 剑指 Offer II 001. 整数除法

给定两个整数 a 和 b ，求它们的除法的商 a/b ，要求不得使用乘号 '*'、除号 '/' 以及求余符号 '%' 。

注意：

整数除法的结果应当截去（truncate）其小数部分，例如struncate(8.345) = 8 以及 truncate(-2.7335) = -2
假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231, 231−1]。本题中，如果除法结果溢出，则返回 231 − 1

```go
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

# 剑指 Offer II 002. 二进制加法

给定两个 01 字符串 `a` 和 `b` ，请计算它们的和，并以二进制字符串的形式输出。

输入为 **非空** 字符串且只包含数字 `1` 和 `0`。

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        # 字符串的存储从左到右高位到低位，为了方便运算，将其倒置
        # 不倒置也可以从右往左加
        a = a[::-1]
        b = b[::-1]
        car = 0 # 默认进位为0
        p = 0 
        lst = [] # 使用列表接收临时答案
        while p < len(a) and p < len(b):
            val = int(car) + int(a[p]) + int(b[p])
            car = val // 2 # 进位位
            val = val % 2 # 值位
            lst.append(str(val))
            p += 1
        if p == len(a): # 可能a扫完了，b没有扫完
            while p < len(b):
                val = int(car) + int(b[p])
                car = val // 2 # 进位位
                val = val % 2 # 值位
                lst.append(str(val))
                p += 1
        elif p == len(b): # 可能b扫完了，a没有扫完
            while p < len(a):
                val = int(car) + int(a[p])
                car = val // 2 # 进位位
                val = val % 2 # 值位
                lst.append(str(val))
                p += 1
        if car == 1: # 如果有进位位遗留，将它加入
            lst.append(str(car))

        return ("".join(lst)[::-1]) # 最终结果需要倒序
```

# 剑指 Offer II 003. 前 n 个数字二进制中 1 的个数

给定一个非负整数 `n` ，请计算 `0` 到 `n` 之间的每个数字的二进制表示中 1 的个数，并输出一个数组。

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        # 方法1: 直接求，注意python中位的特性
        ans = []
        for i in range(n+1):
            ans.append(self.every_nums_bit(i))
        return ans
    
    def every_nums_bit(self,i):
        i = bin(i)[2:] # python会转换为0b*****的字符串
        # 然后返回各个数位的相加总和即可
        temp_sum = 0
        for bit in i:
            temp_sum += int(bit)
        return temp_sum

```

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        # 方法2: 动态规划
        # 源于 i&(i-1)可以把i的二进制中最右边的1变成0。
        # 即：i的二进制比i&(i-1)的二进制多1. 且i一定大于(i&(i-1)),所以只需要递增方向遍历即可
        dp = [0 for i in range(n+1)]
        for i in range(1,n+1):
            dp[i] = dp[(i&(i-1))] + 1
        return dp

```

# 剑指 Offer II 004. 只出现一次的数字 

给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次 。**请你找出并返回那个只出现了一次的元素。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 只有一个元素出现一次，其他元素出现三次，
        # 不限制空间的话一轮扫描+哈希表存储即可获得答案
        # 当限制空间复杂度为常数级时
        # 这些元素都是数字，以二进制考虑，那么以位运算之后对每一位模3，剩下来的则是出现一次的数字
        # 注意考虑越界问题,还需要考虑python的负数的二进制表示方式
        every_bit = []
        for i in range(32):
            bit = sum([num>>i&1 for num in nums])
            bit = bit % 3
            every_bit.append(bit)
        # 此时不相同的那个数的每一位已经被记录，还原它
        # print(every_bit)，数组从左到右存储的是低位到高位
        # 实际上可以将此数的还原写在for循环里，这里为了清晰表示，未合并
        ans = 0
        for index,value in enumerate(every_bit):
            ans += value * 2 ** index 
        if ans > 2**31 - 1: # 由于第32位符号位被置1，转化成标准负数之后返回结果
            ans -= 0xffffffff
            ans -= 1
        return ans
```

# 剑指 Offer II 005. 单词长度的最大乘积

给定一个字符串数组 words，请计算当两个字符串 words[i] 和 words[j] 不包含相同字符时，它们长度的乘积的最大值。假设字符串中只包含英语的小写字母。如果没有不包含相同字符的一对字符串，返回 0。

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
                lst1 = the_dict[words[i]]
                lst2 = the_dict[words[j]]
                for k in range(26):
                    if lst1[k] == True and lst2[k] == True: # 两个同时存在
                        break # 有相同字符则开启下一轮检查
                    if k == 25 and (lst1[k] and lst2[k]) != True: # 检查到了最后一个，并且两个不同时为True
                        longest = max(longest,len(words[i])*len(words[j]))
        return longest
        
```

```python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        # 方法2: 基于int类型制作哈希表，将1～26位的对应2进制置1
        # 注意python的位运算机制
        the_dict = collections.defaultdict(int)
        for w in words:
            val = 0 # 默认所有位都不存在对应字母
            for ch in w: # 扫描单个单词
                off_set = ord(ch) - ord("a") 
                off_set_val = 1 << off_set # 确定单词所对应的位
                val = val|off_set_val # 注意这里的位运算用的是“或”，直接置1
            the_dict[w] = val # 扫描完之后把值存进哈希表
        longest = 0
        for i in range(len(words)):
            for j in range(i,len(words)):
                if the_dict[words[i]] & the_dict[words[j]] == 0: # 所有位不同，则与运算为0
                    longest = max(longest,len(words[i])*len(words[j]))
        return longest

```

# 剑指 Offer II 006. 排序数组中两个数字之和

给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。

函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 0 开始计数 ，所以答案数组应当满足 0 <= answer[0] < answer[1] < numbers.length 。

假设数组中存在且只存在一对符合条件的数字，同时一个数字不能使用两次。

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # 超经典的两数之和，双指针
        # 数组已排序，且保证有解
        left = 0
        right = len(numbers) - 1
        while left < right: # 数字不能使用两次，所以小于号
            if numbers[left] + numbers[right] == target:
                return [left,right] # 返回的是索引
            elif numbers[left] + numbers[right] < target:
                left += 1 # 和小了，那么小的数要变大，小指针右移动
            elif numbers[left] + numbers[right] > target:
                right -= 1 # 和大了，那么大的数要变小，大指针左移动
```

# 剑指 Offer II 007. 数组中和为 0 的三个数

给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a ，b ，c ，使得 a + b + c = 0 ？请找出所有和为 0 且 不重复 的三元组。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 实际上是两数之和加强版。定i之后，寻找另外两个值为j+k == -i
        # 小陷阱：注意去重复,想提升时间效率，则需要排序之后使用对撞指针
        nums.sort() # 预排序
        set_ans = set() # 集合去重
        list_ans = [] # 最终结果收集列表
        for i in range(len(nums)):
            target = -nums[i]
            left = i + 1
            right = len(nums) - 1
            while left < right:
                if target == nums[left] + nums[right]:
                    if (nums[i],nums[left],nums[right]) not in set_ans: # 去重复逻辑
                        set_ans.add((nums[i],nums[left],nums[right]))
                        list_ans.append([nums[i],nums[left],nums[right]])
                        left += 1 # 收集完之后不要忘记移动指针
                        right -= 1 # 收集完之后依旧需要移动指针
                    else:
                        left += 1
                elif nums[left] + nums[right] < target: # 数值和偏小，左指针右移
                    left += 1
                elif nums[left] + nums[right] > target: # 数值和偏小，右指针左移
                    right -= 1
        return list_ans
```

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 加强版两数之和
        # 目的是找j,k == -i
        nums.sort()
        ans = []
        for i in range(len(nums)):
            # 极其重要的去重逻辑
            if i >= 1 and nums[i-1] == nums[i]: # 如果这一个数和前一个相等，则直接跳过
                continue 
            target = -nums[i]
            left = i + 1
            right = len(nums) - 1
            while left < right:
                if nums[left] + nums[right] == target:
                    ans.append([nums[i],nums[left],nums[right]])
                    # 然后需要移动left指针直到不等于现在的left为止
                    # 移动right指针直到不等于现在的right为止
                    left += 1
                    right -= 1
                    while left < right and nums[left-1] == nums[left]:
                        left += 1
                    while right > left and nums[right+1] == nums[right]:
                        right -= 1 
                elif nums[left] + nums[right] < target: # 和偏小，left右移
                    left += 1
                elif nums[left] + nums[right] > target: # 和偏大，right左移 
                    right -= 1
        return ans
```

# 剑指 Offer II 008. 和大于等于 target 的最短子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # 子数组，采用滑动窗口解法
        window = 0 # 记录窗口内数值
        window_length = 0 # 初始化窗口长度
        left = 0
        right = 0
        min_length = 0xffffffff # 初始化一个不可能达到的窗口值，
        while right < len(nums):
            add_num = nums[right] # 记录要加入窗口的元素
            right += 1
            window += add_num
            window_length += 1
            if window >= target:
                min_length = min(min_length,window_length) # 如果符合条件 收集答案
            while left < right and window >= target: # 如果数字大于等于，则收缩窗口
                delete_num = nums[left]
                left += 1
                window -= delete_num
                window_length -= 1
                if window >= target:
                    min_length = min(min_length,window_length) # 如果符合条件 收集答案
        # 注意返回值，如果整个过程中值都没有被刷新过依旧是不合法的值，根据题意返回0
        return min_length if min_length != 0xffffffff else 0 

```

# 剑指 Offer II 009. 乘积小于 K 的子数组 

给定一个正整数数组 `nums`和整数 `k` ，请找出该数组内乘积小于 `k` 的连续的子数组的个数。

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

```java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1){
            return 0;
        }
        int left = 0;
        int right = 0;
        int window = 1;
        int ans = 0;
        int n = nums.length;
        while (right < n){
            int add_num = nums[right];
            window *= add_num;
            right += 1;
            while (left < right && window >= k){ // 收缩
                int delete_num = nums[left];
                window /= delete_num;
                left += 1;
            }
            ans += (right-1 - left + 1);
        }
        return ans;

    }
}
```



# 剑指 Offer II 010. 和为 k 的子数组

给定一个整数数组和一个整数 `k` **，**请找到该数组中和为 `k` 的连续子数组的个数。

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

# 剑指 Offer II 011. 0 和 1 个数相同的子数组

给定一个二进制数组 `nums` , 找到含有相同数量的 `0` 和 `1` 的最长连续子数组，并返回该子数组的长度。

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        # 把0看作-1，本题变化为和为0的最长子数组
        # 前缀和收集到目前序号之前到总和
        # 每次检查当前值是否在之前出现过，只需要记录当前值第一次出现的位置
        for i in range(len(nums)): # 预处理
            if nums[i] == 0:
                nums[i] = -1
        pre_dict = collections.defaultdict(int)
        pre_dict[0] = -1 # 注意这一行的运用
        temp_sum = 0
        longest = 0
        for i in range(len(nums)):
            temp_sum += nums[i]
            if temp_sum not in pre_dict:
                pre_dict[temp_sum] = i # 只需记录第一次出现的值
            elif temp_sum in pre_dict:
                longest = max(longest,i - pre_dict[temp_sum])
        return longest
```

```java
class Solution {
    public int findMaxLength(int[] nums) {
        // 把0当作-1后，这一题可以看作和为0的最长子数组
        // 先预处理
        for(int i = 0; i < nums.length; i += 1){
            if(nums[i] == 0) nums[i] = -1;
        }
        // 然后使用前缀和,记录下每个元素第一次出现的下标
        Map <Integer,Integer> pre_dict = new HashMap<Integer,Integer>();
        int temp_sum = 0;
        int longest = 0;
        pre_dict.put(0,-1); // 
        for(int i = 0; i < nums.length; i += 1){
            temp_sum += nums[i];
            if (pre_dict.get(temp_sum) == null)
                pre_dict.put(temp_sum,i);
            else if (pre_dict.get(temp_sum) != null)
                longest = Math.max(longest,i - pre_dict.get(temp_sum));
        }
        return longest;
    }
}
```

# 剑指 Offer II 012. 左右两边子数组的和相等

给你一个整数数组 nums ，请计算数组的 中心下标 。

数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。

如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。这一点对于中心下标位于数组最右端同样适用。

如果数组有多个中心下标，应该返回 最靠近左边 的那一个。如果数组不存在中心下标，返回 -1 。

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        # 前缀和思想,记录的是不包含本位置元素的前缀和
        temp_sum = 0
        prefix = [] # 构建前缀和数组，注意填充完毕之后，prefix长于nums，最后一位是总和
        for i in nums:
           prefix.append(temp_sum)
           temp_sum += i
        all_sum = temp_sum # 最后一位是总和，不必重复使用sum(nums)计算了
        # 该元素的左边的和为prefix[i],右边的和为all_sum - nums[i] - prefix[i]
        for i in range(0,len(nums)):
            if prefix[i] == all_sum - nums[i] - prefix[i]:
                return i # 找到则返回
        return -1 # 找不到返回-1

```

# 剑指 Offer II 013. 二维子矩阵的和

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
# 如果查询时候的左上角不是0，0，则返回:记忆矩阵的右下角-上长条-做长条+左上角
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



# 剑指 Offer II 014. 字符串中的变位词

给定两个字符串 `s1` 和 `s2`，写一个函数来判断 `s2` 是否包含 `s1` 的某个变位词。

换句话说，第一个字符串的排列之一是第二个字符串的 **子串** 。

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # s1不能长于s2，否则一定不是子串
        # 显然使用滑动窗口来做，而且这一个滑动窗口可以固定窗口大小，很方便
        if len(s1) > len(s2):
            return False
        # 将s1设置为模版，固定窗口大小之后进行比对,由于是只需要判断变位词，窗口内字符数相等即可
        template = collections.defaultdict(int)
        for i in s1:
            template[i] += 1
        window = collections.defaultdict(int)
        for i in s2[:len(s1)]:
            window[i] += 1 # 初始化窗口
        if template == window: # 如果初始化的窗口已经满足条件，则返回True
            return True
        # 否则开始滑动窗口
        left = 0
        right = 0 + len(s1) # 此时right是第一个将要被删除的字符
        while right < len(s2):
            add_char = s2[right] # 记录即将加入的字符
            right += 1
            window[add_char] += 1 # 加入到窗口中
            delete_char = s2[left]
            left += 1
            window[delete_char] -= 1
            if window[delete_char] == 0:
                del window[delete_char] # 为了防止有key:0这种键值对的存在无法判断模板是否与窗口相等
            if window == template:
                return True
        return False # 循环完之后都没有匹配上，则False


```

# 剑指 Offer II 015. 字符串中的所有变位词

给定两个字符串 s 和 p，找到 s 中所有 p 的 变位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

变位词 指字母相同，但排列不同的字符串。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 只需要收集起始位置索引，并且需要扫描完全部的值
        # 如果 s比p还短，直接返回空列表
        if len(s) < len(p):
            return []
        # 以p做模版，在s上滑动窗口
        template = collections.defaultdict(int)
        for i in p:
            template[i] += 1
        window = collections.defaultdict(int)
        for i in s[:len(p)]:
            window[i] += 1
        ans = [] # 收集结果
        if template == window:
            ans.append(0)
        left = 0 # left指向将要删除的第一个字符
        right = len(p) # right指向将要加入的第一个字符
        while right < len(s):
            add_char = s[right] # 记录将要加入的字符
            right += 1
            window[add_char] += 1
            delete_char = s[left] # 记录将要删除的字符
            left += 1
            window[delete_char] -= 1
            if window[delete_char] == 0: # 这一行很重要，防止键值为，k:0的值干扰判断
                del window[delete_char]
            if window == template: # 收集合理结果
                ans.append(left) # 只需要存储索引
        return ans 

```

# 剑指 Offer II 016. 不含重复字符的最长子字符串

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长连续子字符串** 的长度。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 滑动窗口，内容包含数字+字母+空格，使用字典记录窗口更好
        if len(s) == 0: # 处理空串
            return 0
        max_length = 0 # 初始化最大长度为0
        window_length = 0 # 其实可以用左右游标的差值代替，但是为了清楚表达，故对它进行直接表达
        window = collections.defaultdict(int)
        left = 0 
        right = 0 # 初始化为首个要进入窗口的元素
        while right < len(s):
            add_char = s[right] # 记录要加入的元素
            right += 1
            window[add_char] += 1
            window_length += 1 # 提升窗口大小
            if len(window) == window_length: # 说明窗口内字符都不相同
                max_length = max(window_length,max_length) # 收集答案，需要的是更长的那个
            while left < right and len(window) < window_length: # 说明里面有重复字符
                delete_char = s[left]
                left += 1
                window[delete_char] -= 1
                window_length -= 1 # 减少窗口大小
                if window[delete_char] == 0: del window[delete_char] # 这一句话很重要
                if len(window) == window_length: # 再次检查，如果窗口内字符都不相同，其实可以不写，因为它不会被执行
                   max_length = max(window_length,max_length) # 收集答案，需要的是更长的那个
        return max_length

        
```

# 剑指 Offer II 017. 含有所有字符的最短字符串
给定两个字符串 s 和 t 。返回 s 中包含 t 的所有字符的最短子字符串。如果 s 中不存在符合条件的子字符串，则返回空字符串 "" 。

如果 s 中存在多个符合条件的子字符串，返回任意一个。

注意： 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s): # 长度越界，肯定不够
            return ""
        # 滑动窗口法
        left = 0
        right = 0
        valid = 0 # 初始化默认合法字符数目为0
        template = collections.defaultdict(int)
        for i in t:
            template[i] += 1
        window = collections.defaultdict(int)
        length = len(s) + 1 # 初始化超过全长
        while right < len(s):
            add_char = s[right]
            if add_char in template:
                window[add_char] += 1
                if window[add_char] == template[add_char]: valid += 1 # 当字符数相等的时候
            right += 1          
            while left < right and valid == len(template):
                if right - left < length:
                    left_index = left # 存储起始位置
                    length = right - left # 存储长度
                delete_char = s[left]
                if delete_char in template:
                    window[delete_char] -= 1
                    if window[delete_char] < template[delete_char]: 
                        valid -= 1
                left += 1
        if length > len(s): # 如果始终没有过有效的收缩，则返回空串
            return ""
        return s[left_index:left_index+length] # 否则返回切片
```



# 剑指 Offer II 018. 有效的回文

给定一个字符串 `s` ，验证 `s` 是否是 **回文串** ，只考虑字母和数字字符，可以忽略字母的大小写。

本题中，将空字符串定义为有效的 **回文串** 。

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # 一种性能稍低的解决方案是，获取一个新字符串，新字符串是由原字符串格式化而成
        # 然后双指针对撞判断
        # 格式化的原则是，把所有字母变成小写之后加入，忽略所有空格和标点。需要扫两轮原字符串的大小
        # 使用ascii码作为判别
        new_string = []
        for i in s:
            if 48<=ord(i)<=57 or 97<=ord(i)<=122: # 数字和小写字母ASCII码
                new_string.append(i)
            elif 65<=ord(i)<=90: # 大写字母ASCII码
                new_string.append(i.lower())
        left = 0
        right = len(new_string) - 1
        while left < right:
            if new_string[left] != new_string[right]:
                return False
            left += 1
            right -= 1
        return True

        # 如果采取一轮扫描对撞，则需要每次判断字符是否都是数字或者都是字母，都是字母的情况下都格式化成小写再进行判断，性能稍高
```

# 剑指 Offer II 019. 最多删除一个字符得到回文

给定一个非空字符串 `s`，请判断如果 **最多** 从字符串中删除一个字符能否得到一个回文字符串。

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        # 有一次容错机会，在容错的时候由于不知道是哪边回文，那么进行两次判定
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            elif s[left] != s[right]:
                break               
        return left >= right or self.isPalindrome(s,left+1,right) or self.isPalindrome(s,left,right-1)
    
    def isPalindrome(self,s,p1,p2): # p1,p2是闭区间
        while p1 < p2:
            if s[p1] != s[p2]:
                break
            else:
                p1 += 1
                p2 -= 1
        return p1 >= p2        

```

# 剑指 Offer II 020. 回文子字符串的个数

给定一个字符串 `s` ，请计算这个字符串中有多少个回文子字符串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # 动态规划，dp[i][j]的含义是s[i:j+1]是否为回文【注意左闭右开】
        # 申请二维dp数组
        # 需要考虑的是包括主对角线右上角矩阵元素里有几个True
        dp = [[False for i in range(len(s))] for k in range(len(s))]
        # 显然在数组中对角线都是回文子串，填充 True
        count = 0 # 存储是否是回文串
        for i in range(len(s)):
            dp[i][i] = True
            count += 1
        # 状态转移为，如果需要dp[i][j]为回文串，那么需要掐头去尾是回文串且新加入的字符是相同的
        # 即dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
        # 画九宫格，发现dp[i][j]由左下方的状态确定，当只有一条主对角线时候，右平行第一条线无法状态转移
        # 那么直接判断
        for i in range(len(s)-1):
            dp[i][i+1] = (s[i] == s[i+1])
            if dp[i][i+1]:
                count += 1
        # 有了这两条线后可以开始状态转移了，所有需要填充的格子都有了状态转移的来源
        # 填充顺序为从左到右的纵列,一列列填充
        # 所以外循环是纵列，内循环是横行，画图辅助确定填充范围
        for j in range(2,len(s)): # j只需要从2开始填
            for i in range(0,j-1): # 注意左闭右开，i的停止在j-1之前即可
                dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
                if dp[i][j]: # 如果为True 计数
                    count += 1
        return count


```

# 剑指 Offer II 021. 删除链表的倒数第 n 个结点

给定一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # 注意这一题的节点标号不是从0开始的
        # 使用快慢指针，由于n不越界，所以始终让快指针领先即可
        # 方便删除节点处理使用哑节点统一语法
        # n 限定的取值范围为 1 ～ sz
        dummy = ListNode()
        dummy.next = head
        fast = dummy
        count = n + 1
        while count > 0: # 快指针先走
            fast = fast.next # 
            count -= 1
        slow = dummy
        while fast != None: # 直到快指针越界位置，两指针同时走，他们的间隔恒定
            slow = slow.next
            fast = fast.next
        # 此时slow的下一个节点是要删除的节点
        # 简单的删除语法，python无需回收节点
        slow.next = slow.next.next
        return dummy.next
```

# 剑指 Offer II 022. 链表中环的入口节点

给定一个链表，返回链表开始入环的第一个节点。 从链表的头节点开始沿着 next 指针进入环的第一个节点为环的入口节点。如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

说明：不允许修改给定的链表。

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        # 不允许修改链表，经典的Floyd判环法，快慢指针
        # 画图！设起点为A，入环点为B，碰头点为C，环长为circle
        # 第一次点运行方向是 A - B -> C 
        # 快指针的运行方向是 A - B -> C -> B -> C
        # 根据等时写出路程方程,设慢指针速率为1，快指针速率为2。t = s / v
        #  (AB+BC) / 1 = (AB + Circle + BC) / 2
        # 显然环长为 AB + BC ，所以C -> B = A - B
        # 碰头后原慢指针不变，新建一个慢指针从头开始，两者一起运动。
        # 那么A->B等于C->B,即俩指针碰头时候的点为入环点

        fast = head
        slow = head
        # 如果快指针出环了，则无环
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if fast == slow: # 如果在有环的情况下，一定会有俩指针碰头
                new_slow = head
                while new_slow != slow:
                    new_slow = new_slow.next
                    slow = slow.next
                return slow # 找到了则返回
        return None # 快指针出循环了则无环

```

# 剑指 Offer II 023. 两个链表的第一个重合节点

给定两个单链表的头节点 `headA` 和 `headB` ，请找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `null` 。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 经典的双指针法,创建两个游标指针,两个进行同速度移动
        cur1 = headA
        cur2 = headB
        while cur1 != cur2: # 当俩游标没有指向同一个指针时候
        # 切换赛道之后两者的总路程一定相等，
        # 下面的语句其实可以用三目运算符简化
            if cur1 != None:
                cur1 = cur1.next
            else: # 切换
                cur1 = headB
            if cur2 != None:
                cur2 = cur2.next
            else: # 切换
                cur2 = headA
        return cur1
```

# 剑指 Offer II 024. 反转链表

给定单链表的头节点 `head` ，请反转链表，并返回反转后的链表的头节点。

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # 迭代法
        # 双指针辅助反转,初始化画图时，把cur1指在头节点的左边，为None，cur2为头节点
        cur1 = None
        cur2 = head
        while cur2 != None: # 开始扫描
            temp = cur2.next # 存cur2的下一个节点
            cur2.next = cur1 # 使得cur2指向cur1
            cur1 = cur2  # cur1右边移动一格
            cur2 = temp # cur2借助刚刚存下来的节点右移一格
        # 循环时候，完成上述四步之后，cur1始终指向每一步过程中的头节点
        return cur1
```

# 剑指 Offer II 025. 链表中的两数相加

给定两个 非空链表 l1和 l2 来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

可以假设除了数字 0 之外，这两个数字都不会以零开头。

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 如果允许翻转链表，则很方便处理，翻转之后，头头加和并且注意判断进位
        # 不允许翻转链表的话需要使用栈存储节点值，以节点值构建新的链表
        # 迭代法
        stack1 = [] # 收集l1的节点值
        stack2 = [] # 收集l2的节点值
        cur1 = l1
        while cur1 != None:
            stack1.append(cur1.val)
            cur1 = cur1.next
        cur2 = l2
        while cur2 != None:
            stack2.append(cur2.val)
            cur2 = cur2.next
        dummy = ListNode(-1) # 创建一个哑节点
        cur = dummy
        carry = 0 # 初始化进位为0
        while stack1 != [] and stack2 != []:
            val = stack1.pop() + stack2.pop() + carry # 计算这一个节点的值
            carry = val // 10 # 大于等于10则有进位
            val = val % 10 # 取模10后的一位数
            cur.next = ListNode(val)
            cur = cur.next
        stack = stack1 if stack1 else stack2 # 可能还有一个栈非空，需要继续弹出
        while stack != []:
            val = stack.pop() + carry # 不要忘了有之前可能留下来的进位
            carry = val // 10 # 大于等于10则有进位
            val = val % 10 # 取模10后的一位数
            cur.next = ListNode(val)
            cur = cur.next
        # 注意可能在处理完之后，car位还有存值1，则需要再新建一个节点,这一点很容易被忽略！！
        if carry == 1:
            cur.next = ListNode(carry)
            cur = cur.next
        # 此时dummy后的节点需要翻转，
        cur1 = None
        cur2 = dummy.next
        while cur2 != None:
            temp = cur2.next # 存下一个节点
            cur2.next = cur1
            cur1 = cur2
            cur2 = temp

        return cur1

```

# 剑指 Offer II 026. 重排链表

给定一个单链表 L 的头节点 head ，单链表 L 表示为：

 L0 → L1 → … → Ln-1 → Ln 
请将其重新排列后变为：

L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …

不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

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
        # 观察结果链表，发现如果想要靠蛮力找的话较难联结节点关系
        # 方法1: 如果使用大量内存空间，可以使用哈希表蛮力存节点，然后用k-v对进行查找连接
        # 方法2: 观察结果链表，发现其可以分成奇、偶链的关系
        # step1: 奇数链是单调递增的序号，偶数链是单调递减的序号，那么把原链表先分成奇偶链
        # 奇数链是前半部分原链表，偶数链是后半部分原链表
        # step2: 偶数链表倒置
        # step3: 之后再和奇数链表，循环弹出头部进行链接
        # 快慢指针找中点
        slow = head
        fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        # 此时slow指向的是后半部分链表的头
        # 翻转链表,从slow后面翻转可以防止同一个节点出现在两个链表中！
        memo = slow # 注意这一行！
        cur1 = None
        cur2 = slow.next
        while cur2 != None:
            temp = cur2.next
            cur2.next = cur1
            cur1 = cur2
            cur2 = temp
        memo.next = None
        # 此时cur1指向翻转链表后的头部，把它作为偶数链头
        even_head = cur1
        odd_head = head # 奇数链头还是原链头
        # print(odd_head,even_head) # 这里print检查一下,有没有重复节点
        cur1 = odd_head
        cur2 = even_head
        while cur1 != None and cur2 != None: # 合并链表
            temp1 = cur1.next
            temp2 = cur2.next 
            cur1.next = cur2
            cur2.next = temp1
            cur1 = temp1
            cur2 = temp2
```

# 剑指 Offer II 027. 回文链表

给定一个链表的 **头节点** `head` **，**请判断其是否为回文链表。

如果一个链表是回文，那么链表节点序列从前往后看和从后往前看是相同的。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        # 利用快慢指针得到中点，然后翻转中点之后到链表
        # 由于本题无需保证原始结构不变，所以可以选择不复原结构
        fast = head
        slow = head
        while fast != None and fast.next != None: # 注意and的短路运算，这俩不能换位置
            fast = fast.next.next
            slow = slow.next
        # 此时slow指向中点,翻转slow之后的节点,由于不需要复原链表，直接断开即可
        cur1 = None
        cur2 = slow
        while cur2 != None:
            temp = cur2.next # 存下cur2的下一个节点
            cur2.next = cur1
            cur1 = cur2
            cur2 = temp
        # 此时cur1指向原尾巴节点
        cur3 = head # 开始比对
        while cur1 != None and cur3 != None:
            if cur1.val != cur3.val: # 如果不相同，则返回False
                return False
            cur1 = cur1.next
            cur3 = cur3.next
        return True

```

# 剑指 Offer II 028. 展平多级双向链表

多级双向链表中，除了指向下一个节点和前一个节点指针之外，它还有一个子链表指针，可能指向单独的双向链表。这些子列表也可能会有一个或多个自己的子项，依此类推，生成多级数据结构，如下面的示例所示。

给定位于列表第一级的头节点，请扁平化列表，即将这样的多级双向链表展平成普通的双向链表，使所有结点出现在单级双链表中。

```python
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
        # 这一题用迭代做可能比较困难，发现其有较好的递归解法
        # 显然，没有孩子的话直接跨过这个节点即可
        # 有孩子的话，需要把child也抹平
        # 那么如何设计这个递归，是先处理还是先抛给下一个，显然是要先抛给下一个，因为只有把下一个处理完了才知道这一个的next怎么链接
    def flatten(self, head: 'Node') -> 'Node':
        if head == None: # 空链表直接返回
            return None
        cur = head # 走主链
        while cur != None:
            if cur.child == None: #没有child链，继续走
                cur = cur.next
            elif cur.child != None:
                next_cur = cur.next # 存下下一个主链上的节点
                temp = cur.child # 存下child的头节点
                cur.child = None # 扁平化需要置child为None
                temp.prev = cur
                cur.next = self.flatten(temp) # 扁平化孩子链
                tail = temp # 找孩子链的尾巴
                while tail.next != None:
                    tail = tail.next
                if next_cur != None: # 注意这一条，如果下一个节点是None，则不需要链接
                    tail.next = next_cur
                    next_cur.prev = tail
                cur = next_cur
        return head
```

```go
/**
 * Definition for a Node.
 * type Node struct {
 *     Val int
 *     Prev *Node
 *     Next *Node
 *     Child *Node
 * }
 */

func flatten(root *Node) *Node {
    // go翻译版
    if root == nil {
        return root
    }
    cur := root 
    for cur != nil {
        if cur.Child != nil {
            temp := cur.Next
            tempHead := flatten(cur.Child)
            cur.Next = tempHead
            tempHead.Prev = cur 
            cur.Child = nil

            tail := cur
            for tail.Next != nil {
                tail = tail.Next
            }
            tail.Next = temp
            if temp != nil {
                temp.Prev = tail
            }
        } else {
            cur = cur.Next
        }        
    }
    return root    
}
```

# 剑指 Offer II 029. 排序的循环链表

给定循环升序列表中的一个点，写一个函数向这个列表中插入一个新元素 insertVal ，使这个列表仍然是循环升序的。

给定的可以是这个列表中任意一个顶点的指针，并不一定是这个列表中最小元素的指针。

如果有多个满足条件的插入位置，可以选择任意一个位置插入新的值，插入后整个列表仍然保持有序。

如果列表为空（给定的节点是 null），需要创建一个循环有序列表并返回这个节点。否则。请返回原先给定的节点。

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

# 剑指 Offer II 030. 插入、删除和随机访问都是 O(1) 的容器

设计一个支持在平均 时间复杂度 O(1) 下，执行以下操作的数据结构：

insert(val)：当元素 val 不存在时返回 true ，并向集合中插入该项，否则返回 false 。
remove(val)：当元素 val 存在时返回 true ，并从集合中移除该项，否则f返回 true 。
getRandom：随机返回现有集合中的一项。每个元素应该有 相同的概率 被返回。

```python
class RandomizedSet:
# 数组 + 哈希表
# 插入删除O(1)往哈希表方面想
# 
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.arr = [] # 数组，主要用途是用来获取随机索引
        self.dict = {} # 哈希表，主要用途是用来插入和删除

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val not in self.dict:
            self.dict[val] = len(self.arr) # k-v设置为 值:数组索引 对
            self.arr.append(val) # 将其添加到数组中
            return True
        return False

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.dict:
            index = self.dict[val] # 获取它的索引
            # 删除方式是，把数组中最后一个位置的元素复制到这个索引上，然后删除数组的最后一个元素
            last_element = self.arr[-1]
            self.arr[index] = last_element
            self.arr.pop() # 删除数组中的元素
            self.dict[last_element] = index # 更新原来最后一个元素的kv对
            del self.dict[val] # 删除需要删除元素的键
            return True
        return False

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        i = random.randint(0,len(self.arr)-1)
        return self.arr[i]
```

# 剑指 Offer II 031. 最近最少使用缓存

运用所掌握的数据结构，设计和实现一个  LRU (Least Recently Used，最近最少使用) 缓存机制 。

实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。

```
LRU基于双向链表和哈希表
最关键的点在于，访问过后就把那个节点提前

如何设计哈希表？

哈希表需要存的是简单的k,v键值对？
不行！
如果只存k,v。那么和链表有啥关系？根本就没有办法将哈希表和链表沟通起来呀？
那哈希表存的k是key，对应的值存个节点。节点是Node(val)？
假设这么做，的确沟通了哈希表和双向链表

链表当然要节点，那么节点存的是啥呢？只需要key？只需要val？还是都要？

分析节点需要存啥: 哈希表能够快速访问key获取值

假设节点只存了k,形成的是k-k沟通。这有啥用呀。所以不对
假设节点只存了v，形成的是k-v沟通。可以通过k访问到v。但是到了要删除的时候，需要被丢弃的节点还得在map中删除。你从链表中获取到要被删除的节点只能获取到v，而删除map是需要用key来删除的。
所以，节点存key又存val
```

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

# 剑指 Offer II 032. 有效的变位词

给定两个字符串 s 和 t ，编写一个函数来判断它们是不是一组变位词（字母异位词）。

注意：若 s 和 t 中每个字符出现的次数都相同且字符顺序不完全相同，则称 s 和 t 互为变位词（字母异位词）。

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # 如果俩词长度不等一定不是变位词
        if len(s) != len(t):
            return False
        # 由于字符都在ASCII码范围内，直接排序比较即可,注意排除两词相等的情况
        if s == t:
            return False
        return sorted(s) == sorted(t)
```

# 剑指 Offer II 033. 变位词组

给定一个字符串数组 strs ，将 变位词 组合在一起。 可以按任意顺序返回结果列表。

注意：若两个字符串中每个字符出现的次数都相同且字符顺序不完全相同，则称它们互为变位词。

示例 1:

输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 利用python的字典，键为排序后的单词，只要是变位词则键相等，值为填充当前值
        dic = collections.defaultdict(list)
        for i in strs:
            key = ''.join(sorted(i)) # 注意键必须是不可变元素，将其变成字符串
            dic[key].append(i) # 值是列表，补充上本次扫描到的值即可
        # 此时字典中的values即为单个列表
        ans = [] # 收集答案
        for lst in dic.values():
            ans.append(lst)
        return ans
```

# 剑指 Offer II 034. 外星语言是否排序

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

# 剑指 Offer II 035. 最小时间差

给定一个 24 小时制（小时:分钟 **"HH:MM"**）的时间列表，找出列表中任意两个时间的最小时间差并以分钟数表示。

```python
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        # 写一个将time标准化为分钟数的函数，然后转换后排序
        lst = [self.toMinute(time) for time in timePoints]
        n = len(lst)
        lst.sort() # 排序，最小差一定出现在两个相邻的数的差值中
        # 注意时间差取的是最近的差值。比如"23:59","00:00"取的是1而不是1439，所以考虑取模
        p = 0
        min_gap = 0xffffffff # 初始化为极大值
        for i in range(n):  # 这里其实可以省略掉中间的比较pre和nxt，实际只需比较头尾的pre和nxt，统一语法就没有改
            prev = (lst[i] - lst[i-1]) % 1440 # 与前一个的时间差
            nxt = (lst[(i+1)%n] - lst[i]) % 1440 # 与后一个的时间差
            min_gap = min(min_gap,prev,nxt) # 取最小的
        return min_gap
   
    def toMinute(self,time): # 要是为了简洁可以直接全部压缩语句到return里
        the_sum = 0
        hour = time[:2]
        mini = time[3:]
        the_sum = int(hour) * 60 + int(mini)
        return the_sum
```



# 剑指 Offer II 036. 后缀表达式

根据 逆波兰表示法，求该后缀表达式的计算结果。

有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

说明：

整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        # 后缀表达式使用栈匹配
        # 从左到右扫，遇到值则压入栈
        # 遇到符号则弹出两个值，计算之后压入栈
        # 给定的表达式必定有效，省去了很多合法性检查，自己面试的时候一定要注意
        stack = []
        for i in tokens:
            if self.isValidNum(i): #
                stack.append(i)
            else:
                value1 = int(stack.pop())
                value2 = int(stack.pop())
                if i == '+':
                    stack.append(str(value1+value2))
                elif i == '-':
                    stack.append(str(value2-value1))
                elif i == '*':
                    stack.append(str(value1*value2))
                elif i == '/': # 注意这一行处理不能使用value2//value1 否则在负数除法中有误
                    stack.append(str(int(value2/value1)))
        # 最后栈中留下一个数
        return int(stack[0])

    
    def isValidNum(self,s): # 检查传入的字符串是否是合法数值
        if s.isdigit():
            return True
        elif s[1:].isdigit(): # 切片检查负数，去掉符号位
            return True
        else:
            return False

```

# 剑指 Offer II 037. 小行星碰撞

给定一个整数数组 asteroids，表示在同一行的小行星。

对于数组中的每一个元素，其绝对值表示小行星的大小，正负表示小行星的移动方向（正表示向右移动，负表示向左移动）。每一颗小行星以相同的速度移动。

找出碰撞后剩下的所有小行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。

```python
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        # 栈匹配问题，正数往右跑，负数往左跑。数值仅仅代表质量大小
        stack = []
        for i in asteroids:
            if len(stack) == 0:
                stack.append(i)
            elif i >= 0:  # 向右边飞行的直接入栈
                stack.append(i)
            elif i < 0: # 向左边飞行的考虑是否需要处理栈顶,一路撞到自己没了或者正数没了
                while len(stack) > 0 and stack[-1] > 0 and abs(stack[-1]) < abs(i): # 自己还在
                    stack.pop() 
                if len(stack) > 0 and stack[-1] > 0 and abs(stack[-1]) == abs(i): # 自己没了
                    stack.pop()
                elif len(stack) == 0: # 正数没了，数都没了
                    stack.append(i)
                elif stack[-1] > 0: # 如果正数还有，它进不去
                    pass
                elif stack[-1] <= 0: # 如果正数没了，但是还有数，它进去
                    stack.append(i)
        return stack
```

# 剑指 Offer II 038. 每日温度

请根据每日 气温 列表 temperatures ，重新生成一个列表，要求其对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # 单调栈，但该调用单调递减还是单调递增呢？【从栈底->栈头】
        # 假设调用的是单调递增，那么每次遇到小于栈顶的时候需要特殊处理，可是遇到的是小温度，和题目中的要求遇到的是大温度不符合。
        # 那么调用的是单调递减，那么每次遇到大于栈顶的时候需要特殊处理，原栈顶的下一个更高温度是不是就找到了呢？
        # 所以维护单调栈，维护【从栈底->栈头】的单调递减栈
        # 为了方便，这里把入栈元素设置为【索引，值】的二元数组
        # 实际上有索引就够了，可以利用索引在temperatures里面查值，但是为了清晰一点，把俩都加进去
        decrease_monotony_stack = [] # 递减单调栈
        ans = [0 for i in range(len(temperatures))] # 收集列表,由于气温不升高用0代替，直接初始化为全0
        # 扫描temperatures
        p = 0
        while p < len(temperatures):
            if len(decrease_monotony_stack) == 0: # 栈空则直接入栈
                decrease_monotony_stack.append([p,temperatures[p]])
                p += 1
                continue
            if temperatures[p] > decrease_monotony_stack[-1][1]:
                while decrease_monotony_stack != [] and temperatures[p] > decrease_monotony_stack[-1][1]:
                    e = decrease_monotony_stack.pop() # 只要执行过pop，说明在位置p找到了栈顶的下一个更大元素，
                    # 那么需要收集结果，ans[原栈顶顶索引位置] = 当前位置p - 原栈顶顶索引位置
                    ans[e[0]] = p - e[0]               
            decrease_monotony_stack.append([p,temperatures[p]])
            p += 1
        return ans
```

# 剑指 Offer II 039. 直方图最大矩形面积

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

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 单调栈法，保证单调递增
        # 由于需要面积，所以加入的是索引
        # 每次面积计算是算，当前柱子高度*【前面比他小的最近的-后面比他小的最近的横向差】
        # 当遇见下降的时候，栈中前一根是离他最近的比他小的最近的，当前扫描到的是后面比他小小的最近的，计算面积
        maxArea = 0
        n = len(heights)
        # 为了减少边界处理，单调栈前面加入一个-1，则它不会为空
        stack = [-1] # 它同时还代表着索引
        for i in range(n):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                h = heights[stack.pop()]
                w = i-stack[-1]-1
                maxArea = max(maxArea,h*w)
            stack.append(i)
        while stack[-1] != -1:
            h = heights[stack.pop()]
            w = n-stack[-1]-1
            maxArea = max(maxArea,h*w)
        return maxArea

```

# 剑指 Offer II 040. 矩阵中最大的矩形

给定一个由 `0` 和 `1` 组成的矩阵 `matrix` ，找出只包含 `1` 的最大矩形，并返回其面积。

**注意：**此题 `matrix` 输入格式为一维 `01` 字符串数组。

```python
class Solution:
    def maximalRectangle(self, matrix: List[str]) -> int:
        if len(matrix) == 0:
            return 0
        if len(matrix[0]) == 0:
            return 0

        def calcArea(lst):
            maxArea = 0
            stack = [-1]
            n = len(lst)
            for i in range(n):
                while stack[-1] != -1 and lst[stack[-1]] >= lst[i]:
                    h = lst[stack.pop()]
                    w = i-stack[-1]-1
                    maxArea = max(maxArea,h*w)
                stack.append(i)
            while stack[-1] != -1:
                h = lst[stack.pop()]
                w = n - stack[-1] - 1
                maxArea = max(maxArea,h*w)
            return maxArea
        
        m = len(matrix)
        n = len(matrix[0])
        maxArea = 0
        lst = [0 for j in range(n)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    lst[j] += int(matrix[i][j])
                else:  # 注意这个重置
                    lst[j] = 0
            maxArea = max(maxArea,calcArea(lst))
        return maxArea
```

# 剑指 Offer II 041. 滑动窗口的平均值

给定一个整数数据流和一个窗口大小，根据该滑动窗口的大小，计算滑动窗口里所有数字的平均值。

实现 MovingAverage 类：

MovingAverage(int size) 用窗口大小 size 初始化对象。
double next(int val) 成员函数 next 每次调用的时候都会往滑动窗口增加一个整数，请计算并返回数据流中最后 size 个值的移动平均值，即滑动窗口里所有数字的平均值。

```python
# 方法1: 一种简单思想，模拟，它窗口里有啥我都要了,维护窗口大小即可

class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.window = collections.deque()
        self.sum_num = 0 # 窗口内数值总和，初始化为0
        self.limit = size # 限制窗口大小

    def next(self, val: int) -> float:
        if len(self.window) >= self.limit:
            e = self.window.popleft() # 收集弹出的元素值
            self.sum_num -= e # 窗口内的数值减小了

        self.window.append(val) # 窗口中加入数值
        self.sum_num += val # 窗口内的数值加大了
        return float(self.sum_num/len(self.window))

```

```python
# 方法2: 循环队列的方法
# 开始就直接申请长度为size的窗口，每次更新采用取模的方式填充位置之后，更新下一次要填充的位置的索引
class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.window = [0 for i in range(size)] # 
        self.sum_num = 0 # 初始化为0
        self.mark = 0 # 初始化为需要填充的位置的索引
        self.add_times = 0 # 记录加入了几次

    def next(self, val: int) -> float:
        self.add_times += 1 # 加入次数加一
        delete_num = self.window[self.mark] # 判断这个位置以前是否有数字，有的话收集起来
        self.window[self.mark] = val # 填充
        self.mark = (self.mark + 1) % len(self.window) # 更新,如果遇到队尾了，则由于模运算跳转到了头部
        self.sum_num = self.sum_num - delete_num + val # 把删去的数减掉，把加入的数加上
        if self.add_times < len(self.window): # 如果加入次数还没有满窗口
            return float(self.sum_num/self.add_times) # 则平均值的分母是加入的元素个数
        return float(self.sum_num/len(self.window)) # 填满了自然是除以全长
        
```

# 剑指 Offer II 042. 最近请求次数

写一个 RecentCounter 类来计算特定时间范围内最近的请求。

请实现 RecentCounter 类：

RecentCounter() 初始化计数器，请求数为 0 。
int ping(int t) 在时间 t 添加一个新请求，其中 t 表示以毫秒为单位的某个时间，并返回过去 3000 毫秒内发生的所有请求数（包括新请求）。确切地说，返回在 [t-3000, t] 内发生的请求数。
保证 每次对 ping 的调用都使用比之前更大的 t 值。

```python
class RecentCounter:
# 双端队列，先加入入队的值，
# 循环检查：入队后如果头部与现在的尾部差值大于三千，则弹出头部
# 维护这样的一个队列
# 既然是简单题就不必手写双端队列了，python的双端队列是基于链式结构的
    def __init__(self):
        self.my_queue = collections.deque() # 使用双端队列作为容器

    def ping(self, t: int) -> int:
        self.my_queue.append(t)
        while (self.my_queue[-1] - self.my_queue[0]) > 3000:
            self.my_queue.popleft() # 出队
        return len(self.my_queue)

```

# 剑指 Offer II 043. 往完全二叉树添加节点

完全二叉树是每一层（除最后一层外）都是完全填充（即，节点数达到最大，第 n 层有 2n-1 个节点）的，并且所有的节点都尽可能地集中在左侧。

设计一个用完全二叉树初始化的数据结构 CBTInserter，它支持以下几种操作：

CBTInserter(TreeNode root) 使用根节点为 root 的给定树初始化该数据结构；
CBTInserter.insert(int v)  向树中插入一个新节点，节点类型为 TreeNode，值为 v 。使树保持完全二叉树的状态，并返回插入的新节点的父节点的值；
CBTInserter.get_root() 将返回树的根节点。

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

# 剑指 Offer II 044. 二叉树每层的最大值

给定一棵二叉树的根节点 `root` ，请找出该二叉树中每一层的最大值。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        # BFS搜索
        if root == None: # 考虑空树
            return []
        ans = [] # 收集答案
        queue = [root] # 借助队列管理
        while len(queue) != 0:
            level = [] # 收集本层元素
            new_queue = [] # 收集下一层节点
            for i in queue:
                if i != None:
                    level.append(i.val)
                    new_queue.append(i.left)
                    new_queue.append(i.right)
            if len(level) != 0:
                ans.append(max(level)) # 收集本层最大值
            queue = new_queue # 把下一层节点传入
        return ans

```

# 剑指 Offer II 045. 二叉树最底层最左边的值

给定一个二叉树的 **根节点** `root`，请找出该二叉树的 **最底层 最左边** 节点的值。

假设二叉树中至少有一个节点。

```python
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        # 题目已经按需要按层来取数了，BFS一定是最方便的写法
        # 返回最底层的第一个数即可，题目条件给出了树非空
        ans = [] # 收集每层答案
        queue = [root] # 利用队列进行管理
        while len(queue) != 0:
            level = [] # 收集本层节点的值
            new_queue = [] # 收集下一层节点
            for i in queue:
                if i != None:
                    level.append(i.val)
            for i in queue:
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            ans.append(level) # 把收集完的本层路径添加进去
            queue = new_queue # 把下一层将要扫描的节点传入
        # 此时，最后一层的第一个值即为答案
        return ans[-1][0]

```

# 剑指 Offer II 046. 二叉树的右侧视图

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

```python
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        # BFS
        # 注意考虑空树
        if root == None:
            return []
        ans = [] # 收集答案
        queue = [root]
        while len(queue) != 0:
            level = [] # 收集本层所有节点结果
            new_queue = [] # 收集下一层要扫描的节点
            for i in queue:
                if i != None:
                    level.append(i.val)
                if i.left != None:
                    new_queue.append(i.left)
                if i.right != None:
                    new_queue.append(i.right)
            ans.append(level[-1]) # 添加最右边的值
            queue = new_queue # 下一层要扫描的节点传入
        return ans 

```

# 剑指 Offer II 047. 二叉树剪枝

给定一个二叉树 根节点 root ，树的每个节点的值要么是 0，要么是 1。请剪除该二叉树中所有节点的值为 0 的子树。

节点 node 的子树为 node 本身，以及所有 node 的后代。

```python
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        # 先序遍历删除法
        # 看总和是否为0
        countZero = [0] # 数树中的0的数量
        treeSize = [0] # 数树中的有效节点数量

        def calcZero(root): # 
            if root == None:
                return 
            if root.val == 0:
                countZero[0] += 1
            treeSize[0] += 1
            calcZero(root.left)
            calcZero(root.right)

        calcZero(root) # 调用
        if countZero[0] == treeSize[0]: # 全0，则直接全删了
            return None 

        def dfs(root,parent): # 非全0，根节点不会被删掉
            if root == None:
                return
            if root.val == 0 and root.left == None and root.right == None: # 删除操作
                if parent.left == root:
                    parent.left = None
                elif parent.right == root:
                    parent.right = None
            dfs(root.left,root)
            dfs(root.right,root)

        for i in range(countZero[0]): # 最多删除次数就是0的个数
            dfs(root,None)

        return root
```

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

# 剑指 Offer II 048. 序列化与反序列化二叉树

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

# 剑指 Offer II 049. 从根节点到叶节点的路径数字之和

给定一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。

每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的 所有数字之和 。

叶节点 是指没有子节点的节点。

```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        # 数据范围不大，直接dfs暴搜
        # 使用辅助函数将路径转换成数字
        ans = [] # 收集每一条路径
        path = [] # 
        def dfs(node): # 到叶子节点就把路径添加到ans中
            if node == None:
                return 
            path.append(str(node.val)) # 收集成字符类型方便转换成数字
            if node.left == None and node.right == None: # 到叶子节点则收集
                ans.append(path[:]) # 注意不能传path，要传值而不是传引用
            dfs(node.left)
            dfs(node.right)
            path.pop() # 回溯时弹出本次选择的元素
        dfs(root) # 调用搜索方法，此时ans中是每条路径
        the_sum = 0 # 最终需要返回的结果
        for lst in ans:
            the_sum += self.toNum(lst)
        return the_sum
           
    def toNum(self,lst):# 传入参数为列表,返回结果为数值
        # 由于数据值不大，直接先转车过字符串再直接拼接
        return int(''.join(lst))

```

# 剑指 Offer II 050. 向下的路径节点之和

给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

```python
# 方法1:
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        # 双重dfs，一个是遍历节点的扫法，一个是选取为目标值的扫法
        # 注意参数命名，root和node自己要分清楚
        ans = 0 # 收集结果

        def dfs(node,targetSum):
            nonlocal ans
            if node == None:
                return                 
            if node.val == targetSum:
                ans += 1
            dfs(node.left,targetSum-node.val)
            dfs(node.right,targetSum-node.val)

        def pre_order(node):
            if node == None:
                return 
            dfs(node,targetSum)
            pre_order(node.left)
            pre_order(node.right)

        pre_order(root) # 开始搜索
        return ans
```

```python
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        # 哈希法
        preDict = collections.defaultdict(int)
        preDict[0] = 1
        ans = 0
        pre = 0
        def dfs(root):
            nonlocal ans
            nonlocal pre
            if root == None:
                return 
            pre += root.val # 注意这里有➕
            preDict[pre] += 1 # 加入字典
            aim = pre - targetSum

            # 注意这里的处理，如果值相等，要排除自身这一次
            if aim in preDict and aim != pre:
                ans += preDict[aim]
            elif aim in preDict and aim == pre:
                ans += preDict[aim]-1

            dfs(root.left)
            dfs(root.right)
            preDict[pre] -= 1 # 退出字典
            pre -= root.val # 注意这里有-

        dfs(root)
        return ans
```

# 剑指 Offer II 052. 展平二叉搜索树

给你一棵二叉搜索树，请 **按中序遍历** 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        # 较朴素的常规方法，没有在中序遍历过程中直接修改节点指向
        if root == None: # 处理空树
            return None
        inorder_lst = []
        def inorder(node): # 将所有节点以中序遍历的形式加入列表
            if node == None:
                return 
            inorder(node.left)
            inorder_lst.append(node)
            inorder(node.right)
        inorder(root) # 调用方法后填充中序遍历列表
        # 对中序列表的各个节点的指针进行调整
        for index,node in enumerate(inorder_lst[:-1]): # 先不调整最后一个节点
            inorder_lst[index].left = None # 左指针置空
            inorder_lst[index].right = inorder_lst[index + 1] # 右指针置为下一个节点
        # 最后一个节点俩都置空
        inorder_lst[-1].left = None
        inorder_lst[-1].right = None
        return inorder_lst[0] # 返回头节点


```

# 剑指 Offer II 053. 二叉搜索树中的中序后继

给定一棵二叉搜索树和其中的一个节点 p ，找到该节点在树中的中序后继。如果节点没有中序后继，请返回 null 。

节点 p 的后继是值比 p.val 大的节点中键值最小的节点，即按中序遍历的顺序节点 p 的下一个节点。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        # 明确什么是中序后继
        # 1. 如果这个节点有右孩子，则中序后继是先移动到右孩子，再尽可能的往左边移动找到左孩子
        # 2. 如果这个节点没有右孩子，那么中序后继是先移动到父节点，直到这个节点为空或者这个节点是它的父节点的左孩子节点
        # 返回这个节点的父节点
        node = p
        if node.right != None: # 1. 如果这个节点有右孩子，则中序后继是先移动到右孩子，再尽可能的往左边移动找到左孩子
            # print("1===")
            node = node.right
            while node.left != None:
                node = node.left
            return node
        # 2. 如果这个节点没有右孩子，那么中序后继是先移动到父节点，直到这个节点为空或者这个节点是它的父节点的左孩子节点
        # 返回这个节点的父节点
        elif node.right == None:
            # print("2===")
            self.find_parent = dict()
            self.find(root,None,p) # 调用方法填充node - parent字典
            while node != None : # 当这个节点不为空的时候
                if self.find_parent[node] == None: # 如果这个节点的父节点是空，即已经到了原树的根节点
                    return None
                if node == self.find_parent[node].left: # 如果这个节点是它父节点的左孩子，返回它的父节点
                    return self.find_parent[node]
                node = self.find_parent[node] # 否则继续移动
            return None

    def find(self,root,root_parent,target):  # 这一题已经保证p是树中的节点，面试的时候注意考虑p是否在树中
        self.find_parent[root] = root_parent # 填充找父节点的哈希表
        if target.val == root.val:
            return root,root_parent # 返回值是节点，节点的父节点【实际上用不到】
        if target.val > root.val: # 值更大，往右边找
            return self.find(root.right,root,target)
        elif target.val < root.val: # 值更小，往左边找
            return self.find(root.left,root,target)
    

```

# 剑指 Offer II 054. 所有大于等于节点的值之和

给定一个二叉搜索树，请将它的每个节点的值替换成树中大于或者等于该节点值的所有节点值之和。


提醒一下，二叉搜索树满足下列约束条件：

节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        # 一个很方便的方法：
        # 中序遍历收集总和之后，依次以中序遍历填充节点
        # 填充方法是，存下节点值，将其替换为剩余的总和值，然后总和值减去这个节点存下来的原值
        in_order_lst = []
        self.sum_num = 0
        def inorder(node):
            if node == None:
                return 
            inorder(node.left)
            in_order_lst.append(node)
            self.sum_num += node.val
            inorder(node.right)
        # 调用inorder方法，填充
        inorder(root)
        for node in in_order_lst:
            temp = node.val
            node.val = self.sum_num
            self.sum_num -= temp
        return root


```

# 剑指 Offer II 055. 二叉搜索树迭代器

```python
class BSTIterator:
# 方法1: 直接中序遍历展平
    def __init__(self, root: TreeNode):
        self.inorderList = []
        
        def inOrder(node):
            if node == None:
                return 
            inOrder(node.left)
            self.inorderList.append(node.val)
            inOrder(node.right)
        
        inOrder(root) # 调用

        self.ptr = 0 # 数组索引指针

    def next(self) -> int:
        val = self.inorderList[self.ptr]
        self.ptr += 1
        return val

    def hasNext(self) -> bool:
        return self.ptr < len(self.inorderList)

```

```python
class BSTIterator:
    # 方法2: 利用迭代中序遍历和具体的栈
    def __init__(self, root: TreeNode):
        self.cur = root 
        self.stack = []

    def next(self) -> int:
        while self.cur != None:
            self.stack.append(self.cur)
            self.cur = self.cur.left

        self.cur = self.stack.pop()
        val = self.cur.val 
        self.cur = self.cur.right 
        return val

    def hasNext(self) -> bool:
        return self.cur != None or len(self.stack) != 0

```

# 剑指 Offer II 056. 二叉搜索树中两个节点之和

给定一个二叉搜索树的 **根节点** `root` 和一个整数 `k` , 请判断该二叉搜索树中是否存在两个节点它们的值之和等于 `k` 。假设二叉搜索树中节点的值均唯一。

```python
class Solution:
    def findTarget(self, root: TreeNode, k: int) -> bool:
        # 两数之和换皮版本，二叉搜索树为了得到排序数组，且同一个位置的数不能使用两次
        inorderlst = [] # 收集升序数组
        def in_order(node): # 中序遍历收集升序数组
            if node == None:
                return 
            in_order(node.left)
            inorderlst.append(node.val)
            in_order(node.right)
        in_order(root)
        # 双指针
        left= 0 
        right = len(inorderlst) - 1
        while left < right: # 两个不能相等
            if inorderlst[left] + inorderlst[right] == k:
                return True
            elif inorderlst[left] + inorderlst[right] < k: # 总和偏小，小指针右移
                left += 1
            elif inorderlst[left] + inorderlst[right] > k: # 总和偏大，大指针左移
                right -= 1
        return False # 没有找到结果

```

# 剑指 Offer II 057. 值和下标之差都在给定的范围内

给你一个整数数组 nums 和两个整数 k 和 t 。请你判断是否存在 两个不同下标 i 和 j，使得 abs(nums[i] - nums[j]) <= t ，同时又满足 abs(i - j) <= k 。

如果存在则返回 true，不存在返回 false。

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        # 本方法参考了：负雪明烛
        # 链接：https://leetcode-cn.com/problems/contains-duplicate-iii/solution/fu-xue-ming-zhu-hua-dong-chuang-kou-mo-b-jnze/

        if len(nums) < 2:
            return False
        # python使用sortedcontainers
        from sortedcontainers import SortedSet
        theSet = SortedSet()
        # 滑动窗口，注意由于这个Set会自动去重，所以不能采用维护窗口内元素个数而只能维护索引间隔当作窗口大小
        left = 0
        right = 0
        n = len(nums)
        while right < n:
            if right - left > k:
                theSet.remove(nums[left])
                left += 1
            index = bisect.bisect_left(theSet,nums[right]-t) # 找到小于等于nums[right]-t的最大值在theSet中的索引
            if len(theSet) != 0 and 0<=index<len(theSet) and abs(theSet[index]-nums[right]) <= t:
                return True
            theSet.add(nums[right])
            right += 1
        return False
```

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        import sortedcontainers
        theSet = sortedcontainers.SortedList() # 模拟TreeSet。
        # SortedSet会自动去重。。。所以选择用SortedList
        # k是窗口间隔，t是容许数值的大小
        # 只需要往前找索引就行
        n = len(nums)
        for i in range(n):
            # 找比当前值小于等于的最大数，找比当前值大于等于的最小数
            # print(theSet)
            index = theSet.bisect_left(nums[i])
            if 0<=index<len(theSet): # index是如果把nums[i]插入之后，它的序号。看它的前一个是否符合要求
            # 插入之后会把原来的数字往后顶，那么theSet[index]代表比当前值大于等于的最小数
                if abs(theSet[index]-nums[i]) <= t:
                    return True
            # theSet[inde-1]是代表比当前值小于等于的最大数
            if 0<=index-1<len(theSet):
                if abs(theSet[index-1]-nums[i]) <= t:
                    return True
            theSet.add(nums[i])
            if len(theSet) > k:
                theSet.remove(nums[i-k])
        return False
```

```

```



# 剑指 Offer II 058. 日程表

请实现一个 MyCalendar 类来存放你的日程安排。如果要添加的时间内没有其他安排，则可以存储这个新的日程安排。

MyCalendar 有一个 book(int start, int end)方法。它意味着在 start 到 end 时间内增加一个日程安排，注意，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end。

当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生重复预订。

每次调用 MyCalendar.book方法时，如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 true。否则，返回 false 并且不要将该日程安排添加到日历中。

请按照以下步骤调用 MyCalendar 类: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)

```python
class MyCalendar:

    def __init__(self):
        from sortedcontainers import SortedDict
        self.soDict = SortedDict() # k-v关系为 ：开始时间-结束时间

    def book(self, start: int, end: int) -> bool:
        # 找比start小的
        #print(self.soDict)
        index = self.soDict.bisect(start)
        # 如果插入不越头尾界
        if len(self.soDict) == 0:
            self.soDict[start] = end
            return True
        if 0 < index < len(self.soDict):
            # 检查
            if self.soDict.values()[index-1]<=start and end <=self.soDict.keys()[index]:
                self.soDict[start] = end
                return True
         #只需要检查后一个点
        if index == 0:
            if end <= self.soDict.keys()[index]:
                self.soDict[start] = end
                return True
         # 只需要检查前一个点
        if index == len(self.soDict):
            if self.soDict.values()[index-1]<=start:
                self.soDict[start] = end
                return True
        return False

```

# 剑指 Offer II 059. 数据流的第 K 大数值

设计一个找到数据流中第 k 大元素的类（class）。注意是排序后的第 k 大元素，不是第 k 个不同的元素。

请实现 KthLargest 类：

KthLargest(int k, int[] nums) 使用整数 k 和整数流 nums 初始化对象。
int add(int val) 将 val 插入数据流 nums 后，返回当前数据流中第 k 大的元素。

```python
class KthLargest:
# 寻找排序后的第k大元素，可以使用小根堆
# 既然是简单题就直接使用内置小根堆了～
# 但是还是需要掌握自己手写小根堆的方式。最关键的是堆向上冒泡和堆向下冒泡
# 找topK使用小根堆堆过筛方式为：比堆顶小，则舍去，因为找的是大数，比小根堆的堆顶小则不可能是大数。
# 堆里面存的是k个大数！
    def __init__(self, k: int, nums: List[int]):
        # 维护堆的大小为k
        self.min_heap = []
        self.k = k # 记录堆容量
        if len(nums) <= k: # 比k小则直接堆化
            self.min_heap = nums
            heapq.heapify(self.min_heap)
        elif len(nums) > k: # 比k大，先堆化前k个，然后过筛
            temp = nums[:k]
            self.min_heap = temp
            heapq.heapify(self.min_heap)
            for num in nums[k:]: # 开始过筛
                if num > self.min_heap[0]: # 比堆顶大，则弹出一个，加入一个，维护堆容量为k
                    heapq.heappop(self.min_heap)
                    heapq.heappush(self.min_heap,num)

    def add(self, val: int) -> int:
        if len(self.min_heap) < self.k: # 小于则直接入堆
            heapq.heappush(self.min_heap,val)
            return self.min_heap[0]
        elif len(self.min_heap) >= self.k: # 堆容量够了，则过筛
            if val > self.min_heap[0]: # 比堆顶大，则弹出一个，加入一个，维护堆容量为k
                heapq.heappop(self.min_heap)
                heapq.heappush(self.min_heap,val)
            return self.min_heap[0]


```

# 剑指 Offer II 060. 出现频率最高的 k 个数字

给定一个整数数组 `nums` 和一个整数 `k` ，请返回其中出现频率前 `k` 高的元素。可以按 **任意顺序**返回答案。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 经典topK问题，使用堆排序,这一题题目限制了k不越界
        # 关键是如何转化成堆问题，堆的元素需要如何设计
        ct = collections.Counter(nums)
        # 找topK大，使用小根堆,过筛元素，如果比堆顶小，则不可能比堆里其他数大，也就不会是前k大
        # 舍弃比堆顶小的元素，只处理比堆顶大的元素
        min_heap = []
        for key in ct: # key是元素，ct[key]是频率，维护堆大小为k
            if len(min_heap) < k:
                heapq.heappush(min_heap,[ct[key],key])
            elif len(min_heap) >= k:
                if ct[key] > min_heap[0][0]:
                    heapq.heappop(min_heap)
                    heapq.heappush(min_heap,[ct[key],key])
        ans = [] # 收集结果，需要的是元素，而不是频率
        while len(ans) != k:
            ans.append(heapq.heappop(min_heap)[1])
        return ans # 可以按照任何顺序返回

```



# 剑指 Offer II 061. 和最小的 k 个数对

给定两个以升序排列的整数数组 nums1 和 nums2 , 以及一个整数 k 。

定义一对值 (u,v)，其中第一个元素来自 nums1，第二个元素来自 nums2 。

请找到和最小的 k 个数对 (u1,v1),  (u2,v2)  ...  (uk,vk) 。

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        # 分析一下，由于俩数组一定有序，那么和最小的k个数对一定来自于第一组的k个和第二组的k的组合
        # 最多只需要筛k*k的字符量
        # 使用堆去筛,python内置堆是小根堆
        # 注意堆的排序规则而选择键值构造，当等值时，优先把数值大的放返回序列的前面
        # 选择堆内元素构造为[[i+j],[j],[i,j]]
        if k > len(nums1) * len(nums2): # 越界则锁定为最多数量
            k = len(nums1) * len(nums2)
        lst1 = nums1[:k] # python切片，切片无需考虑越界问题
        lst2 = nums2[:k]
        min_heap = []
        for i in lst1:
            for j in lst2:
                min_heap.append([[i+j],[j],[i,j]]) # 注意这个堆的元素构成原则，前两个辅助排序用
        heapq.heapify(min_heap) # 堆化
        ans = [] # 收集结果
        # 维护k大小的堆
        while len(ans) != k:
           ans.append(heapq.heappop(min_heap)[2]) # 注意只需要取弹出数组的索引为2的值【索引0，1是辅助堆内排序用的】
        return ans
```

# 剑指 Offer II 062. 实现前缀树

Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。

```python
class TrieNode:

    def __init__(self):
        self.children = [None for i in range(26)] # 创建序号为0～25的节点
        self.isWord = False # 初始化为False

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode() # 实例化对象

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root # 从根节点扫起
        for char in word:
            index = (ord(char)-ord("a")) # 得到索引
            if node.children[index] == None : # 如果这个节点没有被创建，则创建它
                node.children[index] = TrieNode()
            node = node.children[index] # 移动node指针
        node.isWord = True # 扫描完毕之后，将尾巴标记为True，代表以他终止的时候是单词


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for char in word:
            index = (ord(char) - ord("a"))
            if node.children[index] == None: # 如果扫描的路上发现了None，则不会是单词
                return False
            node = node.children[index] # 移动node指针
        return node.isWord # 返回判断它是否是单词的标记 


    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for char in prefix:
            index = (ord(char) - ord("a"))
            if node.children[index] == None: # 扫描的路上不能有断层
                return False
            node = node.children[index]
        return node != None # 最终位置如果非空，则说明有这一条前缀路线


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

# 剑指 Offer II 063. 替换单词

在英语中，有一个叫做 词根(root) 的概念，它可以跟着其他一些词组成另一个较长的单词——我们称这个词为 继承词(successor)。例如，词根an，跟随着单词 other(其他)，可以形成新的单词 another(另一个)。

现在，给定一个由许多词根组成的词典和一个句子，需要将句子中的所有继承词用词根替换掉。如果继承词有许多可以形成它的词根，则用最短的词根替换它。

需要输出替换之后的句子。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.prefix = None

class Trie:
    def __init__(self):
        self.root = TrieNode() # 初始化一个实例
    
    def insertprefix(self,pre): # 插入前缀
        node = self.root
        for char in pre:
            index = (ord(char)-ord("a"))
            if node.children[index] == None:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.prefix = pre # 将这个前缀存入
    
    def creat_all(self,dictionary): # 将所有词典中的词插入到前缀树中
        for word in dictionary:
            self.insertprefix(word)
    
    def get_shortest(self,word): # 找到最短的合法前缀
        node = self.root
        for char in word:
            index = (ord(char)-ord("a"))
            if node.children[index] == None: # 开头就碰壁，直接返回原单词
                return word
            # 如果开头没有碰壁，说明一定存在合法路径，找到最短的就返回。
            # 第二个elif就是找到最短的，
            elif node.children[index] != None and node.children[index].prefix == None: 
                node = node.children[index]
            elif node.children[index] != None and node.children[index].prefix != None:
                node = node.children[index] 
                return node.prefix
        return word # 可能会有公共路径但是没有公共结尾，此时直接返回原词

class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        # 将sentence间隔成列表，由于测试用例的sentence很规整，不需要格式化处理
        lst = sentence.split(' ')
        dic = Trie() # 创建一个字典树实例
        dic.creat_all(dictionary) 
        for index in range(len(lst)): # 将列表中的每一个词替换
            lst[index] = dic.get_shortest(lst[index])
        ans = ' '.join(lst) # 重新链接成字符串
        return ans
```

# 剑指 Offer II 064. 神奇的字典

设计一个使用单词列表进行初始化的数据结构，单词列表中的单词 互不相同 。 如果给出一个单词，请判定能否只将这个单词中一个字母换成另一个字母，使得所形成的新单词存在于已构建的神奇字典中。

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

# 剑指 Offer II 065. 最短的单词编码

单词数组 words 的 有效编码 由任意助记字符串 s 和下标数组 indices 组成，且满足：

words.length == indices.length
助记字符串 s 以 '#' 字符结尾
对于每个下标 indices[i] ，s 的一个从 indices[i] 开始、到下一个 '#' 字符结束（但不包括 '#'）的 子字符串 恰好与 words[i] 相等
给定一个单词数组 words ，返回成功对 words 进行编码的最小助记字符串 s 的长度 。

```python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.isPath = False
        self.depth = 0 # 当前深度

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

# 剑指 Offer II 066. 单词之和

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
    
    def insert(self,key,val): # 插入单词和相应的值
        node = self.root
        for char in key:
            index = (ord(char)-ord("a"))
            if node.children[index] == None: # 如果这个孩子是空的，创建它
                node.children[index] = TrieNode() 
            node = node.children[index]
        node.value = val
        # print(node,node.value)

    def find_prefix(self,prefix): # 找所有合法前缀
        node = self.root
        for char in prefix:
            index = (ord(char)-ord("a"))
            if node.children[index] == None: # 如果这个孩子是空的，创建它
                node.children[index] = TrieNode()
            node = node.children[index]
        the_sum = 0
        queue = [node] # BFS搜结果，借助队列管理，明确queue里面存的都是节点
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
        self.Tree = TrieTree() # 实例化一个字典树

    def insert(self, key: str, val: int) -> None:
        self.Tree.insert(key,val)

    def sum(self, prefix: str) -> int:
        ans = self.Tree.find_prefix(prefix)
        return ans
```

# 剑指 Offer II 067. 最大的异或

给定一个整数数组 `nums` ，返回 `nums[i] XOR nums[j]` 的最大运算结果，其中 `0 ≤ i ≤ j < n` 。

```python
class TrieNode:
    def __init__(self):
        self.children = [None,None]
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,n):
        node = self.root 
        # 注意bit位，要倒着来,才能保证从上往下是从高位到低位
        for i in range(31,-1,-1):# 
            bit = (n>>i)&1
            if node.children[bit] == None:
                node.children[bit] = TrieNode() 
            node = node.children[bit]
    
    def insertAll(self,lst):
        for n in lst:
            self.insert(n)
            
class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        # 利用字典树。和32位int完成
        maxXor = 0
        tree = Trie()
        for n in nums:
            tree.insert(n)

        for n in nums: # 对于每个数，进行检查，优先往另一个分支走
            xor = 0 # 用字符串接收也可以，用数接收也可以。字符串的接收还需要写一个字符串转int的逻辑
            node = tree.root 
            for i in range(31,-1,-1):
                bit = (n>>i)&1 # 注意这里一个处理，bit只能是1或者0，优先走相反的可以用1-bit统一语法
            
            # 解释一下xor的变化逻辑。
            # 比如：某轮循环得到xor == 10101 ，然后下一位相同的情况下。
            # 然后下一位相同的情况下。新xor应该为 10101(0) 即 xor == xor << 1 
            # 下一位不同的情况下，新xor应该为 10101(1) 即 xor == (xor << 1) + 1
                if node.children[1-bit] != None: # 优先,此时异或后这一位为1
                    node = node.children[1-bit] 
                    xor = xor*2 + 1 # 写成 xor = (xor << 1) + 1 也可以，必须要打括号，位运算优先级低
                else:
                    node = node.children[bit]
                    xor = xor*2 # 写成 xor = xor << 1 也可以
            maxXor = max(maxXor,xor)
        return maxXor  
```

# 剑指 Offer II 068. 查找插入位置

给定一个排序的整数数组 nums 和一个整数目标值 target ，请在数组中找到 target ，并返回其下标。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 O(log n) 的算法。

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # 经典二分搜索
        # 搜索区间为左闭右闭,nums无重复元素！
        left = 0
        right = len(nums) - 1
        while left <= right: # 左闭右闭区间的允许小于等于号，因为要搜索左右指向同一个位置的数
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target: # 说明中值偏大，在左边搜
                right = mid - 1
            elif nums[mid] < target: # 说明中值偏小，在右边搜
                left = mid + 1       
        return left
```

# 剑指 Offer II 069. 山峰数组的顶部

符合下列属性的数组 arr 称为 山峰数组（山脉数组） ：

arr.length >= 3
存在 i（0 < i < arr.length - 1）使得：
arr[0] < arr[1] < ... arr[i-1] < arr[i]
arr[i] > arr[i+1] > ... > arr[arr.length - 1]
给定由整数组成的山峰数组 arr ，返回任何满足 arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1] 的下标 i ，即山峰顶部。

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        # 已知它已经是一个山脉数组，那么只需要开始二分
        # 注意要求的是严格山峰
        left = 1
        right = len(arr) - 2
        while left <= right: # 闭区间二分,由于已经告诉我们它是山脉数组，所以循环内不需要==判断
            mid = (left + right)//2
            if arr[mid] < arr[mid+1]:
                left = mid + 1
            elif arr[mid] > arr[mid+1]:
                right = mid - 1
        return left
```

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        # 已知其是严格山脉数组
        left = 0
        right = len(arr)-1
        while left <= right:
            mid = (left + right)//2
            if arr[mid] < arr[mid+1]: # 收缩左边界
                left = mid + 1
            elif arr[mid] > arr[mid+1]: # 收缩右边界
                right = mid - 1
        return left
```

# 剑指 Offer II 072. 求平方根

给定一个非负整数 x ，计算并返回 x 的平方根，即实现 int sqrt(int x) 函数。

正数的平方根有两个，只输出其中的正数平方根。

如果平方根不是整数，输出只保留整数的部分，小数部分将被舍去。

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        # 二分法
        if x <= 1: # 
            return x
        left = 0
        right = x
        while left <= right:
            mid = (left+right)//2
            if mid**2 <=x and (mid+1)**2 > x:
                return mid
            elif mid**2 < x: # 数值偏小，需要扩大
                left = mid + 1
            elif mid**2 > x: # 数值偏大，需要缩小
                right = mid - 1

```

# 剑指 Offer II 073. 狒狒吃香蕉

狒狒喜欢吃香蕉。这里有 N 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 H 小时后回来。

狒狒可以决定她吃香蕉的速度 K （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 K 根。如果这堆香蕉少于 K 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉，下一个小时才会开始吃另一堆的香蕉。  

狒狒喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。

返回她可以在 H 小时内吃掉所有香蕉的最小速度 K（K 为整数）。

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

# 剑指 Offer II 074. 合并区间

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 先按照左端点起始值排序
        intervals.sort(key = lambda x:x[0])
        # 画图分析情况，每相邻的一个区间，前一个叫prev,后一个叫now
        # 1. 后一个区间完全在前一个区间内。即，prev[0]<= now[0];prev[1] >= now[1] 只需要前一个
        # 2. 后一个区间的一部分在前一个区间内，即 prev[0]<= now[0];prev[1] < now[1] 合并
        # 3. 后一个区间和前一个区间没有重合，毫不相干。 now[0] <= prev[1]
        if len(intervals) == 1:
            return intervals
        ans = [] # 收集答案
        prev = intervals[0]
        ans.append(prev)
        p = 1
        while p < len(intervals):
            now = intervals[p]
            if prev[0] <= now[0] and prev[1] >= now[1]:
                pass # 情况1
            elif prev[0] <= now[0] and prev[1] >= now[0] and prev[1] < now[1]:
                ans.pop()
                ans.append([prev[0],now[1]]) # 情况2
            elif now[0] >= prev[1]: # 情况[3]: 俩不相干，直接加入
                ans.append(now)
            prev = ans[-1] # prev更新
            p += 1
        return ans
```

# 剑指 Offer II 075. 数组相对排序

给定两个数组，arr1 和 arr2，

arr2 中的元素各不相同
arr2 中的每个元素都出现在 arr1 中
对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1 的末尾。

```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        # 方法1 ： 双向映射法，调用了排序api和两次映射
        the_map = {val:index for index,val in enumerate(arr2)}
        mirror_map = {index:val for index,val in enumerate(arr2)}
        seive = [] # 筛分出不在arr2中的元素
        temp = []
        for i in arr1:
            if i not in the_map:
                seive.append(i)
            else:
                temp.append(the_map[i])
        seive.sort() # 不在arr2中的元素需要升序
        temp.sort() # 排序temp
        ans = []
        for i in temp:
            ans.append(mirror_map[i]) # 映射回原值
        ans += seive # 把seive拼接回去
        return ans
```

# 剑指 Offer II 076. 数组中的第 k 大的数字

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `**k**` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 利用内置堆解决，面试时候推荐手写堆,k已知不越界
        # python的内置堆正好是小根堆
        # 已知堆容量的话手写堆逻辑时采取数组存取更好
        # 找第k大的数字，那么使用小根堆只需要维护堆堆大小为k即可
        # 维护k的逻辑是，比堆顶小的都不要，则整个堆里都是大数字，堆顶最小，

        # 大根堆维护len(nums)大小的堆，弹出k个即可
        # 以小根堆为例，小根堆性能更强
        min_heap = [i for i in nums[:k]] # 取前k个      
        heapq.heapify(min_heap) # 堆化
        for i in nums[k:]: # k个之后的过筛,比堆顶小的都不要
            if i > min_heap[0]: # 比堆顶大
                heapq.heappush(min_heap,i)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        # 返回堆顶即可。整个堆里都是大数字，共k个，堆顶最小，是第k大的数字
        return min_heap[0]
        
        
```

# 剑指 Offer II 077. 链表排序

给定链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 找到它的中点
        if head == None or head.next == None:
            return head
        cur1 = head
        cur2 = self.divideList(head)
        # 处理完之后调用递归
        cur1 = self.sortList(cur1)
        cur2 = self.sortList(cur2)
        return self.merge(cur1,cur2)

    def divideList(self,head):
        if head == None or head.next == None:
            return head

        dummy = ListNode()
        dummy.next = head
        slow = dummy
        fast = dummy
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        temp = slow.next # 分离两个链表，先暂存slow.next
        slow.next = None # 断开前后链表链接
        return temp # 返回后链表的头节点
        
    def merge(self,lst1,lst2): # 递归归并
        if lst1 == None: return lst2
        if lst2 == None: return lst1
        if lst1.val < lst2.val:
            lst1.next = self.merge(lst1.next,lst2)
            return lst1
        else:
            lst2.next = self.merge(lst1,lst2.next)
            return lst2
```

```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 递归合并,另一种写法
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

# 剑指 Offer II 078. 合并排序链表

给定一个链表数组，每个链表都已经按升序排列。

请将所有链表合并到一个升序链表中，返回合并后的链表。

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

# 剑指 Offer II 079. 所有子集

给定一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # 空集可以被返回
        # 经典回溯
        ans = [[]] # 预先加入空集
        path = [] # 路径收集
        def backtracking(lst): # 为了防止重复选择，需要合理限制参数,传入参数为选择列表
            if lst == []:
                return 
            path.append(lst[0]) # 做选择
            ans.append(path[:]) # 收集路径结果
            backtracking(lst[1:]) # 只在其后面的元素进行选择,注意这一条是接着上面那条路径的结果往下搜
            path.pop() # 取消选择
            backtracking(lst[1:]) # 只在其后面的元素进行选择，注意这一条是放弃了上面那条路径往下搜

        backtracking(nums)
        return ans # 不要求顺序，直接返回即可，无需整理
```

# 剑指 Offer II 080. 含有 k 个元素的组合

给定两个整数 `n` 和 `k`，返回 `1 ... n` 中所有可能的 `k` 个数的组合。

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # 回溯
        lst = [i+1 for i in range(n)]
        ans = []

        def backtracking(index,path):
            if index == n + 1:
                return 
            if len(path) == k:
                ans.append(path[:])
                return 
            for i in range(index,n):
                path.append(lst[i])
                backtracking(i+1,path)
                path.pop()
                
        backtracking(0,[])
        return ans
```

# 剑指 Offer II 081. 允许重复选择元素的组合

给定一个无重复元素的正整数数组 candidates 和一个正整数 target ，找出 candidates 中所有可以使数字和为目标数 target 的唯一组合。

candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是唯一的。 

对于给定的输入，保证和为 target 的唯一组合数少于 150 个。

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # 回溯，无重复元素，根据剩余值凑成目标
        ans = []
        path = []
        candidates.sort() # 预先排序，
        # 收集逻辑为target == 0

        def backtracking(index,path,target):
            if index >= len(candidates) or target < 0:
                return 
            if target == 0: # 收集条件
                ans.append(path[:])
                return    
            for i in range(index,len(candidates)):  # 注意可以重复收集          
                path.append(candidates[i])  # 做选择
                backtracking(i,path,target-candidates[i])
                path.pop() # 取消选择
         
        backtracking(0,[],target)
        return ans
```

# 剑指 Offer II 082. 含有重复元素集合的组合

给定一个可能有重复数字的整数数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次，解集不能包含重复的组合。 

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # 回溯，注意去重逻辑
        candidates.sort()
        n = len(candidates)
        ans = []
        used = [False for i in range(n)]

        def backtracking(choice,path,aim,index):
            if aim == 0:
                ans.append(path.copy())
                return 
            if aim < 0:
                return 

            for i in range(index,n):
                if used[i] == True: # 这个数被用过
                    continue  # 不要手误写成return
                if i > 0 and choice[i] == choice[i-1] and used[i-1] == False: 
                    # 这里不能是used[i-1] == True，因为是顺序选取而不是可以倒着选
                    continue  # 不要手误写成return
                used[i] = True 
                path.append(choice[i])
                backtracking(choice,path,aim-choice[i],i+1) # 注意这里是i+1而不是index+1
                path.pop()
                used[i] = False 
        
        backtracking(candidates,[],target,0)
        return ans

```

# 剑指 Offer II 083. 没有重复元素集合的全排列

给定一个不含重复数字的整数数组 `nums` ，返回其 **所有可能的全排列** 。可以 **按任意顺序** 返回答案。

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # 经典回溯算法【这一题使用递归交换也可以】
        ans = [] # 收集结果
        path = [] # 收集路径
        n = len(nums)

        def backtracking(lst):  # 本题不包含重复数字,且数组较短可以使用remove直接去除下一层选择不需要的值
            if len(path) == n: # 路径达到上限，收集结果
                ans.append(path[:])
                return 
            for i in lst:
                copy_lst = lst.copy() # 需要拷贝一个数组，之后的选择在新数组中做出选择
                copy_lst.remove(i) # 新数组中不会选择之前选过的数，移除
                # 若不想使用remove，则直接用for循环拷贝，其间不拷贝选择过的数
                path.append(i) # 做选择
                backtracking(copy_lst) # 在新数组中回溯
                path.pop() # 取消选择

        backtracking(nums)
        return ans
```

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # 经典回溯算法【非数组拷贝版本】
        ans = [] # 收集结果
        path = [] # 收集路径
        n = len(nums)
        used = [False for i in range(n)]

        def backtracking(choice,path):  #
            if len(path) == n:
                ans.append(path[:])
                return 
            for i in range(n):
                if used[i] == True:
                    continue 
                used[i] = True
                path.append(choice[i])
                backtracking(choice,path)
                path.pop()
                used[i] = False 
        
        backtracking(nums,[])
        return ans
```

# 剑指 Offer II 084. 含有重复元素集合的全排列 

给定一个可包含重复数字的整数集合 `nums` ，**按任意顺序** 返回它所有不重复的全排列。

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

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # 方法二：在选择过程中，不合法的元素直接舍去，在选择中途过筛
        # 这种过筛法最好是先把元素排序,方便之后的处理
        nums.sort()
        path = []
        ans = []
        visited = [0 for i in range(len(nums))]
        n = len(nums)
        def backtracking(path,nums):
            if len(path) == n:
                ans.append(path[:]) # 传值而不是传引用
                return                
            for i in range(len(nums)):
                if visited[i] == 1: # 扫下一个数
                # 如果这个数已经使用过，跳过
                    continue
                if i > 0 and nums[i] == nums[i-1] and visited[i-1] == 1:
                # 如果这一个数和前一个数相等，并且前一个数已经使用过，则跳过
                # 画递归树的时候会发现，对于1'，1''，1'''，1''''，2，3这样的，当选择第1个1之后，第二个1一定不会被选
                # 最终收集到的顺序一定是 1'''' , 1''',1'',1',2,3，这个剪枝虽然剪了，但是不是特别充分的剪枝
                # 如果写成 if i > 0 and nums[i] == nums[i-1] and visited[i-1] == 0: 
                # 得到的是1'，1''，1'''，1''''，2，3，它的剪枝效率是更高的
                    continue
                visited[i] = 1
                path.append(nums[i])
                backtracking(path,nums)
                visited[i] = 0
                path.pop()

        backtracking(path,nums)
        return (ans)
```

# 剑指 Offer II 085. 生成匹配的括号

正整数 `n` 代表生成括号的对数，请设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

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

# 剑指 Offer II 086. 分割回文子字符串

给定一个字符串 `s` ，请将 `s` 分割成一些子串，使每个子串都是 **回文串** ，返回 s 所有可能的分割方案。

**回文串** 是正着读和反着读都一样的字符串。

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        # 类似于020 还是使用dp，dp[i][j]的含义是 s[i:j+1] 左闭右开是否为回文串
        # 初始化，然后由于需要所有的可能方案，借助dp矩阵进行dfs收集答案
        n = len(s)
        dp = [[False for j in range(n)]for i in range(n)] # 申请二维dp数组
        # 状态转移为掐头去尾是回文串且s[i] == s[j]
        # dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])，需要左下方的已知状态来进行状态转移
        # 对角线可以被初始化，它的所有元素为True
        for i in range(n):
            dp[i][i] = True 
        for i in range(n-1): # 再初始化一条线
            dp[i][i+1] = (s[i] == s[i+1])
        # 可以开始状态转移，所有空缺格子可以由已知状态来确定
        # 填充顺序为从左到右的纵列，所以外层是j循环，内层是i循环，注意边界
        for j in range(2,n):
            for i in range(0,j-1): # 注意左闭右开
                dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
        # 此时dp已经初始化完成
        ans = [] # 收集结果
        path = [] # 记录路径
        def dfs(index): # dfs深度优先遍历搜索结果，要搜索的是字符串，用索引传入
            if index == n: # 索引可以成功到最后一位
                ans.append(path[:]) # 添加至结果集
                return 
            for i in range(index,n):
                if dp[index][i] == False: # s[index:i+1]是否是回文串，是的话才有继续搜的必要
                    continue # 不能写出return，否则不会再搜索了，只能是跳过这一轮循环，搜下一轮
                else:
                    path.append(s[index:i+1]) # 注意左闭右开,做选择
                    dfs(i+1) 
                    path.pop() # 取消选择，回溯
        dfs(0) # 从0开始检查
        return ans

```

# 剑指 Offer II 087. 复原 IP 

给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能从 s 获得的 有效 IP 地址 。你可以按任何顺序返回答案。

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

# 剑指 Offer II 088. 爬楼梯的最少成本

数组的每个下标作为一个阶梯，第 i 个阶梯对应着一个非负数的体力花费值 cost[i]（下标从 0 开始）。

每当爬上一个阶梯都要花费对应的体力值，一旦支付了相应的体力值，就可以选择向上爬一个阶梯或者爬两个阶梯。

请找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 0 或 1 的元素作为初始阶梯。

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # 动态规划，可以状态压缩，
        # 如果需要状态压缩，记录dp[i-1] 和dp [i-2]即可
        # 如果len(cost) <= 2: 则返回最小值即可
        if len(cost) <= 2:
            return min(cost)
        # 动态规划问题，为了方便处理边界条件，给cost数组的尾巴加上极大值的墙,最终需要踩到墙上
        cost = cost + [0xffffffff]
        dp = [0 for i in range(len(cost))] # 申请dp数组为原长+1
        # dp[i]的含义是，踩到这一阶已经耗费的最少体力值[方便理解，把两基态除外【包含在内也可以】]
        # dp[i] = min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2]) 【当i>=2】
        # 初始化dp[0],dp[1] = 0
        for i in range(2,len(dp)):
            dp[i] = min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])
        return dp[-1] # 返回踩到墙上【即超出原边界所耗费的最少体力】

```

# 剑指 Offer II 089. 房屋偷盗

一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响小偷偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组 nums ，请计算 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 无状态压缩的常规dp写法
        # 压缩时候只需要两个临时变量代替dp[i-1],dp[i-2]即可,可以把空间降低到O(1)
        # dp[i]为到i的时候的能获取到的最大值
        # dp[i] = max(dp[i-1],dp[i-2]+nums[i]),可以发现它进行状态转移必须从i=2开始[索引不能越界]
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        dp = [0 for i in range(len(nums))] # 先申请数组
        # 再考虑初始化问题，dp[0]显然为当前值，dp[1]取俩者中较大的那个
        dp[0],dp[1] = nums[0],max(nums[0],nums[1])
        # 状态转移，填充dp数组
        for i in range(2,len(nums)):
            dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        return dp[-1] # 返回最后一个
```

# 剑指 Offer II 090. 环形房屋偷盗

一个专业的小偷，计划偷窃一个环形街道上沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组 nums ，请计算 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 打家劫舍加强版
        # 去了0则不能去n-1，那么对0～n-2使用dp
        # 不去0，那么对1～n-1使用dp
        # 两者的较大值为结果
        # 状态转移方程：dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        # 切片调用
        def calc_dp(lst,start,end):# 计算dp，start和end都是闭区间参数,lst是需要被计算的dp
            calc_lst = lst[start:end+1]
            if len(calc_lst) == 0:
                return 0
            elif len(calc_lst) == 1:
                return calc_lst[0]
            elif len(calc_lst) == 2:
                return max(calc_lst[0],calc_lst[1])
            dp = [0 for i in range(len(calc_lst))]
            dp[0] = calc_lst[0]
            dp[1] = max(calc_lst[0],calc_lst[1])

            for i in range(2,len(calc_lst)): 
                dp[i] = max(dp[i-1],dp[i-2]+calc_lst[i]) # 注意这个dp用的是calc_lst作为传入，不要写错
            return dp[-1] # 返回最后一个

        n = len(nums)
        if n == 1:
            return nums[0]
        # 否则开始dp计算
        situation1 = calc_dp(nums,0,n-2)
        situation2 = calc_dp(nums,1,n-1)
        return max(situation1,situation2)
```

# 剑指 Offer II 091. 粉刷房子

假如有一排房子，共 n 个，每个房子可以被粉刷成红色、蓝色或者绿色这三种颜色中的一种，你需要粉刷所有的房子并且使其相邻的两个房子颜色不能相同。

当然，因为市场上不同颜色油漆的价格不同，所以房子粉刷成不同颜色的花费成本也是不同的。每个房子粉刷成不同颜色的花费是以一个 n x 3 的正整数矩阵 costs 来表示的。

例如，costs\[0][0] 表示第 0 号房子粉刷成红色的成本花费；costs\[1][2] 表示第 1 号房子粉刷成绿色的花费，以此类推。

请计算出粉刷完所有房子最少的花费成本。

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        # dp数组分为三行
        n = len(costs)
        dp = [[0xffffffff for j in range(n)] for i in range(3)]
        for i in range(3):
            dp[i][0] = costs[0][i] # 初始化
        
        # 状态转移。
        # dp[k][j]为当第j次选择到k颜色时候，所花费的最小值。
        # 由于颜色不相邻，只能在另外两行中找，进行状态转移，取最小值
        for j in range(1,n):
            dp[0][j] = min(dp[1][j-1],dp[2][j-1]) + costs[j][0]
            dp[1][j] = min(dp[0][j-1],dp[2][j-1]) + costs[j][1]
            dp[2][j] = min(dp[0][j-1],dp[1][j-1]) + costs[j][2]
        
        # 取最后一列的最小值
        ans = []
        for i in range(3):
            ans.append(dp[i][-1])
        return min(ans)
```

# 剑指 Offer II 092. 翻转字符

如果一个由 '0' 和 '1' 组成的字符串，是以一些 '0'（可能没有 '0'）后面跟着一些 '1'（也可能没有 '1'）的形式组成的，那么该字符串是 单调递增 的。

我们给出一个由字符 '0' 和 '1' 组成的字符串 s，我们可以将任何 '0' 翻转为 '1' 或者将 '1' 翻转为 '0'。

返回使 s 单调递增 的最小翻转次数。

```python
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        # 单序列的二维dp
        # dp[i][j]的意思是对于前i个字符，开两行数组
        # 一行表示把0~j翻转成结尾为0的最少次数，二行表示把0～i翻转成结尾为1的最少次数
        n = len(s)
        dp = [[0xffffffff for j in range(n+1)] for i in range(2)]
        # 初始化dp[0][0] = 0 ,dp[1][0] = 0
        dp[0][0],dp[1][0] = 0,0
        for j in range(1,n+1):
            # 如果当前位置为0:
            # dp[0][j] = dp[0][j-1] # 直接继承，无需翻转
            # dp[1][j] = min(dp[0][j-1],dp[1][j-1])+1,# 需要把当前位置翻转成1，加一步
            # 如果当前位置为1： 
            # dp[0][j] = dp[0][j-1]+1 ,因为要把当前位置翻成0
            # dp[1][j] = min(dp[0][j-1],dp[1][j-1]) 直接继承，无需翻转
            if s[j-1] == "0":
                dp[0][j] = dp[0][j-1]
                dp[1][j] = min(dp[0][j-1],dp[1][j-1]) + 1
            elif s[j-1] == "1":
                dp[0][j] = dp[0][j-1]+1
                dp[1][j] = min(dp[0][j-1],dp[1][j-1])
        # 返回最后一纵列里较小值
        return min(dp[0][-1],dp[1][-1])
                
```

# 剑指 Offer II 093. 最长斐波那契数列

如果序列 X_1, X_2, ..., X_n 满足下列条件，就说它是 斐波那契式 的：

n >= 3
对于所有 i + 2 <= n，都有 X_i + X_{i+1} = X_{i+2}
给定一个严格递增的正整数数组形成序列 arr ，找到 arr 中最长的斐波那契式的子序列的长度。如果一个不存在，返回  0 。

（回想一下，子序列是从原序列  arr 中派生出来的，它从 arr 中删掉任意数量的元素（也可以不删），而不改变其余元素的顺序。例如， [3, 5, 8] 是 [3, 4, 5, 6, 7, 8] 的一个子序列）

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

# 剑指 Offer II 094. 最少回文分割

给定一个字符串 `s`，请将 `s` 分割成一些子串，使每个子串都是回文串。

返回符合要求的 **最少分割次数** 。

```python
class Solution:
    def minCut(self, s: str) -> int:
        # 先使用dp将其预先处理
        n = len(s)
        dp = [[False for i in range(n)] for j in range(n)]
        # dp[i][j] 是 s[i:j+1]是否为回文，注意左闭右开
        # dp[i][j] 是 s[i,j]闭区间是否为回文
        # 显然对角线上为回文
        for i in range(n):
            dp[i][i] = True
        # 状态转移为，掐头去尾为回文且新加入的两个字符相等
        # dp[i][j] = dp[i+1][j-1] and s[i] == s[j],转移方向为从左下到右上
        # 只有一条主对角线无法转移。需要补充主对角线的右边平行线
        for i in range(n-1):
            dp[i][i+1] = (s[i]==s[i+1])
        # 画图辅助后，从左到右，从上到下纵列填充
        for j in range(2,n):
            for i in range(0,j-1):
                dp[i][j] = dp[i+1][j-1] and s[i] == s[j]
        # 此时dp预处理完毕，可以由dp来判断s[i:j+1]是否为回文
        cuts = 0

        # 再次dp。定义数组为gp
        gp = [0xffffffff for i in range(n)]
        # gp[i]的含义是，[0,i]闭区间的最小分隔次数，初始化为极大值
        gp[0] = 0 # 一个字符显然不需要分
        for i in range(n):
            if dp[0][i] == True: # 本身是回文串，无需分隔
                gp[i] = 0
            elif dp[0][i] != True: # 本身不是回文串
            # 需要找到s[0,j]闭区间为回文串，s[j+1,i]闭区间也为回文串
                group = []
                for j in range(i):
                    if dp[j+1][i] == True:
                        group.append(gp[j])
                gp[i] = min(group) + 1
        return gp[-1]
```

# 剑指 Offer II 095. 最长公共子序列

给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # dp 动态规划就是填写表格，俩字符串当然最容易想到的二维表格
        m = len(text1) # text1竖着摆放
        n = len(text2) # text2横着摆放
        dp = [ [0 for j in range(n+1)] for i in range(m+1)] # 初始化大小为(n+1)*(m+1),这样方便边界条件处理
        # 确定dp[i][j] 的含义，即当i取text1的text1[:i]的时候和j取text2[:j]的时候最长的LCS
        # 例如。dp[1][1] 是俩字符串的第一对字母的LCS
        # 确定状态转移: 显然，取决于下一对需要比对的字符是否相等。即text1[i-1] 和 text2[j-1] 的关系
        # 1. 如果相等 dp[i][j] = dp[i-1][j-1] + 1
        # 2. 如果不相等 dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        for j in range(1,n+1):
            for i in range(1,m+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                elif text1[i-1] != text2[j-1]:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        return dp[-1][-1]
```

# 剑指 Offer II 097. 子序列的数目

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



# 剑指 Offer II 098. 路径的数目

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # dp经典问题，为了使得边界条件方便处理，制作dp矩阵时候补上左边和上边的一条边
        # dp矩阵的大小为 (m+1) * (n+1)
        # dp[i][j]是到m行n个的路径方案数目
        # 状态转移为 dp[i][j] = dp[i-1][j] + dp[i][j-1]
        # 先dp[0]全部初始化为0，左边界全部初始化为[0]
        dp = [[0 for j in range(n+1)] for i in range(m+1)]
        # 考虑到转移，把dp[0][1] = 1
        dp[0][1] = 1 # 注意这个初始化条件
        # 按照一行一行来填充       
        for i in range(1,m+1):
            for j in range(1,n+1):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1] # 返回右下角的值即可
```

# 剑指 Offer II 099. 最小路径之和

给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：一个机器人每次只能向下或者向右移动一步。

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # 二维dp，dp[i][j]的含义是，到达第i行第j个的时候，所累加的最小路径
        # 为了方便状态转移，补上左边边界和右边边界
        m = len(grid)
        n = len(grid[0])
        dp = [[0xffffffff for j in range(n+1)] for i in range(m+1)] # 申请dp数组
        # 状态转移方程为dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i-1][j-1]
        # 注意dp比grid大一圈
        dp[0][1] = 0 # 初始化dp的第0行第1个为0，使得在grid中具有填充入口
        # 按照一行行来填充
        for i in range(1,m+1):
            for j in range(1,n+1):
                dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i-1][j-1]
        return dp[-1][-1] # 返回右下角的数即可
```

# 剑指 Offer II 100. 三角形中最小路径之和

给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 方法1: 先考虑常规方法二维dp，先大空间dp，空间复杂度为O（n^2) 自顶向下
        # dp[i][j] 的含义是，到了triangle[i][j]的位置上当前的最小路径和
        # 状态转移方程dp[i][j] = min(dp[i-1][j],dp[i-1][j-1]) + triangle[i][j]
        # 注意处理越界问题
        # 先初始化二维dp为极大值
        n = len(triangle) # 行数
        dp = [[0xffffffff for j in range(n)] for i in range(n)]
        # 显然入口dp[0][0] = triangle[0][0]
        dp[0][0] = triangle[0][0]
        # 开始填充dp数组，注意填充范围只需要填充左下角的三角形
        for i in range(1,n): # 从索引1行开始扫起即可，
            for j in range(0,i+1):
                if j - 1 < 0: # 回顾状态转移方程，其索引越界的时候的情况
                    dp[i][j] = dp[i-1][j] + triangle[i][j]
                else:
                    dp[i][j] = min(dp[i-1][j],dp[i-1][j-1]) + triangle[i][j]
        # 最后一行的每个出口位置都可以是出口，找最后一行的最小值为答案
        return min(dp[-1])
        # 实际上状态转移只需要上一行的数据，那么进行状态压缩仅仅保留上一行即可完成O(n)的空间复杂度
```

# 剑指 Offer II 101. 分割等和子串

给定一个非空的正整数数组 `nums` ，请判断能否将这些数字分成元素和相等的两部分。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # 子集背包问题
        sumNum = sum(nums)
        if sumNum % 2 == 1:
            return False 
        n = len(nums)
        sumNum //= 2
        # dp数组外加一圈
        # dp[i][j]的含义是，选取前i个数，是否能凑到j
        # 状态转移方程: dp[i][j] 不选那个数，则dp[i][j] = dp[i-1][j]
        # 选取第i个数，dp[i][j] = dp[i-1][j-nums[i-1]]。第四个数的索引是3，第i个数的索引是i-1
        dp = [[False for j in range(sumNum+1)] for i in range(n+1)]
        # 初始化
        # dp[i][0]为true。代表不选总能得到true
        for i in range(n+1):
            dp[i][0] = True
        for i in range(1,n+1):
            for j in range(1,sumNum+1):
                if j-nums[i-1] < 0: # 越界，塞不下，只能放弃塞
                    dp[i][j] = dp[i-1][j]
                elif j-nums[i-1] >= 0:
                    dp[i][j] = (dp[i-1][j] or dp[i-1][j-nums[i-1]]) # 有一个成立都是True
        return dp[-1][-1]
```

# 剑指 Offer II 102. 加减的目标值

给定一个正整数数组 nums 和一个整数 target 。

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

# 剑指 Offer II 103. 最少的硬币数目

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

```python
## 超时
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp动态规划
        # 二维dp,dp[i][j]代表前n个硬币凑总数为j的最少硬币个数
        n = len(coins)
        dp = [[0xffffffff for j in range(amount+1)] for i in range(n+1)]
        # 有dp[i][0] = 0
        for i in range(n+1):
            dp[i][0] = 0
        # 状态转移方程为：dp[i][j] = group{dp[i-1][j-k*coins[i-1]]+k}
        for i in range(1,n+1):
            for j in range(amount+1):
                group = []
                k = 0 # 从0开始
                while j - k*coins[i-1] >= 0:
                    group.append(dp[i-1][j-k*coins[i-1]]+k)
                    k += 1
                if len(group) != 0:
                    dp[i][j] = min(group)
        
        return dp[-1][-1] if dp[-1][-1] != 0xffffffff else -1
```

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 状态压缩
        # dp[j]的含义是凑出j的最小硬币数
        dp = [0xffffffff for j in range(amount+1)]
        dp[0] = 0 # 
        # 状态转移为dp[j] = dp[j-coins[i]*k]+k
        for coin in coins:
            for j in range(coin,amount+1):
                dp[j] = min(dp[j],dp[j-coin]+1) # 这里不必采取k的形式
        return dp[-1] if dp[-1] != 0xffffffff else -1
```

```go
//go翻译版
func coinChange(coins []int, amount int) int {
    dp := make([]int,amount+1)
    inf := 999999
    for j:=0;j<amount+1;j++ {
        dp[j] = inf
    }
    dp[0] = 0 // 代表凑出0的需要数量为0

    min := func(a,b int)int{
        if a < b {
            return a
        } else {
            return b
        }
    }
    
    for _,coin := range coins {
        for j := coin; j < amount + 1; j ++ {
            dp[j] = min(dp[j],dp[j-coin]+1)
        }
    }
    if dp[amount] == inf {
        return -1
    } else {
        return dp[amount]
    }
}
```

# 剑指 Offer II 104. 排列的数目

给定一个由 不同 正整数组成的数组 nums ，和一个目标整数 target 。请从 nums 中找出并返回总和为 target 的元素组合的个数。数组中的数字可以在一次排列中出现任意次，但是顺序不同的序列被视作不同的组合。

题目数据保证答案符合 32 位整数范围。

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # 顺序不同视为不同组合，那么遍历的时候需要考虑for的两层顺序
        # for 背包
        #     for 物品 。其中不保证物品先后顺序，从而得到的是排列数
        # 
        # for 物品
        #       for 背包，限制了物品的先后次序，从而得到的是组合数
        
        dp = [0 for i in range(target+1)]
        dp[0] = 1
        for j in range(1,target+1):
            for n in nums:
                if j-n>=0:
                    dp[j] = dp[j] + dp[j-n]
        return dp[-1]
```

```go
func combinationSum4(nums []int, target int) int {
    // 一维dp
    dp := make([]int,target+1)
    dp[0] = 1 // 初始化
    // 确定遍历顺序
    // 方案1: 求组合数
    // for 物品{
    //    for 价值 
    //    dp[价值] += dp[价值-物品]， 
    //   这样遍历出来的时候，物品一定严格按照相对顺序
    //  例如，用[1，3]凑4， 3一定在1的后面  
    // 方案2: 求排列数
    // for 价值
    //     for 物品
    //    dp[价值] += dp[价值-物品]
    
    // 本题实际上求的是排列数目【数学里面的排列】
    // dp[j]的含义是凑出j的排列数目个数
    for j:=0 ; j < target+1; j ++ {
        for _,n := range nums {
            if j-n >= 0 {
                dp[j] += dp[j-n]
            }
        }
    }
    return dp[target]
}
```

# 剑指 Offer II 105. 岛屿的最大面积

给定一个由 0 和 1 组成的非空二维数组 grid ，用来表示海洋岛屿地图。

一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # 超级dfs
        maxArea = 0
        direc = [(0,1),(1,0),(-1,0),(0,-1)]
        area = 0
        m = len(grid)
        n = len(grid[0])
        visited = [[False for j in range(n)] for i in range(m)]

        def judgeValid(i,j):
            if 0<=i<m and 0<=j<n and grid[i][j] == 1: # 注意这里是数字1。。
                return True
            return False
        
        def dfs(i,j):
            nonlocal area 
            if not judgeValid(i,j):
                return 
            if visited[i][j] == True:
                return 
            visited[i][j] = True
            area += 1
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                dfs(new_i,new_j)
        
        for i in range(m):
            for j in range(n):
                dfs(i,j) # 调用,调用之后面积变更
                if area != 0: # 如果面积变更了，收集结果
                    maxArea = max(maxArea,area) # 取值
                    area = 0 # 重置
        
        return maxArea
```

# 剑指 Offer II 106. 二分图

存在一个 无向图 ，图中有 n 个节点。其中每个节点都有一个介于 0 到 n - 1 之间的唯一编号。

给定一个二维数组 graph ，表示图，其中 graph[u] 是一个节点数组，由节点 u 的邻接节点组成。形式上，对于 graph[u] 中的每个 v ，都存在一条位于节点 u 和节点 v 之间的无向边。该无向图同时具有以下属性：

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

# 剑指 Offer II 107. 矩阵中的距离

给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

```python
要求找最短，第一反应就是使用BFS方法。
朴素想法：按照题目要求，对所有的1，找到最近的0。
如果采用bfs寻找，并且找到结果就终止bfs的方法，需要以每个1为起点bfs。需要多次调用bfs。超时【我试过了】

然后因为没有学过超级源点这个概念以自己的水平实在优化不动了参考了官解

思路改进：如果用0为起点，找到所有的1，并且bfs时候是更新而不是终止，则只需要调用一次bfs。【自己写的时候想到了路途中更新，但是写不出来具体函数】

细节：队列中初始化放入的是所有0的坐标。其实visited数组和ansMat可以合并。但是为了代码易读性没有合并。

```

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        # BFS
        # 题目保证了至少有一个0
        queue = []
        m = len(mat)
        n = len(mat[0])
        visited = [[False for j in range(n)] for i in range(m)]

        # 题目要求对所有的1，找到最近的0， 如果采用bfs寻找，并且找到结果就终止bfs的方法 以每个1为起点bfs。需要多次调用bfs。超时【我试过了】，然后因为没有学过超级源点这个概念以自己的水平实在优化不动了参考了官解
        # 思路修改为：
        # 如果用0为起点，找到所有的1，并且bfs是更新而不是终止，则只需要调用一次bfs
        # 

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0: # 以0为起点
                    queue.append((i,j))
                    visited[i][j] = True # 注意这一行
        
        direc = [(0,1),(0,-1),(1,0),(-1,0)]

        ansMat = [[0 for j in range(n)] for i in range(m)]
        steps = 0

        while len(queue) != 0:
            new_queue = []
            for pair in queue:
                x,y = pair
                if mat[x][y] == 1 and visited[x][y] == False:
                    # 注意这里还需要ansMat没有被赋值过，防止覆盖
                    # 可以用if mat[x][y] == 1 and ansMat[x][y] == 0:
                    # 也可以用visited[x][y] == False来判断
                    ansMat[x][y] = steps
                    visited[x][y] = True
                for di in direc:
                    new_x = x + di[0]
                    new_y = y + di[1]
                    if 0<=new_x<m and 0<=new_y<n and visited[new_x][new_y] == False:
                        new_queue.append((new_x,new_y))
            steps += 1
            queue = new_queue

        return ansMat
```

# 剑指 Offer II 108. 单词演变

在字典（单词列表） wordList 中，从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：

序列中第一个单词是 beginWord 。
序列中最后一个单词是 endWord 。
每次转换只能改变一个字母。
转换过程中的中间单词必须是字典 wordList 中的单词。
给定两个长度相同但内容不同的单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # 方法1: 单向bfs
        word_set = set(wordList)
        if len(word_set) == 0 or endWord not in word_set:
            return 0
        
        queue = [beginWord]
        steps = 1
        visited = set(beginWord)
        n = len(beginWord)

        while len(queue) != 0:
            new_queue = []
            for word in queue:
                if word == endWord:
                    return steps
                new_word_lst = list(word)
                for i in range(n):
                    originChar = new_word_lst[i] # 暂存
                    for t in range(26):
                        change = chr(ord('a')+t) # 变更
                        new_word_lst[i] = change
                        tempWord = ''.join(new_word_lst)
                        if tempWord in word_set:
                            if tempWord not in visited:
                                visited.add(tempWord)
                                new_queue.append(tempWord)
                    new_word_lst[i] = originChar # 复原
            queue = new_queue 
            steps += 1
        return 0
```



# 剑指 Offer II 109. 开密码锁

一个密码锁由 4 个环形拨轮组成，每个拨轮都有 10 个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。

列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 target 代表可以解锁的数字，请给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。

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
                for temp in tempList:
                    if temp not in deadends:
                        new_queue.append(temp)
                        deadends.add(temp)
            queue = new_queue
            steps += 1
        return -1

```

# 剑指 Offer II 110. 所有路径

给定一个有 n 个节点的有向无环图，用二维数组 graph 表示，请找到所有从 0 到 n-1 的路径并输出（不要求按顺序）。

graph 的第 i 个数组中的单元都表示有向图中 i 号节点所能到达的下一些结点（译者注：有向图是有方向的，即规定了 a→b 你就不能从 b→a ），若为空，就是没有下一个节点了。

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # 注意这一题是有向图，则无需使用visited集合来辅助是否之间探查过
        # 因为【 0 -》 1 和 1 -〉0 是两个不同的路线】
        # 方法1: DFS
        self.ans = [] # 收集结果
        path = [0] # 记录路径
        def dfs(path):
            if path[-1] == len(graph) - 1: # 终点是 n - 1，到达中点收集路径
                self.ans.append(path[:]) # 注意是传值而不是传引用
                return 
            neighbors = graph[path[-1]] # 记录当前路径的最后一个节点的邻居
            for neigh in neighbors:
                path.append(neigh) # 选择它加入
                dfs(path) # 在加入的基础上进行dfs
                path.pop() # 本个邻居的任务已经完成了，开启下一次for循环前它就要出去了
        
        dfs(path)
        return self.ans

```

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # BFS解决,队列里存的是路径
        path = [0]
        self.ans = []
        queue = [path]
        while len(queue) != 0:
            e = queue.pop(0) # 这个元素是路径
            if e[-1] == len(graph) - 1:
                self.ans.append(e[:])
            neighbors = graph[e[-1]] # 路径的最后一个元素的邻居需要加入
            for neigh in neighbors: # 每一个尾巴的邻居，都和原队列组装好之后加入
                queue.append(e + [neigh]) # 这一着重理解一下
        return self.ans


```

# 剑指 Offer II 111. 计算除法

给定一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。

另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。

返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。

注意：输入总是有效的。可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # 约分后即相当于找邻居,先转化成图
        # 已知不存在矛盾。【如果存在矛盾这一题需要收集所有可行路径然后比对所有路径值是否相等】

        edges = collections.defaultdict(list)
        n = len(equations)
        for i in range(n):
            pair = equations[i]
            edges[pair[0]].append((pair[1],values[i]))
            edges[pair[1]].append((pair[0],1/values[i]))
        
        # 由于需要记录路径权重，所以使用dfs
        def dfs(start,end,visited): # 回溯
            if start not in edges or end not in edges:
                return -1
            if start == end:
                return 1
            visited.add(start) # 标记为访问
            for neigh in edges[start]:
                theNeighbor = neigh[0] # 节点名
                if theNeighbor not in visited:
                    val = dfs(theNeighbor,end,visited)
                    if val > 0:
                        return neigh[1] * val # neigh[1]为节点值
            visited.remove(start) # 取消标记
            return -1

        ans = [] # 收集答案用            
        for q in queries:
            start = q[0]
            end = q[1]
            value = dfs(start,end,set())
            ans.append(value)
        return ans
```

# 剑指 Offer II 112. 最长递增路径

给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        # 记忆化搜索。一种后续遍历的思想
        direc = [(0,1),(0,-1),(1,0),(-1,0)]
        m = len(matrix)
        n = len(matrix[0])
        grid = [[0 for j in range(n)] for i in range(m)] # 存的是以该格子为起点的最长

        def dfs(i,j):
            if grid[i][j] != 0: # 说明它已经被更新过了，直接返回这个值就行
                return grid[i][j]
            group = [1] 
            for di in direc:
                new_i = i + di[0]
                new_j = j + di[1]
                if 0<=new_i<m and 0<=new_j<n and matrix[new_i][new_j] > matrix[i][j]:
                    group.append(dfs(new_i,new_j)+1)
            grid[i][j] = max(group)
            return grid[i][j]
        
        longest = 1
        # 遍历每个格子
        for i in range(m): 
            for j in range(n):
                grid[i][j] = dfs(i,j)
                longest = max(longest,grid[i][j])
        return longest
```

# 剑指 Offer II 113. 课程顺序

现在总共有 numCourses 门课需要选，记为 0 到 numCourses-1。

给定一个数组 prerequisites ，它的每一个元素 prerequisites[i] 表示两门课程之间的先修顺序。 例如 prerequisites[i] = [ai, bi] 表示想要学习课程 ai ，需要先完成课程 bi 。

请根据给出的总课程数  numCourses 和表示先修顺序的 prerequisites 得出一个可行的修课序列。

可能会有多个正确的顺序，只要任意返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

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

# 剑指 Offer II 114. 外星文字典

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

# [剑指 Offer II 115. 重建序列](https://leetcode-cn.com/problems/ur2n8P/)

请判断原始的序列 org 是否可以从序列集 seqs 中唯一地 重建 。

序列 org 是 1 到 n 整数的排列，其中 1 ≤ n ≤ 104。重建 是指在序列集 seqs 中构建最短的公共超序列，即  seqs 中的任意序列都是该最短序列的子序列。

```python
class Solution:
    def sequenceReconstruction(self, org: List[int], seqs: List[List[int]]) -> bool:
        # 即考虑拓扑排序是否唯一
        # 子序列不要求连续
        n = len(org)
        graph = collections.defaultdict(list)
        inDegree = collections.defaultdict(int)
        for i in range(n): # 激活全部
            inDegree[i+1]
            graph[i+1]

        visited = set()
        # a -> b -> c -> d 
        for each in seqs:
            p = 1
            while p < len(each):
                ch1 = each[p-1]
                ch2 = each[p]
                graph[ch1].append(ch2)
                inDegree[ch2] += 1 
                p += 1
            for ch in each:
                visited.add(ch)
        
        # seq中会存在没有出现过的数
        # if len(visited) != n:
        #     return False 
        if visited != set(org):
            return False 
        
        ans = []
        queue = []
        
        for key in inDegree:
            if inDegree[key] == 0:
                queue.append(key)
        #print(graph,inDegree)
        while len(queue) != 0:     
            new_queue = []
            #print(queue)
            if len(queue) != 1:
                return False 
            for node in queue:
                ans.append(node)
                for neigh in graph[node]:
                    inDegree[neigh] -= 1
                    if inDegree[neigh] == 0:
                        new_queue.append(neigh)         
            queue = new_queue 
        # print(ans,org)
        return ans == org
                
                
```



# 剑指 Offer II 116. 朋友圈

一个班上有 n 个同学，其中一些彼此是朋友，另一些不是。朋友关系是可以传递的，如果 a 与 b 直接是朋友，且 b 与 c 是直接朋友，那么 a 与 c 就是间接朋友。

定义 朋友圈 就是一组直接或者间接朋友的同学集合。

给定一个 n x n 的矩阵 isConnected 表示班上的朋友关系，其中 isConnected[i][j] = 1 表示第 i 个同学和第 j 个同学是直接朋友，而 isConnected[i][j] = 0 表示二人不是直接朋友。

返回矩阵中 朋友圈的数量。

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

# 剑指 Offer II 117. 相似的字符串

如果交换字符串 X 中的两个不同位置的字母，使得它和字符串 Y 相等，那么称 X 和 Y 两个字符串相似。如果这两个字符串本身是相等的，那它们也是相似的。

例如，"tars" 和 "rats" 是相似的 (交换 0 与 2 的位置)； "rats" 和 "arts" 也是相似的，但是 "star" 不与 "tars"，"rats"，或 "arts" 相似。

总之，它们通过相似性形成了两个关联组：{"tars", "rats", "arts"} 和 {"star"}。注意，"tars" 和 "arts" 是在同一组中，即使它们并不相似。形式上，对每个组而言，要确定一个单词在组中，只需要这个词和该组中至少一个单词相似。

给定一个字符串列表 strs。列表中的每个字符串都是 strs 中其它所有字符串的一个 字母异位词 。请问 strs 中有多少个相似字符串组？

字母异位词（anagram），一种把某个字符串的字母的位置（顺序）加以改换所形成的新词。

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



# 剑指 Offer II 118. 多余的边

树可以看成是一个连通且 无环 的 无向 图。

给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间，且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges ，edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。

请找出一条可以删去的边，删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案，则返回数组 edges 中最后出现的边。

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
```

```python
class UnionFind: # 并查集
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
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        # 并查集方法
        # 由于提示里有：n == edges.length 即并查集方法仅仅需要删去一条边
        # 使用并查集时，成的第一个环就是需要删去的最后一条边
        uf = UnionFind(len(edges))
        for e in edges:
            if not uf.is_connect(e[0]-1,e[1]-1): # 并的时候使用序号-1
                uf.union(e[0]-1,e[1]-1)
            else:
                return [e[0],e[1]] # 返回的时候返回原序号
```

# 剑指 Offer II 119. 最长连续序列

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

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

# 附加力扣勋章登记查询

```python
#!/usr/bin/env python3

import json
import requests

# 力扣目前勋章的配置
RATING = 1600
GUARDIAN = 0.05
KNIGHT = 0.25
# 查询的地址为全国还是全球？很关键
GLOBAL = False
# 二分查找的右端点(可自调)
RIGHT = 3000


class RankingCrawler:
    URL = 'https://leetcode.com/graphql' if GLOBAL else 'https://leetcode-cn.com/graphql'

    _REQUEST_PAYLOAD_TEMPLATE = {
        "operationName": None,
        "variables": {},
        "query":
r'''{
    localRanking(page: 1) {
        totalUsers
        userPerPage
        rankingNodes {
            currentRating
            currentGlobalRanking
        }
    }
}
''' if not GLOBAL else
r'''{
    globalRanking(page: 1) {
        totalUsers
        userPerPage
        rankingNodes {
            currentRating
            currentGlobalRanking
        }
    }
}
'''
    }

    def fetch_lastest_ranking(self, mode):
        l, r = 1, RIGHT
        retry_cnt = 0
        ansRanking = None
        while l < r:
            cur_page = (l + r + 1) // 2
            try:
                payload = RankingCrawler._REQUEST_PAYLOAD_TEMPLATE.copy()
                payload['query'] = payload['query'].replace('page: 1', 'page: {}'.format(cur_page))
                resp = requests.post(RankingCrawler.URL,
                    headers = {'Content-type': 'application/json'},
                    json = payload).json()

                resp = resp['data']['localRanking'] if not GLOBAL else resp['data']['globalRanking']
                # no more data
                if len(resp['rankingNodes']) == 0:
                    break
                if not mode:
                    top = int(resp['rankingNodes'][0]['currentRating'].split('.')[0])
                    if top < RATING:
                        r = cur_page - 1
                    else:
                        l = cur_page
                        ansRanking = resp['rankingNodes']
                else:
                    top = int(resp['rankingNodes'][0]['currentGlobalRanking'])
                    if top > mode:
                        r = cur_page - 1
                    else:
                        l = cur_page
                        ansRanking = resp['rankingNodes']

                print('The first contest current rating in page {} is {} .'.format(cur_page, resp['rankingNodes'][0]['currentRating']))
                retry_cnt = 0
            except:
                # print(f'Failed to retrieved data of page {cur_page}...retry...{retry_cnt}')
                retry_cnt += 1
        ansRanking = ansRanking[::-1]
        last = None
        if not mode:
            while ansRanking and int(ansRanking[-1]['currentRating'].split('.')[0]) >= 1600:
                last = ansRanking.pop()
        else:
            while ansRanking and int(ansRanking[-1]['currentGlobalRanking']) <= mode:
                last = ansRanking.pop()
        return last


if __name__ == "__main__":
    crawler = RankingCrawler()
    ans = crawler.fetch_lastest_ranking(0)
    n = int(ans['currentGlobalRanking'])
    guardian = crawler.fetch_lastest_ranking(int(GUARDIAN * n))
    knight = crawler.fetch_lastest_ranking(int(KNIGHT * n))
    if not GLOBAL:
        guardian['currentCNRanking'] = guardian['currentGlobalRanking']
        guardian.pop('currentGlobalRanking')
        knight['currentCNRanking'] = knight['currentGlobalRanking']
        knight.pop('currentGlobalRanking')

    print("Done!")
    print()
    print("目前全{}1600分以上的有{}人".format("球" if GLOBAL else "国",n))
    print("根据这个人数，我们得到的Guardian排名及分数信息为:{}".format(guardian))
    print("根据这个人数，我们得到的Knight排名及分数信息为:{}".format(knight))

作者：Benhao
链接：https://leetcode-cn.com/circle/discuss/6gnvEj/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


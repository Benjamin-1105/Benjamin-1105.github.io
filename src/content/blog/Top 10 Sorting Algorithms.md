---
title: "十大排序算法"
description: "Lorem ipsum dolor sit amet"
pubDate: "Mar 05 2024"
---

## 一、冒泡排序：

> **1、思路：**通过对待排序序列从前向后（从下标较小的元素开始）,依次对相邻两个元素的值进行两两比较，若发现前一个数大于后一个数则交换，使值较大的元素逐渐从前移向后部，就如果水底下的气泡一样逐渐向上冒。
> 
> **2、先以一个数组讲解一下，然后再写代码：**
> 
>       待排序数组：3，9，-1，10，20
> 
>        第一轮排序：
> 
>         （1）3，9，-1，10，20      ----3跟9比较，不交换
> 
>         （2）3，-1，9，10，20      ----9比 -1大，所以9跟 -1交换
> 
>         （3）3，-1，9，10，20      ----9跟10比较，不交换
> 
>         （4）3，-1，9，10，20      ----10跟20比较，不交换
> 
>            第一轮过后，将20这个最大的元素固定到了最后的位置。
> 
>            在第二轮的时候20不参与冒泡。
> 
>        第二轮排序：
> 
>             因为20的位置已经固定，所以只对前4个进行排序即可：
> 
>         （1）-1，3，9，10，20      ----3比 -1大，进行交换
> 
>         （2）-1，3，9，10，20      ----3跟9比较，不交换
> 
>         （3）-1，3，9，10，20      ----9跟10比较，不交换
> 
>             第二轮过后，将第二大的元素固定到了倒数第二个位置
> 
>        第三轮排序：
> 
>             10和20的位置已经确定，只需对前三个进行排序
> 
>         （1）-1，3，9，10，20      ----3和-1比较，不交换
> 
>         （2）-1，3，9，10，20      ----3和9比较，不交换
> 
>             第三轮过后，将第三大的元素位置确定
> 
>        第四轮排序：
> 
>             只对前两个元素进行排序
> 
>         （1）-1，3，9，10，20      ----3和-1比较，不交换
> 
>        第四轮过后，将第四大的元素位置确定，此时总共5个元素，已经排序好4个，从而最后一个自然而然就是排好序的了
> 
> 小结：
> 
> 设总的元素个数为n，那么由上边的排序过程可以看出：
> 
> （1）总计需要进行（n-1）轮排序，也就是（n-1）次大循环
> 
> （2）每轮排序比较的次数逐轮减少
> 
> （3）如果发现在某趟排序中，没有发生一次交换， 可以提前结束冒泡排序
> 
> （4）时间复杂度是O(N^2)      在有序的时候，很快，因为有exchange变量优化了代码
> 
>          在乱序的时候很慢很慢。

```c
#include<stdio.h>
void swap(int* a, int* b)
{
	int tmp = *a;
	*a = *b;
	*b = tmp;
}
//冒泡排序
void BubbleSort(int* a, int n)
{
	int end = n - 1;//不能是n，不然会越界
	while(end)
	{
		int exchange = 0;//优化，比较之后没有交换，说明已经排好了，就break循环
		for (int i = 0; i < end; i++)
		{
			if (a[i] < a[i + 1])
			{
				swap(&a[i], &a[i + 1]);
				exchange++;
			}
		}
		if (exchange == 0) break;
		end--;
	}
}
```
<img
  src="https://i-blog.csdnimg.cn/blog_migrate/bfe2e655008710a852274f86240c6963.gif"
  referrerPolicy="no-referrer"
  alt=""
/>


___

## 二、插入排序：

> **1、思路：**
> 
>         在待排序的元素中，假设前n-1个元素已有序，现将第n个元素插入到前面已经排好的序列中，使得前n个元素有序。按照此法对所有元素进行插入，直到整个序列有序。
> 
>   但我们并不能确定待排元素中究竟哪一部分是有序的，所以我们一开始只能认为第一个元素是有序的，依次将其后面的元素插入到这个有序序列中来，直到整个序列有序为止。
> 
> **2、举例：**
> 
>         如下图的插入扑克牌，当摸到7的时候，会不自觉的与前面的数比较，如果比7大，把大的数向后挪动（swap），然后在第一个小于7的后面插入7
<img
  src="https://i-blog.csdnimg.cn/blog_migrate/0f1657fb7d6a355cdfddc24902f52cb1.png"
  referrerPolicy="no-referrer"
  alt=""
/>
<img
  src="https://i-blog.csdnimg.cn/blog_migrate/7b23514289fd4c0fa3eeb9343a58baad.gif"
  referrerPolicy="no-referrer"
  alt=""
/>

```c
//插入排序
void InsertSort(int* a, int n)
{
	for (int i = 1; i < n; i++)
	{
		if (a[i] < a[i - 1])//先判断，如果i下标的值大于前面的数，就不进入
		{
			int tmp = a[i];
			int j;
			for (j = i - 1; j >= 0 && a[j] >tmp; j--)
			{
				a[j+1] = a[j];
			}
			a[j+1] = tmp;
		}
	}
}
//两次循环就可以实现
//内部循环完成一趟的插入
//外层循环完成插入排序
```

___

## 

## 三、选择排序：

> **思路：**
> 
> 1.内层循环一趟找出最小值的下标，与第一个数交换。重复找小，交换的两个操作。
> 
> 2.实际上，我们可以一趟选出两个值，一个最大值一个最小值，然后将其放在序列开头和末尾，这样可以使选择排序的效率快一倍。
> 
> 但时间复杂度还是O（N^2），效率还是不高

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/5d50f9a16c5bb410be12eee453d31ef2.gif"
  referrerPolicy="no-referrer"
  alt=""
/>
```c
//选择排序
void SelectSort(int* a, int n)
{
	for (int i = 0; i < n-1; i++)//i<n-1当它是最后一个数的时候不需要进行交换排序
	{
		int min = i;
		int j;
		for (j = i; j < n; j++)
		{
			if (a[j] < a[min])
			{
				min=j;
			}
		}
		swap(&a[i], &a[min]);//交换函数，前面的代码中有出现，我就不重复写了
	}
}
```

___

## 

## 四、希尔排序：

> **思路：**
> 
> 1.插入排序的优化版，有一个预排序的过程。让大的数快速的跳到后面，小的数快速的跳到前面。
> 
> 2.使待排序列接近有序，然后再对该序列进行一次插入排序。
> 
> 3.相当于把直接插入排序中的1换成gap而已。 

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/93fae629acfc03414c2645e4c160ff01.gif"
  referrerPolicy="no-referrer"
  alt=""
/>
```c
//希尔排序
/*步骤：
1.先选定一个小于N的整数gap作为第一增量，然后将所有距离为gap的元素分在同一组，并对每一组的元素进行直接插入排序。然后再gap--，重复上述操作。
2.当gap==1时就是直接插入排序，就相当于整个序列被分到一组，进行一次直接插入排序，排序完成。*/
 
void ShellSort(int* a, int n)
{
	//这里相当于把插入排序的1换成gap
	int gap = n;
	while (gap>1)
	{
		gap = gap / 3 + 1;
		for (int i = gap; i < n; i++)
		{
			if (a[i] < a[i - gap])
			{
				int tmp = a[i];
				int j;
				for (j = i - gap; j >= 0 && a[j] > tmp; j-=gap)//这里是j-=gap
				{
					a[j + gap] = a[j];
				}
				a[j + gap] = tmp;
			}
		}
	}
}
```

___

## 

## 五、堆排序：

> **先来认识堆：**
> 
>    1.什么是堆？
> 
>           大堆：父亲大于儿子          小堆：父亲小于儿子（父亲，儿子是二叉树的概念）
> 
>    2.堆的物理结构和逻辑结构？
> 
>           物理结构：数组         逻辑结构：完全二叉树

> **堆排序包括建堆（向下调整+循环）  堆排序（交换+向下调整）**
> 
>   1.建堆：
> 
>         要建大堆，堆顶的元素和最后一个数交换，然后把size--，就不会破坏堆的结构
> 
>   2.向下调整算法：
> 
>         比较两个孩子的大小，选出大的孩子，与父亲比较，如果孩子大于父亲，交换。然后把parent=child，child=parent\*2+1；向下调整算法一共会调整h-1次

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/f81018d0dccacef5e724a0c54499d3cc.gif"
  referrerPolicy="no-referrer"
  alt=""
/>
```c
//向下调整算法（要满足它下面的都满足堆，才能用）
void AdjustDown(int* a, int n, int root)
{
	int parent = root;
	int child = parent * 2 + 1;
	while (child < n)
	{
		if (child + 1 < n && a[child] < a[child + 1]) child+=1;//把他移到右孩子那里
		if (a[child] > a[parent])
		{
			swap(&a[child], &a[parent]);
			parent = child;
			child = parent * 2 + 1;
		}
		else break;
	}
}
 
堆排序
void HeapSort(int* arr, int n)
{
	//建大堆
    //从最后一个根开始，就相当于它下面的都满足堆，就可以用向下调整算法
	for (int i = (n-1-1)/2; i >= 0; i--)//n-1-1是因为数组的最后一个元素下标是n-1
	{
		AdjustDown(arr, n, i);
	}
	//排序
	for (int i = n; i > 1; i--)
	{
		swap(&arr[0],&arr[i - 1]);
		AdjustDown(arr, i-1, 0);
	}
}
```

___

## 

## 六、快速排序：

> **三种快排方法：**（一定要自己尝试着去写，会有一些坑，自己写才可以体会）
> 
> 1.挖坑法
> 
> 2.左右指针法
> 
> 3.前后指针法

___

> ### 6.1挖坑法：
> 
> **1.思想：**
> 
>         记第一个数为key，要调整key的位置，使得左边的都要比key的小，右边的数都比key的大。
> 
> **2.步骤：**
> 
>         选出一个数据（一般是最左边或是最右边的）存放在key变量中，在该数据位置形成一个坑
> 
>         还是定义一个left和一个right，left从左向右走（当遇到大于key的值时停下来）。right从右向左走（当遇到小于key的值时停下来）。（若在最左边挖坑，则需要right先走；若在最右边挖坑，则需要left先走） 
> 
>         把right的那个小的数放在坑中，在把left那个位置的值放在right那个位置中
> 
>         重复操作，直到left>right时结束，完成一趟，把key放在了正确的位置
> 
>         最后用分治思想，分成左边和右边，递归。

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/ba616900a5db7f10dee0484bb5ca393d.gif"
  referrerPolicy="no-referrer"
  alt=""
/>
```c
//1.挖坑法的快速排序
void QuickSort(int* a,int left,int right)
{
	if (left >= right)//不能写成pivot==left，pivot-1与left不匹配，会报错
	{
		return;
	}
	int begin = left,end = right;
	int key = a[begin];//挖了一个关键字
	int pivot = begin;//挖了一个坑
	while (begin < end)
	{
		//右边找小，一定要先右边找小,放在pivot
		while (begin < end&&a[end] >= key)//在这里也要判断begin < end,因为这里面end--
		{
			end--;
		}
		//小的放在左边的坑里，然后形成新的坑位
		a[pivot] = a[end];
		pivot = end;
		//左边找大
		while (begin < end && a[begin] <= key)
		{
			begin++;
		}
		a[pivot] = a[begin];
		pivot = begin;	
	}
	//begin==end	
	a[pivot] = key;
 
	//[left,right]
	//[left,pivot-1]  pivot  [pivot+1,right]
	//如果左子区间和右子区间都有序，就全部有序。那就分治递归。
	QuickSort(a, left, pivot - 1);
	QuickSort(a, pivot+1, right);
}
```

___

> ### 6.2左右指针法
> 
> **思路：**
> 
> 1、选出一个key，一般是最左边或是最右边的。
> 
> 2、定义一个begin和一个end，begin从左向右走，end从右向左走。（需要注意的是：若选择最左边的数据作为key，则需要end先走；若选择最右边的数据作为key，则需要bengin先走）（考虑到最后的时候相遇点的和key交换）。
> 
> 3、在走的过程中，若end遇到小于key的数，则停下，begin开始走，直到begin遇到一个大于key的数时，将begin和right的内容交换，end再次开始走，如此进行下去，直到begin和end最终相遇，此时将相遇点的内容与key交换即可。（选取最左边的值作为key）
> 
> 4.此时key的左边都是小于key的数，key的右边都是大于key的数
> 
> 5.将key的左序列和右序列再次进行这种单趟排序，如此反复操作下去，直到左右序列只有一个数据，或是左右序列不存在时

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/8fbca9b2fbc89695ca1a7493fbfa87f3.gif"
  referrerPolicy="no-referrer"
  alt=""
/>
```c
void QuickSort(int* a, int left, int right)
{
	if (left >= right)
	{
		return;
	}
	int begin = left, end = right;
	int key = begin;//这里与挖坑法不同的地方，因为要交换key的那个数组中那个位置的数，而不是值
	while (begin < end)
	{
		while (begin < end && a[end] >= a[key])
		{
			end--;
		}
		while (begin < end && a[begin] <= a[key])
		{
			begin++;
		}
		Swap(&a[begin], &a[end]);
	}
	Swap(&a[begin], &a[key]);
	QuickSort(a, left, begin - 1);
	QuickSort(a, begin + 1, right);
}
```

___

> ### 6.3前后指针法：
> 
> **思路：**
> 
> 1、选出一个key，一般是最左边。
> 
> 2、起始时，prev指针指向序列开头，cur指针指向prev+1。
> 
> 3、让cur一直向前走，当遇到小于a\[key\]时，让prev向前走一格（这个值一定大于a\[key\]，因为是cur走过的），然后a\[cur\]和a\[prev\]交换。
> 
> 经过一次单趟排序，最终也能使得key左边的数据全部都小于key，key右边的数据全部都大于key。
> 
> 然后也还是将key的左序列和右序列再次进行这种单趟排序，如此反复操作下去
> 
> 用if(left>right)
> 
> {
> 
>         reurn;
> 
> }//跳出递归


<img
  src="https://i-blog.csdnimg.cn/blog_migrate/a19f716829dfc860f76e11a8aa24955c.gif"
  referrerPolicy="no-referrer"
  alt=""
/>
```c
void QuickSort(int* a, int left,int right)
{
	if (left >= right)
	{
		return;
	}
	int index=GetMidIndex(a,left, right);
	swap(&a[left], &a[index]);
	int key = left;
	int prev = left;
	int cur = left+1;
	while (cur <= right)
	{
		if (a[cur] < a[key])
		{
			prev++;
			swap(&a[cur], &a[prev]);
		}
		/*可以简写成cur++，   
		但是当时一定要注意不要放在if语句的前面，因为if语句里面有让cur与prev交换的，cur==right跳出循环，但是a[cur]超过数组的范围，会越界范围。
        while (cur<=right&&a[cur] >= a[key])   
		{
			cur++;
		}*/
        cur++;
	}
	swap(&a[prev], &a[key]);
	QuickSort(a, left, prev - 1);
	QuickSort(a, prev+1,right);
}
```

___

###  6.4优化快排

三数取中法：取左端、中间、右端三个数，然后进行比较，将中值数当做key

否则有序时 时间复杂度 为O(N^2)

三数取中法可以套入三种方法中，这里我就写一种

```c
//三数取中
int GetMidIndex(int* a, int left, int right)
{
	int mid = (left + right) / 2;
	if (a[mid] >= a[left])
	{
		if (a[mid] <= a[right])
		{
			return mid;
		}
		else
		{
			if (a[right] >= a[left])
			{
				return right;
			}
			else
			{
				return left;
			}
		}
	}
	else//a[left]>a[mid]
	{
		if (a[right] >= a[left])
		{
			return left;
		}
		else
		{
			if (a[right] >= a[mid])
			{
				return right;
			}
			else
			{
				return mid;
			}
		}
	}
}
 
//交换
void swap(int* a, int* b)
{
	int tmp = *a;
	*a = *b;
	*b = tmp;
}
 
//前后指针法
void QuickSort(int* a, int left,int right)
{
	if (left >= right)
	{
		return;
	}
	int index=GetMidIndex(a,left, right);
	swap(&a[left], &a[index]);
	int key = left;
	int prev = left;
	int cur = left+1;
	while (cur <= right)
	{
		if (a[cur] < a[key])
		{
			prev++;
			swap(&a[cur], &a[prev]);
		}
		cur++;
	}
	swap(&a[prev], &a[key]);
	QuickSort(a, left, prev - 1);
	QuickSort(a, prev+1,right);
}
```

___

## 七、 归并排序 ：

> **思路：**
> 
> 1.不断的分割数据，让数据的每一段都有序（一个数据相当于有序）
> 
> 2.当所有子序列有序的时候，在把子序列归并，形成更大的子序列，最终整个数组有序。


<img
  src="https://i-blog.csdnimg.cn/blog_migrate/20bca66489ea8615dfa2c44c05820c4f.png"
  referrerPolicy="no-referrer"
  alt=""
/>

 ！！！需要开一个\_MergeSort,而不是直接在MergeSort中直接递归，是因为MergeSort中有一个malloc

归并排序很像二叉树中的后序思想，先递归，递归到最后的时候再 合并 。！！！


<img
  src="https://i-blog.csdnimg.cn/blog_migrate/17847222181095acdab5c4a31b307b16.gif"
  referrerPolicy="no-referrer"
  alt=""
/>

```c
//归并排序
void _MergeSort(int* a, int left, int right, int* tmp)//在这个函数中调用递归{
	if (left >= right)
	{
		return;
	}
	int mid = (left + right) >> 1;
	_MergeSort(a, left, mid, tmp);
	_MergeSort(a, mid+1, right, tmp);
	//合并
	int begin1 = left, end1 = mid;
	int begin2 = mid + 1, end2 = right;
	int i = left;
	while (begin1 <= end1 && begin2 <= end2)
	{
		if (a[begin1] <= a[begin2])
		{
			tmp[i++] = a[begin1++];
		}
		else
		{
			tmp[i++] = a[begin2++];
		}
	}
	while (begin1 <= end1)
	{
		tmp[i++] = a[begin1++];
	}
	while (begin2 <= end2)
	{
		tmp[i++] = a[begin2++];
	}
	for (int j = left; j <= right; j++)
	{
		a[j] = tmp[j];
	}
}
void MergeSort(int* a, int n)
{
	int* tmp = (int*)malloc(sizeof(int) * n);
	_MergeSort(a, 0, n - 1, tmp);
	free(tmp);
}
```

___

## 

## 八、桶排序：

> **思路：**大问题化小
> 
> **桶排序 (Bucket sort)**或所谓的**箱排序**，是一种分块的[排序算法](https://baike.baidu.com/item/%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95/5399605 "排序算法")，工作的原理是将数组分到有限数量的桶里，每个桶的大小都相等。每个桶再个别排序（有可能再使用别的[排序算法](https://baike.baidu.com/item/%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95/5399605 "排序算法")或是以递归方式继续使用[桶排序](https://so.csdn.net/so/search?q=%E6%A1%B6%E6%8E%92%E5%BA%8F&spm=1001.2101.3001.7020 "桶排序")进行排序）
> 
> 把待排序序列（数组）中的数据根据函数映射方法分配到若干个桶中，在分别对各个桶进行排序，最后依次按顺序取出桶中的数据。
> 
> 适用于数据分配均匀，数据比较大，相对集中的情况。

```c
//桶排序
void bucket_sort(int a[],int size,int bucket_size) {
	int i,j;        //数组，数组长度，桶的大小
 
                        //定义动态的指针数组
	KeyNode **bucket_num = (KeyNode **)malloc(bucket_size * sizeof(KeyNode*));
 
	for(i = 0;i < bucket_size;i++) 
	{
		bucket_num[i] = (KeyNode*)malloc(sizeof(KeyNode));//为每个链表定义头结点 
		bucket_num[i]->num = 0;   
		bucket_num[i]->next = NULL;   //指针变量初始化为空
	}
 
	for(j = 0;j < size;j++) //准备插入
	{
		KeyNode *node = (KeyNode *)malloc(sizeof(KeyNode));//定义一个节点 
		node->num = a[j];    //数据域存数据 
		node->next = NULL;	//指向空
		
		int index = a[j]/100;  //映射函数 计算桶号
		
		KeyNode *p = bucket_num[index];//p指向链表的头
       
        //链表结构的插入排序
		while(p->next != NULL && p->next->num <= node->num)
		{
			p = p->next;	//1.链表为空，p->next==NULL，进入不了循环 
		}					//2.链表不为空，因为链表从无开始按顺序插入，数据为有序的，
							//可以找到    前一个节点 <= node <=后一个节点
		
		//节点插入链表 
		node->next = p->next;
		p->next = node;
		(bucket_num[index]->num)++;	//记录一下该链表中有几个有效节点 
 
	}
	//打印结果
	KeyNode * k = NULL;  //定义一个空的结构体指针用于储存输出结果
	for(i = 0;i < bucket_size;i++)
    {
        //for(k = bucket_num[i]->next;k!=NULL;k=k->next)//通过最后一个指针指向空
        k = bucket_num[i]->next;
        for(int m=0;m<bucket_num[i]->num;m++)   //通过头指针记录节点数
        {
            printf("%d ",k->num);
            k=k->next;
        			
    }		
	printf("\n");
}
```

___

## 

## 九、计数排序：

一种特殊的排序，唯一种没有比较的排序（指没有前后比较，还是有交换的）

以 数组 的下标当做数值，有这个数的时候a\[i\]++;

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/45ab0354442c067c3fc0814100c3fca4.png"
  referrerPolicy="no-referrer"
  alt=""
/>
 局限：适用于整数。数要求集中（否则空间的浪费大）

### 9.1绝对映射：

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/d1c44becef9d4d22e6ae62b712db4a66.png"
  referrerPolicy="no-referrer"
  alt=""
/>
```c
int * countingSort1(int arr[],int count,int max) {
    int index = 0;
    int *tmpArr = (int *)malloc(max*sizeof(int));
    int *result = (int *)malloc(max*sizeof(int));
    
    for(int k = 0;k<max;k++) {
        tmpArr[k] = 0;
    }
    
    for (int i = 0; i<count; i++) {
        tmpArr[arr[i]]++;
    }
    for (int j = 0; j<max; j++) {
        while (tmpArr[j]) {
            result[index++] = j;
            tmpArr[j]--;
        }
    }
    free(tmpArr);
    tmpArr = NULL;
    return result;
}
```

### 9.2现对映射：

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/8288b108f52d377eef696341beddfe27.png"
  referrerPolicy="no-referrer"
  alt=""
/>

```c
void CountSort(int* a, int n)
{
	int max = a[0], min = a[0];
	for (int i = 0; i < n; i++)
	{
		if (a[i] > max) max = a[i];
		if (a[i] < min) min = a[i];
	}
	int range = max - min + 1;
	int* count = (int*)malloc(sizeof(int) * range);
	memset(count, 0, sizeof(int) * range);
	for (int i = 0; i < n; i++)
	{
		count[a[i] - min]++;
	}
	int i = 0;
	for (int j = 0; j < range; j++)
	{
		while (count[j]--)
		{
			a[i++] = j + min;
		}
	}
	free(count);
}
```

___

## 十、基数排序： 

> 原理：是将整数按位数切割成不同的数字，然后按每个位数分别比较。

<img
  src="https://i-blog.csdnimg.cn/blog_migrate/792053044b35d4d7ea5eb05501c1606f.png"
  referrerPolicy="no-referrer"
  alt=""
/>


```c
#include<math.h>
testBS()
{
    inta[] = {2, 343, 342, 1, 123, 43, 4343, 433, 687, 654, 3};
    int *a_p = a;
    //计算数组长度
    intsize = sizeof(a) / sizeof(int);
    //基数排序
    bucketSort3(a_p, size);
    //打印排序后结果
    inti;
    for(i = 0; i < size; i++)
    {
        printf("%d\n", a[i]);
    }
    intt;
    scanf("%d", t);
}
//基数排序
voidbucketSort3(int *p, intn)
{
    //获取数组中的最大数
    intmaxNum = findMaxNum(p, n);
    //获取最大数的位数，次数也是再分配的次数。
    intloopTimes = getLoopTimes(maxNum);
    inti;
    //对每一位进行桶分配
    for(i = 1; i <= loopTimes; i++)
    {
        sort2(p, n, i);
    }
}
//获取数字的位数
intgetLoopTimes(intnum)
{
    intcount = 1;
    inttemp = num / 10;
    while(temp != 0)
    {
        count++;
        temp = temp / 10;
    }
    returncount;
}
//查询数组中的最大数
intfindMaxNum(int *p, intn)
{
    inti;
    intmax = 0;
    for(i = 0; i < n; i++)
    {
        if(*(p + i) > max)
        {
            max = *(p + i);
        }
    }
    returnmax;
}
//将数字分配到各自的桶中，然后按照桶的顺序输出排序结果
voidsort2(int *p, intn, intloop)
{
    //建立一组桶此处的20是预设的根据实际数情况修改
    intbuckets[10][20] = {};
    //求桶的index的除数
    //如798个位桶index=(798/1)%10=8
    //十位桶index=(798/10)%10=9
    //百位桶index=(798/100)%10=7
    //tempNum为上式中的1、10、100
    inttempNum = (int)pow(10, loop - 1);
    inti, j;
    for(i = 0; i < n; i++)
    {
        introw_index = (*(p + i) / tempNum) % 10;
        for(j = 0; j < 20; j++)
        {
            if(buckets[row_index][j] == NULL)
            {
                buckets[row_index][j] = *(p + i);
                break;
            }
        }
    }
    //将桶中的数，倒回到原有数组中
    intk = 0;
    for(i = 0; i < 10; i++)
    {
        for(j = 0; j < 20; j++)
        {
            if(buckets[i][j] != NULL)
            {
                *(p + k) = buckets[i][j];
                buckets[i][j] = NULL;
                k++;
            }
        }
    }
}
```

___

        各位大佬支持一下，点赞关注加收藏，你的支持就是我创作的动力！
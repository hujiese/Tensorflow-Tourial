## Tensorflow 基础 ##

### 一、Tensorflow基本概念 ###

- 使用图（graphs）来表示计算任务
- 在被称之为会话（Session）的上下文（context）中执行图
- 使用tensor表示数据
- 通过变量（Variable）维护状态
- 使用feed和fetch可以为任意的操作赋值或者从其中获取数据

Tensorflow是一个编程系统，使用图（graphs）来表示计算任务，图（graphs）中的节点称之为op
（operation），一个op获得0个或多个Tensor，执行计算，产生0个或多个Tensor。Tensor 看作是
一个 n 维的数组或列表。图必须在会话（Session）里被启动。Tensorflow结构结构如下图所示：

![](https://i.imgur.com/ShTgzgm.png)

下面将通过代码来演示这个过程：

导入tensorflow：

	import tensorflow as tf

创建两个常量op：

	m1 = tf.constant([[3, 3]])
	m2 = tf.constant([[2], [3]])

创建一个矩阵乘法Op，把m1和m2传入：

	product = tf.matmul(m1, m2)
	print(product)

打印product结果如下：

	Tensor("MatMul:0", shape=(1, 1), dtype=int32)

可见，直接运行tenserflow中常量算术操作的得到的结果是一个张量。

接下来创建一个会话，启动默认图：

	sess = tf.Session()

调用sess的run方法来执行矩阵乘法op，run(product)触发了图中3个op：

	result = sess.run(product)
	print(result)
	sess.close()

打印结果如下：

	[[15]]

可见，真正要进行运算还需要使用会话操作。

当然也可以使用下面方法，使用该方法不需要sess.close()操作：

	with tf.Session() as sess:
	    result = sess.run(product)
	    print(result)

同样的打印结果如下：

	[[15]]

### 二、Tensorflow使用变量 ###

案例一：

	import tensorflow as tf
	
	# 定义变量x
	x = tf.Variable([1, 2])
	y = tf.constant([3, 3])
	
	# 定义减法运算op
	sub = tf.subtract(x, y)
	# 定义加法运算op
	add = tf.add(x, sub)
	
	# 定义变量初始化器
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
	    # 初始化所有变量
	    sess.run(init)
	    print(sess.run(sub))
	    print(sess.run(add))

运行结果如下：

	[-2 -1]
	[-1  1]

需要注意的是，如果使用了变量，那么需要使用tf.global\_variables_initializer()来初始化全局变量。

案例二：

	import tensorflow as tf
	# 给变量state起一个别名counter
	state = tf.Variable(0, name='counter')
	# state+1操作
	new_value = tf.add(state, 1)
	# 将new_value赋值给state
	update = tf.assign(state, new_value)
	
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
	    sess.run(init)
	    print(sess.run(state))
	    # 循环五次累加
	    for _ in range(5):
	        sess.run(update)
	        print(sess.run(state))

运行结果如下：

	0
	1
	2
	3
	4
	5

### 三、Fetch和Feed ###

#### Fetch指在一个会话中执行多个语句op： ####

	import tensorflow as tf
	
	input1 = tf.constant(3.0)
	input2 = tf.constant(2.0)
	input3 = tf.constant(5.0)
	
	add = tf.add(input2, input3)
	mul = tf.multiply(input1, add)
	
	with tf.Session() as sess:
	    result = sess.run([mul, add])
	    print(result)

运行结果如下：

	[21.0, 7.0]

其中语句：

	result = sess.run([mul, add])

先执行了mul语句，之后再执行add语句，这便是Feed。

#### Feed的数据以字典的形式传入： ####

	import tensorflow as tf
	
	# 创建占位符
	input1 = tf.placeholder(tf.float32)
	input2 = tf.placeholder(tf.float32)
	output = tf.multiply(input1, input2)
	
	with tf.Session() as sess:
	    # Feed的数据以字典的形式传入
	    print(sess.run(output, feed_dict={input1:[8.], input2:[2.]}))

运行结果如下：

	[ 16.]

程序开始创建的input1和input2两个占位符一开始只有类型没有赋值，然后再后面的运算中使用：

	feed_dict={input1:[8.], input2:[2.]}

给这两个占位符赋值并完成了后面的output运算。

### 四、使用Tensorflow完成梯度下降线性回归模型参数优化 ###

	import tensorflow as tf
	import numpy as np
	
	# 使用numpy随机生成100个点
	x_data = np.random.rand(100)
	y_data = x_data * 0.1 + 0.2
	
	# 构建一个线性模型
	b = tf.Variable(0.)
	k = tf.Variable(0.)
	y = k * x_data + b
	
	# 二次代价函数
	loss = tf.reduce_mean(tf.square(y_data - y))
	# 定义一个梯度下降法来进行训练的优化器
	optimizer = tf.train.GradientDescentOptimizer(0.2)
	# 最小化代价函数
	train = optimizer.minimize(loss)
	# 初始化变量
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
	    sess.run(init)
	    for step in range(201):
	        sess.run(train)
	        if step%20 == 0:
	            print(step, sess.run([k, b]))

运行结果如下：

	0 [0.051365104, 0.099253416]
	20 [0.10158841, 0.19917615]
	40 [0.10088656, 0.19954024]
	60 [0.10049484, 0.19974338]
	80 [0.10027619, 0.19985677]
	100 [0.10015415, 0.19992006]
	120 [0.10008604, 0.19995537]
	140 [0.10004802, 0.19997509]
	160 [0.1000268, 0.1999861]
	180 [0.10001495, 0.19999224]
	200 [0.10000835, 0.19999567]

设定的方程参数k = 0.1，b = 0.2，最后训练的结果和这个类似。
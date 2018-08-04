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

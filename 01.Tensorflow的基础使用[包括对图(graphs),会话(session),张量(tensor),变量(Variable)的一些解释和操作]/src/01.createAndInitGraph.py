# 导入tensorflow
import tensorflow as tf

# 创建两个常量op
m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])
# 创建一个矩阵乘法Op，把m1和m2传入
product = tf.matmul(m1, m2)
print(product)

# 创建一个会话，启动默认图
sess = tf.Session()
# 调用sess的run方法来执行矩阵乘法op
# run(product)触发了图中3个op
result = sess.run(product)
print(result)
sess.close()

# 使用该方法不需要sess.close()操作
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

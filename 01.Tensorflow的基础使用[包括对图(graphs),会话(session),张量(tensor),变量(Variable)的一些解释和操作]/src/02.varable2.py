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

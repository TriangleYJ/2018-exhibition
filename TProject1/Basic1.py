import tensorflow as tf

def main():
    import tensorflow as tf

    # 변수를 0으로 초기화
    state = tf.Variable(0, name="counter")

    # state에 1을 더할 오퍼레이션 생성
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    # 그래프는 처음에 변수를 초기화해야 합니다. 아래 함수를 통해 init 오퍼레이션을 만듭니다.
    init_op = tf.global_variables_initializer()

    # 그래프를 띄우고 오퍼레이션들을 실행
    with tf.Session() as sess:
        # 초기화 오퍼레이션 실행
        sess.run(init_op)
        # state의 초기 값을 출력
        print(sess.run(state))
        # state를 갱신하는 오퍼레이션을 실행하고, state를 출력
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))

if __name__ == "__main__":
    main()
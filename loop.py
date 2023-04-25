from net_t2l import NetT2L


if __name__ == '__main__':
    net = NetT2L()

    print('== Training loop ==')
    while True:
        prompt = input('\n\n> ')
        ans = net.predict(prompt)
        print(f'>> {ans}')
        correction = input(f'>> Correct class (one of: {net.data_train.sorted_labels}): ')

        if correction:
            if correction in net.data_train.sorted_labels:
                net.learn(samples=[prompt], labels=[correction], verbose=False)

                print(f'(after training) >> {prompt} => {net.predict(prompt)}')
            else:
                print('E: Not a class.')
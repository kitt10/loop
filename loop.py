from net_t2l import NetT2L


if __name__ == '__main__':
    net = NetT2L()

    print('== Training loop ==')
    while True:
        prompt = input('\n\n> ')
        ans = net.predict(prompt)
        print(f'>> {ans}')
        correction = input(f'>> Correct class (one of: {net.data.labels}): ')

        if correction:
            if correction in net.data.labels:
                net.learn(sample=prompt, label=correction, verbose=False)

                print(f'(after training) >> {prompt} => {net.predict(prompt)}')
            else:
                print('E: Not a class.')
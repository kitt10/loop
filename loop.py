from net_t2l import NetT2L


if __name__ == '__main__':
    net = NetT2L()

    print('== Training loop ==')
    while True:
        prompt = input('\n\n> ')
        ans = net.predict(prompt)
        print(f'>> {ans}')
        correction = input(f'>> Correct class: ')

        if correction:
            net.learn(sample=prompt, label=correction, verbose=False)
            print(f'(after training) >> {prompt} => {net.predict(prompt)}')
        else:
            if prompt not in net.data.data['train']['samples']:
                net.learn(sample=prompt, label=ans, verbose=False)
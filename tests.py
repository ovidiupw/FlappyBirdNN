from collections import deque

import time


def test_assert_equals():
    ti = time.time()
    help = range(100000)
    while help:
        del help[:1]

    print("Del time: {t}".format(t=time.time() - ti))

    ti = time.time()

    help = deque(range(100000))
    while help:
        help.popleft()

    print("Popleft time: {t}".format(t=time.time() - ti))

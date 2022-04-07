import sys
import numpy as np
import re


if __name__ == '__main__':
    src_result_path = sys.argv[1]
    rooms = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
    result_all = np.zeros((5, 2))
    with open(src_result_path, 'r') as f:
        lines = f.readlines()
        for i in range(5):
            tmp = re.split('[: ]', lines[i*2+1].strip())
            print(tmp)
            result_all[i] = float(tmp[2]), float(tmp[4])

    with open(src_result_path, 'a') as f:
        f.write('\n'*4)
        f.write(' '.join([f'{item}' for item in result_all[:, 0]]))
        f.write('\n')
        f.write(' '.join([f'{item}' for item in result_all[:, 1]]))

import numpy as np








if __name__ == '__main__':
    arr = np.array([1,2,3,4,5,6,7,8,9,10])
    arr2 = np.array([1,2,3,4,5,6,7,8,9,10])
    arr3 = np.array([1,2,3,4,5,6,7,8,9,10])
    np.savez_compressed('test.npz',arr,arr2,arr3)
    data = np.load('test.npz')
    print(data['arr_0'])
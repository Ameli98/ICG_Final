import numpy as np 
import cv2
import pandas as pd
'''
references
https://blog.csdn.net/qq_32424059/article/details/1008743581
chatgpt
https://medium.com/@zhihaoshi1729/%E8%AE%80%E5%8F%96csv%E6%AA%94%E5%88%B0pandas-dataframe%E7%9A%84%E7%B0%A1%E6%98%93%E6%93%8D%E4%BD%9C-deecd2357f3f
https://ithelp.ithome.com.tw/m/articles/10193421
https://vocus.cc/article/62258b3efd89780001ce9272
https://blog.csdn.net/weixin_38605247/article/details/78736417
''' 
img1 = cv2.imread('./Rie.jpg')
img2 = cv2.imread('./Megumin.png')
coordinates = pd.read_csv("./csv")
len_of_point = len(coordinates)

points = np.zeros((len_of_point, int(8)))
for index, row in coordinates.iterrows():
    points[index][0], points[index][1] = row['Src1'].strip('()').split(', ')
    points[index][2], points[index][3] = row['Src2'].strip('()').split(', ')
    points[index][4], points[index][5] = row['Dst1'].strip('()').split(', ')
    points[index][6], points[index][7] = row['Dst2'].strip('()').split(', ')
print(points)


    

width = img1.shape[1]
height = img1.shape[0]



def cal_u(X, P, Q):
    #calculate u
    # pq is p to q
    pq = Q-P
    u = (X-P).dot((pq))/(np.linalg.norm(pq))**2

    return u

def cal_v(X, P, Q):
    #calculate v
    # pq is p to q
    pq = Q-P
    v = np.dot((X-P), np.array([-pq[1], pq[0]]))/(np.linalg.norm(pq))

    return v

def cal_weight(length, p, a, dist, b):
    return ((length ** p)/ (a + dist)) ** b

def cal_X_prime(image1, lines_source, lines_destination):
    y_index, x_index = np.indices((height, width))
    X = np.stack((y_index, x_index), axis=-1)
    transformed = np.zeros_like(image1)
    Dsum = np.zeros_like(X, dtype=float)
    weightsum = np.zeros((height, width), dtype=float)
    num_of_line = len(lines_source)
    for k in range(num_of_line):
        p = lines_destination[k][0]
        q = lines_destination[k][1]
        p_prime = lines_source[k][0]
        q_prime = lines_source[k][1]
        u = cal_u(X, p, q)
        v = cal_v(X, p, q)
        dist = np.abs(v)
        p_prime_q_prime = q_prime - p_prime
        X_prime = p_prime + u[..., np.newaxis] * (q_prime-p_prime) + v[..., np.newaxis] * (np.array([-p_prime_q_prime[1], p_prime_q_prime[0]]))/np.linalg.norm(q_prime-p_prime)
        D_i = X_prime - X
        dist = np.abs(v)
        #dist = np.linalg.norm(v, axis=-1)
        ''' 可以調整的參數
        '''
        weight = cal_weight(np.linalg.norm(q-p), 1.0, 0.0001, dist, 1.0)
        Dsum += D_i * weight[..., np.newaxis]
        weightsum += weight
        X_prime = X + Dsum / weightsum[..., np.newaxis]
        X_prime=np.round(X_prime).astype(int)
        X_prime = np.clip(X_prime, 0, [height-1, width -1])
        transformed[y_index, x_index] = image1[X_prime[..., 0], X_prime[..., 1]]
        
    return transformed



def img_morphing(image1, image2, frames, points):
    lines_source = [np.array([[points[i][0],points[i][1]], [points[i][2], points[i][3]]]) for i in range(len_of_point)]
    lines_destination = [np.array([[points[i][4],points[i][5]], [points[i][6], points[i][7]]]) for i in range(len_of_point)]
    num_of_pairs = len(lines_source)
    for i in range(frames):
        mix = i / (frames - 1)

        interpolation_line = [np.array(lines_source[i]) * (1-mix) + np.array(lines_destination[i]) * mix for i in range(num_of_pairs)]
        trans_source = cal_X_prime(image1, lines_source, interpolation_line)
        trans_destination = cal_X_prime(image2, lines_destination, interpolation_line)
        final_img = trans_source * (1-mix) + trans_destination * mix
        # refer to https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
        # rever to https://docs.python.org/zh-tw/3.10/tutorial/inputoutput.html
        print(i)
        cv2.imwrite(f'./mid_{i}.png', final_img)
  
    return 1


img_morphing(img1, img2, int(100), points)
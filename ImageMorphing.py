import numpy as np 
import cv2
# reference https://blog.csdn.net/qq_32424059/article/details/100874358
# refer to chatgpt
img1 = cv2.imread('./Rie.jpg')
img2 = cv2.imread('./Megumin.png')

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
    transformed = np.zeros_like(image1)
    for i in range(height):
        for j in range(width):
            X = np.array([i, j])
            Dsum = np.array([0.0,0.0])
            weightsum = 0.0
            num_of_line = len(lines_source)
            for k in range(num_of_line):
                p = lines_destination[k][0]
                q = lines_destination[k][1]
                p_prime = lines_source[k][0]
                q_prime = lines_source[k][1]
                u = cal_u(X, p, q)
                v = cal_v(X, p, q)
                p_prime_q_prime = q_prime - p_prime
                X_prime = p_prime + u * (q_prime-p_prime) + v * (np.array([-p_prime_q_prime[1], p_prime_q_prime[0]]))/np.linalg.norm(q_prime-p_prime)
                D_i = X_prime - X
                dist = np.linalg.norm(v)
                weight = cal_weight(np.linalg.norm(u), 1.0, 0.0001, dist, 1.0)
                
                Dsum += D_i * weight
                weightsum += weight
            if weightsum != 0:
                X_prime = X + Dsum / weightsum
                X_prime = np.round(X_prime).astype(int)
                if X_prime[0] >= height:
                    X_prime[0] = height - 1
                if X_prime[0] < 0 : 
                    X_prime[0] = 0
                if X_prime[1] >= width:
                    X_prime[1] = width - 1
                if X_prime[1] < 0 : 
                    X_prime[1] = 0
                transformed[i, j] = image1[X_prime[0], X_prime[1]]
    return transformed



def img_morphing(image1, image2, frames):
    lines_source = [np.array([[5,10], [100, 105]])]
    lines_destination = [np.array([[10, 30], [100, 190]])]
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


img_morphing(img1, img2, int(5))
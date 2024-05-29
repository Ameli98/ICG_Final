import numpy as np 
import cv2
from PIL import Image
import pandas as pd
import argparse
from tqdm import tqdm


def cal_u(X, P, Q):
    pq = Q - P
    u = np.sum((X - P) * pq, axis=2) / (np.sum(pq ** 2) )
    return u

def cal_v(X, P, Q):
    pq = Q - P
    v = np.sum((X - P) * np.array([- pq[1], pq[0]]), axis=2) / (np.linalg.norm(pq))
    return v

def cal_weight(length, p, a, dist, b):
    return ((length ** p)/ (a + dist)) ** b

def cal_X_prime(image1, lines_source, lines_destination):
    y_index, x_index = np.indices((image1.shape[0], image1.shape[1]))
    X = np.stack((y_index, x_index), axis=-1)
    transformed = np.zeros_like(image1)
    Dsum = np.zeros_like(X, dtype=np.float64)
    weightsum = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.float64)
    for k in range(lines_source.shape[0]):
        p, q, p_prime, q_prime = lines_source[k, :2], lines_source[k, 2:], lines_destination[k, :2], lines_destination[k, 2:]
        u, v = cal_u(X, p, q), cal_v(X, p, q)
        p_prime_q_prime = q_prime - p_prime
        X_prime = p_prime + np.stack((u, u), axis=-1) * p_prime_q_prime + np.stack((v, v), axis=-1) * (np.array([- p_prime_q_prime[1], p_prime_q_prime[0]])) / np.linalg.norm(p_prime_q_prime)
        dist = np.abs(v)
        # dist : hyperparameter
        # dist = np.linalg.norm(v, axis=-1)
        weight = cal_weight(np.linalg.norm(q-p), 1.0, 0.0001, dist, 1.0)
        Dsum += (X_prime - X) * np.expand_dims(weight, axis=-1)
        weightsum += weight
    X_prime = X + Dsum / np.expand_dims(weightsum, axis=-1)
    X_prime = np.round(X_prime).astype(int)
    X_prime = np.clip(X_prime, 0, [image1.shape[0]-1, image1.shape[1] -1])
    transformed[y_index, x_index] = image1[X_prime[..., 0], X_prime[..., 1]]
        
    return transformed



def img_morphing(image1, image2, frames, points):
    lines_source, lines_destination = points[:, :4], points[:, 4:]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Video = cv2.VideoWriter(args.OutVideo, fourcc, 60, (image1.shape[1], image1.shape[0]))
    for i in tqdm(range(frames)):
        mix = i / (frames - 1)
        interpolation_line = lines_source * (1 - mix) + lines_destination * mix
        trans_source = cal_X_prime(image1, lines_source, interpolation_line)
        trans_destination = cal_X_prime(image2, lines_destination, interpolation_line)
        final_img = trans_source * (1 - mix) + trans_destination * mix
        # cv2.imwrite(f'./mid_{i}.png', final_img)
        Video.write(final_img.astype(np.uint8))
    Video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("SrcImage", help="Source Image")
    parser.add_argument("DstImage", help="Destination Image")
    parser.add_argument("Csv", help="Line Pair Data")
    parser.add_argument("OutVideo", help="Output video path")
    args = parser.parse_args()
    # swap the input and output
    img2 = cv2.imread(args.SrcImage)
    img1 = Image.open(args.DstImage)

    img2 = img2.resize((img1.shape[1], img1.shape[0]))
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    points = pd.read_csv(args.Csv)
    points = np.array(points.values)
    len_of_point = len(points)

    img_morphing(img1, img2, 100, points)
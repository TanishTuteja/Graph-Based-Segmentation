from unittest import result
import cv2
import numpy as np
from unionfind import unionfind
import argparse
import os
from sklearn.neighbors import NearestNeighbors


def getWt(myImage, p1, p2):
    return abs(int(myImage[p1[1]][p1[0]]) - int(myImage[p2[1]][p2[0]]))


def maxWeight(graph, node, parent):
    maxWt = 0
    nNodes = 1
    for i in range(len(graph[node])):
        if(graph[node][i][0] != parent):
            maxWt = max(maxWt, graph[node][i][1])
            neighMaxWt, neighNodes = maxWeight(graph, graph[node][i][0], node)
            maxWt = max(maxWt, neighMaxWt)
            nNodes += neighNodes
    return maxWt, nNodes


def internalD(graph, v, k):
    maxWt, nNodes = maxWeight(graph, v, v)
    return maxWt + k/nNodes


def generateSegs(edges, k):
    edges.sort(key=lambda e: e[2])

    h = myImage.shape[0]
    w = myImage.shape[1]

    segments = unionfind(h*w)
    segsGraph = [[] for i in range(h*w)]

    for i in range(len(edges)):
        a, b, wt = edges[i]
        v1 = a[1]*w + a[0]
        v2 = b[1]*w + b[0]
        if(not segments.issame(v1, v2)):
            if(wt <= min(internalD(segsGraph, v1, k), internalD(segsGraph, v2, k))):
                segments.unite(v1, v2)
                segsGraph[v1].append((v2, wt))
                segsGraph[v2].append((v1, wt))

    finalSegs = segments.groups()
    return finalSegs


def buildGridGraph(myImage):

    h = myImage.shape[0]
    w = myImage.shape[1]

    graph = [[[] for i in range(myImage.shape[1])]
             for j in range(myImage.shape[0])]
    edges = []
    for j in range(myImage.shape[0]):
        for i in range(myImage.shape[1]):
            if(j != 0):

                if(i != 0):
                    wt = getWt(myImage, (i, j), (i-1, j-1))
                    graph[j][i].append(((i-1, j-1), wt))
                    edges.append(((i, j), (i-1, j-1), wt))

                wt = getWt(myImage, (i, j), (i, j-1))
                graph[j][i].append(((i, j-1), wt))
                edges.append(((i, j), (i, j-1), wt))

                if(i != w-1):
                    wt = getWt(myImage, (i, j), (i+1, j-1))
                    graph[j][i].append(((i+1, j-1), wt))
                    edges.append(((i, j), (i+1, j-1), wt))

            if(i != 0):
                wt = getWt(myImage, (i, j), (i-1, j))
                graph[j][i].append(((i-1, j), wt))
                edges.append(((i, j), (i-1, j), wt))
    return (graph, edges)


def gridGraphSegMono(myImage, k):
    graph, edges = buildGridGraph(myImage)
    return generateSegs(edges, k)


def intersectRGB(segsR, segsG, segsB, w, h):
    rBins = [0 for i in range(w*h)]
    gBins = [0 for i in range(w*h)]
    bBins = [0 for i in range(w*h)]

    for i in range(len(segsR)):
        for j in range(len(segsR[i])):
            rBins[segsR[i][j]] = i
    for i in range(len(segsG)):
        for j in range(len(segsG[i])):
            gBins[segsG[i][j]] = i
    for i in range(len(segsB)):
        for j in range(len(segsB[i])):
            bBins[segsB[i][j]] = i

    segs = []
    done = [False for i in range(h*w)]
    for x in range(h*w):
        while(x < h*w and done[x]):
            x += 1
        if(x == h*w):
            break
        currSeg = []
        currVal = (rBins[x], gBins[x], bBins[x])
        for j in range(h):
            for i in range(w):
                if((rBins[j*w + i], gBins[j*w + i], bBins[j*w + i]) == currVal):
                    currSeg.append(j*w+i)
                    done[j*w + i] = True
        segs.append(currSeg)
    return segs


def gridGraphSeg(myImage, k):

    h = myImage.shape[0]
    w = myImage.shape[1]
    bluePixels = np.array([[myImage[i][j][0]
                            for j in range(myImage.shape[1])] for i in range(myImage.shape[0])])
    greenPixels = np.array([[myImage[i][j][1]
                            for j in range(myImage.shape[1])] for i in range(myImage.shape[0])])
    redPixels = np.array([[myImage[i][j][2]
                           for j in range(myImage.shape[1])] for i in range(myImage.shape[0])])

    bSegs = gridGraphSegMono(bluePixels, k)
    gSegs = gridGraphSegMono(greenPixels, k)
    rSegs = gridGraphSegMono(redPixels, k)

    finalSegs = intersectRGB(rSegs, gSegs, bSegs, w, h)

    return finalSegs


def buildNNGraph(myImage, nNeighbors):
    h = myImage.shape[0]
    w = myImage.shape[1]

    graph = [[[] for i in range(myImage.shape[1])]
             for j in range(myImage.shape[0])]
    edges = []
    vertices = [[myImage[i][j][0], myImage[i][j][1], myImage[i][j][2], (i*255)/h, (j*255)/w]
                for j in range(myImage.shape[1]) for i in range(myImage.shape[0])]
    X = np.array(vertices)
    nbrs = NearestNeighbors(n_neighbors=nNeighbors+1,
                            algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    for i in range(myImage.shape[0]):
        for j in range(myImage.shape[1]):
            c = i*w + j
            for x in range(1, nNeighbors + 1):
                neigh = (indices[c][x] % w, int(indices[c][x]/w))

                if((i, j) not in [tup[0] for tup in graph[neigh[0]][neigh[1]]]):
                    graph[i][j].append(
                        (neigh, distances[c][x]))
                    edges.append(((i, j), neigh, distances[c][x]))

    return graph, edges


def nnGraphSeg(myImage, k, nNeighbors):
    graph, edges = buildNNGraph(myImage, nNeighbors)
    return generateSegs(edges, k)


parser = argparse.ArgumentParser()
parser.add_argument(
    'input', help='Image to be segmented')
parser.add_argument(
    'output', help='Segmented image output')
parser.add_argument(
    '--k', help='Algorithm parameter indicating size of segments generated. Higher the k, larger the segments generated. Default = 1000', default='1000'
)
parser.add_argument(
    '--sigma', help='Sigma used by Gaussian blurring before running the segmentation algorithm. Default = 0 (Sigma calculated based on kernel size automatically)', default='0'
)
parser.add_argument(
    '--method', help='Method to be used for generating graph, either grid graph or nearest neighbors in RGBXY space. Default = nn', choices=["grid", "nn"], default='nn'
)
parser.add_argument(
    '--num_neighbors', help='Number of neighbors to be used to build grpah if NN method is used. Default = 8', default='8'
)
opt = parser.parse_args()

k = int(opt.k)
sigma = int(opt.sigma)
nNeighbors = int(opt.num_neighbors)

fileName = opt.input
myImage = cv2.imread(fileName, cv2.IMREAD_COLOR)
myImage = cv2.GaussianBlur(myImage, (5, 5), sigma)
print(myImage.shape)

h = myImage.shape[0]
w = myImage.shape[1]

if(opt.method == "grid"):
    finalSegs = gridGraphSeg(myImage, k)
else:
    finalSegs = nnGraphSeg(myImage, k, nNeighbors)

result = np.zeros((h*w, 3))
for i in range(len(finalSegs)):
    colorR = np.random.randint(0, 256)
    colorG = np.random.randint(0, 256)
    colorB = np.random.randint(0, 256)
    for j in range(len(finalSegs[i])):
        result[finalSegs[i][j]] = [colorB, colorG, colorR]
result = np.reshape(result, (h, w, 3))

outputDir = os.path.abspath(os.path.join(opt.output, os.pardir))

if(not os.path.isdir(outputDir)):
    os.makedirs(outputDir)

cv2.imwrite(opt.output, result)

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.figure(figsize=(25, 15))
def drawBoxes( filename ): 
    boxFile = open(filename, 'r')
    rects = boxFile.readlines()
    for line in rects :
        num = list(map(float, line.split()))
        i = 0
        x, y = [], [] 
        for pt in num :
            if i%2 == 0 :
                x.append(pt)
            else :
                y.append(pt)
            i+=1
        x.append(num[0])
        y.append(num[1])
        plt.plot(x, y)
def drawPoints( filename ):
    boxFile = open(filename, 'r')
    points = boxFile.readlines()
    x, y = [], [] 
    for point in points :
        num = list( map(float, point.split() ))
        x.append(num[0])
        y.append(num[1])
    plt.plot(x, y, 'ro', markersize=1)
if __name__ == '__main__':
    drawPoints('10K.txt')
    drawBoxes('Quadtree(10K.txt).txt')
    plt.show()
    # plt.savefig('10PointQuadtree.png')
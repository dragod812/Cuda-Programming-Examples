import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.figure(figsize=(10, 10))
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
def drawPoints( filename , col):
    boxFile = open(filename, 'r')
    points = boxFile.readlines()
    x, y = [], [] 
    for point in points :
        num = list( map(float, point.split() ))
        x.append(num[0])
        y.append(num[1])
    plt.plot(x, y, col+'o', markersize=0.7)
def drawLines( filename ):
    boxFile = open(filename, 'r')
    lines = boxFile.readlines()
    for line in lines :
        num = list(map(float, line.split()))
        i = 0
        x, y = [], [] 
        for pt in num :
            if i%2 == 0 :
                x.append(pt)
            else :
                y.append(pt)
            i+=1
        plt.plot(x, y)

if __name__ == '__main__':
    drawPoints('2.5width_4patels.txt', col='y')
    drawPoints('OuterPoints(2.5width_4patels.txt).txt', col='b')
    drawPoints('InnerPoints(2.5width_4patels.txt).txt', col='r')
    # drawBoxes('OuterPointsBox(2.5width_4patels.txt).txt')
    drawLines('CloseLines.txt')
    plt.show();
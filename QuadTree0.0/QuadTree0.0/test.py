def calcChildren(x, y, cx, cy):
    NEcnt, NWcnt, SWcnt, SEcnt = 0, 0, 0, 0
    for i in range(1142):
        if x[i] >= cx and y[i] >= cy :
            NEcnt+=1
        elif x[i] < cx and y[i] >= cy :
            NWcnt+=1
        elif x[i] < cx and y[i] < cy :
            SWcnt+=1
        elif x[i] >= cx and y[i] < cy :
            SEcnt+=1
    print('NEcnt: ', NEcnt, 'NWcnt: ', NWcnt, 'SWcnt: ', SWcnt, 'SEcnt: ', SEcnt)
if __name__ == '__main__':
    boxFile = open('2.5width_4patels.txt', 'r')
    points = boxFile.readlines()
    x, y = [], [] 
    for point in points :
        num = list( map(float, point.split() ))
        x.append(num[0])
        y.append(num[1])
    calcChildren(x, y, 0.0, 0.0)
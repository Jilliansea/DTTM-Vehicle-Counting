def IOU(Reframe,GTframe):
    """
    input:
    ReFrame:(x1,y1,x1',y1')
    GTFrame:(x2,y2,x2',y2')
    """
    x1 = float(Reframe[0])
    y1 = float(Reframe[1])
    x1_= float(Reframe[2])
    y1_= float(Reframe[3])
    width1 = abs(x1_-x1)
    height1 = abs(y1_-y1)

    x2 = float(GTframe[0])
    y2 = float(GTframe[1])
    x2_ = float(GTframe[2])
    y2_ = float(GTframe[3])
    width2 = abs(x2_ - x2)
    height2 = abs(y2_ - y2)

    insection_w=width1+width2-(max(x1_,x2_)-min(x1,x2))
    insection_h=height1+height2-(max(y1+height1,y2+height2)-min(y1,y2))

    if(insection_w<=0 or insection_h<=0):
        IOU_S=0
    else:
        IOU_S=(insection_w*insection_h)/(width1*height1+width2*height2-insection_w*insection_h)
    return IOU_S
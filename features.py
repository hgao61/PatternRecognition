
import cv2
import pickle, os
import numpy
from copy import deepcopy

current_directory = os.path.abspath(os.path.join(os.path.dirname(__file__)))

EPSILON = 0.00000001
#SHAPE_THRESHOLD = 0.5

SHAPE_THRESHOLD = 1.0

moment_keys =[   'nu02',
                 'nu03',
                 'nu11',
                 'nu12',
                 'nu20',
                 'nu21',
                 'nu30']


#                  'm00',
#                  'm01',
#                  'm02',
#                  'm03',
#                  'm10',
#                  'm11',
#                  'm12',
#                  'm20',
#                  'm21',
#                  'm30',
#                  'mu02',
#                  'mu03',
#                  'mu11',
#                  'mu12',
#                  'mu20',
#                  'mu21',
#                  'mu30',

garbage_contours = [  {'area': 6.9711371324725429,  'fill': 48.7891617273497, 'pos_x': 91.03603861279943,  'pos_y': 92.03632463353593,  'shape': [3.0573051275792578, 6.50625667227858, 8.0, 8.0, -8.0, -8.0, 8.0]} ,
                      {'area': 7.0440942258384052,  'fill': 37.109356014580804, 'pos_x': 99.67636880484488,  'pos_y': 99.67636880484488,  'shape': [2.9679886736562482, 6.3186309436582571, 8.0, 8.0, 8.0, 8.0, 0.0]} ]


def data_to_features(data):

    def scale_moments(moments):
        result = []
        for i in range(len(moments)):
            x = moments[i]
            if numpy.abs(x) < EPSILON:
                x = numpy.sign(x) * EPSILON
            x = -numpy.sign(x)*numpy.log10(numpy.abs(x))
            result.append(numpy.nan_to_num(x))
            
        return result


    moments,hierarchy,mean,rect = data

    y = {}
    y['area'] = numpy.log(numpy.sqrt(moments['m00']/numpy.pi))
    y['pos_x'] = moments['m10'] / moments['m00']
    y['pos_y'] = moments['m01'] / moments['m00']
    u_11 = moments['mu11'] #/ moments['m00']
    u_20 = moments['mu20'] #/ moments['m00']
    u_02 = moments['mu02'] #/ moments['m00']
    if numpy.abs(u_20 - u_02) < EPSILON:
        u_20 = EPSILON
        u_02 = 0.0

    moment_rotation = 0.5 * numpy.arctan((2.0 * u_11) / (u_20 - u_02))
    moment_rotation = int( moment_rotation / (2.0 * numpy.pi) * 360.0)

    rect_rotation = rect[2]
    #if abs(rect_rotation) > abs(moment_rotation):
    y['rotation'] = rect_rotation
    #else:
        #y['rotation'] = moment_rotation
    y['moments'] = scale_moments(numpy.nan_to_num([moments[index] for index in moment_keys]))

    y['shape'] = scale_moments(numpy.nan_to_num(cv2.HuMoments(moments).flatten()))
    
    y['parent'] = hierarchy[3]
    y['fill'] = mean
    return y


def figure_to_features(figure):

    def create_masks(img_shape,contour):
        mask = numpy.zeros(img_shape,numpy.uint8)
        cv2.drawContours(mask,[contour],0,255,-1)
        return mask

    def subtract_child_masks(masks,hierarchy):
        for i in range(len(hierarchy)):
            for j in range(len(hierarchy)):
                parent = hierarchy[j][3]
                if parent == i:
                    masks[i] = masks[i] - masks[j]

    def mean_fill(img,mask):
        x = numpy.ma.array(255-img,mask=255-mask)
        return x.mean()


    y = {}
    y['filename'] = figure.fullpath
    y['img'] = 255-cv2.imread(current_directory + '/' +  y['filename'],0)
    y['img_shape'] = y['img'].shape
    #y['ret'],y['thresh'] = cv2.threshold(y['img'],127,255,0)
    #Deep copy is needed since cv2 blasts the original img.
    y['contours'], y['hierarchy'] = cv2.findContours(deepcopy(y['img']),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if not y['contours']:
        return {}
    y['masks'] = [create_masks(y['img_shape'],c) for c in y['contours']]
    y['moments'] = [cv2.moments(x) for x in y['masks']]
    y['hu_moments'] = [numpy.nan_to_num(cv2.HuMoments(x)) for x in y['moments']]
    subtract_child_masks(y['masks'],y['hierarchy'][0])
    y['mean_fills'] = [mean_fill(y['img'],mask) for mask in y['masks']]
    y['rect'] = [cv2.minAreaRect(cnt) for cnt in y['contours']]
    y['features'] = [data_to_features(x) for x in zip(y['moments'],y['hierarchy'][0],y['mean_fills'],y['rect'])]

    return y



def shape_match(x,parent):

    def match(x,y):
        diff = 0
        diff += abs(x['pos_x'] - y['pos_x']) / 10.0
        #print diff
        diff += abs(x['pos_y'] - y['pos_y']) / 10.0
        #print diff
        diff += abs(x['area'] - y['area']) 
        #print diff
        for z in zip(x['shape'],y['shape']):
            if z[0] and z[1] > EPSILON:
                diff += abs(1/z[0] - 1/z[1])
        #diff += (x['rotation'] - y['rotation']) ** 2.0
        #print diff
        if diff < SHAPE_THRESHOLD:
            return True
        return False

    if match(x,parent):
        return True

    for g in garbage_contours:
        if match(parent,g):
            return True

    return False
    
    

def clean_figure_features(feature_list):
    #Delete additional contours caused by empty fill. 
    #if shape is the same as parent and same area
    
    #Mark contours from deletion. 
    #While we do that we alter the hierarchy in preparation.
    #print len(feature_list)
    
    marked = set()
    for i in xrange(len(feature_list)):
        shape = feature_list[i]
        if shape['parent'] != -1:
            parent = feature_list[shape['parent']]
            #print i
            if shape_match(shape,parent) and parent['fill'] < 128:
                marked.add(shape['parent'])
                #print 'marking',shape['parent']
                shape['parent'] = parent['parent']
                #feature_list.pop(mark)
                #clean_figure_features(feature_list)

    #Delete unwanted contours
    #print marked
    new_feature_list = []
    for i in range(len(feature_list)):
        shape = feature_list[i]
        count = 0
        for m in marked:
            if shape['parent'] > m:
                count = count+1
        shape['parent'] = shape['parent'] - count

        if i not in marked:
            new_feature_list.append(shape)

    return new_feature_list
    



def problem_to_features(problem):
    x = {}
    for k,v in problem.figures.iteritems():
        x[k] = figure_to_features(v)        
    return x

def reduce_features(x):
    features = {k:v for k,v in x.iteritems()}
    for k,v in features.iteritems():
            if v:
                features[k] = clean_figure_features(v['features'])
    return features
    

import numpy

from shape_kb import shape_kb 
from angle_kb import angle_kb 


def score_against_kb(vector,kb):

    def score(x,y):
        z = zip(x,y)
        result = 0.0
        for i,j in z:
            result += abs((i-j))
        return result
    
    scores = []
    for k,v in kb:
        scores.append((k,score(vector,v)))
    scores = sorted(scores, key = lambda x: x[1])

    return scores[0]

shape_iterator = 0

SIZE_THRESHOLD = 0.18
POSITION_THRESHOLD = -40
SHAPE_THRESHOLD = 0.05
#Required to stop attrib replacements during mapping.
SHAPE_LABEL_PREFIX = "S_"



def is_filled(s):
    if s['fill'] < 128:
        return {'yes'}
    else:
        return {'no'}

def determine_shape(s):
    global known_shapes
    global shape_iterator

    score = score_against_kb(s['shape'],shape_kb)

    if score[0] == 'pac-man':
        print "Pacman found"
        print s['shape']
        
    return {score[0]}

    if score[1] < SHAPE_THRESHOLD:
        return {score[0]}
    else:
        print "Unknown Shape"
        label = str(shape_iterator)
        shape_iterator = shape_iterator + 1
        shape_kb.append((shape_iterator , s['shape']))
        return {label}



def relative_sizes(data):
    size_types = {}
    for s, a in data:
        #find all the sizes of each class
        areas = size_types.get(s,[])
        areas.append(a)
        size_types[s] = areas
        
    #Sort in size order
    for s in size_types.keys():
        size_types[s] = sorted(size_types[s])
        #Collapse sizes into cluster based on threshold
        reduced_sizes = []
        for i in size_types[s]:
            if not reduced_sizes:
                reduced_sizes.append(i)
            else:
                if abs(i - reduced_sizes[-1]) > SIZE_THRESHOLD:
                    reduced_sizes.append(i)
        size_types[s] = reduced_sizes
    return size_types


def assign_size(size_types, shape_type, area):
    size_list = size_types[shape_type]
    diff =[(area-x)**2.0 for x in size_list]
    return {str(numpy.argmin(diff))}


def determine_inside(s,shapes,count=0):
    if s['parent'] == -1 or count > 10:
        return []
    inside = [SHAPE_LABEL_PREFIX+str(s['parent'])] + determine_inside(shapes[s['parent']],shapes,count+1)

    if inside[-1] == None:
        inside = inside[:-1]
    return inside

def determine_left_of(s,shapes):
    left_of = []
    for i in range(len(shapes)):
        if (s['pos_x'] - shapes[i]['pos_x']) < POSITION_THRESHOLD:
            left_of.append(SHAPE_LABEL_PREFIX+str(i))
    return left_of

def determine_above(s,shapes):
    above = []
    for i in range(len(shapes)):
        if (s['pos_y'] -shapes[i]['pos_y']) < POSITION_THRESHOLD:
            above.append(SHAPE_LABEL_PREFIX+str(i))
    return above

def determine_angle(s,attributes):

    def quantize_angle(angle,unit):
        if (angle > 0):
            angle = int(numpy.ceil(angle/unit) * unit)
        elif( angle < 0):
            angle = int(numpy.floor(angle/unit) * unit)
        else:
            angle = 0
        angle = angle % 360
        return angle


    x = attributes['shape'].copy().pop()

    if attributes['shape'] == {'circle'}:
        return str(0)

    if x in angle_kb:
        score = score_against_kb(s['moments'],angle_kb[x])
        return str(score[0])


    angle = quantize_angle(s['rotation'],15.0)
    
    #if attributes['shape'] == {'square'} and angle % 45 == 0:
    #    attributes['shape'] = {'diamond'}
    #    angle = 0

    return str(angle)

def features_to_net(features):

    shape_label = 0
    net = {}

    #This is a list of sizes for every shape in the problem.
    shape_sizes = []

    for k,shapes in features.iteritems():
        if not shapes:
            net[k] = {}
        shape_labels = [SHAPE_LABEL_PREFIX+str(i) for i in range(len(shapes))]
        shape_attributes = []
        for s in shapes:
            attributes = {}
            attributes['fill'] = is_filled(s)
            attributes['shape'] = determine_shape(s)
            attributes['size'] = s['area']

            inside = determine_inside(s,shapes)
            if inside:
                attributes['inside'] = set(inside)

            left_of = determine_left_of(s,shapes)
            if left_of:
                attributes['left-of'] = set(left_of)

            above = determine_above(s,shapes)
            if above:
                attributes['above'] = set(above)

            attributes['angle'] = { determine_angle(s,attributes) }
            
            if attributes['shape'] == {'half-arrow'}:
                attributes['vertical-flip'] = {'no'}

            if attributes['angle'] == {'180-flip'}:
                attributes['angle'] = {180}
                attributes['vertical-flip'] = {'yes'}

            if attributes['angle'] == {'0-flip'}:
                attributes['angle'] = {0}
                attributes['vertical-flip'] = {'yes'}

            shape_attributes.append(attributes)
            shape_sizes.append((attributes['shape'].copy().pop(),s['area']))
            
        shape_net = {k:v for k,v in zip(shape_labels,shape_attributes)}
        net[k] = shape_net


    #We have a list of 
    relative_size_types = relative_sizes(shape_sizes)
    
    for k,figure in net.iteritems():
        for label,shape in figure.iteritems():
            shape_type = shape['shape'].copy().pop()
            area = shape['size']
            shape['size'] = assign_size(relative_size_types,shape_type,area)


    return net


def text_to_net(problem):
    
    def figure_to_net(figure):
        y = {}

        for object in figure.objects:
            y[object.name] = {}
            for attrib in object.attributes:
                y[object.name][attrib.name] = set(attrib.value.split(','))
        return y

    x = {}
    for k,v in problem.figures.iteritems():
        x[k] = figure_to_net(v)            
    return x
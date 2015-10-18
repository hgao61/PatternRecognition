#An agent for solving Raven's Progressive Matrices. 



from copy import deepcopy, copy
from itertools import permutations
from math import factorial
from features import problem_to_features, reduce_features
from net import features_to_net, text_to_net
import pickle, os

current_directory = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class Agent:

    def __init__(self):

        #Pre-existing knowledge and cognitive biases go in here.       
        self.NUMERICAL_HANDLER = None
        self.numerical_handlers = ['subtraction','addition']
        self.rotational_invariance = { 'circle': 1, 'plus' : 90, 'square' : 90, 'triangle': 120, 'diamond':90, 'rectangle':180 }
        self.zero_angle_facing_right = [{ 'half-arrow' }, {'arrow'}, {'Pac-Man'} ]
        self.right_triangle = [{'right-triangle'}]

        #Do this when rotating 180?
        self.vertical_flip_bias = [{ 'half-arrow' }, {'arrow'} ]

        #Various limits
        self.map_object_limit = 4

        #Data-structures used for learning.
        self.memory = self.load_memory()



    def load_memory(self):
        print current_directory
        if os.path.isfile(current_directory+'/agent_memory.pkl'):
            with open(current_directory+'/agent_memory.pkl','rb') as f:
                self.memory = pickle.load(f)
        else:
            print 'No memory'
            self.memory = {}
        return self.memory


    # @param problem the RavensProblem your agent should solve
    # @return your Agent's answer to this problem (as a string)
    def Solve(self,problem):
       
        def submit_solution(solution):
            correct_answer = problem.checkAnswer(solution)
            if correct_answer != solution:
                print "Learning", correct_answer
                self.learn(problem.problemType,net,correct_answer)
            return solution
            
        #A helper to turn the problem into our preferred semantic net format.
        print "working on ",problem.name
        net = self.problem_to_net(problem)
        recalled_solution = self.check_memory(problem.problemType,net)
        if recalled_solution:
            return recalled_solution        

        if problem.problemType == '2x1 (Image)' or problem.problemType == '2x1':

            check = self.no_transform_check(net)
            if check:
                return check

            inputs = [net['A'],net['B'],net['C']]
            answers = [net['1'],net['2'],net['3'],net['4'],net['5'],net['6']]

            possible_solutions = self.compute_solutions(*inputs)
            solution, cost, _ = self.solve_2x1(possible_solutions,inputs,answers)
            if solution:
                return submit_solution(str(solution))

            else:
                print "Guesssing"
                return submit_solution("1")
        

        elif problem.problemType == '2x2 (Image)' or problem.problemType == '2x2':

            inputs = [net['A'],net['B'],net['C']]
            answers = [net['1'],net['2'],net['3'],net['4'],net['5'],net['6']]
            h_possible_solutions = self.compute_solutions(*inputs)
            horizontal_solution, horizontal_cost, h_solution_data = self.solve_2x1(h_possible_solutions,inputs,answers)

            #Vertical
            inputs = [net['A'],net['C'],net['B']]
            v_possible_solutions = self.compute_solutions(*inputs)
            vertical_solution, vertical_cost, v_solution_data = self.solve_2x1(v_possible_solutions,inputs,answers)

            tw_solution, tw_cost, tw_data = self.solve_two_ways(h_possible_solutions,v_possible_solutions,answers)
            if tw_solution:
                return submit_solution(str(tw_solution))

            if horizontal_solution and vertical_solution:
                if horizontal_cost <= vertical_cost:
                    return submit_solution(str(horizontal_solution))
                else:
                    return submit_solution(str(vertical_solution))

            if horizontal_solution:
                return submit_solution(str(horizontal_solution))
            
            if vertical_solution:
                return submit_solution(str(vertical_solution))
            print "Agent is guessing"
            return submit_solution("1")
            
        else:
            #cant handle other problems yet
            return "Don't know, can't handle larger problems yet."




    def solve_two_ways(self,h_solutions,v_solutions, answers):

        h = h_solutions[0]['solution']
        v = v_solutions[0]['solution']

        solution = {k:v for k,v in h.items()}
        for k,v in v.items():
            if k not in solution:
                solution[k] = v

        for check_spatial in [True,False]:
            best_solution, cost, data = self.find_best_solution([{'analogy':None,'solution':solution,'cost':1}], answers, check_spatial)
            if best_solution:
                return best_solution, cost, data

        return None, None, None



    def solve_2x1(self,possible_solutions,inputs,answers):
        for check_spatial in [True,False]:
            #Calculate batches of possible solutions and return a single solution to see if any are valid
            possible_solutions = self.compute_solutions(*inputs)
            best_solution, cost, data = self.find_best_solution(possible_solutions, answers, check_spatial)
            if best_solution:
                return best_solution, cost, data

        #Try reverse lookup
        best_solution, cost, data = self.solve_2x1_reverse(inputs,answers)
        if best_solution:
            return best_solution, cost, data
        return None, None, None


    def solve_2x1_reverse(self,inputs,answers):
        for check_spatial in [True,False]:
            #If we didn't find anything, see if we can work it out backwards via going from solution to question.
            #Solution is B
            reverse_answers = [inputs[1]]
            for a in range(len(answers)):
                #C is Solution as A is to ?
                reverse_inputs = [inputs[2], answers[a], inputs[0]] 
                possible_solutions = self.compute_solutions(*reverse_inputs)
                best_solution, cost, data = self.find_best_solution(possible_solutions, reverse_answers, check_spatial)
                if best_solution:
                    return a+1, cost, data
        return None, None, None

    ####Set things up
    def problem_to_net(self,problem):
        if problem.problemType[-len('(Image)'):] == '(Image)':
            x = problem_to_features(problem)
            x = reduce_features(x)
            x = features_to_net(x)
        else:
            x = text_to_net(problem)
        return x

    def no_transform_check(self,net):
        #Just in case there is no transformation taking place.
        if self.test_figures_are_equivalent(net['A'],net['B']):
            #there is no transformation between A and B just return whatever looks like C
            for i in range(1,7):
                if self.test_figures_are_equivalent(net['C'],net[str(i)]):
                    return str(i)
                else:
                    return "Got no idea."
        return None

    ###Memory and Learning
    def check_memory(self,problem_type, net):
        def check_figures(net,case):
            for figure in ['A','B','C']:
                if not self.test_figures_are_equivalent(net[figure],case[figure]):
                    return None
            #We found a match
            for figure in ['1','2','3','4','5','6']:
                if self.test_figures_are_equivalent(net[figure],case["ANSWER"]):
                    print "Recalled"
                    return figure
            return None

        if problem_type in self.memory:
            for case in self.memory[problem_type]:
                result = check_figures(net,case)    
                if result:
                    return result
        return None
                
    def learn(self, problem_type, net, correct_answer):
        case = {}
        for figure in ['A','B','C']:
            case[figure] = net[figure]
        case['ANSWER'] = net[correct_answer]
        
        if problem_type not in self.memory:
            self.memory[problem_type] = []
                
        self.memory[problem_type].append(case)

        with open(current_directory+'/agent_memory.pkl','wb') as f:
            pickle.dump(self.memory,f)

        return None            


    ####Logic
    def map_objects(self,x,y):


        def map_attributes(mapping,attributes):
            new_attributes = {}
            for k,v in attributes.iteritems():
                new_v = [x if x not in mapping else mapping[x] for x in v]
                new_attributes[k] = set(new_v)
            return new_attributes



        x_objects = x.keys()
        y_objects = y.keys()

        #Try shape and size based mapping first
        map_count = 0
        for p in permutations(y_objects):
            mapping = zip(p,x_objects)
            mapping = {k:v for k,v in mapping}

            z = {mapping[k]:map_attributes(mapping,v) for k,v in y.iteritems()}
            #Stop making maps if the solution space is too large
            map_count = map_count + 1
            if map_count > factorial(self.map_object_limit):
                break

            yield mapping,z    


    def test_figures_are_equivalent(self,x,y,check_spatial=True):
        #Fast basic test.
        x_objects = x.keys()
        y_objects = y.keys()
        if len(x_objects) != len(y_objects):
            return False

        #Complete test - uses pythons internal dict hashing to see if things are the same.
        for _,z in self.map_objects(x,y):    
            if z == x:
                return True

        #Check for rotational invariance on shape. 
        if 'angle' in x[x.keys()[0]]:
            def align_rotations(x):
                for obj in x:

                    if 'angle' in x[obj] and 'shape' in x[obj]:
                        shape = x[obj]['shape'].copy().pop()
                        angle = int(x[obj]['angle'].copy().pop())
                        
                        if shape in self.rotational_invariance:
                            x[obj]['angle'] =  { str(angle % self.rotational_invariance[shape]) }
                            #print x[obj]['angle']
                return x
            x = align_rotations(deepcopy(x))
            y = align_rotations(deepcopy(y))
            for _,z in self.map_objects(x,y):    
                if z == x:
                    return True


        #TODO - see if this needs a flag     
        if not check_spatial:
            def remove_spatial(x):
                for obj in x:
                    if 'left-of' in x[obj]:
                        del x[obj]['left-of']

                    if 'above' in x[obj]:
                        del x[obj]['above']

                    if 'inside' in x[obj]:
                        del x[obj]['inside']
                    
                        
                return x

            x = remove_spatial(deepcopy(x))
            y = remove_spatial(deepcopy(y))

            for _,z in self.map_objects(x,y): 
                if z == x:
                    return True


        return False


    def transform_figure(self,input_x,input_y):
        #Yields an iterator of possible transformations between state x and state y with a cost associated with each one.

        x = deepcopy(input_x)
        y = deepcopy(input_y)


        if len(x.keys()) != len(y.keys()):
            #Need to do object deletion or replacement

            additional_objects = len(y.keys()) - len(x.keys())
            if additional_objects > 0:
                for additional_object in range(additional_objects):
                    x['PROXY_'+str(additional_object)] = {}
            else:
                for additional_object in range(additional_objects*-1):
                    y['PROXY_'+str(additional_object)] = {}


        x_objects = x.keys()
        y_objects = y.keys()

        global_transforms = None

        for mapping,z in self.map_objects(x,y):
            #z is the shapes in y mapped to labels from x
            transformations = []
            
            for z_object in z:
                for attribute in z[z_object]:

                    if attribute in x[z_object]:
                        if x[z_object][attribute] != z[z_object][attribute]:
                            value_variants = self.difference_attribute(attribute, x[z_object][attribute], z[z_object][attribute],x[z_object])
                            transformations.append({"object":z_object,"attribute":attribute,"value":value_variants})

                    else:
                        transformations.append({"object":z_object,"attribute":attribute,"value":z[z_object][attribute]})

                for attribute in x[z_object]:
                    if attribute not in z[z_object]:
                        transformations.append({"object":z_object,"attribute":attribute,"value":{"DELETE"}})     
                    
            global_transforms = self.check_if_global_transformation(x_objects,transformations)
            if global_transforms:
                yield {'global':True,'mapping':None,'start_state':x ,'end_state':y, 'mapped_end_state':z,'transformations':global_transforms,'cost':len(global_transforms)}                
            
            yield {'global':False,'mapping':mapping,'start_state':x ,'end_state':y, 'mapped_end_state':z,'transformations':transformations,'cost': self.cost_transformations(transformations)}


    def check_if_global_transformation(self,objects,transformations):
        #Given a set of transforms, see if every object is transformed the same way, if so this is a global transform.

        base_object = objects[0]

        base_transform = sorted([t for t in transformations if t['object'] == base_object])
        if not base_transform:
            #This is required so that if one of the objects didnt change and the other did it will trigger a "not global" response
            base_transform = [{'attribute':'proxy','value':'proxy'}]
        for test_object in objects[1:]:
            test_transform = sorted([t for t in transformations if t['object'] == test_object])
            if not test_transform:
                test_transform = [{'attribute':'proxy','value':'proxy'}]
            comparison = zip(base_transform,test_transform)
            for t,b in comparison:
                if t['attribute'] != b['attribute'] or t['value'] != b['value']:
                    return None
        else:
            return base_transform


    def cost_transformations(self,transformations):
        cost = 0
        for t in transformations:
            #These are are used to clean up left over transforms after repositioning, so we don't cost them.
            if t['value'] != {"DELETE"}:
                cost = cost + 1
            if t['attribute'] == "size":
                cost = cost + 1
            if t['attribute'] == "shape":
                cost = cost + 1
                
        return cost


     
    def alter_attribute(self,new_state,transform):
        t = transform
        if not new_state[t['object']]:
            #new_state is empty object, so transforms have no meaning.
            #new_state object doesnt have attribute, so transfrom has no meaning.
            return new_state

        if t['attribute'] == "fill":
            #Fill requires a special case because its additive.
            if 'overwrite' not in t['value']:
                if t['value'] != {'no'} and t['value'] != {'yes'}:
                    for value in t['value']:
                        if t['attribute'] in new_state[t['object']]:
                            new_state[t['object']][t['attribute']].add(value)
                        else:
                            new_state[t['object']][t['attribute']] = set()
                            new_state[t['object']][t['attribute']].add(value)
                    if 'no' in new_state[t['object']][t['attribute']]:
                        new_state[t['object']][t['attribute']].remove('no')
                else:
                    new_state[t['object']][t['attribute']] = t['value']
            else:

                new_state[t['object']][t['attribute']] = t['value'].copy()
                new_state[t['object']][t['attribute']].remove('overwrite')

        elif t['attribute'] == "angle":
            #Angle requires a special case because its applying a rotation to an object that might already be rotated.
            if t['attribute'] in new_state[t['object']]:
                initial_value = int(new_state[t['object']][t['attribute']].copy().pop())
            else:
                #Sometimes there is no angle present in the data
                initial_value = 0


            value = t['value'].copy().pop()

            if value == "reflect_on_y":
                if new_state[t['object']]['shape'] in self.zero_angle_facing_right:
                    #0 = 180, 45 = 135, 90 = 90, 135 = 45, 180 = 0, 225 = 315
                    value = (180 - initial_value) % 360 
                elif new_state[t['object']]['shape'] in self.right_triangle:
                    value = (270 - initial_value) % 360
                else:
                    value = (360 - initial_value) % 360
                new_state[t['object']][t['attribute']] = {str(value)}

                if "vertical-flip" in new_state[t['object']]:
                    if new_state[t['object']]['vertical-flip'] == {"no"}:
                        new_state[t['object']]['vertical-flip'] = {"yes"}
                    else:
                        new_state[t['object']]['vertical-flip'] = {"no"}


            elif value == "reflect_on_x":
                if new_state[t['object']]['shape'] in self.zero_angle_facing_right:
                    value = (360 - initial_value) % 360 
                elif new_state[t['object']]['shape'] in self.right_triangle:
                    value = (90 - initial_value) % 360
                else:
                    value = (180 - initial_value) % 360
                
                new_state[t['object']][t['attribute']] = {str(value)}

                if "vertical-flip" in new_state[t['object']]:
                    if new_state[t['object']]['vertical-flip'] == {"no"}:
                        new_state[t['object']]['vertical-flip'] = {"yes"}
                    else:
                        new_state[t['object']]['vertical-flip'] = {"no"}

            elif value == "DELETE":
                del new_state[t['object']][t['attribute']]

            else:
                add_value = int(value)
                value = initial_value + add_value
                value = value % 360
                new_state[t['object']][t['attribute']] = {str(value)}

                #Check to see if possible ando
                #Handle arrow flip bias on rotation.
                if add_value > 90 and add_value < 270:
                    if 'shape' in new_state[t['object']]:
                        if new_state[t['object']]['shape'] in self.vertical_flip_bias:
                            if "vertical-flip" in new_state[t['object']]:
                                if new_state[t['object']]['vertical-flip'] == {"no"}:
                                    new_state[t['object']]['vertical-flip'] = {"yes"}
                                else:
                                    new_state[t['object']]['vertical-flip'] = {"no"}
            

        else:
            #Mostly we just overwrite the attribute with whatever is in the transform.
            new_state[t['object']][t['attribute']] = t['value']
     
        return new_state

    def difference_attribute(self,attribute,x,y, object_data):

        if attribute == 'fill':
            if y == {'yes'} or y == {'no'}:
                return y
            elif x == {'no'}:
                return y
            else:
                additional_fill = y.difference(x)
                removed_fill = x.difference(y)
                if removed_fill == x and not additional_fill:
                    return {'no'}
                if removed_fill:
                    #This is to handle fill that goes missing between states
                    additional_fill.add('overwrite')
                return additional_fill

        
        if attribute == 'angle':
            x = int(x.copy().pop())
            x = x % 360
            y = int(y.copy().pop())
 

            #Handle inconsitencies in rotation angle in input data.
            if object_data['shape'] in self.zero_angle_facing_right:    
                reflection_on_y = ( 180 - x ) % 360
                if y == reflection_on_y:
                    return {'reflect_on_y'}
            elif object_data['shape'] in self.right_triangle:    
                reflection_on_y = (270 - x) % 360
                if y == reflection_on_y:
                    return {'reflect_on_y'}
            else:    
                # 0 = 0, 45 = 315, 90=270, 180 = 180, 225=315, 315 = 225
                reflection_on_y = (360 - x) % 360
                if y == reflection_on_y:
                    return {'reflect_on_y'}
            
            if object_data['shape'] in self.zero_angle_facing_right:    
                reflection_on_x = (360 - x) % 360
                if y == reflection_on_x:
                    return {'reflect_on_x'}
            elif object_data['shape'] in self.right_triangle:    
                reflection_on_x = (90 - x) % 360
                if y == reflection_on_x:
                    return {'reflect_on_x'}
            else:
                #0 = 180, 45 = 135, 90=90, 135=45, 180 = 0, 225=315, 275=275, 315 = 225
                reflection_on_x = (180 - x) % 360
                if y == reflection_on_x:
                    return {'reflect_on_x'}

            
            if self.NUMERICAL_HANDLER == 'subtraction':
                difference = y - x
                while difference < 0:
                    difference = difference + 360 
                result = difference

            elif self.NUMERICAL_HANDLER == 'addition':
                addition = y + x
                addition = addition % 360 
                result = addition

            return  {str(result)}

        else:
            return  y 


    def apply_global_transformations(self,state,transforms):
        new_state = deepcopy(state)
        objects = new_state.keys()

        for t in transforms:
            for x in objects:
                new_state = self.alter_attribute(new_state,{"object":x,"attribute":t['attribute'],"value":t["value"]})

        return new_state 

    def apply_transformations(self,state,transforms):

        new_state = deepcopy(state)
        for t in transforms:
            if t['value'] == {"DELETE"}:
                if t['attribute'] in new_state[t['object']]:
                    del new_state[t['object']][t['attribute']]
            else:

                if t['object'] in new_state:
                    new_state = self.alter_attribute(new_state,t)
                    
                else:
                    #Sometimes we need to make a new object and fill it with attributes.
                    #This will occur then the target state more objects that the starting state.
                    new_state[t['object']] = {}
                    new_state[t['object']][t['attribute']] = t['value']
              
        for x in new_state.keys():
            #An empty dict is left over when a proxy object is used and all its attributes get deleted
            #This will occur then the target state has less objects that the starting state.
            if new_state[x] == {}:
                del new_state[x]

        return new_state



    def compute_solutions(self,a,b,c):
        #Compute solutions to a is to b then c is to ?
        solutions = []

        
        for h in self.numerical_handlers:
            
            self.NUMERICAL_HANDLER = h

            transformations = [x for x in self.transform_figure(a,b) ]
            global_transformations = [x for x in transformations if x['global'] ]
            transformations = [x for x in transformations if not x['global'] ]

            possible_solutions = []

            
            for g in global_transformations:
                #Global transforms require no analogy.
                solution = self.apply_global_transformations(c,g['transformations'])
                result = {'solution':solution,'cost':g['cost'],'analogy':None,'transformation':g}
                possible_solutions.append(result)

            #For other solutions in case the global one doesn't exist or doesn't work out.
            analogies = [x for x in self.transform_figure(a,c)]
            analogies = [x for x in analogies if not x['global'] ]

            for t in transformations:
                for analogy in analogies:
                    solution = self.apply_transformations(analogy['mapped_end_state'],t['transformations'])
                    result = {'solution':solution,'cost':analogy['cost']+t['cost'],'analogy':analogy,'transformation':t}
                    possible_solutions.append(result)

            #return possible_solutions
            solutions.extend(sorted(possible_solutions,key=lambda x: x['cost']))
        return solutions



    def find_best_solution(self, possible_solutions, available_solutions, check_spatial=True):

        for p in possible_solutions:

            #This bit tests to make sure a transformation actually took place.
            if p['analogy']:
                #analogy is not present on global transforms
                if self.test_figures_are_equivalent(p['analogy']['end_state'],p['solution'],check_spatial):
                    #Transformation did nothing so is not a valid solution as we have already assured something does change between A and B.
                    #Skip this solution.
                    continue

            #Test the generated solution and see if it matches one of the solutions available.
            for i in xrange(len(available_solutions)):           
                if self.test_figures_are_equivalent(p['solution'],available_solutions[i],check_spatial):
                    #solution labels start at 1 not 0
                    return i+1,p['cost'],p

        return None, None, None

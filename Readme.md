An AI agent that can answer 2x1 and 2x2 visual analogy problems based on visual representations. The problems were based on those found in the Raven's Progressive Matrices Human Intelligence test. The agent is trained based on 40 samples. 
The solving of visual analogy problem for 2x2 problems are based on the solving of 2x1 problems. The 2x2 problem is reduced into separate 2x1 problems and then combine the solutions.

Solving a visual analogy problem for a 2x1 question  
1.  Convert image problem data into a set of shape descriptors 
2.  Convert shape descriptors into custom semantic network representation 
3.  Compute a set of possible solutions 
    a.  Calculate possible transformations from A to B (transforms) 
    b.  Calculate possible transformations from A to C (analogies)   
    c.  Combine analogies with transformations to build a set of possible solutions. 
    d.  Sort possible solutions based on overall cost 
4.  Test each solutions against the answer set in ascending order of cost 
    a.  If a solution matches one of the answers return the solution. 
5.  If no solution has been found, try using reversing the problem by working backwards from 
potential answers. 
6.  If valid solution still hasn¡¯t been found, make an arbitrary guess.  


Solving a visual analogy problem for a 2x2 question 
1.  Convert image problem data into a set of shape descriptors 
2.  Convert shape descriptors into custom semantic network representation 
3.  Solve the horizontal 2x1 problem using the 2x1 method above. 
4.  Solve the vertical 2x1 problem using the 2x1 method above. 
5.  Try to combine the solutions from both flows into a single solution. 
6.  If a valid solution cannot be found via combining the 2x1 problems, rank valid solutions from 
both sets of 2x1 problems and solution with the cheapest cost. 
7.  If no solution is considered valid, make an arbitrary guess 
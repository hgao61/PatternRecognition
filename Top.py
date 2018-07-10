

import os
from Agent import Agent
from ProblemSet import ProblemSet

def main():
    sets=[] # The variable 'sets' stores multiple problem sets.
            # Each problem set comes from a different folder in /Problems/


    for file in os.listdir("Problems"): # One problem set per folder in /Problems/
        newSet = ProblemSet(file)       # Each problem set is named after the folder in /Problems/
        sets.append(newSet)
        for problem in os.listdir("Problems" + os.sep + file):  # Each file in the problem set folder becomes a problem in that set.
            f = open("Problems" + os.sep + file + os.sep + problem) # Make sure to add only problem files to subfolders of /Problems/
            newSet.addProblem(f)

    # Initializing problem-solving agent from Agent.java
    agent=Agent()   # The agent will be initialized with its default constructor.

    # Running agent against each problem set
    results=open("Results.txt","w")     # Results will be written to Results.txt.
                                        # Note that each run of the program will overwrite the previous results.
                                        # Do not write anything else to Results.txt during execution of the program.
    for set in sets:
        results.write("%s\n" % set.getName())   # The agent will solve one problem set at a time.
        results.write("%s\n" % "-----------")   # Problem sets will be individually categorized in the results file.

        for problem in set.getProblems():   # Your agent will solve one problem at a time.
            problem.setAnswerReceived(agent.Solve(problem))     # The problem will be passed to your agent as a RavensProblem object as a parameter to the Solve method
                                                                # The agent should return its answer at the conclusion of the execution of Solve.
                                                                # Note that if your agent makes use of RavensProblem.check to check its answer, the answer passed to check() will be used.
                                                                # The agent cannot change its answer once it has checked its answer.

            result=problem.getName() + ": " + problem.getGivenAnswer() + " " + problem.getCorrect() + " (Correct Answer: " + problem.checkAnswer("") + ")"

            results.write("%s\n" % result)
        results.write("\n")

if __name__ == "__main__":
    main()

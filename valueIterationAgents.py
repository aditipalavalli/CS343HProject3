# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        mdp = self.mdp
        iterations = self.iterations

        for i in range (iterations):
            nextValues = self.values.copy()

            for state in mdp.getStates():
                if mdp.isTerminal(state):
                    nextValues[state] = 0
                else: 
                    best = self.computeActionFromValues(state)

                    if best != None:
                        nextValues[state] = self.computeQValueFromValues(state, best)

            self.values = nextValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        discount = self.discount
        values = self.values

        q = 0
        for next, prob in mdp.getTransitionStatesAndProbs(state, action):
            reward = mdp.getReward(state, action, next)
            q += prob * (reward + discount * values[next])
        return q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp

        if mdp.isTerminal(state) or mdp.getStates() is None:
            return None
        
        best = None
        max = float('-inf')

        for action in mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)

            if q > max:
                max = q
                best = action

        return best
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        iterations = self.iterations
        mdp = self.mdp
        states = mdp.getStates()
        values = self.values

        for i in range(iterations):
            state = states[i % len(states)]

            actions = mdp.getPossibleActions(state)

            if mdp.isTerminal(state) or actions is None:
                continue

            bestValue = float("-inf")
            for action in actions:
                if action is None:
                    values[state] = 0
                    continue
                else:
                    bestValue = max(bestValue, self.computeQValueFromValues(state, action))
            values[state] = bestValue
            

            


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        values = self.values

        # compute previous values
        predecessors = {}
        for state in mdp.getStates():
            if mdp.isTerminal(state):
                continue
            for action in mdp.getPossibleActions(state):
                for next, prob in mdp.getTransitionStatesAndProbs(state, action):
                    
                    if predecessors.get(next) is None:
                        predecessors[next] = set()

                    predecessors[next].add(state)


        # initialize empty pq
        pq = util.PriorityQueue()

        for state in mdp.getStates():

            if mdp.isTerminal(state):
                continue

            
            actions = mdp.getPossibleActions(state)
            if actions is None:
                continue
            maxValue = float("-inf")
            for action in actions:
                maxValue = max(maxValue, self.computeQValueFromValues(state, action))

            diff = abs(maxValue - values[state])
            pq.push(state, -diff)

        for i in range(self.iterations):
            if (pq.isEmpty()):
                break
                
            state = pq.pop()

            if mdp.isTerminal(state):
                continue
            
            actions = mdp.getPossibleActions(state)
            if actions is None:
                continue
            maxValue = float("-inf")
            for action in actions:
                maxValue = max(maxValue, self.computeQValueFromValues(state, action))

            
            values[state] = maxValue

            for predecessor in predecessors[state]:

                actions = mdp.getPossibleActions(predecessor)
                if actions is None:
                    continue
                maxValue2 = float("-inf")
                for action in actions:
                    maxValue2 = max(maxValue2, self.computeQValueFromValues(predecessor, action))

                diff2 = abs(values[predecessor]- maxValue2)

                if diff2 > self.theta:
                    pq.update(predecessor, -diff2)
            





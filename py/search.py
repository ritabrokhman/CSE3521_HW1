# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from util import heappush, heappop
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
      """
      Returns the start state for the search problem
      """
      util.raiseNotDefined()

    def isGoalState(self, state):
      """
      state: Search state

      Returns True if and only if the state is a valid goal state
      """
      util.raiseNotDefined()

    def getSuccessors(self, state):
      """
      state: Search state

      For a given state, this should return a list of triples,
      (successor, action, stepCost), where 'successor' is a
      successor to the current state, 'action' is the action
      required to get there, and 'stepCost' is the incremental
      cost of expanding to that successor
      """
      util.raiseNotDefined()

    def getCostOfActions(self, actions):
      """
      actions: A list of actions to take

      This method returns the total cost of a particular sequence of actions.  The sequence must
      be composed of legal moves
      """
      util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure that you implement the graph search version of DFS,
    which avoids expanding any already visited states. 
    Otherwise your implementation may run infinitely!
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Using a stack to follow DFS behavior (LIFO: Last In, First Out)
    from util import Stack

    stack = Stack()
    # Push the start state with an empty path
    stack.push((problem.getStartState(), []))
    # A set to keep track of visited states
    visited = set()

    while not stack.isEmpty():
        # Get the most recently added state
        state, path = stack.pop()
        # If we have already visited this state, skip it
        if state in visited:
            continue
        # Mark the state as visited
        visited.add(state)

        # If we found the goal, return the path taken
        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            # Add each successor state to the stack, appending the action to the path
            stack.push((successor, path + [action]))
    # If no solution is found, return an empty list
    return []
    

def breadthFirstSearch(problem):
    # Using a queue for BFS behavior (FIFO: First In, First Out)
    from util import Queue

    queue = Queue()
    # Start with the initial state and empty path
    queue.push((problem.getStartState(), []))
    # Set to track visited states
    visited = set()

    while not queue.isEmpty():
        # Get the oldest state added to the queue
        state, path = queue.pop()
        # Skip if already visited
        if state in visited:
            continue
        # Mark as visited
        visited.add(state)

        # If the goal is reached, return the path
        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            # Add each successor to the queue, appending the action to the path
            queue.push((successor, path + [action]))
    # If no path to the goal is found, return an empty list
    return []

def uniformCostSearch(problem):
    # Using a priority queue to expand lowest-cost paths first
    from util import PriorityQueue

    pq = PriorityQueue()
    # Start with initial state and cost of 0
    pq.push((problem.getStartState(), []), 0)
    # Dictionary to store the lowest cost to reach each state
    visited = {}

    while not pq.isEmpty():
        # Get the state with the lowest cost
        state, path = pq.pop()

        # If we already visited this state with a lower cost, skip it
        if state in visited and visited[state] <= problem.getCostOfActions(path):
            continue
        # Update the lowest cost to reach this state
        visited[state] = problem.getCostOfActions(path)

        # If the goal is reached, return the path
        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            # Append the new action to the current path
            new_path = path + [action]
            # Add to priority queue with updated cost
            pq.push((successor, new_path), problem.getCostOfActions(new_path))
    # Return an empty list if no solution is found
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    # Using a priority queue to expand paths based on cost + heuristic
    from util import PriorityQueue

    pq = PriorityQueue()
    # Start with the initial state and cost of 0
    pq.push((problem.getStartState(), []), 0)
    # Dictionary to store the lowest cost to reach each state
    visited = {}

    while not pq.isEmpty():
        # Get the state with the lowest estimated cost (cost + heuristic)
        state, path = pq.pop()

        # If we already visited this state with a lower cost, skip it
        if state in visited and visited[state] <= problem.getCostOfActions(path):
            continue
        # Update the lowest cost to reach this state
        visited[state] = problem.getCostOfActions(path)

        # If we reached the goal, return the path
        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            # Append the new action to the current path
            new_path = path + [action]
            # Get the total path cost and add heuristic estimate
            cost = problem.getCostOfActions(new_path) + heuristic(successor, problem)
            # Add to priority queue
            pq.push((successor, new_path), cost)
    # Return an empty list if no solution is found
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

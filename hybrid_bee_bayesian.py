import numpy as np
from bee_optimization import BeeOptimization
from bayesian_optimization import BayesianOptimization

class HybridBeeBayesianOptimizer:
    """
    Hybrid optimization that combines Bee Optimization for global exploration
    and Bayesian Optimization for local refinement.
    """
    
    def __init__(self, search_space, objective_function, bee_params=None, bayesian_params=None):
        self.search_space = search_space
        self.objective_function = objective_function
        
        # Initialize Bee Optimization with given parameters
        self.bee_optimizer = BeeOptimization(search_space, objective_function, **(bee_params or {}))
        
        # Initialize Bayesian Optimization with given parameters
        self.bayesian_optimizer = BayesianOptimization(search_space, objective_function, **(bayesian_params or {}))
        
    def optimize(self, n_bee_iter=40, n_bayesian_iter=60, top_k=8):
        """
        Two-stage optimization:
        Stage 1: Bee Optimization for global exploration (n_bee_iter iterations)
        Stage 2: Bayesian Optimization for local refinement (n_bayesian_iter iterations)
        """
        
        # Stage 1: Bee Optimization
        print("Stage 1: Bee Optimization (Global Exploration)")
        bee_candidates, bee_scores = self.bee_optimizer.optimize(n_iterations=n_bee_iter, return_top_k=top_k)
        
        # Stage 2: Bayesian Optimization initialized with top-k candidates from Bee
        print("Stage 2: Bayesian Optimization (Local Refinement)")
        self.bayesian_optimizer.initialize_with_candidates(bee_candidates, bee_scores)
        bayesian_result = self.bayesian_optimizer.optimize(n_iterations=n_bayesian_iter)
        
        return bayesian_result


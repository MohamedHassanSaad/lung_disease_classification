import numpy as np

class BeeOptimization:
    """
    Bee Optimization Algorithm for hyperparameter tuning.
    """
    
    def __init__(self, search_space, objective_function, num_scouts=15, num_onlookers=20, 
                 local_iterations=20, neighborhood_shrink=0.9):
        self.search_space = search_space
        self.objective_function = objective_function
        self.num_scouts = num_scouts
        self.num_onlookers = num_onlookers
        self.local_iterations = local_iterations
        self.neighborhood_shrink = neighborhood_shrink
        
    def optimize(self, n_iterations, return_top_k=8):
        """
        Run Bee Optimization for a given number of iterations.
        """
        # Initialize scout bees randomly
        scouts = self._initialize_scouts()
        best_candidates = []
        best_scores = []
        
        for iteration in range(n_iterations):
            # Evaluate scout bees
            scout_scores = [self.objective_function(**scout) for scout in scouts]
            
            # Select the best sites for local search (onlooker bees)
            best_indices = np.argsort(scout_scores)[-self.num_onlookers:]
            onlooker_sites = [scouts[i] for i in best_indices]
            
            # Local search around the best sites
            for site in onlooker_sites:
                for _ in range(self.local_iterations):
                    candidate = self._local_search(site, iteration)
                    score = self.objective_function(**candidate)
                    # Update if better candidate found
                    if score > min(best_scores) if best_scores else True:
                        # Maintain the top-k candidates
                        best_candidates.append(candidate)
                        best_scores.append(score)
                        # Keep only top-k
                        if len(best_scores) > return_top_k:
                            idx = np.argmin(best_scores)
                            best_candidates.pop(idx)
                            best_scores.pop(idx)
            
            # Update scout bees for next iteration (randomly explore new areas)
            scouts = self._update_scouts(scouts, scout_scores)
            
            # Shrink neighborhood for local search
            self.neighborhood_shrink *= self.neighborhood_shrink
        
        # Return the top-k candidates and their scores
        return best_candidates, best_scores
    
    def _initialize_scouts(self):
        # Generate random scout bees within the search space
        scouts = []
        for _ in range(self.num_scouts):
            scout = {}
            for param, bounds in self.search_space.items():
                if isinstance(bounds, list):
                    # Categorical parameter
                    scout[param] = np.random.choice(bounds)
                else:
                    # Continuous parameter
                    scout[param] = np.random.uniform(bounds[0], bounds[1])
            scouts.append(scout)
        return scouts
    
    def _local_search(self, site, iteration):
        # Generate a candidate in the neighborhood of the current site
        candidate = {}
        for param, value in site.items():
            if isinstance(value, (int, float)):
                # Continuous parameter: perturb with Gaussian noise
                neighborhood_size = (self.search_space[param][1] - self.search_space[param][0]) * (self.neighborhood_shrink ** iteration)
                candidate[param] = np.clip(np.random.normal(value, neighborhood_size), 
                                          self.search_space[param][0], 
                                          self.search_space[param][1])
            else:
                # Categorical parameter: randomly choose from the options
                candidate[param] = np.random.choice(self.search_space[param])
        return candidate
    
    def _update_scouts(self, scouts, scores):
        # Replace the worst scouts with random new scouts
        worst_indices = np.argsort(scores)[:len(scouts)//2]  # replace half
        new_scouts = self._initialize_scouts()
        for i, idx in enumerate(worst_indices):
            scouts[idx] = new_scouts[i]
        return scouts

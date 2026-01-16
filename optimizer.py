# optimizer.py - Recommendation Engine for Hycarane
import numpy as np
from scipy.optimize import differential_evolution, minimize
import joblib
import pandas as pd
from typing import Dict, List, Tuple, Optional

class HycaraneOptimizer:
    """
    Optimization engine that recommends optimal reactor parameters
    to maximize specific targets while respecting constraints.
    """
    
    def __init__(self, models: Dict, feature_cols: List[str], 
                 feature_bounds: pd.DataFrame):
        """
        Initialize optimizer with trained models and parameter bounds.
        
        Args:
            models: Dictionary of trained models {target_name: model}
            feature_cols: List of feature column names
            feature_bounds: DataFrame with min/max values for each feature
        """
        self.models = models
        self.feature_cols = feature_cols
        self.feature_bounds = feature_bounds
        
        # Create bounds array for optimization
        self.bounds = [(row['min'], row['max']) 
                      for _, row in feature_bounds.iterrows()]
    
    def predict_all_targets(self, X: np.ndarray) -> Dict[str, float]:
        """Predict all targets for given input parameters."""
        predictions = {}
        for target_name, model in self.models.items():
            predictions[target_name] = float(model.predict(X.reshape(1, -1))[0])
        return predictions
    
    def objective_function(self, X: np.ndarray, 
                          optimization_target: str,
                          constraints: Dict[str, Tuple[float, float]],
                          minimize_obj: bool = False) -> float:
        """
        Objective function for optimization.
        
        Args:
            X: Input parameters
            optimization_target: Target to optimize (e.g., 'Net_Profit_Margin_Index')
            constraints: Dict of {target_name: (min_value, max_value)}
            minimize_obj: If True, minimize target; if False, maximize
        
        Returns:
            Objective value (penalized if constraints violated)
        """
        predictions = self.predict_all_targets(X)
        
        # Base objective: the target we want to optimize
        objective = predictions[optimization_target]
        
        # Apply penalty for constraint violations
        penalty = 0
        for target_name, (min_val, max_val) in constraints.items():
            if target_name in predictions:
                pred_val = predictions[target_name]
                
                # Penalty for being below minimum
                if pred_val < min_val:
                    penalty += (min_val - pred_val) ** 2 * 1000
                
                # Penalty for being above maximum
                if pred_val > max_val:
                    penalty += (pred_val - max_val) ** 2 * 1000
        
        # Return objective with penalty
        if minimize_obj:
            return objective + penalty
        else:
            return -(objective - penalty)  # Negative for maximization
    
    def optimize(self, 
                 optimization_target: str,
                 fixed_params: Optional[Dict[str, float]] = None,
                 constraints: Optional[Dict[str, Tuple[float, float]]] = None,
                 minimize: bool = False,
                 method: str = 'differential_evolution') -> Dict:
        """
        Find optimal parameter values.
        
        Args:
            optimization_target: Target to optimize
            fixed_params: Parameters to keep fixed {param_name: value}
            constraints: Constraints on other targets {target: (min, max)}
            minimize: If True, minimize target; if False, maximize
            method: 'differential_evolution' or 'local' (faster but may find local optimum)
        
        Returns:
            Dictionary with optimal parameters and predicted outcomes
        """
        if constraints is None:
            constraints = {}
        
        if fixed_params is None:
            fixed_params = {}
        
        # Create bounds considering fixed parameters
        bounds_to_use = []
        fixed_indices = []
        fixed_values = []
        
        for i, feature in enumerate(self.feature_cols):
            if feature in fixed_params:
                fixed_indices.append(i)
                fixed_values.append(fixed_params[feature])
                bounds_to_use.append((fixed_params[feature], fixed_params[feature]))
            else:
                bounds_to_use.append(self.bounds[i])
        
        # Optimization wrapper
        def obj_wrapper(X):
            return self.objective_function(X, optimization_target, 
                                          constraints, minimize)
        
        # Run optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                obj_wrapper,
                bounds_to_use,
                maxiter=300,
                popsize=15,
                seed=42,
                polish=True,
                atol=1e-6,
                tol=1e-6
            )
        else:  # local optimization (faster)
            # Start from middle of bounds
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_to_use])
            result = minimize(
                obj_wrapper,
                x0,
                method='L-BFGS-B',
                bounds=bounds_to_use
            )
        
        # Get optimal parameters
        optimal_params = {
            feature: float(value) 
            for feature, value in zip(self.feature_cols, result.x)
        }
        
        # Predict all targets at optimal point
        optimal_predictions = self.predict_all_targets(result.x)
        
        return {
            'optimal_parameters': optimal_params,
            'predicted_outcomes': optimal_predictions,
            'optimization_success': result.success,
            'optimization_message': result.message if hasattr(result, 'message') else 'Success',
            'objective_value': -result.fun if not minimize else result.fun,
            'fixed_parameters': fixed_params,
            'constraints_applied': constraints
        }
    
    def multi_objective_optimize(self,
                                 targets: List[str],
                                 weights: Optional[List[float]] = None,
                                 fixed_params: Optional[Dict[str, float]] = None,
                                 constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict:
        """
        Optimize multiple targets simultaneously using weighted sum.
        
        Args:
            targets: List of targets to optimize
            weights: Weights for each target (default: equal weights)
            fixed_params: Parameters to keep fixed
            constraints: Constraints on targets
        
        Returns:
            Optimization results
        """
        if weights is None:
            weights = [1.0 / len(targets)] * len(targets)
        
        if len(weights) != len(targets):
            raise ValueError("Number of weights must match number of targets")
        
        if constraints is None:
            constraints = {}
        
        if fixed_params is None:
            fixed_params = {}
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Create bounds
        bounds_to_use = []
        for i, feature in enumerate(self.feature_cols):
            if feature in fixed_params:
                bounds_to_use.append((fixed_params[feature], fixed_params[feature]))
            else:
                bounds_to_use.append(self.bounds[i])
        
        def multi_obj_wrapper(X):
            predictions = self.predict_all_targets(X)
            
            # Weighted sum of targets (negative for maximization)
            objective = -sum(weights[i] * predictions[target] 
                           for i, target in enumerate(targets))
            
            # Add constraint penalties
            penalty = 0
            for target_name, (min_val, max_val) in constraints.items():
                if target_name in predictions:
                    pred_val = predictions[target_name]
                    if pred_val < min_val:
                        penalty += (min_val - pred_val) ** 2 * 1000
                    if pred_val > max_val:
                        penalty += (pred_val - max_val) ** 2 * 1000
            
            return objective + penalty
        
        result = differential_evolution(
            multi_obj_wrapper,
            bounds_to_use,
            maxiter=300,
            popsize=15,
            seed=42,
            polish=True
        )
        
        optimal_params = {
            feature: float(value) 
            for feature, value in zip(self.feature_cols, result.x)
        }
        
        optimal_predictions = self.predict_all_targets(result.x)
        
        return {
            'optimal_parameters': optimal_params,
            'predicted_outcomes': optimal_predictions,
            'optimization_success': result.success,
            'targets_optimized': targets,
            'weights_used': weights.tolist(),
            'weighted_objective': -result.fun,
            'fixed_parameters': fixed_params,
            'constraints_applied': constraints
        }


# Example usage functions
def load_optimizer_from_models(model_dir: str = '.') -> HycaraneOptimizer:
    """Load trained models and create optimizer."""
    import glob
    
    # Load all models
    models = {}
    for model_file in glob.glob(f'{model_dir}/model_*.pkl'):
        target_name = model_file.split('model_')[1].replace('.pkl', '')
        models[target_name] = joblib.load(model_file)
    
    # Load feature names
    feature_cols = joblib.load(f'{model_dir}/feature_names.pkl')
    
    # Load data to get feature bounds
    df = pd.read_csv(f'{model_dir}/clean_data.csv')
    
    # Calculate bounds for each feature
    bounds_data = []
    for feature in feature_cols:
        bounds_data.append({
            'feature': feature,
            'min': df[feature].min(),
            'max': df[feature].max(),
            'mean': df[feature].mean(),
            'std': df[feature].std()
        })
    
    feature_bounds = pd.DataFrame(bounds_data).set_index('feature')
    
    return HycaraneOptimizer(models, feature_cols, feature_bounds)


def example_optimization_scenarios():
    """Example optimization scenarios."""
    
    optimizer = load_optimizer_from_models()
    
    # Scenario 1: Maximize profit with quality constraints
    print("Scenario 1: Maximize Profit with Quality Constraints")
    result1 = optimizer.optimize(
        optimization_target='Net_Profit_Margin_Index',
        constraints={
            'H2_Purity_Post_': (0.95, 1.0),  # At least 95% purity
            'Carbon_Quality_': (0.90, 1.0),  # At least 90% quality
        },
        minimize=False
    )
    print(f"Optimal Profit: {result1['predicted_outcomes']['Net_Profit_Margin_Index']:.4f}")
    print(f"H2 Purity: {result1['predicted_outcomes']['H2_Purity_Post_']:.4f}")
    
    # Scenario 2: Maximize H2 yield with fixed temperature
    print("\nScenario 2: Maximize H2 Yield at Fixed Temperature")
    result2 = optimizer.optimize(
        optimization_target='H2_Yield_Rate',
        fixed_params={'Reactor_Temper': 800.0},
        minimize=False
    )
    print(f"Optimal H2 Yield: {result2['predicted_outcomes']['H2_Yield_Rate']:.4f}")
    
    # Scenario 3: Multi-objective (profit + yield)
    print("\nScenario 3: Balance Profit and H2 Yield")
    result3 = optimizer.multi_objective_optimize(
        targets=['Net_Profit_Margin_Index', 'H2_Yield_Rate'],
        weights=[0.6, 0.4]  # 60% profit, 40% yield
    )
    print(f"Balanced Profit: {result3['predicted_outcomes']['Net_Profit_Margin_Index']:.4f}")
    print(f"Balanced Yield: {result3['predicted_outcomes']['H2_Yield_Rate']:.4f}")


if __name__ == "__main__":
    example_optimization_scenarios()
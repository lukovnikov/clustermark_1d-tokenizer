import json
import os
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any, Callable


class ExperimentSelector:
    """
    A class to select experiment directories based on their args.json settings.
    
    Supports filtering by exact values, tuples (for multiple values or ranges),
    and string matching patterns.
    """
    
    def __init__(self, parent_directory: Union[str, Path]):
        """
        Initialize the ExperimentSelector.
        
        Args:
            parent_directory: Path to the directory containing experiment folders
        """
        self.parent_directory = Path(parent_directory)
        if not self.parent_directory.exists():
            raise ValueError(f"Parent directory {parent_directory} does not exist")
        
        self._experiment_dirs = self._find_experiment_directories()
    
    def _find_experiment_directories(self) -> List[Path]:
        """Find all subdirectories that contain an args.json file."""
        experiment_dirs = []
        
        for item in self.parent_directory.iterdir():
            if item.is_dir():
                args_file = item / "args.json"
                if args_file.exists():
                    experiment_dirs.append(item)
        
        return experiment_dirs
    
    def _load_args(self, experiment_dir: Path) -> Dict[str, Any]:
        """Load args.json from an experiment directory."""
        args_file = experiment_dir / "args.json"
        try:
            with open(args_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load args.json from {experiment_dir}: {e}")
            return {}
    
    def _matches_criteria(self, value: Any, criteria: Any) -> bool:
        """
        Check if a value matches the given criteria.
        
        Args:
            value: The value from args.json
            criteria: The criteria to match against. Can be:
                     - Exact value for exact match
                     - slice(start, stop) for range matching
                     - tuple or set for multiple acceptable values
                     - string for pattern matching (case-insensitive contains)
        
        Returns:
            True if the value matches the criteria, False otherwise
        """
        if isinstance(criteria, slice):
            # Range matching using slice(start, stop)
            if not isinstance(value, (int, float)):
                return False
            start = criteria.start if criteria.start is not None else float('-inf')
            stop = criteria.stop if criteria.stop is not None else float('inf')
            return start <= value <= stop
        
        elif isinstance(criteria, (tuple, set)):
            # Multiple acceptable values
            return value in criteria
        
        elif isinstance(criteria, str) and isinstance(value, str):
            # String pattern matching (case-insensitive contains)
            return criteria.lower() in value.lower()

        elif isinstance(criteria, Callable):
            # If criteria is a callable (function), use it to check the value
            return criteria(value)
        
        else:
            # Exact match
            return value == criteria
    
    def select(self, **kwargs) -> List[Path]:
        """
        Select experiment directories based on the given criteria.
        
        Args:
            **kwargs: Key-value pairs where keys are parameter names in args.json
                     and values are the criteria to match against.
                     
                     Criteria can be:
                     - Exact values: select(num_clusters=5)
                     - Tuples/sets for multiple values: select(num_clusters=(3, 5, 7)) or select(num_clusters={3, 5, 7})
                     - slice() for ranges: select(wm_red_penalty=slice(0.1, 0.5))
                     - Strings for pattern matching: select(model_type="bert")
        
        Returns:
            List of Path objects for matching experiment directories
        """
        matching_dirs = []
        
        for exp_dir in self._experiment_dirs:
            args = self._load_args(exp_dir)
            
            # Check if all criteria match
            matches_all = True
            for param_name, criteria in kwargs.items():
                if param_name not in args:
                    matches_all = False
                    break
                
                if not self._matches_criteria(args[param_name], criteria):
                    matches_all = False
                    break
            
            if matches_all:
                matching_dirs.append(exp_dir)
        
        return matching_dirs
    
    def get_all_experiments(self) -> List[Path]:
        """Get all experiment directories."""
        return self._experiment_dirs.copy()
    
    def get_experiment_args(self, experiment_dir: Path) -> Dict[str, Any]:
        """Get the args.json content for a specific experiment directory."""
        return self._load_args(experiment_dir)
    
    def list_unique_values(self, param_name: str) -> List[Any]:
        """
        Get all unique values for a specific parameter across all experiments.
        
        Args:
            param_name: Name of the parameter to check
            
        Returns:
            List of unique values found for that parameter
        """
        unique_values = set()
        
        for exp_dir in self._experiment_dirs:
            args = self._load_args(exp_dir)
            if param_name in args:
                unique_values.add(args[param_name])
        
        return sorted(list(unique_values))
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of all experiments and their parameters.
        
        Returns:
            Dictionary with experiment count and parameter summaries
        """
        all_params = set()
        param_values = {}
        
        for exp_dir in self._experiment_dirs:
            args = self._load_args(exp_dir)
            all_params.update(args.keys())
            
            for param, value in args.items():
                if param not in param_values:
                    param_values[param] = set()
                param_values[param].add(value)
        
        # Convert sets to sorted lists
        for param in param_values:
            param_values[param] = sorted(list(param_values[param]))
        
        return {
            'total_experiments': len(self._experiment_dirs),
            'all_parameters': sorted(list(all_params)),
            'parameter_values': param_values
        }


# Example usage
if __name__ == "__main__":
    # Example usage
    selector = ExperimentSelector("experiments_v1/")

    results = selector.select(
        expprefix=lambda x: x.startswith("gen_clean"),
    )
    for result in results:
        print(result)
    
    # Select experiments with exact values
    results = selector.select(
        wm_red_penalty=5,
        num_clusters=64,
    )
    for result in results:
        print(result)
    
    # Select experiments with multiple acceptable values (tuple or set)
    results = selector.select(
        num_samples={2000,} # Using set
    )
    print(results)
    
    # Select experiments with ranges using slice()
    results = selector.select(
        wm_red_penalty=slice(0.1, 0.5),      # Between 0.1 and 0.5
        wm_green_fraction=slice(0.2, 0.8),   # Between 0.2 and 0.8
        num_samples=slice(1000, None)        # 1000 or greater
    )
    print(results)
    
    # Select experiments with string pattern matching
    results = selector.select(
        model_type="bert",  # Contains "bert"
        dataset="mnist"     # Contains "mnist"
    )
    print(results)
    
    # Mixed criteria using different types
    results = selector.select(
        wm_red_penalty=slice(0.1, 0.3),      # Range using slice
        num_clusters={3, 5, 7},              # Multiple values using set
        num_samples=(1000, 2000),            # Multiple values using tuple
        model_type="transformer"             # String pattern
    )
    print(results)
    
    # Open-ended ranges
    results = selector.select(
        wm_red_penalty=slice(0.1, None),     # 0.1 or greater
        num_clusters=slice(None, 10)         # 10 or less
    )
    print(results)
    
    # Get summary of all experiments
    summary = selector.summary()
    print(f"Found {summary['total_experiments']} experiments")
    print(f"Parameters: {summary['all_parameters']}")
    
    # Get unique values for a specific parameter
    unique_clusters = selector.list_unique_values('num_clusters')
    print(f"Unique num_clusters values: {unique_clusters}")
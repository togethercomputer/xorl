"""
Flatten Collator for converting nested batch structures to flat structures.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FlattenCollator:
    """
    A collator that flattens a list of lists of dicts into a single list of dicts.
    
    Input: [[dict1, dict2], [dict3], [dict4, dict5]]
    Output: [dict1, dict2, dict3, dict4, dict5]
    
    This is useful when you have batched data in a nested structure and want to
    flatten it for processing by models that expect a simple list of examples.
    """

    def __call__(self, all_features) -> List[Dict[str, Any]]:
        """
        Flatten a list of lists of dicts into a single list of dicts.
        If the input is already a flat list, return it as-is.
        
        Args:
            all_features: Either a list of lists of dicts, or a flat list of dicts
        
        Returns:
            Flattened list of dicts
        
        Example:
            >>> collator = FlattenCollator()
            >>> # Nested input - will be flattened
            >>> input_data = [
            ...     [{"input_ids": [1, 2], "labels": [3, 4]}, 
            ...      {"input_ids": [5, 6], "labels": [7, 8]}],
            ...     [{"input_ids": [9, 10], "labels": [11, 12]}]
            ... ]
            >>> output = collator(input_data)
            >>> # output = [
            >>> #     {"input_ids": [1, 2], "labels": [3, 4]},
            >>> #     {"input_ids": [5, 6], "labels": [7, 8]},
            >>> #     {"input_ids": [9, 10], "labels": [11, 12]}
            >>> # ]
            >>> 
            >>> # Already flat - returned as-is
            >>> flat_input = [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]
            >>> output = collator(flat_input)  # same as flat_input
        """
        # Check if already flat (empty list or first element is a dict)
        if not all_features or isinstance(all_features[0], dict):
            return all_features
        
        # Flatten nested structure
        flattened = []
        for batch in all_features:
            flattened.extend(batch)
        return flattened


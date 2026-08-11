from typing import Any, Callable, Dict, Sequence


class CollatePipeline:
    def __init__(self, data_collators: Sequence[Callable]):
        """
        Args:
            data_collators: collators to apply in order
        """
        self.data_collators = list(data_collators)

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        """
        process data batch through data collators.

        Args:
            batch: the original input data batch

        Returns:
            batch: the processed data batch

        """
        for data_collator in self.data_collators:
            batch = data_collator(batch)
        return batch

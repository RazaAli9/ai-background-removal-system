# import time

# from app.core import logger


# class Timer:
#     """
#     Context manager for measuring execution time.
#     """

#     def __init__(self, task_name: str):
#         self.task_name = task_name
#         self.start_time = None

#     def __enter__(self):
#         self.start_time = time.perf_counter()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         elapsed = time.perf_counter() - self.start_time

#         logger.info(
#             f"{self.task_name} completed in "
#             f"{elapsed:.2f} seconds."
#         )

import time
from app.core import logger


class Timer:
    """
    Context manager for measuring execution time.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.start_time = None
        self.elapsed_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = (
            time.perf_counter() - self.start_time
        )

        logger.info(
            f"{self.task_name} completed in "
            f"{self.elapsed_time:.2f} seconds."
        )
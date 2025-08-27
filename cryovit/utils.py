"""Utility functions to process data and models in a format recognizable by CryoVIT."""

import string
import random

def id_generator(size: int = 6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
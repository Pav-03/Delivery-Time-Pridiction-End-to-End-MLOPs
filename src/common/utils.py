import numpy as np

def to_python_types(obj):
    """
    Convert numpy/pandas types to native Python types for JSON serialization.
    Usage: json.dump(to_python_types(data), f)
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(item) for item in obj]
    return obj
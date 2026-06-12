def get_batches(lst, n):
    """Yield successive n-sized batches from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

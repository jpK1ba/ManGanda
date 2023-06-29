import joblib, os


def save_pkl(obj, name, prompt=True):
    """Save an object to a pickle file.
    """
    folder = 'pickles'
    ext = '.pkl'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if name[-4:] == ext:
        fp = os.path.join(folder, name)
    else:
        fp = os.path.join(folder, name+ext)
    joblib.dump(obj, fp)
    
    if prompt:
        print('Object pickled for future use.')
    
    return

def load_pkl(name, prompt=False):
    """Load an object from a pickle file.
    """
    folder = 'pickles'
    ext = '.pkl'
    if not os.path.exists(folder):
        raise ValueError("'pickles' folder does not exist.")
    
    if name[-4:] == ext:
        fp = os.path.join(folder, name)
    else:
        fp = os.path.join(folder, name+ext)
    pkl = joblib.load(fp)
    
    if prompt:
        print('Pickle file loaded.')
    
    return pkl

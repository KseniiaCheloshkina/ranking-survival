import numpy as np

def filter_relevant_contragent(t, y, t_i, y_i):
    
    """
    According to definition of Harell's C-index return indices of comparable pairs
    for observation with label y_i and time-to-event t_i
    """
    
    if y_i == 0:
        ind_y = np.where(y == 1)[0]
        ind_t = np.where(t < t_i)[0]
        ind = np.intersect1d(ind_y, ind_t)
    else:
        ind_y1 = np.where(y == 1)[0]
        ind_y2 = np.where(y == 0)[0] 
        ind_t2 = np.where(t > t_i)[0]        
        ind = np.union1d(ind_y1, np.intersect1d(ind_y2, ind_t2))   
    return ind


def get_time_bins(t, n_time_bins):

    """
    Get equal size bins
    """
    
    percent_list = np.linspace(0, 100, n_time_bins + 1, dtype=np.int)
    bins = np.percentile(a=t, q=percent_list[1:-1])
    q = np.digitize(t, bins)
    if n_time_bins != np.unique(q).shape[0]:
        raise Exception("There is too large value for n_time_bins selected")
    return q
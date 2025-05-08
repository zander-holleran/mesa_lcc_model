from scipy.stats import truncnorm

def make_truncnorm(upper, lower, var, mean=None):
    '''
    If a mean is not passed the average of upper and lower will be used
    '''
    if not mean:
        mean = (upper+lower)/2

    return truncnorm((lower - mean)/var, (upper - mean)/var, loc=mean, scale=var)

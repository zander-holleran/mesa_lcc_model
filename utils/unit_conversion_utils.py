
def get_mps(mph):
    """
    Convert miles per hour to meters per second.
    """
    return mph * 1609.34 / 3600

def get_mph(mps):
    """
    Convert meters per second to miles per hour.
    """
    return mps * 3600 / 1609.34

def meters_to_feet(meters):
    """
    Convert meters to feet.
    """
    return meters * 3.28084

def feet_to_meters(feet):
    """
    Convert feet to meters.
    """
    return feet / 3.28084

def meters_to_miles(meters):
    """
    Convert meters to miles.
    """
    return meters / 1609.34

def sec_after_five(hr):
    '''convert hr of day to the secounds after five am'''
    hr_after_five = hr - 5 
    return round(hr_after_five*3600)
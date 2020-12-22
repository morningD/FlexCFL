
class FederatedBase(object):
    '''
    Base class of trainer like fedavg fedgroup and fedsplit
    '''
    def __init__(self, params):
        params = {'dataset': 'mnist'}
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)

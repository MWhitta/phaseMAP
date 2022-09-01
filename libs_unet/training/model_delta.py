import torch

def state_diff(new_param, init_wts):

    """ Function to evaluate changes to tunable model parameters during training.
        Arguments:
            new_param: current named_parameters() iter
            init_wts: prior weights as dict
            
        Returns:
            model: dict with same key structure as inputs, with dict of results
                {node1: {wt_norm:123, wt_var:456, wt_mad:23, grad_norm:123}}
                
    """
    
    results_dict = {}

    #iterators can only be traversed once so transform to dict
    new_dict = {}
    for k, v in new_param:
        new_dict[k] = { 'wt':v, 'grad':v.grad }

    #calculate the corresponding dictionary of element-wise wt changes
    diff_dict = {}
    for k,v in new_dict.items():
        diff_dict[k] = v['wt'] - init_wts[k]
    #process node by node in the model
    for k, v in new_dict.items(): 
        #don't process the counters for batchnorm nodes
        if v['wt'].dtype != torch.int64:
            results_dict[k] = {}
            results_dict[k]['wt_mav'] = torch.mean(torch.abs(v['wt']))
            results_dict[k]['wt_var'] = torch.var(v['wt'])
            results_dict[k]['wtdiff_mav'] = torch.mean(torch.abs(diff_dict[k]))
            results_dict[k]['grad_mav'] = torch.mean(torch.abs(v['grad']))
            
    return results_dict


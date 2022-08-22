import torch

def state_diff(new_dict, old_dict, type = 'model'):

    """ Function to evaluate changes to tunable model parameters during training.
        Arguments:
            new_dict: current state dictionary
            old_dict: prior state dictionary
            type = 'model'|'optimizer'
            
        Returns:
            model: dict with same key structure as inputs, with summary and detail tensors
                summary for each key: size(3,3) 
                dim0: new, old, delta
                dim1: meanabsval, range, var
                #note delta is element-wise subtraction new-old
            optimizer: dict of dict with var_name:{new:value, old:value}
    """
    
    results_dict = {}
    match type:
        case 'model':
            #calculate the corresponding dictionary of element-wise changes
            diff_dict = {}
            for key,value in new_dict.items():
                diff_dict[key] = value - old_dict[key]

            #going to populate a node dict with tensors of results
            #process node by node in the model across all three dicts
            for node in new_dict: #iterate keys, order maintained since python 3.6
              summary = torch.zeros(3,3)
              summary[0,0] = torch.mean(torch.abs(new_dict[node])).item()
              summary[0,1] = torch.max(new_dict[node]).item() - torch.min(new_dict[node]).item()
              summary[0,2] = torch.var(new_dict[node]).item()
              summary[1,0] = torch.mean(torch.abs(old_dict[node])).item()
              summary[1,1] = torch.max(old_dict[node]).item() - torch.min(old_dict[node]).item()
              summary[1,2] = torch.var(old_dict[node]).item()
              summary[2,0] = torch.mean(torch.abs(diff_dict[node])).item()
              summary[2,1] = torch.max(diff_dict[node]).item() - torch.min(diff_dict[node]).item()
              summary[2,2] = torch.var(diff_dict[node]).item()
              results_dict[node] = summary
            return results_dict
        
        case 'optimizer':
            for key, value in new_dict:
                results_dict[key] = {"new":value, "old":old_dict[key]}
            return results_dict
        
        case _:
            raise ValueError(f"Invalid type parameter {type} provided")
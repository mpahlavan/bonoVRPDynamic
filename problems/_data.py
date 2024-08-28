import torch
from torch.utils.data import Dataset

class VRP_Dataset(Dataset):
    CUST_FEAT_SIZE = 3

    @classmethod
    def generate(cls,
            batch_size = 1,
            cust_count = 100,
            veh_count = 25,
            veh_capa = 200,
            veh_speed = 1,
            min_cust_count = None,
            cust_loc_range = (0,101),
            cust_dem_range = (5,41)
            ):
       #batch_size is a parameter that determines the number of instances or samples in a single batch of data. 
       #cust_count represents the number of customers (nodes other than the depot)
       #The total number of nodes in each instance, including the depot, would be cust_count + 1
      
        size = (batch_size, cust_count, 1)

        # Sample locs        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count+1, 2), dtype = torch.float)
        # Sample dems             q_j ~ U(5,  40)
        #generates random integer values for demands (dems) using the specified range.
        dems = torch.randint(*cust_dem_range, size, dtype = torch.float)#TODO dems=1 for all nodes.
         #NOTE (FOR the next step improvement) site triage / numerous patient in each node.

        # Regroup all features in one tensor
        #Concatenates the locs tensor (excluding the first column) with the dems tensor along the third dimension.
        customers = torch.cat((locs[:,1:], dems), 2)
        

        # Add depot node
        #Creates a tensor depot_node with zeros and a size of (batch_size, 1, cls.CUST_FEAT_SIZE).
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
         #Sets the first two columns of depot_node to be equal to the first column of locs.
        depot_node[:,:,:2] = locs[:,0:1]
       #Concatenates depot_node with the customers tensor along the second dimension.
        nodes = torch.cat((depot_node, customers), 1)
        

        if min_cust_count is None:
            cust_mask = None
            #Checks if min_cust_count is None. If so, sets cust_mask to None.
            #Otherwise, generates random integer values for counts and creates a boolean mask cust_mask
        else:
            counts = torch.randint(min_cust_count+1, cust_count+2, (batch_size, 1), dtype = torch.int64)
            cust_mask = torch.arange(cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0
            # this block of code introduces a mechanism to randomly mask (set to zero) a certain number of customer nodes in the dataset.

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        #This line of code creates an instance of the VRP_Dataset class and returns it.
        return dataset
        

    def __init__(self, veh_count, veh_capa, veh_speed, nodes, cust_mask = None):
        self.veh_count = veh_count
        self.veh_capa = veh_capa
        self.veh_speed = veh_speed
        #These are instance variables that store information about the number of vehicles, vehicle capacity, and vehicle speed.
        #Initializes an instance of the class with the provided parameters.

        # This instance variable stores the tensor nodes that contains information about the depot and customer nodes
        self.nodes = nodes
        #The nodes.size() method returns a tuple with three elements: the first element is the batch size, the second element is the number of nodes, and the third element is the number of features (or dimensions) per node. 
        self.batch_size, self.nodes_count, d = nodes.size()
        
        if d != self.CUST_FEAT_SIZE:
            raise ValueError("Expected {} customer features per nodes, got {}".format(
                self.CUST_FEAT_SIZE, d))
        self.cust_mask = cust_mask
        #This block checks if the third dimension (d) of the nodes tensor is equal to self.CUST_FEAT_SIZE. If not, it raises a ValueError indicating that the expected number of customer features per node is not met.
        

    def __len__(self):
        # returns the number of examples or batches in the dataset.
        return self.batch_size
        
    def __getitem__(self, i):
        if self.cust_mask is None:
            return self.nodes[i]
        else:
            return self.nodes[i], self.cust_mask[i]
    # it is used to define the behavior of the indexing operator [] for instances of a class. this method is typically implemented to retrieve a specific example or batch from the dataset.

    def nodes_gen(self):
        if self.cust_mask is None:
            yield from self.nodes
        else:
            yield from (n[m^1] for n,m in zip(self.nodes, self.cust_mask))
            #this line uses a generator expression to yield nodes based on the customer mask. Here, n[m^1] selects the nodes where the corresponding mask is False (using the bitwise XOR ^1 to invert the mask).
    #

    def normalize(self):
        #this part  is responsible for normalizing the location and demand features of the nodes in the dataset

        #self.nodes has shape (batch_size, nodes_count, 3), where the last dimension represents x, y, and demand
        #self.nodes[:,:,:2].max().item(): This extracts the maximum value among the x and y coordinates of all nodes in the dataset.
        #self.nodes[:,:,:2].min().item(): This extracts the minimum value among the x and y coordinates of all nodes in the dataset.
        loc_scl, loc_off = self.nodes[:,:,:2].max().item(), self.nodes[:,:,:2].min().item()
        
        #loc_off(location offset), which is the minimum value of the x and y coordinates.
        #loc_scl( location scale)is difference between the maximum and minimum values of the x and y coordinates.
        loc_scl -= loc_off

        #part of the normalization process.
        #This step ensures that the minimum x and y coordinates become zero after normalization, and the entire set of coordinates is centered around zero.
        self.nodes[:,:,:2] -= loc_off
        #This step ensures that the x and y coordinates are normalized to a common scale after the centering process achieved by subtracting the location offset and scaling by the location scale.
        self.nodes[:,:,:2] /= loc_scl
        # normalizes the demand of each node
        #TODO dems=1 for all nodes.
        self.nodes[:,:, 2] /= self.veh_capa
        

        self.veh_capa = 1
        self.veh_speed = 1

        return loc_scl, 1
        #this part is responsible for normalizing the location features of the nodes.

    def save(self, fpath):
        torch.save({
            "veh_count": self.veh_count,
            "veh_capa": self.veh_capa,
            "veh_speed": self.veh_speed,
            "nodes": self.nodes,
            "cust_mask": self.cust_mask
            }, fpath)
            #when you call save with a file path, it serializes the relevant attributes and tensors of the VRP_Dataset object
            # and saves them to the specified file using the PyTorch torch.save function.

    @classmethod
    def load(cls, fpath):
        return cls(**torch.load(fpath))

from marpdan.problems import VRP_Dataset
import torch

class VRPTW_Dataset(VRP_Dataset):
    CUST_FEAT_SIZE = 6
    #  6 features customer:
    #x-coordinate of the customer's location:dest[:, :, 0]
    #y-coordinate of the customer's location: dest[:, :, 1]
    #Demand of the customer:dest[:, :, 2]
    #%%%Time window start for the customer:dest[:, :, 3] #TODO lower bound TW = 0
    #%%%Time window end for the customer:dest[:, :, 4]
    #Service time for the customer:dest[:, :, 5]

    @classmethod
    def generate(cls,
            batch_size = 1,
            cust_count = 100,
            veh_count = 25,
            veh_capa = 200,
            veh_speed = 1,
            min_cust_count = None,
            cust_loc_range = (0,101),
            cust_dem_range = (5,41),
            horizon = 480,
            cust_dur_range = (10,31),
            tw_ratio = 0.5,
            cust_tw_range = (30,91)
            ):
        
        #batch_size is a parameter that determines the number of instances or samples in a single batch of data. 
       #cust_count represents the number of customers (nodes other than the depot)
       #The total number of nodes in each instance, including the depot, would be cust_count + 1
        size = (batch_size, cust_count, 1)

        # Sample locs        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count+1, 2), dtype = torch.float)
        # Sample dems             q_j ~ U(5,  40)
        dems = torch.randint(*cust_dem_range, size, dtype = torch.float)
        # Sample serv. time       s_j ~ U(10, 30)
        #generates random service times for customers in batch.
        durs = torch.randint(*cust_dur_range, size, dtype = torch.float)

        # Sample TW subset            ~ B(tw_ratio)
        #The condition uses isinstance to check the type of tw_ratio:
        #If it's a simple float, a single Bernoulli distribution is used / If it's a single-element list, the same logic applies./ If it's a list of floats, one ratio is randomly chosen for each batch and expanded to apply to all customers
        if isinstance(tw_ratio, float):
            # binary mask has_tw indicating whether each customer has a time window. The probability of having a time window is determined by tw_ratio.
            has_tw = torch.empty(size).bernoulli_(tw_ratio)
        elif len(tw_ratio) == 1:
            
            has_tw = torch.empty(size).bernoulli_(tw_ratio[0])
        else: # tuple of float
            ratio = torch.tensor(tw_ratio)[torch.randint(0, len(tw_ratio), (batch_size,), dtype = torch.int64)]
            has_tw = ratio[:,None,None].expand(*size).bernoulli() #TODO has_tw=1 for all of nodes

        # Sample TW width        tw_j = H if not in TW subset
        #                        tw_j ~ U(30,90) if in TW subset
        # tws variable: This stores the generated time window widths for all customers.
        # first part fills all elements with the horizon (horizon) for customers without time windows (where has_tw is 0). torch.full(size, horizon) creates a tensor of the same size as the customer data, filled with the horizon value. 
        # second part assigns random widths within the specified range (cust_tw_range) to customers with time windows (where has_tw is 1).
        # torch.randint() generates random integers within the specified range (cust_tw_range) for customers with time windows.
        # the addition operator combines the cases, resulting in a complete tensor of time window widths for all customers.
        tws = (1 - has_tw) * torch.full(size, horizon) \
                + has_tw * torch.randint(*cust_tw_range, size, dtype = torch.float)

        # The resulting tensor tts will have dimensions (batch_size, cust_count, cust_count), where each element [i, j, k] represents the travel time from the depot to customer j in batch i with vehicle k.
        #pow(2): This squares each element in the pairwise difference matrix. 
        #sum(-1): This sums the squared values along the last dimension (-1).

        #locs[:,None,0:1,:] essentially extracts the x-coordinates of all customers and the depot and expands them into a separate dimension for each element. This allows for efficient pairwise comparisons and calculations in subsequent operations, such as calculating travel times or distances between customers and the depot.
        #Imagine you have a spreadsheet with customer locations in different columns. You want to calculate the distance between each customer and a specific point (depot) based on their x-coordinates. To do this efficiently, you would need to isolate and work with the x-coordinate column alone. This is what locs[:,None,0:1,:] accomplishes in the context of VRPTW dataset generation.
        tts = (locs[:,None,0:1,:] - locs[:,1:,None,:]).pow(2).sum(-1).pow(0.5) / veh_speed #????? NOTE why we do not consider distance between two node
        # Sample ready time       e_j = 0 if not in TW subset
        #                         e_j ~ U(a_j, H - max(tt_0j + s_j, tw_j)) 
        #                          a_j is appearence time
        #calculates the ready times (Latest Feasible Ready Time) for all customers based on their time window presence and other factors. 
        # torch.rand(size): This generates random values between 0 and 1 for all customers with time windows.
        rdys = has_tw * (torch.rand(size) * (horizon - torch.max(tts + durs, tws)))#TODO uncertainty
        #  Rounds the ready times down to integers. This ensures that the ready times are integers representing the time slots when the customer can start service.
        rdys.floor_()

        # Regroup all features in one tensor
        customers = torch.cat((locs[:,1:], dems, rdys, rdys + tws, durs), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:,:,:2] = locs[:,0:1]
        depot_node[:,:,4] = horizon
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(min_cust_count+1, cust_count+2, (batch_size, 1), dtype = torch.int64)
            cust_mask = torch.arange(cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        return dataset

    def normalize(self):
        loc_scl, loc_off = self.nodes[:,:,:2].max().item(), self.nodes[:,:,:2].min().item()
        loc_scl -= loc_off
        t_scl = self.nodes[:,0,4].max().item()

        self.nodes[:,:,:2] -= loc_off
        self.nodes[:,:,:2] /= loc_scl
        self.nodes[:,:, 2] /= self.veh_capa
        self.nodes[:,:,3:] /= t_scl

        self.veh_capa = 1
        self.veh_speed *= t_scl / loc_scl

        return loc_scl, t_scl

from marpdan.problems import VRP_Environment
import torch

class VRPTW_Environment(VRP_Environment):
    CUST_FEAT_SIZE = 6
    #  6 features customer:
    #x-coordinate of the customer's location:dest[:, :, 0]
    #y-coordinate of the customer's location: dest[:, :, 1]
    #Demand of the customer:dest[:, :, 2]
    #Time window start for the customer:dest[:, :, 3] #TODO lower bound TW = 0
    #Time window end for the customer:dest[:, :, 4]
    #Service time for the customer:dest[:, :, 5]

    #data: This parameter holds information about the vehicles, their capacity, speed, and the initial state of the problem.
    #nodes: This parameter represents the nodes in the problem, including customer locations, demands, and other information. If not provided, it defaults to None.
    #cust_mask: This is a mask that can be applied to exclude certain customers from the problem. If not provided, it defaults to None.
    #pending_cost: This parameter represents the cost associated with pending deliveries. In the context of the problem, it might be a penalty for deliveries that are not completed on time. It defaults to 2 if not provided.
    #late_cost: This parameter represents the cost associated with late deliveries. It is specific to the VRPTW (Vehicle Routing Problem with Time Windows) and reflects the penalty for delivering a product outside the specified time window. It defaults to 1 if not provided.

    def __init__(self, data, nodes = None, cust_mask = None,
            pending_cost = 2, late_cost = 1):
    #This method is responsible for setting up the basic environment state based on the problem data.
        super().__init__(data, nodes, cust_mask, pending_cost)
        # represents the penalty or cost associated with late deliveries in VRPTW
        self.late_cost = late_cost
    

    def _sample_speed(self):
        return self.veh_speed

    def _update_vehicles(self, dest):
        # dest[:, 0, :2] extracts the x and y coordinates of the destinations for the first vehicle in each batch.
        #dist: Calculates the pairwise Euclidean distance between the current vehicle positions and the destination positions
        dist = torch.pairwise_distance(self.cur_veh[:,0,:2], dest[:,0,:2], keepdim = True)
        #Computes the time it takes to travel from the current positions to the destinations by dividing the distance (dist) by the sampled speed (self._sample_speed()).
        tt = dist / self._sample_speed()
        #Computes the arrival time at each destination by taking the maximum of the sum of the current time (self.cur_veh[:,:,3]) and the travel time (tt) and the specified arrival time for the destination (dest[:,:,3]).
        arv = torch.max(self.cur_veh[:,:,3] + tt, dest[:,:,3]) #TODO in our case dest[:,:,3]=0
        #Calculates the lateness at each destination by subtracting the destination's due time (dest[:,:,4]) from the computed arrival time (arv). The clamp_(min=0) ensures that if the vehicle arrives earlier than the due time, the lateness is set to zero.
        # The clamp_(min=0) ensures the lateness is non-negative.
        late = ( arv - dest[:,:,4] ).clamp_(min = 0) 

        self.cur_veh[:,:,:2] = dest[:,:,:2]#updates the vehicle's position to the chosen customer's location.
        self.cur_veh[:,:,2] -= dest[:,:,2]#updates the vehicle's remaining capacity by subtracting the chosen customer's demand.
        self.cur_veh[:,:,3] = arv + dest[:,:,5]#This line updates the vehicle's arrival time to the calculated arrival time plus the customer's service time.

        # self.vehicles: This tensor represents the state of all vehicles in the environment.
        # scatter(1, ...) : The scatter method is used to scatter/gather values along a particular dimension of the tensor.
        # 1: Specifies that the scattering operation is along the second dimension (the vehicle dimension).
         # self.cur_veh_idx: The indices indicating which vehicle to update for each batch.
         # [:,:,None]: Adds a new dimension to the indices (None is equivalent to adding a dimension).
         # expand(-1,-1,self.VEH_STATE_SIZE): Expands the dimensions to match the size of self.cur_veh. It repeats the indices for each state feature of a vehicle.
        # self.cur_veh: This tensor contains the updated state information for the vehicles.
        self.vehicles = self.vehicles.scatter(1,
                self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE), self.cur_veh)
        # The function then returns dist and late, where dist is the distance traveled, and late represents any lateness in reaching the destinations
        return dist, late


        #this method simulates the movement of vehicles to their destinations, updating their states and the overall state of the environment. The distances traveled by the vehicles are returned as a result.
        

         #self.vehicles = torch.zeros((2, 3, 4))  # Shape: [batch_size, vehicle_count, vehicle_state_size]
         #self.cur_veh_idx = torch.tensor([[0], [2]])  # Shape: [batch_size, 1] #indices = torch.tensor([[[0, 0, 0, 0]], [[2, 2, 2, 2]]])
         #self.cur_veh = torch.tensor([[[10, 20, 30, 40]], [[50, 60, 70, 80]]])  # Shape: [batch_size, 1, vehicle_state_size]
         #                                       v1            v2            v3
         #                                 x   y   c  t    x  y  c  t    x  y  c  t
         #self.vehicles = torch.tensor([[[10, 20, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0]],
         #                               [[0, 0, 0, 0], [0, 0, 0, 0], [50, 60, 70, 80]]])

    def step(self, cust_idx):
    # This step method is part of a reinforcement learning environment
        # This line gathers information about the destinations of the vehicles based on the indices provided by cust_idx. It extracts the customer features for the specified destinations.
        dest = self.nodes.gather(1, cust_idx[:,:,None].expand(-1,-1,self.CUST_FEAT_SIZE))
        # Calls the _update_vehicles method to update the state of the vehicles based on the destinations. It returns the distance traveled (dist) and any lateness in reaching the destinations (late).
        dist, late = self._update_vehicles(dest)
        # Updates information about whether each vehicle is done with its route based on the customer indices.
        self._update_done(cust_idx)
        # Updates a mask that keeps track of which customers have been served and which vehicles are overloaded or done.
        self._update_mask(cust_idx)
        # Updates information about the current state of the vehicles.
        self._update_cur_veh()
        # Calculates the reward for the agent based on the negative distance traveled and the cost associated with any lateness.
        reward = -dist - self.late_cost * late

        # Checks if all vehicles have completed their routes.
        if self.done:
            # If there is an initial customer mask, it adds that to the list of served customers.
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            # Calculates the number of unserved customers. It uses the XOR operator to find customers that haven't been served yet, converts it to float, sums across the last dimension, and subtracts 1.
            pending = (self.served ^ True).float().sum(-1, keepdim = True) - 1
            # Deducts a penalty from the reward based on the number of pending (unserved) customers.
            reward -= self.pending_cost * pending
        return reward

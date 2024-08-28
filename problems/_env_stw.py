from marpdan.problems import VRPTW_Environment
import torch

class SVRPTW_Environment(VRPTW_Environment):
#SVRPTW_Environment is a subclass of VRPTW_Environment, inheriting its properties and methods.

    #
    def __init__(self, data, nodes = None, cust_mask = None,
            pending_cost = 2, late_cost = 1,
            speed_var = 0.1, late_p = 0.05, slow_down = 0.5, late_var = 0.3):
        #Constructor Arguments:
        #data: This is the main data structure containing information about the VRPTW problem, such as customer locations, demands, and service times.
        #nodes: (Optional) This is an alternative representation of the data, potentially providing additional information about the nodes.
        #cust_mask: (Optional) This is a binary mask indicating which customers are initially available for service (True) and which are not (False).
        #pending_cost: The cost incurred for having a customer in the waiting list instead of serving them immediately.
        #late_cost: The cost incurred for serving a customer after their due time.
        
        # super(): Gives access to the parent class methods and initialization
        #.__init__(): Calls the parent __init__ method
        #First initialize the parent VRPTW environment class. Pass the common arguments like problem data to the parent. Then in the child __init__ do additional custom initialization
        super().__init__(data, nodes, cust_mask, pending_cost, late_cost)
        #speed_var: The variance of the speed fluctuation during travel for non-late customers.
        self.speed_var = speed_var
        #late_p: is a parameter that represents the probability of a vehicle being late
        self.late_p = late_p
        # slow_down: The factor by which the speed of a vehicle is reduced. 
        self.slow_down = slow_down
        #late_var: The variance of the speed fluctuation during travel.
        self.late_var = late_var

        

    def _sample_speed(self):
        # Calling .new_empty() on it creates the new tensor with same dtype and device. (self.minibatch_size, 1) specifies the shape
        # .bernoulli_(self.late_p) samples a 1 with probability self.late_p and 0 otherwise for each entry
        # late : Is sampling a Bernoulli trial to determine lateness at each customer node.
        # .new_empty(size): Creates a new tensor with the specified size but uninitialized (empty). The tensor is allocated but not initialized with any specific values.
        late = self.nodes.new_empty((self.minibatch_size, 1)).bernoulli_(self.late_p)
        # rand is a tensor of the same shape as late, filled with random numbers drawn from a standard normal distribution (mean 0, standard deviation 1).
        #This is achieved by calling torch.randn_like(late). These random numbers are used to introduce variability in the speed of the vehicles. The rand tensor is multiplied with either self.late_var or self.speed_var depending on whether the vehicle is late or not, respectively. 
        #This results in a speed that varies around the mean speed (self.slow_down for late vehicles and 1 for on-time vehicles).
        rand = torch.randn_like(late)
        # calculates the speed for each vehicle based on their lateness
        speed = late * self.slow_down * (1 + self.late_var * rand) + (1-late) * (1 + self.speed_var * rand)
        #speed.clamp_(min = 0.1) ensures that the minimum speed is 0.1, preventing unrealistic scenarios
        #speed * self.veh_speed scales the final speed by the base vehicle speed, resulting in the actual travel speed for each vehicle.
        return speed.clamp_(min = 0.1) * self.veh_speed

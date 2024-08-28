from marpdan.problems import SVRPTW_Environment
import torch

class SDVRPTW_Environment(SVRPTW_Environment):
    CUST_FEAT_SIZE = 7
    #  6 features customer:
    #x-coordinate of the customer's location:dest[:, :, 0]
    #y-coordinate of the customer's location: dest[:, :, 1]
    #Demand of the customer:dest[:, :, 2]
    #Time window start for the customer:dest[:, :, 3] #TODO lower bound TW = 0 or a_j (appearence time) 
    #Time window end for the customer:dest[:, :, 4]
    #Service time for the customer:dest[:, :, 5]
    #appearence time: [:, :, 6]


    def _update_hidden(self):
        # Get current time of each vehicle (stored in the 3rd feature of vehicle state)
        #The .clone() method creates a copy of the extracted time tensor, ensuring that the original time information is preserved.
        # time: This newly created tensor time will be used to update the cust_mask, mask, veh_done, and vehicles tensors, reflecting the passage of time and any dynamically revealed customers.
        time = self.cur_veh[:, :, 3].clone()
        # If no initial customer mask was provided: Reveal customers whose arrival time <= current time
        if self.init_cust_mask is None:
            reveal = self.cust_mask & (self.nodes[:,:,6] <= time)
        else:
            # If initial mask was provided: Reveal newly appeared customers based on those not in initial mask AND arrival time < current time
            reveal = (self.cust_mask ^ self.init_cust_mask) & (self.nodes[:,:,6] < time)
        if reveal.any():
            # If reveal mask indicates any new customers:
            # Set flag that there are new customers
            self.new_customers = True
            # Update customer availability mask by flipping revealed bits
            self.cust_mask = self.cust_mask ^ reveal
            # Update vehicle customer masks/ So for each vehicle, it updates the availability masks by XORing with the revealed customer mask.
            self.mask = self.mask ^ reveal[:,None,:].expand(-1,self.veh_count,-1)
            # Mark vehicles done if they still have unseen customers
            self.veh_done = self.veh_done & (reveal.any(1) ^ True).unsqueeze(1)
            # Update elapsed time
            self.vehicles[:, :, 3] = torch.max(self.vehicles[:, :, 3], time)
            # Update vehicle pointers
            self._update_cur_veh()
    # it uses the reveal mask to: Update environment state to include new customers/ Handle impact on vehicles (masks, done status, time)


    def reset(self):
        # Resets vehicles tensor to zeros, shaped (batch, num_vehicles, state_size)
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        # Sets initial vehicle locations to depot locations
        self.vehicles[:,:,:2] = self.nodes[:,0:1,:2]
        # nitializes vehicle capacities to max capacity
        self.vehicles[:,:,2] = self.veh_capa

        #is resetting the done statuses to initialize a new episode:
        # veh_done is a flag indicating which vehicles are done their routes. Setting it to all 0s resets the flag for all vehicles
        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype = torch.bool)
        #
        self.done = False

        #  Creates a boolean mask checking if arrival times are > 0  ,So any customer with a non-zero arrival time will be marked as False/unavailable in cust_mask.
        self.cust_mask = (self.nodes[:,:,6] > 0)
        #Updating cust_mask if initial mask is provided:
        if self.init_cust_mask is not None:
            # This makes any customer unavailable if set unavailable in the initial mask
            #The | performs a bitwise OR operation between the two masks.
            #So any customer that was marked as unavailable in init_cust_mask will now also be marked unavailable in cust_mask.
            self.cust_mask = self.cust_mask | self.init_cust_mask
        # Signals there are new customers appearing in reset
        self.new_customers = True
        #Sets all customers as unserved initially
        self.served = torch.zeros_like(self.cust_mask)

        # Is creating a separate customer availability mask for each vehicle from the base cust_mask.
        # [:,None,:]: Adds a new dimension in the 2nd axis (shape is now batch_size x 1 x num_customers)
        # .repeat(1, self.veh_count, 1): Repeats along 2nd dim to match number of vehicles
        self.mask = self.cust_mask[:,None,:].repeat(1, self.veh_count, 1)

        #Initialize current vehicle index as 0 for all batches
        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype = torch.int64)
        # Use gather() to extract current vehicle state using the index:
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        # Similarly extract availability mask for current vehicle
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))

    #This reset() method is initializing some key parts of the environment's state at the start of each new episode.


    def step(self, cust_idx):
        #Call base step() to get reward and update state
        reward = super().step(cust_idx)
        # Check for newly visible customers, Reveals customers based on elapsed time
        self._update_hidden()
        
        return reward
        #step(self, cust_idx) define in env.py
        #This environment class inherits from a base VRPTW environment class. That base class likely has its own step() method defined
        #When we override step() here, we want to first call the parent step() before adding custom logic
        #super() gives access to the parent class allowing us to call its step method
        #So super().step(cust_idx) will call the base environment's step implementation. This allows:

        #Inheriting any core logic defined there Then augmenting it with additional customization Without super(), we would lose access to the parent step logic when overriding.
        # it chains together parent then child class functionality on a method. Very handy for inheritance!
    
    
    
    
    # So at every step during rollout:
    #Parent step() is called to actually assign and route to customer
    #Then _update_hidden() reveals any new customers dynamically
    #This couples the base routing environment with the dynamic stochastic customer logic.
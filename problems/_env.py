import torch

class VRP_Environment:
    #self.vehicles: Tensor representing the state of all vehicles.
    #self.veh_done: Tensor indicating whether each vehicle has completed its route.
    #self.served: Tensor tracking which customers have been served.
    #self.mask: Tensor representing the mask indicating valid actions for each vehicle.
    #self.cur_veh_idx: Index of the current vehicle.
    #self.cur_veh: State vector of the current vehicle.
    #self.cur_veh_mask: Mask for the current vehicle's actions.



    VEH_STATE_SIZE = 4
    # 4 feature:
    #x-coordinate of the vehicle's location:cur_veh[:, :, 0]
    #y-coordinate of the vehicle's location: cur_veh[:, :, 1]
    #capacity of the vehicles:cur_veh[:, :, 2]
    #available time of the vehicles:cur_veh[:, :, 3]

    CUST_FEAT_SIZE = 3
    # 3 feature:
    #x-coordinate of the customer's location:dest[:, :, 0]
    #y-coordinate of the customer's location: dest[:, :, 1]
    #Demand of the customer:dest[:, :, 2]

    def __init__(self, data, nodes = None, cust_mask = None,
            pending_cost = 2):
         # Initialization of various attributes

         #  number of vehicles in the environment.
        self.veh_count = data.veh_count
         # he capacity of each vehicle in the environment
        self.veh_capa = data.veh_capa
        #speed of each vehicle in the environment.
        self.veh_speed = data.veh_speed
        #If the user provides custom node information (nodes is not None), self.nodes will be set to that custom information
        #If the user does not provide custom node information (nodes is None), self.nodes will be set to the node information from the VRP_Dataset (data.nodes)
        self.nodes = data.nodes if nodes is None else nodes
        #This conditional assignment allows the user to have the option of providing their own customer mask or using the default customer mask from the VRP_Dataset.
        self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
        
        
        #to extract and assign specific dimensions of the tensor size to corresponding variables.
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()
        #size = (batch_size, cust_count, 1)
        
        # It specifies the cost associated with pending tasks.
        self.pending_cost = pending_cost
        #TODO penalty of dead person (remain untreat)

    def _update_vehicles(self, dest):
         #current state of vehicles in the environment
        #calculates the pairwise distances between the current locations of the vehicles and their respective destinations
        # and then computes the time it would take for each vehicle to reach its destination based on the given speed.
        #  [:, 0, :2] extracts the x and y coordinates of the first node for each vehicle (GPT)

        # This represents the current positions of all vehicles in the batch. It is a tensor of shape (batch_size, num_vehicles, 2), where the last dimension holds the x and y coordinates of each vehicle.
        # dest[:,0,:2]: This represents the chosen destinations for each vehicle in the batch.
       dist = torch.pairwise_distance(self.cur_veh[:,0,:2], dest[:,0,:2], keepdim = True)
        tt = dist / self.veh_speed
        


        #an assignment statement in the _update_vehicles method.
        #updates the x and y coordinates of the current locations of the vehicles (self.cur_veh)...
        #to be the same as the x and y coordinates of their respective destinations (dest).
        self.cur_veh[:,:,:2] = dest[:,:,:2]
        
       
       #self.cur_veh:This tensor represents the current state of vehicles.The shape is (batch_size, num_vehicles, vehicle_features)
        #dest:This tensor represents the destination nodes for the vehicles.The shape is (batch_size, num_vehicles, destination_features).
        #I think it is related to the capacity of the vehicle
        #TODO dems=1 
        #NOTE (for future) dems >= 1
        self.cur_veh[:,:,2] -= dest[:,:,2]
        #time to arrive destination 
        self.cur_veh[:,:,3] += tt
      

        #the scatter_ function is an operation that updates elements of a tensor based on indices and values provided
                
        # self.vehicles: This tensor represents the state of all vehicles in the environment.
        # scatter(1, ...) : The scatter method is used to scatter/gather values along a particular dimension of the tensor.
        # 1: Specifies that the scattering operation is along the second dimension (the vehicle dimension).
         # self.cur_veh_idx: The indices indicating which vehicle to update for each batch.
         # [:,:,None]: Adds a new dimension to the indices (None is equivalent to adding a dimension).
         # expand(-1,-1,self.VEH_STATE_SIZE): Expands the dimensions to match the size of self.cur_veh. It repeats the indices for each state feature of a vehicle.
        # self.cur_veh: This tensor contains the updated state information for the vehicles.
        self.vehicles = self.vehicles.scatter(1,
                self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE), self.cur_veh)
                

        return dist
    #this method simulates the movement of vehicles to their destinations, updating their states and the overall state of the environment. The distances traveled by the vehicles are returned as a result.
        

    #self.vehicles = torch.zeros((2, 3, 4))  # Shape: [batch_size, vehicle_count, vehicle_state_size]
    #self.cur_veh_idx = torch.tensor([[0], [2]])  # Shape: [batch_size, 1] #indices = torch.tensor([[[0, 0, 0, 0]], [[2, 2, 2, 2]]])
    #self.cur_veh = torch.tensor([[[10, 20, 30, 40]], [[50, 60, 70, 80]]])  # Shape: [batch_size, 1, vehicle_state_size]
         #                                       v1            v2            v3
         #                                 x   y   c  t    x  y  c  t    x  y  c  t
         #self.vehicles = torch.tensor([[[10, 20, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0]],
         #                               [[0, 0, 0, 0], [0, 0, 0, 0], [50, 60, 70, 80]]])
    
    
    def _update_done(self, cust_idx):
        self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
        self.done = bool(self.veh_done.all())
    # this method updates the completion status of individual vehicles based on whether they have reached the depot (customer index is 0), 
    # and it checks if all vehicles have completed their routes to set the overall completion status (self.done).
        
    def _update_mask(self, cust_idx):
        self.new_customers = False
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        #Calculate overload based on the difference between vehicle capacity and customer demand 
        #TODO we don't have overload.  ??   
        overload = torch.zeros_like(self.mask).scatter_(1,
                self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count),
                self.cur_veh[:,:,None,2] - self.nodes[:,None,:2] < 0)
          
        #This line updates the self.mask tensor by combining information about served customers, overload, and completed vehicles.
        self.mask = self.mask | self.served[:,None,:] | overload | self.veh_done[:,:,None]
        # Set the first column of the mask to 0 (exclude the depot node). 
        #This operation is often performed to ensure that the vehicles do not revisit the depot once they have started their route or ... 
        #...that the depot is not treated as a potential destination for serving customers. The value 0 in the context of a mask often signifies that the corresponding element is "masked out" or not considered.
        #This is a 3D tensor representing a mask. The dimensions are (minibatch_size, veh_count, nodes_count).
        self.mask[:,:,0] = 0 #TODO Multi-trip (a condition should be applied to the vehicles in a way to only mask the ambulance which are not at the depot position. this means that the masked ambulences are those which did not coming back to the depot) caution: this is premitive thinking and it might prone to bugg.
        # 
        
    def _update_cur_veh(self):
         #This line clones the availability values from the vehicles tensor. Availability indicates whether a vehicle is currently available or not.
        avail = self.vehicles[:,:,3].clone()
       # For the vehicles that are already marked as done (self.veh_done), their availability is set to infinity (float('inf')). This ensures that done vehicles are not considered for future assignments.
        avail[self.veh_done] = float('inf')
        # This line finds the indices of the least available vehicles for each minibatch. The argmin function finds the minimum value along dimension 1 (vehicles)
        self.cur_veh_idx = avail.argmin(1, keepdim = True)
        # Gather the information of the current vehicle using the found index
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        # Gather the mask information for the current vehicle
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))

     #_update_cur_veh determines the next vehicle to be considered for assignment based on their availability, updates the current vehicle state, and extracts the corresponding mask. This is crucial for the next steps in the environment, such as assigning customers to the selected vehicle.
  
        

    def reset(self):
         # Creates a tensor filled with zeros to store the state of all vehicles, including their position, capacity, time, etc.
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))

        # Sets the initial position of all vehicles to the depot location (first node).
        self.vehicles[:,:,:2] = self.nodes[:,0:1,:2] 
        # Sets the initial capacity of each vehicle to the specified capacity.
        self.vehicles[:,:,2] = self.veh_capa 
    #
       
        # Creates a tensor of booleans to track whether each vehicle has finished serving all its customers.
        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype = torch.bool) 
        # Sets the overall environment completion flag to False (not done). indicating the episode is not over yet.
        self.done = False 

        #Loads the initial customer mask, indicating which customers are available  for selection.
        self.cust_mask = self.init_cust_mask
        # indicating that there are new customers (this could be used as a signal to the agent).
        self.new_customers = True
        # Creates a new tensor with zeros for all served customers.Creates a tensor to track served customers.
        self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype = torch.bool)

      
        # Initializes the accessibility mask:
        #GPT: self.mask: This is a tensor that represents a mask in the environment. It has a shape of (minibatch_size, veh_count, nodes_count). Each element of this tensor is a boolean value, indicating whether a specific customer (node) is allowed to be serviced by a particular vehicle.
        #self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype=torch.bool): This part creates a tensor of zeros with the same shape as self.mask. The dtype=torch.bool specifies that the tensor will have boolean values.
        # If self.cust_mask is None, meaning there is no specific customer mask provided, then the tensor of zeros is assigned to self.mask.
        #If self.cust_mask is not None, it means a customer mask has been provided.the customer mask is expanded to match the shape of self.mask. The [:, None, :] adds a new dimension to the mask, and repeat(1, self.veh_count, 1) replicates the mask along the second dimension (veh_count) to cover all vehicles.
        self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype = torch.bool) \
                #If self.cust_mask is None, creates a mask of zeros for all customers and vehicles.
                #This creates a new tensor named mask with dimensions representing minibatch size, vehicle count, and customer count.
                #All elements are initialized to False, indicating no customer is initially accessible to any vehicle.
                #This effectively disables any restrictions on customer selection when no initial mask is provided.
                if self.cust_mask is None else self.cust_mask[:,None,:].repeat(1, self.veh_count, 1)
                #self.cust_mask[:,None,:].repeat(1, self.veh_count, 1):This part of the code assumes the existence of a pre-defined cust_mask attribute.
                #[:,None,:]: This extracts the entire customer mask and adds a new dimension at the second position (vehicle dimension).
                #.repeat(1, self.veh_count, 1): This replicates the customer mask for each vehicle in the batch.
                #This effectively creates an accessibility mask where each element indicates whether a specific customer is accessible to a specific vehicle based on the initial mask.

        
        # Creates a new tensor with zeros for the current vehicle indices (initially no vehicles are active).
        #This tensor will hold the index of the current active vehicle for each instance in the batch.
        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype = torch.int64)
        # Extracts a zero-state vector for the "current vehicle" (since none are active).
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        # Extracts a zero accessibility mask for the "current vehicle" (since none are active). indicating which customers are currently accessible.
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))

    #he reset function efficiently prepares the environment for a fresh VRP simulation with the initial vehicle states, customer availability,
    # and masking information. This allows for subsequent steps and actions to be taken within this clean slate environment.


    def step(self, cust_idx):
        # Retrieves the chosen customer features (including their location) based on the cust_idx.
        dest = self.nodes.gather(1, cust_idx[:,:,None].expand(-1,-1,self.CUST_FEAT_SIZE))
        # Update the vehicles' positions and calculate the distance traveled
        dist = self._update_vehicles(dest)
        # Update the completion status based on the customer indices
        # Updates the veh_done flag based on whether the chosen customer served them to their full capacity.
        self._update_done(cust_idx)
        # Update the mask based on the current state
        # Updates the mask indicating which customers are accessible to each vehicle after serving the chosen customers. It also considers overload situations and vehicle completion.
        self._update_mask(cust_idx)
        # Update information about the current vehicle
        #Updates the cur_veh and cur_veh_mask to reflect the current active vehicle and its accessible customers after the actions.
        self._update_cur_veh()
        # Sets the immediate reward as the negative of the total travel distance for all vehicles.
        reward = -dist
        #If all vehicles are done, it checks for the init_cust_mask and adds any initially unavailable customers to the served count.
        if self.done:
            # Update the served customers based on the initial customer mask
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            # Calculate the number of pending customers (not served) based on the served mask and penalize the agent. -1 indigate depot cosideration
            pending = (self.served ^ True).float().sum(-1, keepdim = True) - 1
            # If there are pending customers, it subtracts a penalty based on the pending_cost for each vehicle.
            reward -= self.pending_cost * pending
        return reward
    
    #this step function progresses the environment based on chosen customer visits, updates vehicle states and accessibility,
    # and calculates the reward considering travel distances and potential penalties.

    
    def state_dict(self, dest_dict = None):
    #This function, as you mentioned, saves the current state of the VRP_Environment to a dictionary. It does this in two ways:
        #Creating a new dictionary: If you don't provide a dest_dict, it creates a new dictionary and populates it with key-value pairs representing the state components and their corresponding tensors. This essentially creates a snapshot of the environment at that moment.
        #Updating an existing dictionary: If you provide a dest_dict, it assumes it contains a previously saved state. It then iterates over the keys in this dictionary and copies the relevant state data from the environment's attributes to the corresponding entries in the dest_dict. This effectively overwrites the existing state in the dictionary with the current one.

        if dest_dict is None:
            #The key-value pairs include all the crucial elements needed to resume or analyze the environment later:
            dest_dict = {
                    "vehicles": self.vehicles,#Stores the current state vectors for all vehicles.
                    "veh_done": self.veh_done,#  Indicates whether each vehicle has finished serving customers (boolean tensor).
                    "served": self.served,# Tracks which customers have been served by any vehicle (boolean tensor).
                    "mask": self.mask,# Represents the accessibility of each customer for each vehicle (boolean tensor).
                    "cur_veh_idx": self.cur_veh_idx# Identifies the currently active vehicle for each minibatch.
                    }
        else:
            dest_dict["vehicles"].copy_(self.vehicles)
            dest_dict["veh_done"].copy_(self.veh_done)
            dest_dict["served"].copy_(self.served)
            dest_dict["mask"].copy_(self.mask)
            dest_dict["cur_veh_idx"].copy_(self.cur_veh_idx)
        return dest_dict

    
    def load_state_dict(self, state_dict):
     #This function allows you to restore the environment's state from a previously saved dictionary (state_dict). It performs the opposite operation of state_dict:
    #It iterates over the key-value pairs in the state_dict and copies the corresponding data back into the environment's attributes.
    #This effectively overwrites the current state of the environment with the information stored in the dictionary.
    #Additionally, it updates cur_veh and cur_veh_mask based on the loaded cur_veh_idx to ensure consistency with the restored state.   


        # Copy the saved state components into the environment's state tensors
        self.vehicles.copy_(state_dict["vehicles"])
        self.veh_done.copy_(state_dict["veh_done"])
        self.served.copy_(state_dict["served"])
        self.mask.copy_(state_dict["mask"])
        self.cur_veh_idx.copy_(state_dict["cur_veh_idx"])

        # Update information about the current vehicle based on the restored state
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))
        # It recomputes the current vehicle state (cur_veh) and the corresponding mask (cur_veh_mask). This is done using the gathered values from the vehicles tensor based on the indices in cur_veh_idx. The gather method is used to select the relevant portions of the vehicles tensor.
        # it updates cur_veh and cur_veh_mask based on the loaded cur_veh_idx to ensure consistency with the restored state.

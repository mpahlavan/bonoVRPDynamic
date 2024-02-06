#!/usr/bin/env python3

from marpdan import *
from marpdan.problems import *
from marpdan.baselines import *
from marpdan.externals import *
from marpdan.dep import *
from marpdan.utils import *
from marpdan.layers import reinforce_loss

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import time
import os
from itertools import chain
#Imports various modules and functions from the marpdan package, as well as essential PyTorch modules, and other standard libraries.


def train_epoch(args, data, Environment, env_params, bl_wrapped_learner, optim, device, ep):
#Conducts a single epoch of training for a reinforcement learning model within the marpdan framework

    #This line sets the learner (the model being trained) to training mode. this is necessary to enable training-specific behavior like dropout.
    bl_wrapped_learner.learner.train()
    #Initializes a PyTorch DataLoader to iterate over the training data (data) in batches of size args.batch_size. 
    #The True argument indicates that the data should be shuffled during each epoch.
    loader = DataLoader(data, args.batch_size, True)

    #Initialization of Epoch-level Variables:
    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0
    #Key Variables:
    #args: Object containing configuration arguments for training.
    #data: Dataset to train on.
    #Environment: Environment class for interaction (specific implementation depends on the VRP variant).
    #env_params: Additional parameters for environment setup.
    #bl_wrapped_learner: Wrapper combining the model with a chosen baseline for variance reduction.
    #optim: Optimizer for updating model parameters.
    #device: Computation device (CPU or GPU).
    #ep: Current epoch number.
    
    with tqdm(loader, desc = "Ep.#{: >3d}/{: <3d}".format(ep+1, args.epoch_count)) as progress:
    #This line creates a progress bar (progress) that wraps around the loader. The progress bar is configured with a description
    # (displayed on the left side of the bar) that includes the current epoch (ep+1) and the total number of epochs (args.epoch_count).
        
        #This starts a loop over the batches provided by the loader. The progress bar (progress) updates with each iteration.
        for minibatch in progress:

            #This conditional block checks if the data object has a cust_mask attribute. If it does, it assumes that the minibatch is a tuple 
            #where the first element is custs and the second element is mask. If cust_mask is None, it means that the data doesn't have a customer mask,
            # and the entire minibatch is assigned to custs with mask set to None.
            if data.cust_mask is None:
                custs, mask = minibatch.to(device), None
            else:
                custs, mask = minibatch[0].to(device), minibatch[1].to(device)
                #The .to(device) is used to move the data to the specified device (CPU or GPU).

            #This line creates an instance of the Environment class (which was determined based on the problem type, such as VRP, VRPTW, etc.).
            #It is initialized with the provided arguments (data, custs, mask, and the elements of env_params).  
            #This object encapsulates the problem state and dynamics.
            dyna = Environment(data, custs, mask, *env_params)
            #Passes the environment (dyna) to the wrapped learner, which likely executes the following:
            #Uses the model to generate actions based on the current state.
            #Calculates log probabilities of those actions.
            #Interacts with the environment using those actions, obtaining rewards.
            #Uses the baseline to estimate value functions for the states.
            actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
            #Computes the reinforcement learning loss using the REINFORCE algorithm, likely involving:
            #The log probabilities of actions (logps)/The received rewards (rewards)/The baseline values (bl_vals), which help reduce variance in gradient updates
            loss = reinforce_loss(logps, rewards, bl_vals)

            #torch.stack(logps): This function is used to stack a sequence of tensors along a new dimension. If logps is a list of tensors,
            # it will stack them along a new dimension (dimension 0 by default), creating a new tensor with one additional dimension. This is often used to combine results from different time steps or sequences.
            #Converts log probabilities (logps) to probabilities using exp().Sums probabilities across trajectories within the batch for each action.
            #sum(0): This method computes the sum of the tensor along the newly added dimension (dimension 0).
            #.mean(): This method calculates the mean of the tensor. It adds up all the elements and divides by the total number of elements. It provides a measure of the central tendency of the tensor.
            #So, torch.stack(logps).sum(0).mean() computes the mean of the summed log probabilities along the specified dimension. 
            #This type of computation is common in reinforcement learning when dealing with sequences of probabilities or rewards over time.
            prob = torch.stack(logps).sum(0).exp().mean()
            #torch.stack(rewards): Stacking a list of tensors along a new dimension. This is likely a collection of rewards for different actions or steps.
            #.sum(0): Summing along the newly created dimension. This computes the total reward for each action or step across the batches.
            #.mean(): Computing the mean of the total rewards. This could represent the average total reward across all actions or steps in a batch.
            val = torch.stack(rewards).sum(0).mean()
            # Calculates the mean of the first baseline value (bl_vals[0]) across trajectories, representing the average estimated value of the initial state.
            bl = bl_vals[0].mean()
            # This line zeroes out the gradients of all the model parameters. In PyTorch, gradients accumulate by default for each parameter
            # during the backward pass. Before computing the gradients for a new minibatch, it's necessary to zero out the gradients from the previous minibatch.
            # Otherwise, the new gradients would be added to the existing ones.
            optim.zero_grad()
            #
            loss.backward()
            #Performs gradient clipping and optimizer step for gradient-based optimization.
            if args.max_grad_norm is not None:
                #This helps prevent exploding gradients, which can cause training instability and divergence.
                grad_norm = clip_grad_norm_(chain.from_iterable(grp["params"] for grp in optim.param_groups),
                        args.max_grad_norm)
            #Updates model parameters based on the computed gradients, aiming to minimize the loss and improve model performance.
            optim.step()

            #"l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(...) creates a formatted string containing:
            #l: Loss value (rounded to 4 decimal places).
            #p: Probability of actions (rounded to 9 decimal places).
            #val: Average reward (rounded to 4 decimal places).
            #bl: Average baseline value (rounded to 4 decimal places).
            #|g|: Gradient norm (rounded to 4 decimal places).
            #  sets the postfix string of a progress bar object (likely a tqdm progress bar) with the formatted string.
            progress.set_postfix_str("l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
                loss, prob, val, bl, grad_norm))

            
            
            ep_loss += loss.item()#Adds the current loss value to the epoch loss accumulator ep_loss.
            #Adds the current probability value to the epoch probability accumulator ep_prob.
            ep_prob += prob.item()
            #
            ep_val += val.item()
            #
            ep_bl += bl.item()
            #
            ep_norm += grad_norm

    #Iterates through the accumulated statistics (ep_loss, ep_prob, ep_val, ep_bl, ep_norm).
    #Divides each statistic by args.iter_count, likely the number of iterations within the epoch, to obtain average values per iteration.
    #Creates a tuple containing the normalized statistics.
    return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm))


def test_epoch(args, test_env, learner, ref_costs):
#Defines a function named test_epoch consisting: args: Configuration arguments for the test process.test_env: A test environment for evaluating model performance.
#learner: The trained model to be tested.ref_costs: Reference costs for calculating performance gaps.
    
    #Sets model to evaluation mode, disabling dropout and batch normalization layers for deterministic testing.
    #his line sets the model (learner) to evaluation mode using eval(). In evaluation mode, 
    #dropout and batch normalization layers are typically disabled to ensure deterministic behavior during testing.
    learner.eval()
    #This conditional statement checks if the problem type, specified in the configuration arguments (args), starts with the letter "s."
    if args.problem_type[0] == "s":
        #If the problem type starts with "s," a tensor costs is created with zeros, having the same size as the minibatch size in the test environment.
        costs = test_env.nodes.new_zeros(test_env.minibatch_size)
        #A loop runs for 100 iterations, during which the model (learner) is used to generate predictions (rewards) on the test environment 
        #(test_env). The negative sum of rewards is accumulated in the costs tensor. After the loop, the accumulated costs are divided by 100 to compute the mean.
        for _ in range(100):

            _, _, rewards = learner(test_env)
            costs -= torch.stack(rewards).sum(0).squeeze(-1)
        costs = costs / 100
       
        #If the problem type does not start with "s," predictions (rs) are obtained from the model (learner) on the test environment (test_env).
        # The negative sum of these rewards is computed and stored in the costs tensor.
    else:
        _, _, rs = learner(test_env)
        costs = -torch.stack(rs).sum(dim = 0).squeeze(-1)
    mean = costs.mean()
    std = costs.std()
    gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()

    print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
    return mean.item(), std.item(), gap.item()


def main(args):
    #Defines the main function, taking a parsed argument namespace args as input.

    #Selects the appropriate device (GPU if available and enabled, otherwise CPU).
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #Seeds the random number generator for reproducibility if a seed is provided.
    if args.rng_seed is not None:
        torch.manual_seed(args.rng_seed)

    #Creates a conditional function for printing status messages only if verbosity is enabled.
    if args.verbose:
        verbose_print = print
    
    #If verbose is False, verbose_print is defined as a function that accepts any arguments (*args) and keyword arguments (**kwargs) but does nothing. This effectively silences messages.
    else:
        def verbose_print(*args, **kwargs): pass

    # PROBLEM
    
    
    #Creates a dictionary mapping problem types to their corresponding dataset classes.
    #Fetches the appropriate dataset class based on the specified args.problem_type.
    Dataset = {
            "vrp": VRP_Dataset,
            "vrptw": VRPTW_Dataset,
            "svrptw": VRPTW_Dataset,
            "sdvrptw": SDVRPTW_Dataset
            }.get(args.problem_type)
    #Initializes a list to store parameters required for dataset generation.
    #Includes parameters common to all problem types:
    #customers_count: Number of customers in a problem instance.
    #vehicles_count: Number of available vehicles.
    #veh_capa: Capacity of each vehicle.
    #veh_speed: Speed at which vehicles travel.
    #min_cust_count: Minimum number of customers to include in an instance.
    #loc_range: Range of possible customer locations.
    #dem_range: Range of possible customer demand
    gen_params = [
            args.customers_count,
            args.vehicles_count,
            args.veh_capa,
            args.veh_speed,
            args.min_cust_count,
            args.loc_range,
            args.dem_range
            ]
    #If the problem type is not the basic VRP, adds additional parameters:
    #horizon: Scheduling horizon (likely for time-dependent problems).
    #dur_range: Range of possible service durations.
    #tw_ratio: Ratio of time window length to service duration.
    #tw_range: Range of possible time window start times.
    if args.problem_type != "vrp":
        gen_params.extend( [args.horizon, args.dur_range, args.tw_ratio, args.tw_range] )
    if args.problem_type == "sdvrptw":
        gen_params.extend( [args.deg_of_dyna, args.appear_early_ratio] )


    # TRAIN DATA

    #Calls the verbose_print function, which either prints the message or does nothing depending on the verbose setting.
    #...".format(...):Formats a string with placeholders: {}: Placeholder for the total number of training samples.
    #{}: Placeholder for the problem type in uppercase.
    #args.iter_count * args.batch_size:Calculates the total number of samples to be generated.
    #args.problem_type.upper():Converts the problem type to uppercase for better readability.
    #end = " ": Prevents a newline after the message, allowing for continuous output updates.
    #flush = True: Forces immediate printing to the console, avoiding buffering delays.
    verbose_print("Generating {} {} samples of training data...".format(
        args.iter_count * args.batch_size, args.problem_type.upper()),
        end = " ", flush = True)
    train_data = Dataset.generate(
            args.iter_count * args.batch_size,
            *gen_params
            )
    #Normalizes the features within the train_data object.
    #Calls the normalize method, likely defined within the dataset class.Specific implementation varies depending on data types and features.
    #Common normalization techniques include:Min-max scaling: Shifts and scales features to a range of 0 to 1.Standardization: Subtracts the mean and divides by the standard deviation.
    train_data.normalize()
    #Prints a status message indicating completion of normalization (if verbosity is enabled).
    verbose_print("Done.")


    # TEST DATA AND COST REFERENCE
    verbose_print("Generating {} {} samples of test data...".format(
        args.test_batch_size, args.problem_type.upper()),
        end = " ", flush = True)
    test_data = Dataset.generate(
            args.test_batch_size,
            *gen_params
            )
    verbose_print("Done.")

    #If the OR-Tools solver is available, calls ort_solve to obtain optimal reference routes for the generated test data.
    if ORTOOLS_ENABLED:
        ref_routes = ort_solve(test_data)
    #If OR-Tools isn't available but the LKH solver is, calls lkh_solve for the same purpose
    elif LKH_ENABLED:
        ref_routes = lkh_solve(test_data)
    # If neither solver is available, sets ref_routes to None and prints a warning message.
    else:
        ref_routes = None
        print("Warning! No external solver found to compute gaps for test.")
    test_data.normalize()

    # ENVIRONMENT
    #Creates a dictionary mapping problem types to their corresponding environment classes.
    Environment = {
            "vrp": VRP_Environment,
            "vrptw": VRPTW_Environment,
            "svrptw": SVRPTW_Environment,
            "sdvrptw": SDVRPTW_Environment
            }.get(args.problem_type)
    #Initializes a list to store parameters for environment construction.Includes the pending_cost parameter, likely related to waiting costs.
    env_params = [args.pending_cost]
    #If the problem type is not the basic VRP, adds additional parameters:late_cost: Likely a penalty for delayed deliveries.
    #For SDVRPTW, adds parameters related to dynamism and uncertainty:speed_var, late_prob, slow_down, late_var
    if args.problem_type != "vrp":
        env_params.append(args.late_cost)
        if args.problem_type != "vrptw":
            env_params.extend( [args.speed_var, args.late_prob, args.slow_down, args.late_var] )
    #Creates a test environment instance using the retrieved Environment class.Provides the test_data, sets the remaining arguments as None, and passes the env_params.
    test_env = Environment(test_data, None, None, *env_params)

    if ref_routes is not None:
        #If optimal reference routes were obtained earlier:ref_costs = eval_apriori_routes(...): Evaluates the cost of these routes within the test environment.
        ref_costs = eval_apriori_routes(test_env, ref_routes, 100 if args.problem_type[0] == 's' else 1)
        #Prints the mean and standard deviation of the reference costs.
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))
    #Moves the nodes tensor (likely representing customer locations) to the selected device (CPU or GPU)
    test_env.nodes = test_env.nodes.to(dev)
    #If an init_cust_mask exists (possibly indicating initial customer availability):Moves it to the device as well
    if test_env.init_cust_mask is not None:
        test_env.init_cust_mask = test_env.init_cust_mask.to(dev)

    # MODEL
    verbose_print("Initializing attention model...",
        end = " ", flush = True)
    learner = AttentionLearner(
            Dataset.CUST_FEAT_SIZE,
            Environment.VEH_STATE_SIZE,
            args.model_size,
            args.layer_count,
            args.head_count,
            args.ff_size,
            args.tanh_xplor
            )
    learner.to(dev)
    verbose_print("Done.")

    # BASELINE
    verbose_print("Initializing '{}' baseline...".format(
        args.baseline_type),
        end = " ", flush = True)
    if args.baseline_type == "none":
        baseline = NoBaseline(learner)
    elif args.baseline_type == "nearnb":
        baseline = NearestNeighbourBaseline(learner, args.loss_use_cumul)
    elif args.baseline_type == "rollout":
        args.loss_use_cumul = True
        baseline = RolloutBaseline(learner, args.rollout_count, args.rollout_threshold)
    elif args.baseline_type == "critic":
        baseline = CriticBaseline(learner, args.customers_count, args.critic_use_qval, args.loss_use_cumul)
    baseline.to(dev)
    verbose_print("Done.")

    # OPTIMIZER AND LR SCHEDULER

    #Initializes the Adam optimizer (optim) for training the attention model. If the baseline type is "critic," 
    #it sets up separate learning rates for the attention model and the baseline model.
    verbose_print("Initializing Adam optimizer...",
        end = " ", flush = True)
    #Initializes a variable lr_sched to None, which will potentially hold a learning rate scheduler.
    #it sets up a learning rate scheduler (lr_sched) using LambdaLR if a rate decay is specified.
    lr_sched = None
    
    #Creates an Adam optimizer with separate parameter groups:One for the main model's parameters (learner.parameters()) with a learning rate of args.learning_rate.
    #Another for the critic baseline's parameters (baseline.parameters()) with a distinct learning rate of args.critic_rate.
    if args.baseline_type == "critic":
        optim = Adam([
            {"params": learner.parameters(), "lr": args.learning_rate},
            {"params": baseline.parameters(), "lr": args.critic_rate}
            ])
        #Sets up learning rate decay schedules:Determines the critic decay rate (critic_decay).
        #Creates a LambdaLR scheduler to adjust learning rates over time.
        #Uses two lambda functions, one for each parameter group, ensuring independent decay.
        #The learning rate scheduler adjusts the learning rates during training according to a specified schedule
        if args.rate_decay is not None:
            critic_decay = args.rate_decay if args.critic_decay is None else args.critic_decay
            lr_sched = LambdaLR(optim,[
                lambda ep: args.learning_rate * args.rate_decay**ep,
                lambda ep: args.critic_rate * critic_decay**ep
                ])
    
        #If no rate decay is specified, it initializes the Adam optimizer with the specified learning rate for the attention model (learner.parameters()). 
    else:
        optim = Adam(learner.parameters(), args.learning_rate)
        if args.rate_decay is not None:
            lr_sched = LambdaLR(optim, lambda ep: args.learning_rate * args.rate_decay**ep)
    verbose_print("Done.")

    ###
    #Adaptive Optimization: The Adam optimizer dynamically adjusts learning rates for effective training.
    #Baseline-Specific Tuning: Separate learning rates and decay schedules can benefit CriticBaselines.
    #Learning Rate Decay: Gradually reducing learning rates can aid convergence and prevent overfitting.
    #Flexible Decay with LambdaLR: Customizable decay logic for fine-grained control.
    #User Feedback: Status updates like verbose_print("Done.") enhance transparency and user experience


    # CHECKPOINTING
    verbose_print("Creating output dir...",
        end = " ", flush = True)
    #Construct output directory path  with a descriptive name for storing model checkpoints and results.
    args.output_dir = "./output/{}n{}m{}_{}".format(
            args.problem_type.upper(),
            args.customers_count,
            args.vehicles_count,
            time.strftime("%y%m%d-%H%M")
            ) if args.output_dir is None else args.output_dir
    #Create the directory
    os.makedirs(args.output_dir, exist_ok = True)
    #Save training configuration in a JSON file for reproducibility.
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("'{}' created.".format(args.output_dir))

    #If resuming, load checkpoint, Otherwise start from epoch 0
    if args.resume_state is None:
        start_ep = 0
    else:
        start_ep = load_checkpoint(args, learner, optim, baseline, lr_sched)

    #Print status
    verbose_print("Running...")
    #Initialize stat trackers
    train_stats = []
    test_stats = []
    #Main training loop over epochs.
    #Call train and test functions
    #Step LR schedule if enabled
    try:
        for ep in range(start_ep, args.epoch_count):
            train_stats.append( train_epoch(args, train_data, Environment, env_params, baseline, optim, dev, ep) )
            if ref_routes is not None:
                test_stats.append( test_epoch(args, test_env, learner, ref_costs) )

            if args.rate_decay is not None:
                lr_sched.step()
            if args.pend_cost_growth is not None:
                env_params[0] *= args.pend_cost_growth
            if args.late_cost_growth is not None:
                env_params[1] *= args.late_cost_growth
            if args.grad_norm_decay is not None:
                args.max_grad_norm *= args.grad_norm_decay

            #Periodically save checkpoint
            if (ep+1) % args.checkpoint_period == 0:
                save_checkpoint(args, ep, learner, optim, baseline, lr_sched)

    except KeyboardInterrupt:
        save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
    finally:
        export_train_test_stats(args, start_ep, train_stats, test_stats)


if __name__ == "__main__":
    main(parse_args())

import argparse
import random
import time
import warnings
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator


# TODO ignore warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")


argparse = argparse.ArgumentParser()

warnings.filterwarnings("ignore", category=DeprecationWarning)
argparse.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/cifar10/server_fedasync.yaml",
)
argparse.add_argument(
    "--client_config", type=str, default="./resources/configs/cifar10/client_1.yaml"
)
args = argparse.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

with open('metrics_log.txt', 'w') as file:
    pass  # The file is now empty

# TODO :D Define the mapping of rank to delay (in seconds)
rank_delay_mapping = {
    1: 0,
    2: 30,
    3: 60,
    4: 90,
    5: 120,
    6: 75,
    7: 90,
    8: 105,
    9: 120,
    10: 135,
    11: 150,
}

# TODO :D Define the mapping of rank to delay (in seconds)
rank_delay_mapping_FEMNIST = {
    1: 5,
    2: 10,
    3: 15,
    4: 20,
    5: 25,
    6: 30,
    7: 35,
    8: 40,
    9: 45,
    10: 50,
}

random.seed(70)

if rank == 0:
    # Load and set the server configurations
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.scheduler_kwargs.num_clients = num_clients
    if hasattr(server_agent_config.server_configs.aggregator_kwargs, "num_clients"):
        server_agent_config.server_configs.aggregator_kwargs.num_clients = num_clients
    # Create the server agent and communicator
    server_agent = ServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(
        comm, server_agent, logger=server_agent.logger
    )
    # Start the server to serve the clients
    server_communicator.serve()
else:
    # Set the client configurations
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.train_configs.logging_id = f"Client{rank}"
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
    client_agent_config.data_configs.dataset_kwargs.visualization = (
        True if rank == 1 else False
    )
    # Create the client agent and communicator
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(comm, server_rank=0)
    # Load the configurations and initial global model
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)
    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)
    # Send the sample size to the server
    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(
        action="set_sample_size", sample_size=sample_size
    )
    # Generate data readiness report
    if (
        hasattr(client_config, "data_readiness_configs")
        and hasattr(client_config.data_readiness_configs, "generate_dr_report")
        and client_config.data_readiness_configs.generate_dr_report
    ):
        data_readiness = client_agent.generate_readiness_report(client_config)
        client_communicator.invoke_custom_action(
            action="get_data_readiness_report", **data_readiness
        )

    round_num = 0
    # TODO seed randomizer depending on rank
    random.seed(rank)

    # Local training and global model update iterations
    while True:
        round_num += 1  # Increment round number (add this variable initialization before the loop)

        # TODO :D Introduce a delay based on the rank
        #delay = rank_delay_mapping_FEMNIST.get(rank, 0)  # Default to 0 seconds if rank is not in the mapping
        #if delay > 0:
            #time.sleep(delay)

        delay = random.uniform(0, 50)
        time.sleep(delay)

        # Training time
        start_train = time.perf_counter()
        client_agent.train()
        train_time = time.perf_counter() - start_train

        start_serialize = time.perf_counter()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}

        serialize_time = time.perf_counter() - start_serialize

        new_global_model, metadata = client_communicator.update_global_model(
            local_model, **metadata
        )
        if metadata["status"] == "DONE":
            break
        if "local_steps" in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]

        start_deserialize = time.perf_counter()
        client_agent.load_parameters(new_global_model)
        deserialize_time = time.perf_counter() - start_deserialize

        # Log computational times (example: to a file per client)
        with open(f'client_{rank}_log.txt', 'a') as f:
            f.write(
                f"Round: {round_num}, Train time: {train_time:.4f}s, Serialize time: {serialize_time:.4f}s, Deserialize time: {deserialize_time:.4f}s\n")

    client_communicator.invoke_custom_action(action="close_connection")

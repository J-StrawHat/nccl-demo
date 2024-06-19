# NCCL Multi-node Communication Demo without MPI

This project provides sample code demonstrating the use of [NVIDIA's NCCL](https://developer.nvidia.com/nccl) for multi-node communication without the need for [MPI](https://www.open-mpi.org/). The project includes both server and client programs that utilize socket communication to achieve distributed computing across multiple GPUs.

## Prerequisites

- NCCL 2.21.5
- CUDA 11.7
- Server node with at least 2 NVIDIA GPU
- Client node with at least 2 NVIDIA GPU

## Build

To build the project, run the following command:

```bash
make
```

## Usage

To run the server program, execute the following command on the server node: `./server <Server IP>`

```bash
$ ./server 192.168.0.208
Server listening on port 192.168.0.208:12345
NCCL unique ID, nRanks and base rank sent to clients
Init CUDA resources
Before AllReduce:
GPU[0]1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 
GPU[1]1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 
After AllReduce:
GPU[0]6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 
GPU[1]6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 
Server completed successfully
```

To run the client program, execute the following command on each client node: `./client <Server IP>`

```bash
$ ./client 192.168.0.208
Received NCCL unique ID, nRanks=4, baseRank=2
Init CUDA resources
Before AllReduce:
GPU[0]2.000000 2.000000 2.000000 2.000000 2.000000 2.000000 2.000000 2.000000 
GPU[1]2.000000 2.000000 2.000000 2.000000 2.000000 2.000000 2.000000 2.000000 
After AllReduce:
GPU[0]6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 
GPU[1]6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 6.000000 
Client completed successfully
```
#include "util.h"


int main(int argc, char *argv[]) {
    const char* server_ip = (argc > 1) ? argv[1] : SERVER_IP;
    int nRanks = GPUS_NUM * (CLIENTS_NUM + 1); // each node has same number of GPUs

    // Initialize server socket
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt))) {
        perror("setsockopt");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_port = htons(SERVER_PORT);
    if (inet_pton(AF_INET, server_ip, &address.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    printf("Server listening on port %s:%d\n", server_ip, SERVER_PORT);

    // Generating NCCL unique ID
    ncclUniqueId id;
    ncclGetUniqueId(&id);

    // Accepting connections from clients
    for (int i = 0; i < CLIENTS_NUM; ++i) {
        int clientBaseRank = GPUS_NUM + i * GPUS_NUM; //start behind server
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                                 (socklen_t *)&addrlen)) < 0) {
            perror("accept");
            close(server_fd);
            exit(EXIT_FAILURE);
        }
        // Sending NCCL unique ID, nRanks, and its base rank to clients
        send(new_socket, &id, sizeof(id), 0);
        send(new_socket, &nRanks, sizeof(nRanks), 0);
        send(new_socket, &clientBaseRank, sizeof(clientBaseRank), 0);
        close(new_socket);
    }
    printf("NCCL unique ID, nRanks and base rank sent to clients\n");

    float **sendbuff = (float **)malloc(GPUS_NUM * sizeof(float *));
    float **recvbuff = (float **)malloc(GPUS_NUM * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * GPUS_NUM);
    // Initialize CUDA resources
    float* tmpbuff = (float*)malloc(BUFFER_SIZE * sizeof(float));
    for (int i = 0; i < BUFFER_SIZE; i++) tmpbuff[i] = 1;

    for (int I = 0; I < GPUS_NUM; ++I) {
        CUDACHECK(cudaSetDevice(I));
        CUDACHECK(cudaMalloc(sendbuff + I, BUFFER_SIZE * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + I, BUFFER_SIZE * sizeof(float)));
        CUDACHECK(cudaMemcpy(sendbuff[I], tmpbuff, BUFFER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(recvbuff[I], 0, BUFFER_SIZE * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + I));
    }
    printf("Init CUDA resources\n");
    printf("Before AllReduce:\n");
    PrintBuffer((void**)sendbuff);

    ncclComm_t comms[GPUS_NUM];
    // Initializing NCCL
    NCCLCHECK(ncclGroupStart());
    for (int devID = 0; devID < GPUS_NUM; devID++) {
        CUDACHECK(cudaSetDevice(devID));
        NCCLCHECK(
            ncclCommInitRank(comms + devID, nRanks, id, devID));
    }
    NCCLCHECK(ncclGroupEnd());

    // NCCL communication
    NCCLCHECK(ncclGroupStart());
    for (int devID = 0; devID < GPUS_NUM; devID++) {
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[devID], (void *)recvbuff[devID],
                                BUFFER_SIZE, ncclFloat, ncclSum, comms[devID],
                                s[devID]));
    }
    NCCLCHECK(ncclGroupEnd());

    // Synchronizing on CUDA stream
    for (int devID = 0; devID < GPUS_NUM; devID++) {
        CUDACHECK(cudaStreamSynchronize(s[devID]));
    }
    printf("After AllReduce:\n");
    PrintBuffer((void**)recvbuff);

    // Freeing device memory
    for (int devID = 0; devID < GPUS_NUM; devID++) {
        CUDACHECK(cudaFree(sendbuff[devID]));
        CUDACHECK(cudaFree(recvbuff[devID]));
    }

    // Finalizing NCCL
    for (int devID = 0; devID < GPUS_NUM; devID++) {
        ncclCommDestroy(comms[devID]);
    }

    printf("Server completed successfully\n");
    close(server_fd);
    return 0;
}
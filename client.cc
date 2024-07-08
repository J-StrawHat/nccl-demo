#include "util.h"

int main(int argc, char *argv[]) {
    int version;
    NCCLCHECK(ncclGetVersion(&version));
    printf("NCCL version: %d\n", version);

    // Initialize client socket
    const char* server_ip = (argc > 1) ? argv[1] : SERVER_IP;
    int sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(SERVER_PORT);

    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    // Receiving NCCL unique ID from server
    ncclUniqueId id;
    int nRanks;
    int baseRank;
    read(sock, &id, sizeof(id));
    read(sock, &nRanks, sizeof(nRanks));
    read(sock, &baseRank, sizeof(baseRank));
    close(sock);
    printf("Received NCCL unique ID, nRanks=%d, baseRank=%d\n", nRanks, baseRank);

    // Initialize CUDA resources
    float **sendbuff = (float **)malloc(GPUS_NUM * sizeof(float *));
    float **recvbuff = (float **)malloc(GPUS_NUM * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * GPUS_NUM);
    float* tmpbuff = (float*)malloc(BUFFER_SIZE * sizeof(float));
    for (int i = 0; i < BUFFER_SIZE; i++) tmpbuff[i] = 2;

    for (int devID = 0; devID < GPUS_NUM; ++devID) {
        CUDACHECK(cudaSetDevice(devID));
        CUDACHECK(cudaMalloc(sendbuff + devID, BUFFER_SIZE * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + devID, BUFFER_SIZE * sizeof(float)));
        CUDACHECK(cudaMemcpy(sendbuff[devID], tmpbuff, BUFFER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(recvbuff[devID], 0, BUFFER_SIZE * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + devID));
    }
    printf("Init CUDA resources\n");
    printf("Before AllReduce:\n");
    PrintBuffer((void**)sendbuff);

    // Initializing NCCL
    ncclComm_t comms[GPUS_NUM];
    NCCLCHECK(ncclGroupStart());
    for (int devID = 0; devID < GPUS_NUM; devID++) {
        CUDACHECK(cudaSetDevice(devID));
        NCCLCHECK(ncclCommInitRank(comms + devID, nRanks, id, baseRank + devID));
    }
    NCCLCHECK(ncclGroupEnd());

    for (int devID = 0; devID < GPUS_NUM; devID++) {
        int count, device, userRank;
        NCCLCHECK(ncclCommCount(comms[devID], &count));
        NCCLCHECK(ncclCommCuDevice(comms[devID], &device));
        NCCLCHECK(ncclCommUserRank(comms[devID], &userRank));
        printf("comm: %p, CUDADevice: %d, userRank: %d, nRanks: %d\n", comms[devID], device, userRank, count);
    }

    // NCCL communication
    NCCLCHECK(ncclGroupStart());
    for (int devID = 0; devID < GPUS_NUM; devID++) {
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[devID], (void *)recvbuff[devID],
                                BUFFER_SIZE, ncclFloat, ncclSum, comms[devID],
                                s[devID]));
    }
    NCCLCHECK(ncclGroupEnd());

    // Synchronizing on CUDA stream
    for (int i = 0; i < GPUS_NUM; i++) {
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }
    printf("After AllReduce:\n");
    PrintBuffer((void**)recvbuff);

    // Freeing device memory
    for (int i = 0; i < GPUS_NUM; i++) {
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // Finalizing NCCL
    for (int devID = 0; devID < GPUS_NUM; devID++) {
        ncclCommDestroy(comms[devID]);
    }

    printf("Client completed successfully\n");
    return 0;
}
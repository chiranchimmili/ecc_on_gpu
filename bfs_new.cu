#include "bfs_new.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

int no_of_nodes;
int num_of_blocks, num_of_threads_per_block;
Node* d_graph_nodes;
int* d_graph_edges;
bool* d_graph_mask, *d_updating_graph_mask, *d_graph_visited;
int* d_cost;
bool* d_over;

void BFSGraph(int argc, char** argv) {
    char *input_f;
    if(argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        exit(0);
    }

    input_f = argv[1];
    printf("Reading File\n");

    // Read in Graph from a file
    FILE* fp = fopen(input_f, "r");
    if(!fp) {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0;
    fscanf(fp, "%d", &no_of_nodes);

    num_of_blocks = 1;
    num_of_threads_per_block = no_of_nodes;

    // Make execution Parameters according to the number of nodes
    // Distribute threads across multiple Blocks if necessary
    if(no_of_nodes > MAX_THREADS_PER_BLOCK) {
        num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }

    // allocate host memory
    Node* h_graph_nodes = (Node*)malloc(sizeof(Node) * no_of_nodes);
    bool* h_graph_mask = (bool*)malloc(sizeof(bool) * no_of_nodes);
    bool* h_updating_graph_mask = (bool*)malloc(sizeof(bool) * no_of_nodes);
    bool* h_graph_visited = (bool*)malloc(sizeof(bool) * no_of_nodes);

    int start, edgeno;
    // initialize the memory
    for(unsigned int i = 0; i < no_of_nodes; i++) {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i] = false;
        h_updating_graph_mask[i] = false;
        h_graph_visited[i] = false;
    }

    // read the source node from the file
    fscanf(fp, "%d", &source);
    source = 0;

    // set the source node as true in the mask
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;

    int edge_list_size;
    fscanf(fp, "%d", &edge_list_size);

    int* h_graph_edges = (int*)malloc(sizeof(int) * edge_list_size);
    for(int i = 0; i < edge_list_size; i++) {
        int id, cost;
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        h_graph_edges[i] = id;
    }

    fclose(fp);

    printf("Read File\n");

    // Copy the Node list to device memory
    cudaMalloc((void**)&d_graph_nodes, sizeof(Node) * no_of_nodes);
    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice);

    // Copy the Edge List to device Memory
    cudaMalloc((void**)&d_graph_edges, sizeof(int) * edge_list_size);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);

    // Copy the Mask to device memory
    cudaMalloc((void**)&d_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_updating_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    // Copy the Visited nodes array to device memory
    cudaMalloc((void**)&d_graph_visited, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);

    // allocate memory for the result on host side
    int* h_cost = (int*)malloc(sizeof(int) * no_of_nodes);
    for(int i = 0; i < no_of_nodes; i++)
        h_cost[i] = -1;
    h_cost[source] = 0;

    // allocate device memory for result
    cudaMalloc((void**)&d_cost, sizeof(int) * no_of_nodes);
    cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    // make a bool to check if the execution is over
    cudaMalloc((void**)&d_over, sizeof(bool));

    printf("Copied Everything to GPU memory\n");

    // setup execution parameters
    dim3 grid(num_of_blocks, 1, 1);
    dim3 threads(num_of_threads_per_block, 1, 1);

    int k = 0;
    printf("Start traversing the tree\n");
    bool stop;
    // Call the Kernel until all the elements of Frontier are not false
    do {
        // if no thread changes this value then the loop stops
        stop = false;
        cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);
        Kernel<<<grid, threads, 0>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);

        Kernel2<<<grid, threads, 0>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);

        cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
        k++;
    } while(stop);

    printf("Kernel Executed %d times\n", k);

    // copy result from device to host
    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    // Store the result into a file
    FILE* fpo = fopen("result.txt", "w");
    for(int i = 0; i < no_of_nodes; i++)
        fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
    fclose(fpo);
    printf("Result stored in result.txt\n");

    // cleanup memory
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_updating_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);
}

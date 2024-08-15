#ifndef BFS_KERNEL_H
#define BFS_KERNEL_H

#define MAX_THREADS_PER_BLOCK 512

struct Node {
    int starting;
    int no_of_edges;
};

// Function declarations
void BFSGraph(int argc, char** argv);

// // Kernel declarations
__global__ void Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes);
__global__ void Kernel2(bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes);

// External variables
extern int no_of_nodes;
extern int num_of_blocks, num_of_threads_per_block;
extern Node* d_graph_nodes;
extern int* d_graph_edges;
extern bool* d_graph_mask, *d_updating_graph_mask, *d_graph_visited;
extern int* d_cost;
extern bool* d_over;

#endif // BFS_KERNEL_H

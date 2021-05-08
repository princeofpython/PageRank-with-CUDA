#include<stdio.h>
#include <bits/stdc++.h>

using namespace std;

void get_adj_matrix(float** graph, int n, float d, FILE *inputFilePtr ){

    if ( inputFilePtr == NULL )  {
        printf( "input.txt file failed to open." );
        return ;
    }

    int m, indexing;
    
    fscanf(inputFilePtr, "%d", &m);
    fscanf(inputFilePtr, "%d", &indexing);

    
    for(int i = 0; i< n ; i++){
        graph[i] = (float*)malloc(sizeof(float)* n);
        for(int j = 0; j< n; ++j){
            graph[i][j] = (1 - d)/float(n);
        }
    }

    while(m--){
        int source, destin;
        fscanf(inputFilePtr, "%d", &source);
        fscanf(inputFilePtr, "%d", &destin);
        if (indexing == 0){
            graph[destin][source] += d* 1.0;
        }
        else{
            graph[destin - 1][source - 1] += d* 1.0;
        }
    }
}
void manage_adj_matrix(float** graph, int n){

    for(int j = 0; j < n; ++j){
        float sum = 0.0;

        for (int i = 0; i< n; ++i){
            sum += graph[i][j];
        }

        for (int i = 0; i < n; ++i){
            if (sum != 0.0){
                graph[i][j] /= sum;
            }
            else{
                graph[i][j] = (1/(float)n);
            }
        }
    }

}
float norm(float *vect, int n){
    float ans = 0.0;
    for (int i = 0; i < n; ++i){
        ans += abs(vect[i]);
    }
    return ans;
}

void power_method(float **graph, float *r, int n, int max_iter = 1000, float eps = 0.000001 ){
   
    float* r_last = (float*) malloc(n * sizeof(float));
    

    for(int i = 0; i< n; ++i){
        r[i] = (1/(float)n);
    }

    while(max_iter--){
        for(int i = 0; i< n; ++i){
            r_last[i] = r[i];
        }
        for(int i = 0; i< n; ++i){
            float sum = 0.0;

            for (int j = 0; j< n; ++j){
                sum += r_last[j] * graph[i][j];
            }

            r[i] = sum;

        }

        for(int i = 0; i< n; ++i){
            r_last[i] -= r[i];
        }

        if(norm(r_last, n) < eps){
            return;
        }

    }
    return;
}

void top_nodes(float *r, int n, int count = 10){

    priority_queue<pair<float, int>> pq;

    for(int i = 0; i< n; ++i){
        pq.push(make_pair(r[i], i+ 1));
    }
    int rank =1;
    while(rank <= count){
        printf("Rank %d Node is %d\n", rank, pq.top().second);
        rank++;
        pq.pop();
    }

}

int main(int argc, char** argv){

    clock_t start, end;

    FILE *inputFilePtr;

    char * inputfile = argv[1];
    inputFilePtr = fopen(inputfile, "r");

    int n; 
    fscanf(inputFilePtr, "%d", &n);

    float d = 0.85; 

    float** graph = (float**)malloc(n * sizeof(float*));


    float* r = (float*) malloc(n * sizeof(float));

   
    get_adj_matrix(graph, n, d, inputFilePtr);

    start = clock();

    manage_adj_matrix(graph, n);
    power_method(graph, r, n);
    top_nodes(r, n);

    end = clock();

    printf("Time taken :%f for sequential implementation with %d nodes.\n", float(end - start), n);
    return 0;
}
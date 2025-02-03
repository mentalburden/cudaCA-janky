#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX 99999
#define PORT 6379
#define RUNCOUNT 6000
#define EDGESIZE 512
#define SOCKY struct sockaddr


const int runcount = RUNCOUNT;
const int edgesize = EDGESIZE;
const int bordersize = 8;
const int redisbuffer = (edgesize*edgesize+200); //epoch size + extra for redis header


//----------------------------------------------------------------------
//utility functions -- host side

int redisSet(char* key, char* val)
{
        int sockfd;
        struct sockaddr_in servaddr;
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1)
        {
                printf("Cant build socket \n");
                exit(0);
        }
        else
        {
                bzero(&servaddr, sizeof(servaddr));
        }
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = inet_addr("172.16.0.2");
        servaddr.sin_port = htons(PORT);
        if (connect(sockfd, (SOCKY*)&servaddr, sizeof(servaddr)) != 0)
        {
                printf("Cant connect \n");
                exit(0);
        }

//---------------- janky way to build redis set tcp payload
		int keysize = strlen(key);
		char keysizeholder[123];
		int valsize = strlen(val);
		char valsizeholder[123];

		char header[42] = "*3\r\n$3\r\nset\r\n$";
		sprintf(keysizeholder, "%d", keysize);
		strcat(header, keysizeholder);
		strcat(header, "\r\n");

		char middle[42] = "\r\n$";
		sprintf(valsizeholder, "%d", valsize);
		strcat(middle, valsizeholder);
		strcat(middle, "\r\n");

		char end[42] =  "\r\n";
		int totalsize = (strlen(end) + strlen(middle) + strlen(header) + keysize + valsize);
		char buff[redisbuffer] = {};
		strcat(buff, header); //terrible and dangerous, fix this
		strcat(buff, key);
		strcat(buff, middle);
		strcat(buff, val);
		strcat(buff, end);
//------------------------------------------------------
	write(sockfd, buff, totalsize);
	close(sockfd);
	return 0;
}


double psudoRand(float randmax)
{
	return rand() % (int)randmax;
}

//----------------------------------------------------------------------
//ca grid -- device kernel -- this one is int based

__global__ void runCUDAModel(int gridedge, int *gridmem)
{
	int neighborcount;
	int dY = blockDim.y * blockIdx.y + threadIdx.y;
	int dX = blockDim.x * blockIdx.x + threadIdx.x;
	int dID = dY * (gridedge) + dX;
	if (dX > bordersize && dX < gridedge - (bordersize) && dY>bordersize && dY < gridedge - (bordersize))
	{
		neighborcount = (gridmem[dID+gridedge]) + (gridmem[dID-gridedge])
				+ (gridmem[dID+1]) + (gridmem[dID-1] )
				+ (gridmem[dID+(gridedge+1)] + gridmem[dID-(gridedge+1)] )
				+ (gridmem[dID+(gridedge-1)] + gridmem[dID-(gridedge-1)] );
		if (gridmem[dID] == 1 && neighborcount < 3)
		{
			gridmem[dID] = 0;
		}
		else if (gridmem[dID] == 1 && (neighborcount == 3 || neighborcount == 4))
		{
			gridmem[dID] = 1;
		}
		else if (gridmem[dID] == 1 && neighborcount  >= 7)
		{
			gridmem[dID] = 0;
		}
		else
		{
			gridmem[dID] = gridmem[dID];
		}
	}
}

//----------------------------------------------------------
//model handler -- host side

int runModel(int times, int edgesize)
{
	int *gridmem, *gridmemtemp, *devicegrid, *devicegridtemp;
	double *coinflip = (double*)malloc(sizeof(double));
	int gridsize = edgesize * edgesize;
	size_t gridmemsize = sizeof(int)*gridsize;
	gridmem = (int *)malloc(gridmemsize);
	gridmemtemp = (int *)malloc(gridmemsize);
	// build randomized init grid
	for (int x=0; x < edgesize; x++)
	{
		for (int y=0; y < edgesize; y++)
		{
			if (y>bordersize && x>bordersize && y<edgesize-bordersize && x<edgesize-bordersize)
			{
				*coinflip = psudoRand(13);
				gridmem[x*edgesize+y] = *coinflip;
			}
			else
			{
				gridmem[x*edgesize+y] = 0;
			}
		}
	}
	//device memory
	cudaMalloc(&devicegrid, gridmemsize);
	cudaMalloc(&devicegridtemp, gridmemsize);
	cudaMemcpy(devicegrid, gridmem, gridmemsize, cudaMemcpyHostToDevice);
	int threadchonk = (int)ceil(edgesize/32);
	dim3 blocky(edgesize, edgesize, 1);
	dim3 gridy(threadchonk, threadchonk, 1);
	for (int run = 0; run <= times; run++)
	{
		//fire kernel
		runCUDAModel<<<1, blocky>>>(edgesize, devicegrid);
		cudaDeviceSynchronize();
		cudaMemcpy(gridmemtemp, devicegrid, gridmemsize, cudaMemcpyDeviceToHost);
		char lilchonk[edgesize*edgesize] = {}; //use a char array to hold epoch state after firing
		for (int x=0; x < edgesize; x++)
		{
			for (int y=0; y < edgesize; y++)
			{
				if ((int)ceil(gridmemtemp[x*edgesize+y]) >= 1)
				{
					lilchonk[(x*edgesize+y)] = '1'; //push to char array
				}
				else
				{
					lilchonk[(x*edgesize+y)] = '0';
				}
			}
		}
		char runholder[42] = {};
		sprintf(runholder, "%d", run);
		redisSet(runholder, lilchonk); //send to redis
		printf("epoch %s sent to redis\n", runholder);
	}
	free(coinflip);
	free(gridmem);
	free(gridmemtemp);
	cudaFree(devicegrid);
	cudaFree(devicegridtemp);
	return 0;
}

int main()
{
	clock_t startcuda, stopcuda;
	double cudatotaltime;
	//init rand
	srand(time(NULL));
	//run ca model (times, size)
	startcuda = clock();
	runModel(runcount,edgesize);
	stopcuda = clock();
	cudatotaltime = ((double)(stopcuda-startcuda)) / CLOCKS_PER_SEC;
	printf("CUDA CA Model done in: %.3f seconds\n", cudatotaltime);
	return 0;
}

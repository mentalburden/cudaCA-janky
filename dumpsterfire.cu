#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>

//handy gif library from https://github.com/charlietangora/gif-h
#include "gif-h.h"
#include <math.h>

const char* fileobj = "testout.jank";
const char* giffile = "cudaCA.gif";
char* on = "1";
char* off = "0"; //add other named state types here later
const int runcount = 4000;
const int edgesize = 512;
static uint8_t imageholder[ edgesize * edgesize * 4 ];
char* bigchonk[(edgesize*edgesize)*runcount] = {};

//----------------------------------------------------------------------
//utility functions -- host side

int clearfile(const char *filename)
{
	FILE *thisfile;
	thisfile = fopen(filename, "w");
	fprintf(thisfile, "%s", "");
	fclose(thisfile);
	return 0;
}

void SetPixel(int x, int y, uint8_t red, uint8_t green, uint8_t blue )
{
	uint8_t* pixel = &imageholder[(x*edgesize+y)*4];
	pixel[0] = red;
	pixel[1] = green;
	pixel[2] = blue;
	pixel[3] = 255;
}

int makeGifMem(const char *outfile, int edgesize, int times)
{
	GifWriter writegif = {};
	GifBegin(&writegif, outfile, edgesize, edgesize, 2, 8, true );
	for (int run = 0; run < times; run++)
	{
		for (int x = 0; x < edgesize; x++)
		{
			for (int y = 0; y < edgesize; y++)
			{
				if (bigchonk[(x*edgesize+y)*run] == on)
				{
					SetPixel(x, y, 128, 0, 0);
				}
				else
				{
					SetPixel(x, y, 0, 0, 0);
				}
			}
		}
		GifWriteFrame(&writegif, imageholder, edgesize, edgesize, 6, 8, true);
		for (int clearout = 0; clearout < (edgesize*edgesize*4); clearout++)
		{
			imageholder[clearout] = 0;
		}
	}
	GifEnd(&writegif);
	return 0;
}

int makeGifFileObj(const char *readfile, const char *outfile, int edgesize, int times)
{
	//build gif from fileobj dump
	GifWriter writegif = {};
	GifBegin(&writegif, outfile, edgesize, edgesize, 2, 8, true );
	FILE *thisfile;
	size_t linebufsize = sizeof(char)*(edgesize*edgesize);
	char linebuf[linebufsize];
	thisfile = fopen(readfile, "r+");
	while (fgets(linebuf, linebufsize, thisfile) != NULL)
	{
		for (int x = 0; x < edgesize; x++)
		{
			for (int y = 0; y < edgesize; y++)
			{
				if (linebuf[x*edgesize+y] == '1') //if 1 pixel red
				{
					SetPixel(x, y, 128, 0, 0);
				}
				else
				{
					SetPixel(x, y, 0, 0, 0); //else pixel black
				}
			}
		}
		GifWriteFrame(&writegif, imageholder, edgesize, edgesize, 6, 8, true);
	}
	GifEnd(&writegif);
	return 0;
}

int fileappendstring(const char *filename, const char *payload)
{
	FILE *thisfile;
	thisfile = fopen(filename, "a+");
	fprintf(thisfile, "%s", payload);
	fclose(thisfile);
	return 0;
}

double psudoRand(float randmax)
{
	return rand() % (int)randmax;
}

//----------------------------------------------------------------------
//ca grid -- device kernel

__global__ void runModel(int gridedge, double *gridmem)
{
	double neighborcount;
	int dY = blockDim.y * blockIdx.y + threadIdx.y;
	int dX = blockDim.x * blockIdx.x + threadIdx.x;
	int dID = dY * (gridedge) + dX;
	if (dY>0 && dY < gridedge -2 && dX>0 && dX < gridedge -2)
	{
		neighborcount = (int)(gridmem[dID+(gridedge+2)]) + (gridmem[dID-(gridedge+2)] )
				+ (gridmem[dID+1]) + (gridmem[dID-1] )
				+ (gridmem[dID+(gridedge+2)] + gridmem[dID-(gridedge+3)] )
				+ (gridmem[dID+(gridedge+1)] + gridmem[dID-(gridedge+1)] );

		//need to turn into case switch and branch by start self state
		//floats are used to represent additional state types later on
		if (gridmem[dID] <= 0.6f && neighborcount < 3)
		{
			gridmem[dID] = 0.0f;
		}
		else if (gridmem[dID] >= 0.5f && (neighborcount <= 2 || neighborcount >= 5))
		{
			gridmem[dID] = 1.0f;
		}
//		else if (gridmem[dID] >= 1.0f && neighborcount >= 7)
//		{
//			gridmem[dID] = 0.0f;
//		}
		else if (gridmem[dID] <= 0.5f && neighborcount  == 3)
		{
			gridmem[dID] = 1.0f;
		}
		else
		{
			gridmem[dID] = 0.0f;
		}
	}
	else
	{
		//dead borders, not a torroid
		gridmem[dID] = 0.0f;
	}
}

//----------------------------------------------------------
//model handler -- host side

int runModel(const char *dumpfile, int times, int edgesize)
{
	double *gridmem, *gridmemtemp, *devicegrid, *devicegridtemp;
	double *coinflip = (double*)malloc(sizeof(double));
	int gridsize = edgesize * edgesize;
	size_t gridmemsize = sizeof(double)*gridsize;
	gridmem = (double *)malloc(gridmemsize); //use double to easily add more states
	gridmemtemp = (double *)malloc(gridmemsize);
	// build randomized init grid
	for (int x=0; x < edgesize; x++)
	{
		for (int y=0; y < edgesize; y++)
		{
			*coinflip = psudoRand(13.0f);
			gridmem[x*edgesize+y] = *coinflip;
		}
	}
	cudaMalloc(&devicegrid, gridmemsize);
	cudaMalloc(&devicegridtemp, gridmemsize);
	cudaMemcpy(devicegrid, gridmem, gridmemsize, cudaMemcpyHostToDevice);
	dim3 blocky(32,32,1);
	int threadchonk = (int)ceil(edgesize/32);
	dim3 gridy(threadchonk, threadchonk, 1);
	for (int run = 0; run <= times; run++)
	{
		runModel<<<gridy, blocky>>>(edgesize, devicegrid);
		cudaDeviceSynchronize();
		cudaMemcpy(gridmemtemp, devicegrid, gridmemsize, cudaMemcpyDeviceToHost);
		for (int x=0; x < edgesize; x++)
		{
			for (int y=0; y < edgesize; y++)
			{
				if ((int)ceil(gridmemtemp[x*edgesize+y]) >= 1)
				{
					//printf("\033[48:2::128:0:0m \033[49m");
					bigchonk[(x*edgesize+y)*run] = on;
				}
				else
				{
					//printf("\033[48:2::0:0:0m \033[49m");
					bigchonk[(x*edgesize+y)*run] = off;
				}
			}
		}
	}
	//printf("\e[1;1H\e[2J"); //clear screen
	free(coinflip);
	free(gridmem);
	free(gridmemtemp);
	cudaFree(devicegrid);
	cudaFree(devicegridtemp);
	return 0;
}

int main()
{
	clock_t startcuda, stopcuda, starthost, stophost;
	double cudatotaltime, hosttotaltime;
	//init rand
	srand(time(NULL));
	//run ca model (times, size)
	startcuda = clock();
	runModel(fileobj, runcount,edgesize);
	stopcuda = clock();
	cudatotaltime = ((double)(stopcuda-startcuda)) / CLOCKS_PER_SEC;
	printf("CUDA CA Model done in: %.3f seconds\n", cudatotaltime);
	//make gif
	starthost = clock();
	makeGifMem(giffile, edgesize, runcount);
	stophost = clock();
	hosttotaltime = ((double)(stophost-starthost)) / CLOCKS_PER_SEC;
	printf("Wrote Gif with cpu in: %.3f seconds\n", hosttotaltime);
	return 0;
}

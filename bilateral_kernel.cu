#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

using namespace std;
using namespace cv;

float cpuGaussian[64];

__constant__ float cGaussian[64];

texture<unsigned char, 2, cudaReadModeElementType> texRef;

void updateGaussian(int radius, double delta) {

	float fGaussian[64];
	for (int i = 0; i < 2 * radius + 1; i++) {
		float x = i - radius;
		fGaussian[i] = expf(-(x*x) / (2 * delta * delta));
	}
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2 * radius + 1));
}

// CPU
void updateGaussianCPU(int radius, float delta) {

	for (int i = 0; i < 2 * radius + 1; i++) {
		int x = i - radius;
		cpuGaussian[i] = exp(-(x * x) / (2 * delta * delta));
	}
}

__device__ inline double gaussian(float x, double sigma) {
	return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

// CPU
float euclideanDistance(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2)));
}

__global__ void gpuCalculation(unsigned char* input, unsigned char* output, int width, int height, 
	int r, double euclidean_delta)
{
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	if ((x < width) && (y < height)) {
		double t = 0;
		double sumFactor = 0;
		unsigned char center = tex2D(texRef, x, y);

		for (int i = -r; i <= r; i++) {
			for (int j = -r; j <= r; j++) {

				unsigned char curPix = tex2D(texRef, x + j, y + i);

				double factor = (cGaussian[i + r] * cGaussian[j + r]) * gaussian(center - curPix, euclidean_delta);

				t += factor * curPix;
				sumFactor += factor;
			}
		}
		output[y*width + x] = t / sumFactor;
	}
}

void bilateralFilter(const Mat & input, Mat & output, int filter_radius, double euclidean_delta) {
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int gray_size = input.step*input.rows;

	size_t pitch;
	unsigned char *d_input = NULL;
	unsigned char *d_output;

	updateGaussian(filter_radius, euclidean_delta);

	// Allocate device memory
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char)*input.step, input.rows);
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char)*input.step, sizeof(unsigned char)*input.step, input.rows, cudaMemcpyHostToDevice);
	cudaBindTexture2D(0, texRef, d_input, input.step, input.rows, pitch);
	cudaMalloc<unsigned char>(&d_output, gray_size);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	cudaEventRecord(start, 0);

	gpuCalculation <<<grid, block>>> (d_input, d_output, input.cols, input.rows, filter_radius, euclidean_delta);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);

	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventElapsedTime(&time, start, stop);
	cout << "Time on GPU: " << time << " milliseconds" << endl;
}

// CPU
void bilateralFilterGold(unsigned char *input, unsigned char *output, float euclidean_delta, int width, int height, int r) {
	updateGaussianCPU(r, euclidean_delta);

	float domainDist, colorDist, factor;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)	{
			float t = 0.0f;
			float sum = 0.0f;

			for (int i = -r; i <= r; i++) {
				int neighborY = y + i;
				if (neighborY < 0) {
					neighborY = 0;
				} else if (neighborY >= height) {
					neighborY = height - 1;
				}
				for (int j = -r; j <= r; j++) {
					domainDist = cpuGaussian[r + i] * cpuGaussian[r + j];

					int neighborX = x + j;

					if (neighborX < 0) {
						neighborX = 0;
					} else if (neighborX >= width) {
						neighborX = width - 1;
					}
					colorDist = euclideanDistance(input[neighborY * width + neighborX] - input[y * width + x], euclidean_delta);
					factor = domainDist * colorDist;
					sum += factor;
					t += factor * input[neighborY * width + neighborX];
				}
			}
			output[y * width + x] = t / sum;
		}
	}
}

int main()
{
	Mat input = imread("image.bmp", IMREAD_GRAYSCALE);

	Mat output_own(input.rows, input.cols, CV_8UC1);
	Mat output_cpu(input.rows, input.cols, CV_8UC1);
	bilateralFilter(input, output_own, 7, 100.0);

	clock_t start_s = clock();
	bilateralFilterGold(input.ptr(), output_cpu.ptr(), 100.0, input.rows, input.cols, 7);
	clock_t stop_s = clock();
	cout << "Time on CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " milliseconds" << endl;

	imwrite("result.bmp", output_cpu);
	getchar();
	return 0;
}
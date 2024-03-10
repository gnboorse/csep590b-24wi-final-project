# csep590b-24wi-final-project

sDTW implementaion in AMD ROCm HIP. This repository is configured for analysing the throughput of this implementaiton in a AMD GPU. The steps for analysing the throughput is given below.
It recommented to run it 12 times, by considering the first 2 runs as a warmup of the GPU and take the average of the throughput of the remaining 10 runs. Apart form anlalysing the throughput, we can also test each core components of this project. The explanation and the steps for testing the core components are also given below. 

## Steps for analysing the throughput

### Prepare for the DTW execution. 
Before the signal alignment processing with DTW, we need to create 512 random query strings and the normalized reference string. The below `prepare` step will do all that.
```
./prepare
```
### Running the project N time(Eg :12 ) 
Run & Print time taken by the normalization and DTW in seconds

```
./analyse_throughput 12
```

## Core components
### Normalizer kernel
The normalizer kernel runs in two passes:

1. The first pass will calculate the sum of all elements in the input query, as well as the sum of all squared elements in the input query. 
  The result of this first pass is coalesced on the host and the final sigma (variance) and mu (mean) values are copied to constant memory
  on the device. This pass uses a list reduction algorithm for adding the numbers together.
2. The second pass will then use the mean and variance to normalize the input query data. This pass will simply process some N number of
  values of the input query per thread and write them to the normalized output buffer.

To build the normalizer, run: 
```sh
./build test-components/normalizer.cu normalizer-test
```
To run it, simply run
```
./project_runner normalizer-test
```
The output should look something like:

```
Creaated 512random querys of length 2000 at test-normalizer1710049216query.raw
Number of Queries : 512
Size of each Querie : 2000
mu: 88.8045, sigma: 19.1511
...
...
mu: 88.171, sigma: 19.0935
mu: 89.088, sigma: 19.1927
mu: 88.7375, sigma: 19.1846
normalized 2000 values
```

The 512 data to normalize is generated / randomized and stored in the temporary folder `query-string` in the name `query.raw` 
The normalized output is stored in `normalized_query.raw` in the same directory.



### DTW Kernal
The DTW kernal is created independently in the `test-components/dtw.cu` file and also created a dedicated test(`test-components/test-dtw`) and test runner(`run_tests_devel_dtw`) for it.
#### Testing the DTW kernal
1. Build the `dtw.cu` file
```
./build test-components/dtw.cu dtw_test
```
2. Run the tests on `dtw_test`
```
sbatch run_tests_devel_dtw dtw_test
```
3. The output can be seen in the slurm-xxx.out file created. Eg: `slurm-27634.out`, its content will looks like
```
Passed test-components/test-dtw/0!
1/1 tests passed
```
4. Also the generated DTW out put metrix can be seen in each test folder. Eg : `test-components/test-dtw/0/attempt.raw` 



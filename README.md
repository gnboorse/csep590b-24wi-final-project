# csep590b-24wi-final-project

The implemntation for Normalizer and DTW are handled independently.  
Once we complete the implementations for Normalizer & DTW, both will be added to the main file `starter.cu`.

## Normalizer Kernel
The normalizer kernel runs in two passes:

1. The first pass will calculate the sum of all elements in the input query, as well as the sum of all squared elements in the input query. 
  The result of this first pass is coalesced on the host and the final sigma (variance) and mu (mean) values are copied to constant memory
  on the device. This pass uses a list reduction algorithm for adding the numbers together.
2. The second pass will then use the mean and variance to normalize the input query data. This pass will simply process some N number of
  values of the input query per thread and write them to the normalized output buffer.

The data to normalize is generated / randomized and stored in `test-normalizer/<unix-date>/query.raw` at the beginning of the run. 
The normalized output is stored in `normalized_query.raw` in the same directory.

To build the normalizer, run: 
```sh
# remove the previously-built libwb because of changes made to libwb. 
# this is only required the first time you build the normalizer.
rm -rf libwb/build 

./build normalizer.cu normalizer
```
To run it, simply run
```
./normalizer
```
The output should look something like:

```
[student08@login1 final-project-sdtw]$ ./normalizer
mu: 89.956, sigma: 19.6125
normalized 2000 values
```


## DTW Kernal
The DTW kernal is created independently in the `dtw.cu` file and also created a dedicated test(`test-dtw`) and test runner(`run_tests_devel_dtw`) for it.
### Testing the DTW kernal independently
1. Build the `dtw.cu` file
```
./build dtw.cu dtw_test
```
2. Run the tests on `dtw_test`
```
sbatch run_tests_devel_dtw dtw_test
```
3. The output can be seen in the slurm-xxx.out file created. Eg: `slurm-27634.out`
4. Also the generated DTW out put metrix can be seen in each test folder. Eg : `test-dtw/0/attempt.raw ` 

### Open Issues
1. Its a naive implementation of DTW, so we need to optimize it by using constant memory and possible tiling. 
2. Need to create the necessary tooling for capturing it through-put



## Test run Steps

### Prepare for the DTW execution. 
Before the signal alignment processing with DTW execution, we need to create the 512 random query strings and the normalized reference string. The below step will do all that.
```
./prepare
```
### Running the project N time(Eg :3 ) and print time take in seconds

```
./analyse_throughput 3
```

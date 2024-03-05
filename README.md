# csep590b-24wi-final-project

The implemntation for Normalizer and DTW are handled independently.  
Once we complete the implementations for Normalizer & DTW, both will be added to the main file `starter.cu`.

## Normalizer Kernal
*ToDo*
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
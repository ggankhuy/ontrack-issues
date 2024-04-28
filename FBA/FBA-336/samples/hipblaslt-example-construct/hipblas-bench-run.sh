HIPBLASLT_BENCH=~/extdir/gg/git/fba/FBA/FBA-336/samples/hipblaslt/hipBLASLt/build/release/clients/staging/hipblaslt-bench
$HIPBLASLT_BENCH -m 2304 -n 16384 -k 16384 --compute_type f16_r  --scale_type f32_r --a_type bf16_r --b_type bf16_r --c_type bf16_r

#Invalid value for --compute_type f16_r
# may be due to --compute_type does not support f16_r? 
# hipblaslt-bench --help | grep type: 
# --compute_type <value>     Precision of computation. Options: s,f32_r,x,xf32_r,f64_r,i32_r                     (Default value is: f32_r)

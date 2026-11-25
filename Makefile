# Top-level shortcuts (run from the repository root)
PY      ?= python3
CXX     ?= g++
RTL      = rtl/loeffler_dct8.v rtl/tsc_unit.v rtl/transpose_buffer.v rtl/dct2d_controller.v rtl/dct2d_top.v

.PHONY: smoke test vectors figures rtl rtl-vcd hls-csim hls matlab synth eval clean

smoke:                 ## every available implementation against the golden vectors
	bash scripts/smoke_test.sh

test:                  ## Python unit tests
	$(PY) -m pytest -q python/tests

vectors:               ## regenerate vectors/ from the Python model
	$(PY) python/scripts/gen_vectors.py

rtl:                   ## Icarus Verilog regression
	bash scripts/run_rtl_sim.sh

rtl-vcd:               ## regression + waveform dump build/tb_dct2d_top.vcd
	DCT_VCD=1 bash scripts/run_rtl_sim.sh

figures: rtl-vcd       ## all figures in docs/figures (needs iverilog for the waveform)
	$(PY) python/scripts/make_figures.py --vcd build/tb_dct2d_top.vcd

hls-csim:              ## HLS C-simulation with g++
	@mkdir -p build
	$(CXX) -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label -Ihls \
	    hls/dct2d_loeffler.cpp hls/tb_dct2d_loeffler.cpp -o build/hls_csim
	./build/hls_csim vectors

hls:                   ## Vitis HLS: csim + csynth (DCT_COSIM=1 / DCT_EXPORT=1 for more)
	vitis_hls -f hls/run_hls.tcl

matlab:                ## MATLAB/Octave model against the vectors
	octave --no-gui --quiet --eval "cd matlab; smoke_test_matlab"

synth:                 ## Vivado out-of-context synthesis + implementation on Artix-7
	vivado -mode batch -source rtl/synth/synth_artix7.tcl

eval:                  ## image-quality and arithmetic table
	$(PY) python/scripts/evaluate.py

clean:
	rm -rf build .pytest_cache python/.pytest_cache xsim.dir .Xil *.jou *.log *.pb

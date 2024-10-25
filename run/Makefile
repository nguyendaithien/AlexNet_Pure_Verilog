all:
	python3 testbench_gen.py
	xrun *.v -access +rwc
	python3 convert.py

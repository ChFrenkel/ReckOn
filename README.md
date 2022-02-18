# ReckOn: A Spiking RNN Processor Enabling On-Chip Learning over Second-Long Timescales

ReckOn is a spiking **rec**urrent neural network (RNN) processor enabling **on**-chip learning over second-long timescales based on a modified version of the e-prop algorithm (we released a PyTorch implementation of the vanilla e-prop algorithm for leaky integrate-and-fire neurons [here](https://github.com/ChFrenkel/eprop-PyTorch)). It was prototyped and measured in 28-nm FDSOI CMOS at the Institute of Neuroinformatics, University of Zurich and ETH Zurich, and published at the 2022 *IEEE International Solid-State Circuits Conference (ISSCC)* with the following three main claims:

* ReckOn demonstrates **end-to-end on-chip learning over second-long timescales** while keeping a milli-second temporal resolution,
* it provides a low-cost solution with a 0.45-mm² core area, 5.3pJ/SOP at 0.5V, and a **memory overhead of only 0.8%** compared to the equivalent inference-only network,
* it exploits a **spike-based representation for task-agnostic learning** toward user customization and chip repurposing at the edge.

In case you decide to use the ReckOn HDL source code for academic or commercial use, we would appreciate it if you let us know; **feedback is welcome**.


## Citation

Upon usage of the HDL source code of ReckOn, please cite the associated paper:

> [C. Frenkel and G. Indiveri, "ReckOn: A 28nm sub-mm² task-agnostic spiking recurrent neural network processor enabling on-chip learning over second-long timescales," *IEEE International Solid-State Circuits Conference (ISSCC)*, 2022]


## Documentation

Documentation on the contents, usage and features of the ReckOn HDL source code can be found in the [doc folder](doc/).


## Licenses

> *Copyright (C) 2020-2022 University of Zurich*

> *The HDL source code of ReckOn is under a Solderpad Hardware License v2.1 (see [LICENSE](LICENSE) file or https://solderpad.org/licenses/SHL-2.1/).*

> *The documentation of ReckOn is under a Creative Commons Attribution 4.0 International License (see [doc/LICENSE](doc/LICENSE) file or http://creativecommons.org/licenses/by/4.0/).*

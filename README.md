# Automatic Design Space Algorithm for DNN Accelerators Under Total Resources Constraint
Akarsh Kumar, Elyssa Hofgard, and Omri Lev

### Project Abstract
We study the problem of efficient space and memory allocation design within a Deep Neural Network (DNN) accelerator under a fixed total number of design elements (i.e., a fixed total number of Processing Elements (PEs) and kilobytes of memory). We present a unique design metric, which we optimize by utilizing derivative-free numerical optimization algorithms. We use the Timeloop and the Accelergy tools to evaluate the performance of our optimization algorithm when applied to the Eyeriss architecture with the VGG16 network.

### System Requirements
In order to run the example, you need to be inside a docker with the installed tools (given in `docker-compose.yaml`) or install Timeloop/Accelergy. 

### Repository Organization:
- `genetic_algo_quadratic.py` contains a toy example of the genetic algorithm just using random numbers so the user can explore how it works.
- Workspace contains the main files necessary for replicating our results. The example_designs folder contains the Eyeriss-like design for simulation with Timeloop/Accelergy. The layer_shapes folder includes the VGG16 model. `genetic_algo_timeloop.py` contains the interface and genetic algorithm for optimizing hardware parameters.

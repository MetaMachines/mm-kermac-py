# mm-kermac
> Dynamically compiled hyper semirings for Pytorch using PTX Inject and Stack PTX

This repo provides routines for Semiring and Semiring gradient Tensor operations for PyTorch. It also provides a DSL for writing your own custom Semiring and Semiring gradient routines that may include hyperparameters passed in to the kernel. These hyperparameters can either be single values tensor, single value tensors broadcast to a batch of tensors or a vector of batched hyperparameters applied to a batch of tensors.

## Installation
`mm-kermac` only supports Nvidia cards with `sm_80` or greater:
* For server cards A100 or greater, i.e. A10, H100, B100, BH200
* For consumer cards 3000 series or greater, i.e. 3070, 3090, 4090, 5090

To install, depending on your cuda toolkit version do one of these:
```bash
pip install mm-kermac[cu11]
```
```bash
pip install mm-kermac[cu12]
```
```bash
pip install mm-kermac[cu13]
```
## mm-ptx
This repo relies on [mm-ptx](https://github.com/MetaMachines/mm-ptx-py) for the implemented routines custom routines the user might implement themselves. Please see the repo for details on how `Stack PTX` works, how to use it and simplified examples for using the system.

## TODO
    * Better Docs
    * Benchmarks

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this software in your work, please cite it using the following BibTeX entry (generated from the [CITATION.cff](CITATION.cff) file):
```bibtex
@software{Durham_mm-kermac_2025,
  author       = {Durham, Charlie},
  title        = {mm-kermac: Dynamically compiled hyper semirings for Pytorch using PTX Inject and Stack PTX},
  version      = {0.1.2},
  date-released = {2025-10-19},
  url          = {https://github.com/MetaMachines/mm-kermac-py}
}
```

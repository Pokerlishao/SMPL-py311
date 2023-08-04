 # SMPL Based on Python3.11

This repo is able to load both v1.0.0 and v1.1.0 SMPL models. And you can download the SMPL models in [here](https://smpl.is.tue.mpg.de/).

Support for exporting **OBJ** and **FBX** files

This repo only relies on the following:

 - Python = 3.11
 - scipy = 1.11.1
 - numpy = 1.25.2
 - pyassimp = 5.2.5
 - pytorch = 2.0.1
 - CUDA = 11.8

 ## Perpare
 The `pyassimp` relies on compiled dynamic libraries, and you can run the following command in ubuntu.
```bash
 sudo apt-get install libassimp-dev
```
 In windows, you may need to go to [Assimp Github](https://github.com/assimp/assimp) and download the source code to compile it yourself.

 ## RUN

```bash
python pytorch_main.py
```
or
```bash
python numpy_main.py
```

 ## Todo

- [x] Add a Pytorch implementation
- [ ] Add UV texture mapping
- [ ] Add animation support
- [ ] Save joint data


 ## References

- https://github.com/CalciferZh/SMPL
- https://smpl.is.tue.mpg.de/


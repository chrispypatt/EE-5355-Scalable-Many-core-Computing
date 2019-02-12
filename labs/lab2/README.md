# Lab 2: 7-point Stencil with Thread-coarsening and Register Tiling

The purpose of this lab is to practice thread coarsening and register tiling op-timization techniques using 7-point stencil computation as an example.

## Running code:
```
make
./stencil                #Cubesize = default (512x512x64) 
./stencil <NX> <NY> <NZ> #Cubesize= NX x NY x NZ
```

# UA_GIR_Mani_25-26_P02
Repository to hold the code used to train a RL for the LunarLander problem

-Prepare conda environment:
```cmd
conda create -n mani python=3.11.14 && conda activate mani
```

-Complete environment with all the needed upgrades and packages:
```cmd
pip install --upgrade -y pip setuptools wheel && pip install -y gymnasium "gymnasium[classic_control]" && conda install swig && pip install -y "gymnasium[box2d]" matplotlib
```

-Set conda environment to be added to jupyter notebook:
```cmd
pip install --user ipykernel && python -m ipykernel install --user --name=mani
```

-(DON'T USE, NEEDS MORE TESTING) To uninstall env from jupyter notebook:
```cmd
jupyter kernelspec uninstall mani && conda deactivate && conda remove -n mani --all
```

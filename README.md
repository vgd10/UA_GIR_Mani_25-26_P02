# UA_GIR_Mani_25-26_P02
Repository to hold the code used to train a RL for the LunarLander problem

-Prepare conda environment:
```cmd
conda create -n mani python=3.11.14 && conda activate mani
```

-Complete environment with all the needed upgrades and packages:
```cmd
pip install --upgrade pip setuptools wheel && pip install gymnasium "gymnasium[classic_control]" && conda install swig && pip install "gymnasium[box2d]" matplotlib
```

-Set conda environment to be added to jupyter notebook:
```cmd
python -m ipykernel install --user --name=mani && pip install --user ipykernel 
```

-To uninstall env from jupyter notebook:
```cmd
jupyter kernelspec uninstall mani
```

"""Isolated categorical private-head pilot entry point."""
import pathlib,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(pathlib.Path.home()/"Documents/objective_power_grid/src/research/moe_sboed_categorical_pilot_20261116"))
from run import main
if __name__=="__main__":main()

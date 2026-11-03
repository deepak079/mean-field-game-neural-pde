"""PINN solvers: neural networks trained to satisfy the equations of the game.

hjb.py holds the value network (the HJB for a fixed final mean); fokker_planck.py the density
network (the Fokker-Planck equation under a given position) and the position network it uses;
nets.py the pieces they share.
"""

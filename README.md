# MLBraneWebs

Description:

In these files we study the classification of 5d Superconformal Field Theories arising from brane webs in Type IIB String Theory, using a Siamese Neural Network to identify different webs giving rise to the same theory. We consider two datasets of brane webs with three external legs: one with classes defined under weak equivalence and the other defined under strong equivalence, where weak and strong equivalence are defined as 
Strong equivalence: two webs are strongly equivalent if they can be transformed into each other by means of any combination of SL(2,Z) and HW moves.
Weak equivalence: two webs are weakly equivalent if they have the same number of 7-branes, asymptotic charge invariant and total monodromy up to SL(2,Z).

How to run:

The 2x3 web matrices describing the brane webs of the two weakly and strongly equivalent datasets are saved in the files 3leg_data_X.db and 3leg_data_Y.db respectively.
The SNN can be trained and tested by running the SNN.py file and the TDA can be obtained by running the TDA.py file. 

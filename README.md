# phylo_likelihood_bayesian
Python files that generate trees, caculate the likelihood + MCMC files + representation files \
230612_i0change.py: A python code to generate tree based on initial infected (i0), calculate likelihood, run MCMC \
plotting_sampled.py: based on 230612_i0change.py to have sampled as a trajectory or on the infected line \
230630_simTree.py: a script to simulate a tree. It doesn’t have class and it is written to be called from bash with this command: \
chmod +x /Users/siavashriazi/Desktop/SFU/Codes/Python/simTree_230630.py \
Loop: \
sum=0 \
for i in {1..100}; do \
    size=$(python simTree_230630.py) \
    sum=$(awk "BEGIN {print $sum + $size; exit}") \
done \
mean=$(awk "BEGIN {print $sum / 100; exit}") \
echo "Mean Tree Size: $mean" \
230709_mhChain.py: A script that has a class for theta and a function for mhChian that runs a chain. This script get the tree values from 230630_simTree.py. \
230712_chainMixing.py: A script that takes excel files from 230709_mhChain.py, mixes them, calculated kernel density and plot the final results. \
230725_mhChain.py: modified version of 230709_mhChain to run on clusters. \
230827_mhChain.py: updating likelihood calculation from 230825_likelihood_test.py. The result of posterior is not better than old version. \
230830_mhChain_1parm.py and 23030_chainMixing_1parm.py: updated version of 230827_mhChain.py and 230712_chainMixing.py for one parameter.







mkdir -p results

python experiments_fig1.py &> exp_fig1.log
python exp_small_linreg.py &> exp_small_linreg.log
python exp_small_logreg.py &> exp_small_logreg.log
python exp_big_logreg.py &> exp_big_logreg.log

python fig1_plot.py
python main_plot.py
python appendix_plots.py
python legend_plot.py
